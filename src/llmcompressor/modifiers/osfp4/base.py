# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

from dataclasses import dataclass
from typing import Literal

import torch
from compressed_tensors.offload.dist_utils import is_distributed
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationConfig,
    QuantizationStrategy,
    preset_name_to_scheme,
)
from compressed_tensors.registry import standardize_lookup_name
from compressed_tensors.utils import (
    match_modules_set,
    match_named_modules,
    update_offload_parameter,
)
from loguru import logger
from pydantic import Field, PrivateAttr, field_validator
from torch.nn import Linear, Module
from torch.utils._pytree import tree_leaves
from tqdm import tqdm

from llmcompressor.core import Event, State, active_session
from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag
from llmcompressor.modeling.moe.linearize import (
    get_linearized_moes,
    get_non_linearized_moes,
)
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import update_qparams
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.modifiers.transform.smoothquant.dynamic_mappings import (
    get_layer_mappings_from_model,
)
from llmcompressor.observers.min_max import StaticMinMaxObserver
from llmcompressor.utils.pytorch.module import get_module_to_name_dict

from .activation_subsampling import count_calibration_tokens, subsample_activations
from .calibration_cache import OSFP4CalibrationCache
from .optimization import RTNQuantizationResult, SICQuantizationResult
from .osfp4_quantize import quantize_mapping
from .smoothing import (
    deploy_mapping_smooth_quant_scale,
    update_smoothed_input_global_scale,
)
from .transforms import initialize_diagonal_smoothing

__all__ = ["OSFP4Mapping", "OSFP4Modifier"]

_OSFP4_OBSERVER = "osfp4"

# Relative targets, resolved within each expert in this order.
_EXPERT_BALANCE_GROUPS = (
    ("gate_proj", "up_proj"),
    ("down_proj",),
)
_OPTIONAL_EXPERT_PROJECTIONS = frozenset({"gate_proj"})


@dataclass(frozen=True, slots=True)
class OSFP4Mapping:
    """One complete alpha optimization and SmoothQuant deployment unit."""

    mapping_name: str
    smooth_layer: Module | None
    balance_layers: tuple[Linear, ...]
    is_expert: bool = False

    @property
    def requires_runtime_smoothing(self) -> bool:
        """Return whether the mapping needs an input transform because it has no smooth
        layer.
        """
        return self.smooth_layer is None


class OSFP4Modifier(Modifier, QuantizationMixin):
    """Apply OSFP4 optimization to NVFP4 or NVFP4A16 quantization using RTN or SIC.

    Each mapping is optimized jointly. It contains Linear layers sharing a
    preceding smooth layer, or runtime input transforms. Expert gate/up
    projections share one mapping; each expert down projection has its own.
    LLMC controls whether experts observe every token or only their routed tokens.

    Lifecycle:
        1. on_initialize:
           Validate the setup, resolve smooth-layer and runtime mappings,
           attach permanent input transforms where needed, and initialize
           quantization.

        2. on_calibration_start:
           Start observers, count tokens for activation subsampling when
           possible, and register temporary hooks to capture batch activations.

        3. on_sequential_epoch_end:
           Validate sampling counts before optimizing any mapping. For each
           observed mapping that has not been optimized, prepare optimization
           inputs, run RTN or SIC, apply smoothing and quantization parameters,
           and clear its cached data after successful deployment.

        4. on_calibration_end:
           Remove capture hooks, verify every mapping was optimized, update
           optional KV-cache parameters, and freeze quantization.

        5. on_finalize:
           Verify completion and clear the remaining calibration cache and
           mapping references. Keep deployed weights and input transforms.

    Every calibration token contributes to the RTN or SIC statistics.
    NVFP4 also caches activation values to evaluate reconstruction loss during
    optimization. When subsampling is enabled, capture stores only the selected
    token rows where possible. Routed experts use a bounded reservoir because
    their token counts are unknown in advance. Other unsupported capture paths
    store all rows and select the subset before optimization. NVFP4A16 uses
    statistics without caching activation values.

    Smoothing multiplies each mapping's input channels by the deployed scale
    and divides the corresponding weight columns in every balance layer by
    the same scale. The input multiplication is implemented by updating the
    preceding smooth layer's parameters or through a permanent runtime input
    transform. The updated parameters and runtime transforms remain in the
    model for inference and checkpoint export.
    """

    # Configuration

    # Calibration is always required, so pipeline inference selects sequential.
    requires_calibration_data: Literal[True] = True
    optimization_mode: Literal["rtn", "sic"] = "sic"
    steps: int = Field(default=80, ge=0)
    lr: float = Field(default=0.12, gt=0, allow_inf_nan=False)
    dampening_frac: float = Field(default=0.01, ge=0, lt=1, allow_inf_nan=False)
    offload_hessians: bool = False
    activation_subsample_size: int | Literal["auto"] | None = "auto"

    # Internal state

    _resolved_mappings: list[OSFP4Mapping] = PrivateAttr(default_factory=list)
    _calibration_dataloader: object = PrivateAttr(default=None)
    _calibration_cache: OSFP4CalibrationCache = PrivateAttr(default_factory=OSFP4CalibrationCache)
    _optimized_mapping_names: set[str] = PrivateAttr(default_factory=set)
    _finalization_complete: bool = PrivateAttr(default=False)
    _failed_deployment_mapping_name: str | None = PrivateAttr(default=None)
    _weight_only: bool = PrivateAttr(default=False)
    _activation_subsampling_records: dict[str, dict[str, int | str]] = PrivateAttr(default_factory=dict)

    @field_validator("activation_subsample_size")
    @classmethod
    def _validate_activation_subsample_size(cls, value):
        """Reject nonpositive integer activation sample sizes."""
        if isinstance(value, int) and value <= 0:
            raise ValueError("activation_subsample_size must be positive")
        return value

    def resolve_quantization_config(self) -> QuantizationConfig:
        """Validate NVFP4/NVFP4A16 groups, configure OSFP4 weight observers, and disable
        activation subsampling for weight-only calibration.
        """
        config = QuantizationMixin.resolve_quantization_config(self)
        observer_kwargs = {
            "mode": self.optimization_mode,
            "num_iters": self.steps,
            "lr": self.lr,
        }

        def matches_quantization_args(actual, reference):
            """Compare quantization arguments while ignoring observer names and options.
            """
            if actual is None or reference is None:
                return actual is reference

            excluded = {"observer", "observer_kwargs"}
            return actual.model_dump(exclude=excluded) == reference.model_dump(
                exclude=excluded
            )

        weight_only_groups = set()
        for scheme in config.config_groups.values():
            input_activations = scheme.input_activations
            expected_scheme = preset_name_to_scheme(
                "NVFP4A16" if input_activations is None else "NVFP4", []
            )

            if (
                not matches_quantization_args(scheme.weights, expected_scheme.weights)
                or not matches_quantization_args(
                    input_activations, expected_scheme.input_activations
                )
                or scheme.output_activations is not None
            ):
                raise ValueError("OSFP4 supports only NVFP4 and NVFP4A16 schemes")
            weight_only_groups.add(input_activations is None)
            scheme.weights.observer = _OSFP4_OBSERVER
            scheme.weights.observer_kwargs = observer_kwargs.copy()
        if len(weight_only_groups) > 1:
            raise ValueError("OSFP4 cannot mix NVFP4 and NVFP4A16 config groups")
        self._weight_only = next(iter(weight_only_groups), False)
        if self._weight_only:
            self.activation_subsample_size = None
        return config

    @property
    def resolved_weight_targets(self) -> set[str]:
        """Return weight-target selectors from all resolved quantization groups."""
        return {
            target
            for group in self.resolved_config.config_groups.values()
            for target in group.targets
        }

    # Initialization

    def on_initialize(self, state: State, **kwargs) -> bool:
        """Validate the setup, resolve mappings, reset capture state, attach runtime
        transforms, and initialize quantization.
        """
        self._raise_if_deployment_failed()
        self._validate_initialization(state, **kwargs)

        self._resolved_mappings = self._resolve_mappings(state.model)
        self._reset_calibration_state()
        # State may contain a deep copy; count the loader the pipeline will consume.
        self._calibration_dataloader = kwargs.get("calib_data", state.data.calib)
        self._calibration_cache.offload_hessians = self.offload_hessians
        module_names = get_module_to_name_dict(state.model)
        initialize_diagonal_smoothing(
            state.model,
            [
                module_names[layer]
                for mapping in self._resolved_mappings
                if mapping.requires_runtime_smoothing
                for layer in mapping.balance_layers
            ],
        )
        QuantizationMixin.initialize_quantization(self, state.model)
        return True

    def _validate_initialization(self, state: State, **kwargs) -> None:
        """Reject unsupported execution, recipe, and model configurations before
        modifying the model.
        """
        if is_distributed():
            raise NotImplementedError("OSFP4 calibration is single-process")
        pipeline = kwargs.get("pipeline")
        if pipeline is not None and standardize_lookup_name(pipeline) not in (
            "sequential",
            "independent",
        ):
            raise ValueError("OSFP4 requires the sequential calibration pipeline")
        if getattr(state, "layerwise_decompression", False) or getattr(
            state, "layerwise_compression", False
        ):
            raise ValueError(
                "OSFP4 does not support layerwise compression or decompression; "
                "calibrate from an uncompressed model"
            )
        recipe = active_session().lifecycle.recipe
        if recipe is not None and any(
            modifier is self for modifier in recipe.modifiers
        ):
            if len(recipe.modifiers) != 1:
                raise ValueError("OSFP4 must be the only modifier in its recipe")
        if get_non_linearized_moes(state.model):
            raise ValueError(
                "OSFP4 requires linearized experts; use oneshot or linearize_moe first"
            )

    def _resolve_mappings(self, model: Module) -> list[OSFP4Mapping]:
        """Group selected Linear layers into smooth-layer mappings and create runtime
        mappings for the remaining layers.
        """
        selected = list(
            match_named_modules(model, self.resolved_weight_targets, self.ignore)
        )
        if not selected:
            raise ValueError("No Linear modules matched the OSFP4 targets")
        if any(
            not isinstance(layer, Linear) or layer.in_features % 16
            for _, layer in selected
        ):
            raise ValueError(
                "OSFP4 requires Linear layers with input widths divisible by 16"
            )
        selected_balance_layers = {layer for _, layer in selected}
        module_names = get_module_to_name_dict(model)
        expert_mappings, expert_layers = self._resolve_expert_mappings(
            model, selected_balance_layers, module_names
        )
        mappings = self._resolve_smooth_layer_mappings(
            model, selected_balance_layers - expert_layers, module_names
        )
        smooth_layer_count = len(mappings)
        mappings.extend(expert_mappings)
        covered = {layer for mapping in mappings for layer in mapping.balance_layers}
        mappings.extend(
            OSFP4Mapping(
                mapping_name=mapping_name,
                smooth_layer=None,
                balance_layers=(layer,),
            )
            for mapping_name, layer in selected
            if layer not in covered
        )
        logger.info(
            f"Resolved OSFP4 architecture {model.__class__.__name__}: "
            f"{smooth_layer_count} smooth-layer mappings and "
            f"{len(mappings) - smooth_layer_count} "
            "runtime mappings"
        )
        return mappings

    def _resolve_expert_mappings(
        self,
        model: Module,
        selected_balance_layers: set[Module],
        module_names: dict[Module, str],
    ) -> tuple[list[OSFP4Mapping], set[Module]]:
        """Group expert projections without folding their scales into a shared norm.

        Inspect the linearized expert block and its shared experts. Return all expert
        projections, including ignored ones, to exclude them from norm folding.
        """
        mappings = []
        expert_layers: set[Module] = set()
        visited: set[Module] = set()
        for experts_name, experts in get_linearized_moes(model):
            parent_name = experts_name.rpartition(".")[0]
            parent = model.get_submodule(parent_name) if parent_name else model
            candidates = {
                experts[index]: self._match_expert_balance_groups(experts[index])
                for index in range(experts.num_experts)
            }
            # Shared experts are ordinary MLPs alongside the routed expert container.
            for module in parent.modules():
                if module not in candidates:
                    groups = self._match_expert_balance_groups(module)
                    if groups is not None:
                        candidates[module] = groups
            for expert, groups in candidates.items():
                if expert in visited:
                    continue
                visited.add(expert)
                if groups is None:
                    raise ValueError(
                        f"OSFP4 expert {module_names[expert]!r} requires Linear "
                        f"projections matching {_EXPERT_BALANCE_GROUPS}"
                    )
                for balance_layers in groups:
                    expert_layers.update(balance_layers)
                    selected = set(balance_layers) & selected_balance_layers
                    if not selected:
                        continue
                    if selected != set(balance_layers):
                        raise ValueError(
                            f"OSFP4 expert {module_names[expert]!r} requires the "
                            "projections in each balance group to be selected together"
                        )
                    mappings.append(
                        OSFP4Mapping(
                            mapping_name=module_names[balance_layers[0]],
                            smooth_layer=None,
                            balance_layers=balance_layers,
                            is_expert=True,
                        )
                    )
        return mappings, expert_layers

    @staticmethod
    def _match_expert_balance_groups(
        expert: Module,
    ) -> tuple[tuple[Linear, ...], ...] | None:
        """Resolve projection targets in definition order, or reject the layout.

        Matching each target separately preserves gate/up order regardless of module
        registration order. Ungated experts may omit the optional gate target.
        """
        groups = []
        for targets in _EXPERT_BALANCE_GROUPS:
            layers = []
            for target in targets:
                matches = [layer for _, layer in match_named_modules(expert, [target])]
                if not matches and target not in _OPTIONAL_EXPERT_PROJECTIONS:
                    return None
                if any(not isinstance(layer, Linear) for layer in matches):
                    return None
                layers.extend(matches)
            groups.append(tuple(dict.fromkeys(layers)))
        return tuple(groups)

    def _resolve_smooth_layer_mappings(
        self,
        model: Module,
        selected_balance_layers: set[Module],
        module_names: dict[Module, str],
    ) -> list[OSFP4Mapping]:
        """Build mappings from SmoothQuant specifications whose balance layers are all
        selected for quantization.
        """
        mappings = []
        for spec in get_layer_mappings_from_model(model):
            for *nested_balance_layers, smooth_layers in match_modules_set(
                model, tree_leaves(spec)
            ):
                if len(smooth_layers) > 1:
                    raise ValueError(
                        "OSFP4 must match a single smooth layer per mapping; "
                        f"got {[module_names.get(layer) for layer in smooth_layers]} "
                        f"for {spec}"
                    )
                if not smooth_layers:
                    continue
                smooth_layer = smooth_layers[0]
                balance_layers = tuple(tree_leaves(nested_balance_layers))
                if not balance_layers:
                    logger.warning(
                        f"Skipping OSFP4 mapping for {module_names[smooth_layer]}: "
                        "no balance layers"
                    )
                    continue
                if set(balance_layers) <= selected_balance_layers:
                    mappings.append(
                        OSFP4Mapping(
                            mapping_name=module_names[smooth_layer],
                            smooth_layer=smooth_layer,
                            balance_layers=balance_layers,
                        )
                    )
        return mappings

    def _reset_calibration_state(self) -> None:
        """Clear captured data, optimization progress, and sampling records, and reset
        finalization state.
        """
        self._calibration_cache.clear_all()
        self._optimized_mapping_names.clear()
        self._activation_subsampling_records.clear()
        self._finalization_complete = False

    # Calibration capture

    def on_calibration_start(self, state: State, event: Event, **kwargs):
        """Start observers, count tokens when subsampling is requested, and register
        calibration capture hooks.
        """
        self._raise_if_deployment_failed()
        if is_distributed():
            raise NotImplementedError("OSFP4 calibration is single-process")
        QuantizationMixin.start_calibration(self, state.model)
        total_num_tokens = (
            count_calibration_tokens(self._calibration_dataloader)
            if self.activation_subsample_size is not None and not self._weight_only
            else None
        )
        self._calibration_dataloader = None
        if (
            total_num_tokens is None
            and state.data.calib is not None
            and not self._weight_only
            and self.activation_subsample_size is not None
        ):
            logger.warning(
                "OSFP4 cannot count calibration token rows; "
                "using full activation caching where bounded capture is unavailable"
            )
        self._setup_capture_hooks(total_num_tokens)

    def _setup_capture_hooks(self, total_num_tokens: int | None) -> None:
        """Validate all sampling caps, then register an input capture hook on each
        mapping's first balance layer.
        """
        activation_subsample_sizes = [
            self._resolve_activation_subsample_size(mapping)
            for mapping in self._resolved_mappings
        ]
        for mapping, activation_subsample_size in zip(
            self._resolved_mappings, activation_subsample_sizes
        ):
            can_subsample = (
                activation_subsample_size is not None
                and not self._weight_only
                and all(
                    type(getattr(layer, "input_observer", None)) is StaticMinMaxObserver
                    and layer.input_observer.args.strategy
                    == QuantizationStrategy.TENSOR_GROUP
                    and layer.input_observer.args.dynamic == DynamicType.LOCAL
                    for layer in mapping.balance_layers
                )
            )
            routed_expert = mapping.is_expert and not get_calibrate_all_experts_flag()
            # Routed expert counts are unknown until routing finishes.
            subsample_during_capture = (
                can_subsample and total_num_tokens is not None and not routed_expert
            )
            self.register_hook(
                mapping.balance_layers[0],
                self._calibration_cache.make_capture_hook(
                    mapping.mapping_name,
                    optimization_mode=self.optimization_mode,
                    cache_activations=not self._weight_only,
                    total_num_tokens=total_num_tokens
                    if subsample_during_capture
                    else None,
                    activation_subsample_size=activation_subsample_size,
                    reservoir_sampling=can_subsample and routed_expert,
                ),
                "forward_pre",
            )

    def _resolve_activation_subsample_size(
        self, mapping: OSFP4Mapping
    ) -> int | None:
        """Return the configured token cap, twice the shared input width for auto
        sampling, or None to disable subsampling.
        """
        if self.activation_subsample_size != "auto":
            return self.activation_subsample_size
        input_widths = {layer.weight.shape[1] for layer in mapping.balance_layers}
        if len(input_widths) != 1:
            raise ValueError(
                f"OSFP4 mapping {mapping.mapping_name!r} cannot use auto activation "
                "subsampling because its balance layers have different input widths: "
                f"{sorted(input_widths)}"
            )
        input_width = input_widths.pop()
        return 2 * input_width

    # Mapping optimization and deployment

    def on_sequential_epoch_end(
        self,
        state: State,
        event: Event,
        modules: list[Module],
        **kwargs,
    ):
        """Validate sampling counts for all observed, unoptimized mappings before
        optimizing any of them.
        """
        # Optimize available mappings.
        self._raise_if_deployment_failed()
        available_mappings = [
            mapping
            for mapping in self._resolved_mappings
            if self._calibration_cache.has_calibration_data(mapping.mapping_name)
            and mapping.mapping_name not in self._optimized_mapping_names
        ]
        for mapping in available_mappings:
            self._calibration_cache.validate_sampling(mapping.mapping_name)
        for mapping in tqdm(available_mappings, desc="OSFP4 layer optimization"):
            self._optimize_mapping(mapping)

    def _optimize_mapping(self, mapping: OSFP4Mapping) -> None:
        """Optimize and deploy one mapping, then mark it complete and clear its cache.

        The caller must select a mapping that has not already been optimized.
        """
        self._raise_if_deployment_failed()
        sampled_activations = self._prepare_optimization_activations(mapping)
        result = quantize_mapping(
            mapping,
            self._calibration_cache,
            optimization_mode=self.optimization_mode,
            dampening_frac=self.dampening_frac,
            weight_only=self._weight_only,
            sampled_activations=sampled_activations,
        )
        self._deploy_mapping(mapping, result)
        self._optimized_mapping_names.add(mapping.mapping_name)
        self._calibration_cache.clear_mapping(mapping.mapping_name)

    def _prepare_optimization_activations(
        self, mapping: OSFP4Mapping
    ) -> tuple[torch.Tensor, ...] | None:
        """Prepare activation rows for reconstruction loss and record sampling metadata.

        Reuse rows sampled during capture, or sample the full activation cache. Return
        None when no separate activation subset is needed.
        """
        activation_subsample_size = self._resolve_activation_subsample_size(mapping)
        if activation_subsample_size is None:
            return None

        output_rows = sum(layer.weight.shape[0] for layer in mapping.balance_layers)
        if mapping.mapping_name in self._calibration_cache.sampled_indices:
            selection = self._calibration_cache.prepare_sampled_activations(
                mapping.mapping_name, output_rows
            )
        else:
            # Finish D2H capture before sampling the cached CPU rows.
            self._calibration_cache.wait_for_capture(
                mapping.mapping_name, torch.device("cpu")
            )
            selection = subsample_activations(
                self._calibration_cache.cached_activations[mapping.mapping_name],
                activation_subsample_size,
                output_rows=output_rows,
            )
        provenance = selection.provenance.copy()
        provenance["activation_subsample_size"] = activation_subsample_size
        if self.activation_subsample_size == "auto":
            provenance["policy"] = "auto"
        self._activation_subsampling_records[mapping.mapping_name] = provenance
        return selection.batches

    def _deploy_mapping(
        self,
        mapping: OSFP4Mapping,
        result: RTNQuantizationResult | SICQuantizationResult,
    ) -> None:
        """Apply smoothing, update input scales, and install optimized weight
        parameters.

        Prepare parameters for every balance layer before installing them. If deployment
        fails, mark the modifier unusable because the model may already be partially
        modified.
        """
        def prepare_mapping_parameters() -> (
            list[tuple[Module, dict[str, torch.Tensor]]]
        ):
            """Split optimized weight parameters into per-layer tensors matching the
            destination dtypes and devices.
            """
            balance_layers = mapping.balance_layers
            quantized_weight = (
                result.quantized_weight
                if isinstance(result, SICQuantizationResult)
                else None
            )
            deployment: list[tuple[Module, dict[str, torch.Tensor]]] = []
            row_start = 0
            for layer in balance_layers:
                row_end = row_start + layer.weight.shape[0]
                qparams: dict[str, torch.Tensor] = {}
                if quantized_weight is not None:
                    qparams["weight"] = quantized_weight[row_start:row_end].to(
                        layer.weight
                    )
                qparams["weight_scale"] = result.gamma_w_final[row_start:row_end].to(
                    layer.weight_scale
                )
                qparams["weight_zero_point"] = result.weight_zero_point[
                    row_start:row_end
                ].to(layer.weight_zero_point)
                # Selected block scales carry the scaling; the global multiplier is one.
                qparams["weight_global_scale"] = torch.ones_like(
                    layer.weight_global_scale
                )
                deployment.append((layer, qparams))
                row_start = row_end
            return deployment

        try:
            deployed_scale = deploy_mapping_smooth_quant_scale(mapping, result.alpha)
            if not self._weight_only:
                update_smoothed_input_global_scale(
                    mapping, self._calibration_cache, deployed_scale
                )
            deployment = prepare_mapping_parameters()
            for layer, qparams in deployment:
                for name, value in qparams.items():
                    update_offload_parameter(layer, name, value)
        except BaseException as error:
            # No rollback: even a failed first write may have changed offload state.
            self._failed_deployment_mapping_name = mapping.mapping_name
            raise RuntimeError(self._deployment_failure_message()) from error

    # Calibration completion

    def on_calibration_end(self, state: State, event: Event, **kwargs):
        """Remove capture hooks, verify every mapping was optimized, update optional
        KV-cache parameters, and freeze quantization.
        """
        self.remove_hooks()
        self._require_complete_calibration()
        if self.kv_cache_scheme is not None:
            kv_modules = [
                module
                for _, module in match_named_modules(
                    state.model,
                    self.resolved_targets,
                    self.ignore,
                )
            ]
            update_qparams(kv_modules, ("q", "k", "v"))
        QuantizationMixin.end_calibration(self, state.model)

    def on_finalize(self, state: State, **kwargs) -> bool:
        """Verify calibration completed, clear captured data and mapping references, and
        mark finalization complete.
        """
        if self._finalization_complete:
            return True
        self._require_complete_calibration()
        self._calibration_cache.clear_all()
        self._resolved_mappings.clear()
        self._finalization_complete = True
        return True

    def _require_complete_calibration(self) -> None:
        """Reject failed deployment or mappings that have not been optimized."""
        self._raise_if_deployment_failed()
        expected_mapping_names = {
            mapping.mapping_name for mapping in self._resolved_mappings
        }
        missing_mapping_names = sorted(
            expected_mapping_names - self._optimized_mapping_names
        )
        if missing_mapping_names:
            raise RuntimeError(
                "OSFP4 calibration did not observe every required registry mapping; "
                f"missing {missing_mapping_names}. Ensure calibration reaches every "
                "selected mapping; for routed experts, consider "
                "moe_calibrate_all_experts=True."
            )

    # Deployment failure handling

    def _raise_if_deployment_failed(self) -> None:
        """Prevent further use of a modifier whose deployment may have partially changed
        the model.
        """
        if self._failed_deployment_mapping_name is not None:
            raise RuntimeError(self._deployment_failure_message())

    def _deployment_failure_message(self) -> str:
        """Describe the failed mapping and the requirement to reload a fresh model."""
        return (
            f"OSFP4 deployment failed for mapping {self._failed_deployment_mapping_name!r}; "
            "the model may be partially modified. Reload a fresh model and create "
            "a fresh OSFP4Modifier; this modifier cannot be retried."
        )

    # Sampling records

    @property
    def activation_subsampling_records(self) -> dict[str, dict[str, int | str]]:
        """Return sampling metadata copies for tests and calibration-run comparisons.

        Tests use these records to verify sampling policies, token counts, and
        selected-index hashes remain unchanged across refactors.
        """
        return {
            mapping_name: record.copy()
            for mapping_name, record in self._activation_subsampling_records.items()
        }
