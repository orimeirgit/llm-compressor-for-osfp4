# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Frozen quantization (79fb6332) and modifier setup (70d0a9cb) references.

Do not update this reference to match refactored production helpers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import torch
from torch.nn import Module

from llmcompressor.modifiers.osfp4 import base
from llmcompressor.modifiers.osfp4.calibration_cache import OSFP4CalibrationCache
from llmcompressor.modifiers.quantization.calibration import update_qparams

from ._smoothing_reference import deploy_mapping_smooth_quant_scale
from .optimization._paper_reference import optimize_rtn, optimize_sic

if TYPE_CHECKING:
    from llmcompressor.modifiers.osfp4.base import OSFP4Mapping

__all__ = ["quantize_mapping"]


def _update_smoothed_input_global_scale(
    layers: tuple[Module, ...],
    batches: list[torch.Tensor] | tuple[torch.Tensor, ...],
    scale: torch.Tensor,
) -> None:
    """Recompute the saved input global scale from smoothed calibration inputs."""
    observers = [layer.input_observer for layer in layers]
    scale = scale.detach().float().reshape(1, -1)
    # Discard pre-SmoothQuant statistics before replaying smoothed inputs.
    for observer in observers:
        for statistic in ("min_vals", "max_vals"):
            if hasattr(observer, statistic):
                delattr(observer, statistic)

    for inputs in batches:
        smoothed_inputs = inputs.detach().float() * scale.to(device=inputs.device)
        for observer in observers:
            observer(smoothed_inputs)

    update_qparams(layers, "input")


def quantize_mapping(
    mapping: OSFP4Mapping,
    cache: OSFP4CalibrationCache,
    *,
    optimization_mode: Literal["rtn", "sic"],
    dampening_frac: float,
    weight_only: bool,
    sampled_activations: tuple[torch.Tensor, ...] | None = None,
    _deployment_stage: Callable[[str], None] | None = None,
) -> list[tuple[Module, dict[str, torch.Tensor]]]:
    """Normalize cached statistics, optimize one mapping, and return deployment."""
    layers = mapping.balance_layers
    weight_dtype = layers[0].weight.dtype
    device = layers[0].weight.device
    weight = torch.cat(
        [
            layer.weight.detach().to(device=device, dtype=torch.float32)
            for layer in layers
        ],
        dim=0,
    )
    cache.wait_for_capture(mapping.mapping_name, weight.device)
    alpha_dtype = torch.bfloat16 if mapping.requires_runtime_smoothing else weight_dtype
    observer = layers[0].weight_observer
    input_batches = (
        None if weight_only else cache.cached_activations[mapping.mapping_name]
    )
    optimizer_batches = (
        input_batches if sampled_activations is None else sampled_activations
    )
    token_count = cache.token_count[mapping.mapping_name]
    if optimization_mode == "sic":
        result = optimize_sic(
            weight,
            optimizer_batches,
            cache.hessian[mapping.mapping_name].to(weight.device) / token_count,
            observer=observer,
            alpha_dtype=alpha_dtype,
            weight_dtype=weight_dtype,
            percdamp=dampening_frac,
        )
        quantized_weight = result.quantized_weight
    else:
        sigma_x_squared = (
            cache.sigma_x_squared_sum[mapping.mapping_name].to(weight.device)
            / token_count
        )
        result = optimize_rtn(
            weight,
            optimizer_batches,
            observer=observer,
            alpha_dtype=alpha_dtype,
            sigma_x_squared=sigma_x_squared,
        )
        quantized_weight = None

    if _deployment_stage is not None:
        _deployment_stage("smoothing")
    scale = deploy_mapping_smooth_quant_scale(mapping, result.alpha_star)

    if not weight_only:
        if _deployment_stage is not None:
            _deployment_stage("input-scale replay")
        # Replay reads pinned CPU buffers on the host; finish their D2H copies.
        cache.wait_for_capture(mapping.mapping_name, torch.device("cpu"))
        replay_inputs = cache.cached_activations[mapping.mapping_name]
        if mapping.mapping_name in cache.input_channel_absmax:
            # max(abs(s * x)) = max(abs(s) * max_tokens(abs(x))). Replay
            # symmetric channel extrema using the original FP32 scale arithmetic.
            maximum = cache.input_channel_absmax[mapping.mapping_name].cpu()
            replay_inputs = (torch.stack((-maximum, maximum)),)
        _update_smoothed_input_global_scale(
            layers,
            replay_inputs,
            scale,
        )
    if _deployment_stage is not None:
        _deployment_stage("parameter preparation")
    deployment: list[tuple[Module, dict[str, torch.Tensor]]] = []
    row_start = 0
    for layer in layers:
        row_end = row_start + layer.weight.shape[0]
        qparams: dict[str, torch.Tensor] = {}
        if quantized_weight is not None:
            qparams["weight"] = quantized_weight[row_start:row_end].to(layer.weight)
        qparams["weight_scale"] = result.gamma_w_star[row_start:row_end].to(
            layer.weight_scale
        )
        qparams["weight_zero_point"] = result.weight_zero_point[row_start:row_end].to(
            layer.weight_zero_point
        )
        qparams["weight_global_scale"] = torch.ones_like(layer.weight_global_scale)
        deployment.append((layer, qparams))
        row_start = row_end
    return deployment


class PreRefactorOSFP4Modifier(base.OSFP4Modifier):
    """Initialization and capture setup frozen at 70d0a9cb.

    Only private-name adapters may change; do not call the extracted helpers.
    Other phases are covered by the frozen quantization reference above.
    """

    def on_initialize(self, state: base.State, **kwargs) -> bool:
        self._raise_if_deployment_failed()
        if base.is_distributed():
            raise NotImplementedError("OSFP4 calibration is single-process")
        pipeline = kwargs.get("pipeline")
        if pipeline is not None and base.standardize_lookup_name(pipeline) not in (
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
        recipe = base.active_session().lifecycle.recipe
        if recipe is not None and any(m is self for m in recipe.modifiers):
            if len(recipe.modifiers) != 1:
                raise ValueError("OSFP4 must be the only modifier in its recipe")
        config = getattr(state.model, "config", None)
        if config is not None and hasattr(config, "get_text_config"):
            config = config.get_text_config()
        if any(
            getattr(config, name, 0) or 0
            for name in (
                "num_experts",
                "num_local_experts",
                "moe_num_experts",
                "n_routed_experts",
            )
        ):
            raise ValueError("OSFP4 initially supports dense models only")

        self._resolved_mappings = self._resolve_mappings(state.model)
        self._reset_calibration_state()
        # State may contain a deep copy; count the loader the pipeline will consume.
        self._calibration_dataloader = kwargs.get("calib_data", state.data.calib)
        self._calibration_cache.offload_hessians = self.offload_hessians
        base.initialize_diagonal_smoothing(
            state.model,
            [
                m.mapping_name
                for m in self._resolved_mappings
                if m.requires_runtime_smoothing
            ],
        )
        base.QuantizationMixin.initialize_quantization(self, state.model)
        return True

    def resolve_quantization_config(self) -> base.QuantizationConfig:
        config = base.QuantizationMixin.resolve_quantization_config(self)
        observer_kwargs = {
            "mode": self.optimization_mode,
            "num_iters": self.steps,
            "lr": self.lr,
        }
        weight_only_groups = set()
        for scheme in config.config_groups.values():
            inputs = scheme.input_activations
            expected = base.preset_name_to_scheme(
                "NVFP4A16" if inputs is None else "NVFP4", []
            )

            def matches(actual, reference):
                return (
                    actual is None
                    and reference is None
                    or (
                        actual is not None
                        and reference is not None
                        and actual.model_dump(exclude={"observer", "observer_kwargs"})
                        == reference.model_dump(exclude={"observer", "observer_kwargs"})
                    )
                )

            if (
                not matches(scheme.weights, expected.weights)
                or not matches(inputs, expected.input_activations)
                or scheme.output_activations is not None
            ):
                raise ValueError("OSFP4 supports only NVFP4 and NVFP4A16 schemes")
            weight_only_groups.add(inputs is None)
            scheme.weights.observer = base._OSFP4_OBSERVER
            scheme.weights.observer_kwargs = observer_kwargs.copy()
        if len(weight_only_groups) > 1:
            raise ValueError("OSFP4 cannot mix NVFP4 and NVFP4A16 config groups")
        self._weight_only = next(iter(weight_only_groups), False)
        self.weight_observer = base._OSFP4_OBSERVER
        if self._weight_only:
            self.activation_subsample_size = None
        return config

    def on_calibration_start(self, state: base.State, event: base.Event, **kwargs):
        self._raise_if_deployment_failed()
        if base.is_distributed():
            raise NotImplementedError("OSFP4 calibration is single-process")
        base.QuantizationMixin.start_calibration(self, state.model)
        total_num_tokens = (
            base.count_calibration_tokens(self._calibration_dataloader)
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
            base.logger.warning(
                "OSFP4 cannot count calibration token rows; "
                "using full activation caching"
            )
        activation_subsample_sizes = [
            self._resolve_activation_subsample_size(mapping)
            for mapping in self._resolved_mappings
        ]
        for mapping, activation_subsample_size in zip(
            self._resolved_mappings, activation_subsample_sizes
        ):
            subsample_during_capture = (
                total_num_tokens is not None
                and activation_subsample_size is not None
                and not self._weight_only
                and all(
                    type(getattr(layer, "input_observer", None))
                    is base.StaticMinMaxObserver
                    and layer.input_observer.args.strategy
                    == base.QuantizationStrategy.TENSOR_GROUP
                    and layer.input_observer.args.dynamic == base.DynamicType.LOCAL
                    for layer in mapping.balance_layers
                )
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
                ),
                "forward_pre",
            )
