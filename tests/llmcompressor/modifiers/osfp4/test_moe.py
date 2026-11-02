# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Expert capture, optimization, and transform-aware compressed checkpoint reload."""

import contextlib
import json

import pytest
import torch
from accelerate import init_empty_weights
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization import (
    QuantizationConfig,
    apply_quantization_config,
)
from compressed_tensors.quantization.utils import generate_gparam
from compressed_tensors.transform import TransformConfig, apply_transform_config
from safetensors.torch import load_file
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from llmcompressor.args import DatasetArguments
from llmcompressor.core import Event, State, create_session
from llmcompressor.modeling.moe.context import moe_calibration_context
from llmcompressor.modeling.moe.linear_experts import LinearExperts2D
from llmcompressor.modeling.moe.linearize import (
    get_non_linearized_moes,
    linearize_moe,
)
from llmcompressor.modifiers.osfp4 import OSFP4Modifier
from llmcompressor.modifiers.osfp4.activation_subsampling import (
    _select_activation_indices,
)
from llmcompressor.modifiers.osfp4.smoothing import (
    deploy_mapping_smooth_quant_scale,
    update_smoothed_input_global_scale,
)
from llmcompressor.pipelines import CalibrationPipeline
from llmcompressor.pipelines.sequential import pipeline as sequential
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)


def make_model(num_hidden_layers=1):
    torch.manual_seed(13)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=4,
        num_experts_per_tok=2,
        use_cache=False,
        dtype=torch.bfloat16,
    )
    config._attn_implementation = "eager"
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16).eval()
    linearize_moe(model)
    return model


def make_modifier(**kwargs):
    return OSFP4Modifier(ignore=["lm_head", "re:.*mlp.gate$"], steps=2, **kwargs)


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
    )


def audit_pipeline_layers(model, monkeypatch):
    """Independently retain every observed row and check each deployed layer."""
    names = {module: name for name, module in model.named_modules()}
    inputs = {}
    input_scales = {}
    report = {}
    setup_hooks = OSFP4Modifier._setup_capture_hooks
    deploy = OSFP4Modifier._deploy_mapping

    def capture(layer, args):
        rows = args[0].detach().float().cpu().reshape(-1, layer.in_features)
        if len(rows):
            inputs.setdefault(layer, []).append(rows.clone())

    def setup(self, total_num_tokens):
        setup_hooks(self, total_num_tokens)
        for mapping in self._resolved_mappings:
            for layer in mapping.balance_layers:
                self.register_hook(layer, capture, "forward_pre")

    def check_deployment(self, mapping, result):
        layers = mapping.balance_layers
        weights = [layer.weight.detach().clone() for layer in layers]
        scale = result.alpha.detach().reshape(-1)
        scale = (
            scale.float().to(torch.bfloat16).float()
            if mapping.requires_runtime_smoothing
            else scale.to(layers[0].weight).float()
        )
        smooth_weight = (
            mapping.smooth_layer.weight.detach().clone()
            if mapping.smooth_layer is not None
            else None
        )
        deploy(self, mapping, result)
        if smooth_weight is not None:
            folded = scale.to(smooth_weight)
            smooth_weight[: len(scale)] *= (
                folded if smooth_weight.ndim == 1 else folded.view(-1, 1)
            )
            assert_bytes(mapping.smooth_layer.weight, smooth_weight)
        row_start = 0
        for layer, weight in zip(layers, weights):
            name = names[layer]
            assert name not in report, "A layer was deployed twice"
            observed = torch.cat(inputs[layer])
            assert (
                len(observed)
                == self._calibration_cache.token_count[mapping.mapping_name]
            )
            if mapping.requires_runtime_smoothing:
                assert_bytes(layer.smooth_quant_input.weight, scale.to(torch.bfloat16))
            row_end = row_start + layer.out_features
            expected_weight = (
                result.quantized_weight[row_start:row_end].to(weight)
                if self.optimization_mode == "sic"
                else weight / scale.to(weight).view(1, -1)
            )
            assert_bytes(layer.weight, expected_weight)
            assert_bytes(
                layer.weight_scale,
                result.gamma_w_final[row_start:row_end].to(layer.weight_scale),
            )
            assert_bytes(
                layer.weight_zero_point,
                result.weight_zero_point[row_start:row_end].to(layer.weight_zero_point),
            )
            assert_bytes(
                layer.weight_global_scale, torch.ones_like(layer.weight_global_scale)
            )
            row_start = row_end
            maximum = (observed * scale.cpu()).abs().max().reshape(1)
            report[name] = {
                "token_count": len(observed),
                "smoothed_absmax": maximum.item(),
            }
            if self._weight_only:
                assert not hasattr(layer, "input_global_scale")
                assert not self._calibration_cache.cached_activations
            else:
                # NVFP4 global multiplier: FP8 scale range * FP4 range / absmax.
                expected = (448.0 * 6.0 / maximum).to(layer.input_global_scale)
                assert_bytes(layer.input_global_scale, expected)
                input_scales[name] = expected.detach().clone()
                report[name]["input_global_scale"] = expected.item()

    monkeypatch.setattr(OSFP4Modifier, "_setup_capture_hooks", setup)
    monkeypatch.setattr(OSFP4Modifier, "_deploy_mapping", check_deployment)
    return report, input_scales


@pytest.mark.parametrize("all_tokens", [False, True])
@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_expert_capture_and_input_scales(all_tokens, mode):
    model = make_model()
    modifier = make_modifier(
        scheme="NVFP4", optimization_mode=mode, activation_subsample_size=2
    )
    state, event = State(model=model), Event()
    modifier.on_initialize(state)
    mappings = [m for m in modifier._resolved_mappings if m.is_expert]
    assert len(mappings) == 8
    experts = model.model.layers[0].mlp.experts
    assert not hasattr(model.model.layers[0].mlp.gate, "quantization_scheme")
    modifier.on_calibration_start(state, event)
    modifier.remove_hooks()
    context = moe_calibration_context() if all_tokens else contextlib.nullcontext()
    # Configure the hooks inside the same context used by oneshot.
    with context:
        modifier._setup_capture_hooks(4)
        rows = torch.linspace(-2, 3, 128).reshape(4, 32).to(torch.bfloat16)
        sampled_indices = _select_activation_indices(4, 2)
        outlier_index = next(
            index for index in range(4) if index not in sampled_indices
        )
        rows[outlier_index, 5] = 16
        indices = torch.tensor([[0, 1], [0, 2], [0, 1], [0, 2]])
        with torch.no_grad():
            experts(rows, indices, torch.full((4, 2), 0.5))
    cache = modifier._calibration_cache
    for index in range(4):
        mapping = mappings[2 * index]
        observed = rows if all_tokens else rows[(indices == index).any(dim=1)]
        assert cache.token_count.get(mapping.mapping_name, 0) == len(observed)
        assert cache.has_calibration_data(mapping.mapping_name) == bool(len(observed))
        if not len(observed):
            continue
        if mode == "rtn":
            assert_bytes(
                cache.sigma_x_squared_sum[mapping.mapping_name],
                observed.float().square().sum(0),
            )
        else:
            assert_bytes(
                cache.hessian[mapping.mapping_name],
                observed.float().T @ observed.float(),
            )
        assert mapping.mapping_name in cache.sampled_indices
        assert (
            sum(len(batch) for batch in cache.cached_activations[mapping.mapping_name])
            <= 2
        )
        if all_tokens:
            retained = torch.cat(cache.cached_activations[mapping.mapping_name])
            assert retained.abs().max() < observed.abs().max()
        alpha = torch.linspace(0.613 + index * 0.1, 1.713 + index * 0.1, 32)
        original_weights = [
            layer.weight.detach().clone() for layer in mapping.balance_layers
        ]
        deployed = deploy_mapping_smooth_quant_scale(mapping, alpha)
        assert_bytes(deployed, alpha.to(torch.bfloat16).float())
        for layer, original in zip(mapping.balance_layers, original_weights):
            assert_bytes(layer.weight, original / deployed.to(original).reshape(1, -1))
            assert_bytes(layer.smooth_quant_input.weight, deployed.to(torch.bfloat16))
        assert (
            mapping.balance_layers[0].smooth_quant_input.weight.data_ptr()
            != mapping.balance_layers[1].smooth_quant_input.weight.data_ptr()
        )
        update_smoothed_input_global_scale(mapping, cache, deployed)
        maximum = (observed.float() * deployed).abs().max().reshape(1)
        expected = generate_gparam(-maximum, maximum)
        for layer in mapping.balance_layers:
            assert_bytes(
                layer.input_global_scale, expected.to(layer.input_global_scale)
            )
    modifier.remove_hooks()


@pytest.mark.parametrize("all_tokens", [False, True])
@pytest.mark.parametrize("scheme", ["NVFP4", "NVFP4A16"])
@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_sequential_expert_checkpoint_round_trip(
    all_tokens, scheme, mode, tmp_path, monkeypatch
):
    model = make_model(num_hidden_layers=2)
    layer_report, expected_input_scales = audit_pipeline_layers(model, monkeypatch)
    modifier = make_modifier(
        scheme=scheme, optimization_mode=mode, activation_subsample_size=8
    )
    unchanged = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if ".mlp.gate." in name or "post_attention_layernorm" in name
    }
    loader = torch.utils.data.DataLoader(
        [{"input_ids": torch.randint(0, 64, (1, 16))} for _ in range(4)],
        batch_size=None,
    )
    monkeypatch.setattr(sequential, "get_main_device", lambda: torch.device("cpu"))
    context = moe_calibration_context() if all_tokens else contextlib.nullcontext()
    with context, create_session() as session:
        session.initialize(
            model=model,
            recipe=[modifier],
            start=-1,
            calib_data=loader,
            sequential_targets=["Qwen3MoeDecoderLayer"],
        )
        pipeline = CalibrationPipeline.from_modifiers(
            session.lifecycle.recipe.modifiers
        )
        pipeline(
            model,
            loader,
            DatasetArguments(
                sequential_targets=["Qwen3MoeDecoderLayer"],
                sequential_offload_device="cpu",
                propagate_error=True,
                moe_calibrate_all_experts=all_tokens,
            ),
        )
        with torch.no_grad():
            before = model(input_ids=torch.arange(16).reshape(1, -1)).logits.clone()
        session.finalize()
    assert not modifier._resolved_mappings
    assert not modifier._calibration_cache.token_count
    assert not modifier._calibration_cache.cached_activations
    assert not modifier._calibration_cache.capture_events
    assert not modifier._hooks
    assert len(layer_report) == 32  # Sixteen quantized projections per decoder block.
    for block_index in range(2):
        counts = []
        for index in range(4):
            prefix = f"model.layers.{block_index}.mlp.experts.{index}"
            counts.append(layer_report[f"{prefix}.gate_proj"]["token_count"])
            assert layer_report[f"{prefix}.up_proj"]["token_count"] == counts[-1]
            assert layer_report[f"{prefix}.down_proj"]["token_count"] == counts[-1]
        if all_tokens:
            assert counts == [64] * 4
        else:
            assert sum(counts) == 128
    for name, value in expected_input_scales.items():
        assert_bytes(model.get_submodule(name).input_global_scale, value)
    for name, value in unchanged.items():
        assert_bytes(model.state_dict()[name], value)
    with torch.no_grad():
        expected = model(input_ids=torch.arange(16).reshape(1, -1)).logits.clone()
    assert_bytes(expected, before)
    assert torch.isfinite(expected).all()
    transforms = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if name.endswith("smooth_quant_input.weight")
    }
    assert len(transforms) == 26  # Expert projections and attention o_proj per block.
    for block_index in range(2):
        for index in range(4):
            prefix = f"model.layers.{block_index}.mlp.experts.{index}"
            assert_bytes(
                transforms[f"{prefix}.gate_proj.smooth_quant_input.weight"],
                transforms[f"{prefix}.up_proj.smooth_quant_input.weight"],
            )
    modify_save_pretrained(model)
    model.save_pretrained(tmp_path, save_compressed=True)
    saved = load_file(str(tmp_path / "model.safetensors"))
    for name, value in expected_input_scales.items():
        assert_bytes(saved[f"{name}.input_global_scale"], value)
    qdata = json.loads((tmp_path / "config.json").read_text())["quantization_config"]
    assert qdata["quant_method"] == "compressed-tensors"
    tconfig = TransformConfig.model_validate(qdata.pop("transform_config"))
    qconfig = QuantizationConfig.model_validate(qdata)
    assert set(tconfig.config_groups["smooth_quant"].apply[0].targets) == {
        name.removesuffix(".smooth_quant_input.weight") for name in transforms
    }
    for name, value in transforms.items():
        assert_bytes(saved[name], value)
        prefix = name.removesuffix(".smooth_quant_input.weight")
        assert f"{prefix}.weight_packed" in saved
        assert (f"{prefix}.input_global_scale" in saved) == (scheme == "NVFP4")
    config = Qwen3MoeConfig.from_pretrained(tmp_path)
    config._attn_implementation = "eager"
    with init_empty_weights():
        restored = Qwen3MoeForCausalLM(config).to(torch.bfloat16).eval()
        for name, experts in get_non_linearized_moes(restored):
            cls = LinearExperts2D.get_linear_experts_cls(type(experts))
            restored.set_submodule(name, cls(config))
        apply_quantization_config(restored, qconfig)
        apply_transform_config(restored, tconfig)
        ModelCompressor(
            quantization_config=qconfig, transform_config=tconfig
        ).compress_model(restored)
    restored.load_state_dict(saved, assign=True, strict=True)
    for name, value in expected_input_scales.items():
        assert_bytes(restored.get_submodule(name).input_global_scale, value)
    with torch.no_grad():
        assert_bytes(
            restored(input_ids=torch.arange(16).reshape(1, -1)).logits, expected
        )
    (tmp_path / "pipeline-report.json").write_text(
        json.dumps(
            {
                "scheme": scheme,
                "optimization_mode": mode,
                "moe_calibrate_all_experts": all_tokens,
                "layers": layer_report,
                "reload_output_bytes_equal": True,
            },
            indent=2,
        )
    )


def test_missing_expert_is_not_optimized():
    model = make_model()
    modifier = make_modifier(scheme="NVFP4A16", optimization_mode="rtn")
    state, event = State(model=model), Event()
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, event)
    experts = model.model.layers[0].mlp.experts
    with torch.no_grad():
        experts(
            torch.randn(4, 32).to(torch.bfloat16),
            torch.tensor([[0, 1]] * 4),
            torch.full((4, 2), 0.5),
        )
    modifier.on_sequential_epoch_end(state, event, list(model.modules()))
    assert not any("experts.2." in name for name in modifier._optimized_mapping_names)
    with pytest.raises(RuntimeError, match=r"experts\.2"):
        modifier.on_calibration_end(state, event)


def test_shared_expert_and_partial_selection():
    import copy

    model = make_model()
    model.model.layers[0].mlp.shared_expert = copy.deepcopy(
        model.model.layers[0].mlp.experts[0]
    )
    modifier = make_modifier(scheme="NVFP4")
    modifier.on_initialize(State(model=model))
    shared = [
        m for m in modifier._resolved_mappings if "shared_expert" in m.mapping_name
    ]
    assert len(shared) == 2
    assert all(m.is_expert and m.requires_runtime_smoothing for m in shared)
    model = make_model()
    modifier = make_modifier(scheme="NVFP4")
    modifier.ignore.append("model.layers.0.mlp.experts.0.up_proj")
    with pytest.raises(ValueError, match="selected together"):
        modifier.on_initialize(State(model=model))
    assert not any("smooth_quant_input" in name for name, _ in model.named_modules())


def test_unlinearized_experts_require_conversion():
    config = make_model().config
    model = Qwen3MoeForCausalLM(config)
    with pytest.raises(ValueError, match="linearized experts"):
        make_modifier(scheme="NVFP4").on_initialize(State(model=model))


@pytest.mark.parametrize("gated", [False, True])
def test_projection_groups_preserve_definition_order(gated):
    expert = torch.nn.Module()
    expert.up_proj = torch.nn.Linear(32, 32)
    expert.down_proj = torch.nn.Linear(32, 32)
    if gated:
        expert.gate_proj = torch.nn.Linear(32, 32)
    groups = OSFP4Modifier._match_expert_balance_groups(expert)
    expected_input = (expert.gate_proj, expert.up_proj) if gated else (expert.up_proj,)
    assert groups == (expected_input, (expert.down_proj,))


def test_relative_patterns_resolve_routed_and_shared_experts(monkeypatch):
    import copy

    from llmcompressor.modifiers.osfp4 import base

    model = make_model()
    block = model.model.layers[0].mlp
    block.shared_expert = copy.deepcopy(block.experts[0])
    experts = [block.experts[index] for index in range(block.experts.num_experts)]
    experts.append(block.shared_expert)
    for expert in experts:
        for original, replacement in (
            ("gate_proj", "w1"),
            ("up_proj", "w3"),
            ("down_proj", "w2"),
        ):
            projection = getattr(expert, original)
            delattr(expert, original)
            expert.add_module(replacement, projection)
    monkeypatch.setattr(
        base, "_EXPERT_BALANCE_GROUPS", (("re:^w1$", "re:^w3$"), ("re:^w2$",))
    )
    modifier = make_modifier(scheme="NVFP4")
    modifier.on_initialize(State(model=model))
    mappings = [mapping for mapping in modifier._resolved_mappings if mapping.is_expert]
    assert len(mappings) == 2 * len(experts)
    for index, expert in enumerate(experts):
        assert mappings[2 * index].balance_layers == (expert.w1, expert.w3)
        assert mappings[2 * index + 1].balance_layers == (expert.w2,)
        assert mappings[2 * index].requires_runtime_smoothing


def test_ignored_expert_projections_remain_excluded_from_folding():
    model = make_model()
    modifier = make_modifier(scheme="NVFP4")
    modifier.ignore.append("re:.*experts.0.*")
    modifier.on_initialize(State(model=model))
    assert not any(
        "experts.0." in mapping.mapping_name for mapping in modifier._resolved_mappings
    )
    expert = model.model.layers[0].mlp.experts[0]
    assert not any(hasattr(layer, "smooth_quant_input") for layer in expert.modules())
    selected = {
        layer
        for mapping in modifier._resolved_mappings
        for layer in mapping.balance_layers
    }
    names = {module: name for name, module in model.named_modules()}
    _, protected = modifier._resolve_expert_mappings(model, selected, names)
    assert {expert.gate_proj, expert.up_proj, expert.down_proj} <= protected


def test_expert_norm_folding_is_excluded(monkeypatch):
    from llmcompressor.modifiers.osfp4 import base
    from llmcompressor.modifiers.transform.smoothquant.utils import LayerMap

    model = make_model()
    monkeypatch.setattr(
        base,
        "get_layer_mappings_from_model",
        lambda _: [
            LayerMap(
                ["re:.*experts.*gate_proj", "re:.*experts.*up_proj"],
                "re:.*post_attention_layernorm",
            )
        ],
    )
    modifier = make_modifier(scheme="NVFP4")
    modifier.on_initialize(State(model=model))
    assert all(mapping.smooth_layer is None for mapping in modifier._resolved_mappings)
    layers = [
        layer
        for mapping in modifier._resolved_mappings
        for layer in mapping.balance_layers
    ]
    assert len(layers) == len(set(layers))


@pytest.mark.parametrize("all_tokens", [False, True])
def test_oneshot_honors_expert_calibration_mode(all_tokens, tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from llmcompressor import oneshot
    from llmcompressor.modeling.moe.context import get_calibrate_all_experts_flag

    config = make_model().config
    config._name_or_path = str(tmp_path / "source")
    config.save_pretrained(config._name_or_path)
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16).eval()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1})),
        unk_token="<unk>",
        eos_token="<eos>",
    )
    observed_modes = []
    original = OSFP4Modifier._setup_capture_hooks

    def capture_mode(self, total_num_tokens):
        observed_modes.append(get_calibrate_all_experts_flag())
        return original(self, total_num_tokens)

    monkeypatch.setattr(OSFP4Modifier, "_setup_capture_hooks", capture_mode)
    monkeypatch.setattr(sequential, "get_main_device", lambda: torch.device("cpu"))
    loader = torch.utils.data.DataLoader(
        [{"input_ids": torch.randint(0, 64, (1, 16))} for _ in range(4)],
        batch_size=None,
    )
    # Omitting the flag verifies that LLMC's existing True default is preserved.
    options = {} if all_tokens else {"moe_calibrate_all_experts": False}
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=loader,
        recipe=make_modifier(
            scheme="NVFP4", optimization_mode="rtn", activation_subsample_size=8
        ),
        **options,
    )
    assert observed_modes == [all_tokens]
    assert not get_non_linearized_moes(model)
    with torch.no_grad():
        assert torch.isfinite(
            model(input_ids=torch.arange(16).reshape(1, -1)).logits
        ).all()


@pytest.mark.parametrize("mode", ["rtn", "sic"])
@pytest.mark.parametrize("sample_size", [4, "auto"])
@pytest.mark.parametrize("countable_loader", [False, True])
def test_routed_expert_reservoir_preserves_input_scale(
    mode, sample_size, countable_loader
):
    """An excluded outlier must still determine the deployed input global scale."""
    import random

    model = make_model()
    modifier = make_modifier(
        scheme="NVFP4",
        optimization_mode=mode,
        activation_subsample_size=sample_size,
    )
    loader = [{"input_ids": torch.zeros((1, 16), dtype=torch.long)} for _ in range(8)]
    state = State(model=model)
    modifier.on_initialize(
        state, calib_data=loader if countable_loader else iter(loader)
    )
    modifier.on_calibration_start(state, Event())
    mapping = next(m for m in modifier._resolved_mappings if m.is_expert)
    cap = 64 if sample_size == "auto" else sample_size
    selected = list(range(cap))
    rng = random.Random(42)
    for index in range(cap, 128):
        slot = rng.randrange(index + 1)
        if slot < cap:
            selected[slot] = index
    excluded = next(index for index in range(128) if index not in selected)
    rows = torch.ones((128, 32), dtype=torch.bfloat16)
    rows[excluded, 5] = 100
    cache = modifier._calibration_cache
    name = mapping.mapping_name
    try:
        experts = model.model.layers[0].mlp.experts
        with torch.no_grad():
            for batch in rows.split(16):
                experts(
                    batch,
                    torch.tensor([[0, 1]] * len(batch)),
                    torch.full((len(batch), 2), 0.5),
                )
                assert len(cache.cached_activations[name][0]) <= cap
                assert (
                    cache.cached_activations[name][0].untyped_storage().nbytes()
                    == cap * 32 * 2
                )
        assert name in cache.reservoirs
        assert cache.token_count[name] == 128
        sampled = modifier._prepare_optimization_activations(mapping)
        assert torch.cat(sampled).abs().max().item() == 1
        assert cache.input_channel_absmax[name][5].item() == 100
        deployed = deploy_mapping_smooth_quant_scale(
            mapping, torch.linspace(0.613, 1.713, 32)
        )
        update_smoothed_input_global_scale(mapping, cache, deployed)
        maximum = (rows.float() * deployed).abs().max().reshape(1)
        expected = generate_gparam(-maximum, maximum)
        for layer in mapping.balance_layers:
            assert_bytes(
                layer.input_global_scale, expected.to(layer.input_global_scale)
            )
    finally:
        modifier.remove_hooks()
        cache.clear_all()
