# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects
"""OSFP4 on post-norm Olmo3 decoders: shared runtime transforms, no norm folding."""

import pytest
import torch
from compressed_tensors.quantization import disable_quantization
from transformers import Olmo3Config, Olmo3ForCausalLM

from llmcompressor.args import DatasetArguments
from llmcompressor.core import create_session
from llmcompressor.modifiers.osfp4 import OSFP4Modifier
from llmcompressor.modifiers.osfp4.transforms import TRANSFORM_MODULE
from llmcompressor.modifiers.transform.smoothquant.dynamic_mappings import (
    get_layer_mappings_from_model,
)
from llmcompressor.pipelines import CalibrationPipeline
from llmcompressor.pipelines.sequential import pipeline as sequential
from compressed_tensors.utils import match_modules_set
from torch.utils._pytree import tree_leaves


def _tiny_olmo3(dtype=torch.float32):
    torch.manual_seed(0)
    config = Olmo3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        sliding_window=8,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        pad_token_id=0,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = Olmo3ForCausalLM(config).to(dtype).eval()
    for name, parameter in model.named_parameters():
        if "norm" in name:
            parameter.data.uniform_(0.5, 1.5)
    return model


def _samples():
    generator = torch.Generator().manual_seed(0)
    return [
        {"input_ids": torch.randint(0, 64, (1, 16), generator=generator)}
        for _ in range(4)
    ]


def _run_sequential(model, modifier, monkeypatch):
    samples = _samples()
    loader = torch.utils.data.DataLoader(samples, batch_size=None)
    monkeypatch.setattr(sequential, "get_main_device", lambda: torch.device("cpu"))
    with create_session() as session:
        session.initialize(
            model=model,
            recipe=[modifier],
            start=-1,
            calib_data=loader,
            sequential_targets=["Olmo3DecoderLayer"],
        )
        pipeline = CalibrationPipeline.from_modifiers(session.lifecycle.recipe.modifiers)
        pipeline(
            model,
            loader,
            DatasetArguments(
                sequential_targets=["Olmo3DecoderLayer"],
                sequential_offload_device="cpu",
                propagate_error=True,
            ),
        )
        session.finalize()
    return samples


def test_default_smoothquant_fallback_cannot_describe_olmo3():
    """The registry falls back to pre-norm defaults, which do not match Olmo3."""
    model = _tiny_olmo3()
    spec = get_layer_mappings_from_model(model)[0]
    with pytest.raises(ValueError, match="input_layernorm"):
        list(match_modules_set(model, tree_leaves(spec)))


def test_olmo3_resolves_shared_runtime_groups_without_norm_folding():
    model = _tiny_olmo3()
    modifier = OSFP4Modifier(scheme="NVFP4", ignore=["lm_head"], optimization_mode="rtn")
    mappings = modifier._resolve_mappings(model)

    assert all(mapping.smooth_layer is None for mapping in mappings)
    covered = [layer for mapping in mappings for layer in mapping.balance_layers]
    linears = [m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and n != "lm_head"]
    assert sorted(map(id, covered)) == sorted(map(id, linears))
    for index, layer in enumerate(model.model.layers):
        attention, mlp = layer.self_attn, layer.mlp
        groups = {
            tuple(map(id, mapping.balance_layers))
            for mapping in mappings
            if f"layers.{index}." in mapping.mapping_name
        }
        assert groups == {
            (id(attention.q_proj), id(attention.k_proj), id(attention.v_proj)),
            (id(mlp.gate_proj), id(mlp.up_proj)),
            (id(attention.o_proj),),
            (id(mlp.down_proj),),
        }


def test_olmo3_partially_selected_group_falls_back_to_single_layers():
    model = _tiny_olmo3()
    modifier = OSFP4Modifier(
        scheme="NVFP4", ignore=["lm_head", "re:.*k_proj$"], optimization_mode="rtn"
    )
    mappings = modifier._resolve_mappings(model)
    assert all(len(mapping.balance_layers) == 1 for mapping in mappings
               if "self_attn" in mapping.mapping_name)


@pytest.mark.parametrize("scheme", ["NVFP4", "NVFP4A16"])
@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_olmo3_sequential_calibration_shares_scales(scheme, mode, monkeypatch):
    model = _tiny_olmo3()
    modifier = OSFP4Modifier(
        scheme=scheme, optimization_mode=mode, ignore=["lm_head"], steps=2
    )
    _run_sequential(model, modifier, monkeypatch)

    for layer in model.model.layers:
        for group in (
            (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj),
            (layer.mlp.gate_proj, layer.mlp.up_proj),
        ):
            scales = [getattr(m, TRANSFORM_MODULE).weight for m in group]
            assert not torch.equal(scales[0], torch.ones_like(scales[0]))
            for scale in scales[1:]:
                assert torch.equal(scale, scales[0])
            if scheme == "NVFP4":
                global_scales = [m.input_global_scale for m in group]
                for global_scale in global_scales[1:]:
                    assert torch.equal(global_scale, global_scales[0])
        for single in (layer.self_attn.o_proj, layer.mlp.down_proj):
            assert hasattr(single, TRANSFORM_MODULE)
    # The post-norm weights must never absorb a smoothing scale.
    reference = _tiny_olmo3()
    for name, parameter in reference.named_parameters():
        if "norm" in name:
            assert torch.equal(model.get_parameter(name), parameter), name


def test_olmo3_rtn_smoothing_is_an_exact_equivalence(monkeypatch):
    """RTN keeps balanced source weights, so disabling quantization recovers BF16."""
    model = _tiny_olmo3()
    samples = _samples()
    with torch.no_grad():
        expected = model(**samples[0]).logits
    modifier = OSFP4Modifier(
        scheme="NVFP4", optimization_mode="rtn", ignore=["lm_head"], steps=2
    )
    _run_sequential(model, modifier, monkeypatch)
    model.apply(disable_quantization)
    with torch.no_grad():
        actual = model(**samples[0]).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
