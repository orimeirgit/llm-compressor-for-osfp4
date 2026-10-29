# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.transform import TransformConfig, TransformFactory

from llmcompressor.core import Event, State, create_session
from llmcompressor.modifiers import ModifierFactory, OSFP4Modifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines import CalibrationPipeline
from llmcompressor.pipelines.sequential import SequentialPipeline

from ._testing import LlamaForCausalLM, make_osfp4_modifier, run_modifier


@pytest.mark.parametrize("mode", ["rtn", "sic"])
@pytest.mark.parametrize("scheme", ["NVFP4", "NVFP4A16"])
def test_finalize_preserves_smoothing_and_clears_capture_state(mode, scheme):
    model, modifier = run_modifier(optimization_mode=mode, scheme=scheme, steps=1)
    x = torch.randn(1, 4, 16)
    expected = model(x)
    layer = model.model.layers[0].mlp.down_proj
    before_hooks = list(layer._forward_pre_hooks)
    modifier.on_finalize(State(model=model))
    modifier.on_finalize(State(model=model))
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert list(layer._forward_pre_hooks) == before_hooks
    assert not modifier._calibration.inputs
    assert not modifier._resolved_mappings
    assert isinstance(model.transform_config, TransformConfig)


def test_factory_discovers_modifier_without_external_plugin():
    ModifierFactory.refresh()
    modifier = ModifierFactory.create("OSFP4Modifier", False, False, scheme="NVFP4")
    assert isinstance(modifier, OSFP4Modifier)


@pytest.mark.parametrize("pipeline", [None, "sequential", "independent"])
def test_pipeline_selection(pipeline):
    selected = CalibrationPipeline.from_modifiers([make_osfp4_modifier()], pipeline)
    assert isinstance(selected, SequentialPipeline)


@pytest.mark.parametrize("pipeline", ["basic", "datafree"])
def test_unsupported_pipeline_rejected(pipeline):
    with pytest.raises(ValueError, match="sequential"):
        CalibrationPipeline.from_modifiers([make_osfp4_modifier()], pipeline)


def test_composition_rejected_before_quantization():
    model = LlamaForCausalLM()
    with create_session() as session, pytest.raises(ValueError, match="only modifier"):
        session.initialize(
            model=model,
            recipe=[
                make_osfp4_modifier(),
                QuantizationModifier(scheme="NVFP4"),
            ],
        )
    assert not hasattr(model.model.layers[0].self_attn.q_proj, "weight_scale")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": -1},
        {"lr": 0},
        {"lr": float("nan")},
        {"dampening_frac": -0.1},
        {"dampening_frac": 1},
        {"activation_subsample_size": 0},
        {"optimization_mode": "gptq"},
    ],
)
def test_invalid_options_rejected(kwargs):
    with pytest.raises(ValueError):
        make_osfp4_modifier(**kwargs)


@pytest.mark.parametrize("scheme", ["W4A16", "FP8", "MXFP4"])
def test_other_schemes_rejected(scheme):
    with pytest.raises(ValueError, match="only NVFP4"):
        _ = OSFP4Modifier(scheme=scheme).resolved_config


def test_dense_only_and_distributed_guards(monkeypatch):
    from llmcompressor.modifiers.osfp4 import base

    model = LlamaForCausalLM()
    model.config.num_local_experts = 8
    with pytest.raises(ValueError, match="dense"):
        make_osfp4_modifier().on_initialize(State(model=model))
    monkeypatch.setattr(base, "is_distributed", lambda: True)
    with pytest.raises(NotImplementedError, match="single-process"):
        make_osfp4_modifier().on_initialize(State(model=LlamaForCausalLM()))


def test_missing_dependency_fails_clearly(monkeypatch):
    monkeypatch.setattr(TransformFactory, "registered_names", lambda: [])
    with pytest.raises(ImportError, match="diagonal transform support"):
        make_osfp4_modifier().on_initialize(State(model=LlamaForCausalLM()))


def test_unobserved_target_rejected():
    model = LlamaForCausalLM()
    model.model.layers[0].unused = torch.nn.Linear(16, 16)
    modifier = make_osfp4_modifier(steps=0)
    state = State(model=model)
    modifier.on_initialize(state)
    modifier.on_calibration_start(state, Event())
    model(torch.randn(1, 3, 16))
    modifier.on_sequential_epoch_end(state, Event(), list(model.modules()))
    with pytest.raises(RuntimeError, match="model.layers.0.unused"):
        modifier.on_calibration_end(state, Event())
