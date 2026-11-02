# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from compressed_tensors.quantization import preset_name_to_scheme
from compressed_tensors.transform import TransformConfig, TransformFactory

from llmcompressor.core import Event, State, create_session
from llmcompressor.modifiers import ModifierFactory
from llmcompressor.modifiers.osfp4 import OSFP4Modifier
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines import CalibrationPipeline
from llmcompressor.pipelines.independent.pipeline import IndependentPipeline
from llmcompressor.pipelines.sequential import SequentialPipeline
from llmcompressor.recipe import Recipe

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
    assert not modifier._calibration_cache.cached_activations
    assert not modifier._resolved_mappings
    assert isinstance(model.transform_config, TransformConfig)


def test_factory_discovers_modifier_without_external_plugin():
    ModifierFactory.refresh()
    modifier = ModifierFactory.create("OSFP4Modifier", False, False, scheme="NVFP4")
    assert isinstance(modifier, OSFP4Modifier)


@pytest.mark.parametrize("pipeline", [None, "sequential", "independent"])
def test_pipeline_selection(pipeline):
    selected = CalibrationPipeline.from_modifiers([make_osfp4_modifier()], pipeline)
    expected = IndependentPipeline if pipeline == "independent" else SequentialPipeline
    assert isinstance(selected, expected)


@pytest.mark.parametrize("pipeline", ["Sequential", "INDEPENDENT"])
def test_pipeline_name_is_standardized(pipeline):
    with create_session() as session:
        session.initialize(
            model=LlamaForCausalLM(),
            recipe=[make_osfp4_modifier()],
            pipeline=pipeline,
        )


@pytest.mark.parametrize("pipeline", ["basic", "datafree"])
def test_unsupported_pipeline_rejected(pipeline):
    with create_session() as session, pytest.raises(ValueError, match="sequential"):
        session.initialize(
            model=LlamaForCausalLM(),
            recipe=[make_osfp4_modifier()],
            pipeline=pipeline,
        )


@pytest.mark.parametrize("flag", ["layerwise_decompression", "layerwise_compression"])
def test_layerwise_modes_rejected(flag):
    with create_session() as session, pytest.raises(ValueError, match="layerwise"):
        setattr(session.state, flag, True)
        session.initialize(model=LlamaForCausalLM(), recipe=[make_osfp4_modifier()])


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


@pytest.mark.parametrize("scheme", ["NVFP4", "NVFP4A16"])
@pytest.mark.parametrize("optimization_mode", ["rtn", "sic"])
@pytest.mark.parametrize("configuration", ["preset", "config_groups"])
@pytest.mark.parametrize(
    "observer_options",
    [
        {},
        {"observer": {"weights": "minmax"}},
        {"observer": {"input": "static_minmax"}},
        {"weight_observer": "minmax"},
    ],
    ids=["default", "weight_dict", "input_dict", "weight_argument"],
)
def test_observer_configuration_survives_recipe_reload_and_repeated_resolution(
    scheme, optimization_mode, configuration, observer_options
):
    configuration_options = (
        {"scheme": scheme}
        if configuration == "preset"
        else {"config_groups": {"group_0": preset_name_to_scheme(scheme, ["Linear"])}}
    )
    modifier = OSFP4Modifier(
        **configuration_options,
        **observer_options,
        optimization_mode=optimization_mode,
        steps=3,
        lr=0.03,
    )
    expected = modifier.resolved_config.model_dump(mode="json")

    # Resolving the config must not create conflicting recipe observer options.
    serialized_recipe = Recipe.from_modifiers(modifier).yaml()
    restored = Recipe.create_instance(serialized_recipe).modifiers[0]
    assert restored.resolved_config.model_dump(mode="json") == expected

    for candidate in (modifier, restored):
        assert (
            candidate.resolve_quantization_config().model_dump(mode="json") == expected
        )
        assert candidate.weight_observer == observer_options.get("weight_observer")
        assert candidate.observer == observer_options.get("observer")
        for group in candidate.resolved_config.config_groups.values():
            assert group.weights.observer == "osfp4"
            assert group.weights.observer_kwargs == {
                "mode": optimization_mode,
                "num_iters": 3,
                "lr": 0.03,
            }


def test_distributed_guard(monkeypatch):
    from llmcompressor.modifiers.osfp4 import base

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
