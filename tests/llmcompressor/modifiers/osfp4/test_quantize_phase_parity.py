# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Byte-exact parity against quantization (79fb6332) and setup (70d0a9cb)."""

import json
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch
from compressed_tensors.offload import offload_module, update_offload_parameter
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

from llmcompressor.args import DatasetArguments
from llmcompressor.core import create_session
from llmcompressor.modifiers.osfp4 import OSFP4Modifier, base, osfp4_quantize
from llmcompressor.modifiers.osfp4.base import OSFP4Mapping
from llmcompressor.modifiers.osfp4.calibration_cache import OSFP4CalibrationCache
from llmcompressor.modifiers.osfp4.transforms import initialize_diagonal_smoothing
from llmcompressor.pipelines import CalibrationPipeline
from llmcompressor.pipelines.sequential import pipeline as sequential
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)

from . import _quantize_reference as reference
from ._quantize_phases import run_phases

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.get_device_module("cuda").is_available(),
            reason="CUDA unavailable",
        ),
    ),
]


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
    )


def assert_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        try:
            assert_bytes(actual[name], expected[name])
        except AssertionError as error:
            raise AssertionError(f"Tensor bytes differ: {name}") from error


def snapshot(model):
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def assert_tree_bytes(actual, expected, path="snapshot"):
    if isinstance(expected, torch.Tensor):
        try:
            assert_bytes(actual, expected)
        except AssertionError as error:
            raise AssertionError(f"Tensor bytes differ: {path}") from error
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_tree_bytes(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected), path
        assert len(actual) == len(expected), path
        for index, (value, reference_value) in enumerate(zip(actual, expected)):
            assert_tree_bytes(value, reference_value, f"{path}[{index}]")
    else:
        assert actual == expected, path


@pytest.mark.parametrize("optimization_mode", ["rtn", "sic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["runtime", "folded", "shared"])
@pytest.mark.parametrize("policy", ["full", "sampled", "extrema", "weight_only"])
@pytest.mark.parametrize("offloaded", [False, True])
def test_mapping_phases_match_reference_bytes(
    optimization_mode, dtype, layout, policy, offloaded
):
    def execute(implementation):
        torch.manual_seed(37)
        model = torch.nn.Module()
        model.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(32, rows, dtype=dtype)
                for rows in ((16, 24) if layout == "shared" else (16,))
            ]
        )
        model.norm = torch.nn.LayerNorm(32, dtype=dtype)
        modifier = OSFP4Modifier(
            scheme="NVFP4A16" if policy == "weight_only" else "NVFP4",
            targets=["Linear"],
            optimization_mode=optimization_mode,
            steps=2,
        )
        modifier.initialize_quantization(model)
        if layout == "runtime":
            initialize_diagonal_smoothing(model, ["layers.0"])
        modifier.start_calibration(model)
        if offloaded:
            for layer in model.layers:
                offload_module(layer, onload_device="cpu", offload_device="cpu")
        mapping = OSFP4Mapping(
            "mapping", None if layout == "runtime" else model.norm, tuple(model.layers)
        )
        rows = torch.randn(71, 32).to(dtype)
        rows[0, 0] = -0.0
        batches = list(rows.split([17, 0, 31, 23]))
        sampled = (
            (rows[[1, 6, 19, 29, 39]].clone(),)
            if policy in ("sampled", "extrema")
            else None
        )
        cache = OSFP4CalibrationCache(
            cached_activations={}
            if policy == "weight_only"
            else {"mapping": list(sampled) if policy == "extrema" else batches},
            hessian={"mapping": rows.float().T @ rows.float()},
            sigma_x_squared_sum={"mapping": rows.float().square().sum(0)},
            token_count={"mapping": rows.shape[0]},
            input_channel_absmax={"mapping": rows.float().abs().amax(0)}
            if policy == "extrema"
            else {},
        )
        captured = []
        optimizer_name = (
            "optimize_sic" if optimization_mode == "sic" else "optimize_rtn"
        )
        optimizer = getattr(
            reference
            if implementation is reference.quantize_mapping
            else osfp4_quantize,
            optimizer_name,
        )

        def record(*args, **kwargs):
            result = optimizer(*args, **kwargs)
            captured.append(
                {
                    {"alpha_star": "alpha", "gamma_w_star": "gamma_w_final"}.get(
                        name, name
                    ): value.detach().clone()
                    for name, value in result._asdict().items()
                }
            )
            return result

        with patch.object(reference, optimizer_name, record), patch.object(
            osfp4_quantize, optimizer_name, record
        ):
            deployment = implementation(
                mapping,
                cache,
                optimization_mode=optimization_mode,
                dampening_frac=0.01,
                weight_only=policy == "weight_only",
                sampled_activations=sampled,
            )
        before_install = snapshot(model)
        parameters = []
        observers = {}
        for i, (layer, qparams) in enumerate(deployment):
            assert layer is model.layers[i]
            parameters.append(
                {name: value.detach().clone() for name, value in qparams.items()}
            )
            if policy != "weight_only":
                for name in ("min_vals", "max_vals"):
                    value = getattr(layer.input_observer, name, None)
                    if value is not None:
                        observers[f"{i}.{name}"] = value.detach().clone()
            for name, value in qparams.items():
                update_offload_parameter(layer, name, value)
        modifier.remove_hooks()
        return captured, before_install, parameters, observers, snapshot(model)

    old, new = execute(reference.quantize_mapping), execute(run_phases)
    for old_result, new_result in zip(old[0], new[0], strict=True):
        assert_state(new_result, old_result)
    assert_state(new[1], old[1])
    for old_params, new_params in zip(old[2], new[2], strict=True):
        assert list(new_params) == list(old_params)
        assert_state(new_params, old_params)
    assert_state(new[3], old[3])
    assert_state(new[4], old[4])


def reference_optimize_mapping(self, mapping):
    """Baseline successful modifier path; no refactored phase helpers."""
    self._raise_if_deployment_failed()
    if mapping.mapping_name in self._optimized_mapping_names:
        return
    sampled = self._prepare_optimization_activations(mapping)
    deployment = reference.quantize_mapping(
        mapping,
        self._calibration_cache,
        optimization_mode=self.optimization_mode,
        dampening_frac=self.dampening_frac,
        weight_only=self._weight_only,
        sampled_activations=sampled,
    )
    for layer, qparams in deployment:
        for name, value in qparams.items():
            update_offload_parameter(layer, name, value)
    self._optimized_mapping_names.add(mapping.mapping_name)
    self._calibration_cache.clear_mapping(mapping.mapping_name)


@pytest.mark.parametrize("optimization_mode", ["rtn", "sic"])
@pytest.mark.parametrize("reference_phase", ["quantization", "setup"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("offload_hessians", [False, True])
@pytest.mark.parametrize(
    "scheme,policy",
    [
        ("NVFP4", "full"),
        ("NVFP4", "sampled"),
        ("NVFP4", "streaming"),
        ("NVFP4", "fixed"),
        ("NVFP4A16", "full"),
    ],
)
@pytest.mark.parametrize("device", DEVICES)
def test_sequential_export_matches_reference_bytes(
    optimization_mode,
    reference_phase,
    dtype,
    offload_hessians,
    scheme,
    policy,
    device,
    tmp_path,
):
    def execute(legacy):
        torch.manual_seed(42)
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            use_cache=False,
        )
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config).to(device=device, dtype=dtype).eval()
        generator = torch.Generator().manual_seed(0)
        samples = [
            {"input_ids": torch.randint(0, 64, (1, 16), generator=generator)}
            for _ in range(5)
        ]
        loader = torch.utils.data.DataLoader(samples, batch_size=None)
        modifier_class = (
            reference.PreRefactorOSFP4Modifier
            if legacy and reference_phase == "setup"
            else OSFP4Modifier
        )
        modifier = modifier_class(
            scheme=scheme,
            optimization_mode=optimization_mode,
            ignore=["lm_head"],
            steps=2,
            offload_hessians=offload_hessians,
            activation_subsample_size="auto"
            if policy == "streaming"
            else (7 if policy in ("sampled", "fixed") else None),
        )
        with ExitStack() as stack:
            if legacy and reference_phase == "quantization":
                stack.enter_context(
                    patch.object(
                        OSFP4Modifier, "_optimize_mapping", reference_optimize_mapping
                    )
                )
            if policy == "sampled":
                stack.enter_context(
                    patch.object(
                        base, "count_calibration_tokens", return_value=None
                    )
                )
            stack.enter_context(
                patch.object(
                    sequential, "get_main_device", lambda: torch.device(device)
                )
            )
            session = stack.enter_context(create_session())
            session.initialize(
                model=model, recipe=[modifier], start=-1, calib_data=loader
            )
            module_names = {module: name for name, module in model.named_modules()}
            mappings = [
                (
                    mapping.mapping_name,
                    module_names.get(mapping.smooth_layer),
                    tuple(module_names[layer] for layer in mapping.balance_layers),
                )
                for mapping in modifier._resolved_mappings
            ]
            initialized_state = snapshot(model)
            CalibrationPipeline.from_modifiers([modifier])(
                model,
                loader,
                DatasetArguments(
                    sequential_targets=["LlamaDecoderLayer"],
                    sequential_offload_device="cpu",
                    propagate_error=True,
                ),
            )
            with torch.no_grad():
                before_finalize = (
                    model(input_ids=samples[0]["input_ids"].to(device))
                    .logits.cpu()
                    .clone()
                )
            state_before_finalize = snapshot(model)
            session.finalize()
            modifier.on_finalize(session.state)
        state = snapshot(model)
        assert_state(state, state_before_finalize)
        with torch.no_grad():
            output = (
                model(input_ids=samples[0]["input_ids"].to(device)).logits.cpu().clone()
            )
        folder = tmp_path / str(legacy)
        modify_save_pretrained(model)
        model.save_pretrained(folder, save_compressed=True)
        exported = load_file(str(folder / "model.safetensors"))
        config = json.loads((folder / "config.json").read_text())
        assert_bytes(output, before_finalize)
        cache = modifier._calibration_cache
        assert not modifier._resolved_mappings
        assert modifier._finalization_complete
        assert not any(
            value for name, value in vars(cache).items() if name != "offload_hessians"
        )
        return {
            "mappings": mappings,
            "initialized_state": initialized_state,
            "state_before_finalize": state_before_finalize,
            "state": state,
            "before_finalize": before_finalize,
            "output": output,
            "exported": exported,
            "config": config,
            "sampling_records": modifier.activation_subsampling_records,
            "optimized_mapping_names": sorted(modifier._optimized_mapping_names),
            "hook_counts": {
                name: (len(layer._forward_pre_hooks), len(layer._forward_hooks))
                for name, layer in model.named_modules()
            },
            "rng": torch.get_rng_state().clone(),
        }

    old, new = execute(True), execute(False)
    assert_tree_bytes(new, old)
