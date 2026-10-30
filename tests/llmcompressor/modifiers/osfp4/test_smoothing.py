# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Smoothing algebra and byte-exact parity with the pre-cleanup implementation."""

import pytest
import torch
from compressed_tensors.offload import offload_module

from llmcompressor.modifiers.osfp4.base import OSFP4Mapping
from llmcompressor.modifiers.osfp4.smoothing import deploy_mapping_smooth_quant_scale
from llmcompressor.modifiers.osfp4.transforms import (
    TRANSFORM_MODULE,
    initialize_diagonal_smoothing,
)

from ._smoothing_reference import (
    deploy_mapping_smooth_quant_scale as reference_deploy,
)

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable"
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


def make_mapping(layout, dtype, bias, device="cpu", offloaded=False):
    torch.manual_seed(29)
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList(
        [
            torch.nn.Linear(16, rows, dtype=dtype, bias=bias)
            for rows in ((13,) if layout == "runtime" else (13, 7))
        ]
    )
    if layout == "runtime":
        smooth = None
    elif layout == "linear":
        smooth = torch.nn.Linear(11, 16, bias=bias, dtype=dtype)
    elif layout == "rmsnorm":
        smooth = torch.nn.RMSNorm(16, dtype=dtype)
    else:
        smooth = torch.nn.LayerNorm(16, bias=bias, dtype=dtype)
    if smooth is not None:
        model.smooth = smooth
        if getattr(smooth, "bias", None) is not None:
            with torch.no_grad():
                smooth.bias.copy_(torch.randn_like(smooth.bias))
    with torch.no_grad():
        for layer in model.layers:
            layer.weight[0, 0] = 0.0
            layer.weight[0, 1] = -0.0
    model.to(device).eval()
    if layout == "runtime":
        initialize_diagonal_smoothing(model, ["layers.0"])
    if offloaded:
        # Exercise the cache for both folded parameters and the runtime transform.
        for module in model.modules():
            if any(value is not None for value in module._parameters.values()):
                offload_module(module, onload_device=device, offload_device="cpu")
    inputs = torch.randn(9, 11 if layout == "linear" else 16).to(
        device=device, dtype=dtype
    )
    mapping = OSFP4Mapping("mapping", smooth, tuple(model.layers))
    return model, mapping, inputs


@torch.no_grad()
def outputs(mapping, inputs):
    if mapping.smooth_layer is not None:
        inputs = mapping.smooth_layer(inputs)
    return [layer(inputs).detach().cpu().clone() for layer in mapping.balance_layers]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["norm", "rmsnorm", "linear", "runtime"])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("offloaded", [False, True])
def test_smoothing_matches_reference_bytes(dtype, layout, bias, offloaded, device):
    def execute(deploy):
        model, mapping, inputs = make_mapping(layout, dtype, bias, device, offloaded)
        before = outputs(mapping, inputs)
        # Nonrepresentable values exercise both BF16 and weight-dtype rounding;
        # a grouped CPU tensor also exercises flattening and device placement.
        scale = torch.linspace(0.713, 1.419, 16, dtype=torch.float64).reshape(4, 4)
        original_scale = scale.clone()
        deployed = deploy(mapping, scale)
        assert_bytes(scale, original_scale)
        assert deployed.dtype == torch.float32
        assert not deployed.requires_grad
        if layout == "runtime":
            transform = getattr(mapping.balance_layers[0], TRANSFORM_MODULE)
            assert transform.weight.dtype == torch.bfloat16
        after = outputs(mapping, inputs)
        state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        return deployed, before, after, state

    old = execute(reference_deploy)
    new = execute(deploy_mapping_smooth_quant_scale)
    assert_bytes(new[0], old[0])
    for actual, expected in zip(new[1] + new[2], old[1] + old[2], strict=True):
        assert_bytes(actual, expected)
    assert new[3].keys() == old[3].keys()
    for name, expected in old[3].items():
        assert_bytes(new[3][name], expected)


@pytest.mark.parametrize("layout", ["norm", "rmsnorm", "linear", "runtime"])
@pytest.mark.parametrize("bias", [False, True])
def test_smoothing_preserves_outputs_with_exact_scales(layout, bias):
    # Powers of two and FP64 avoid rounding/underflow in this algebra check.
    _, mapping, inputs = make_mapping(layout, torch.float64, bias)
    before = outputs(mapping, inputs)
    scale = torch.tensor([0.5, 1.0, 2.0, 4.0] * 4)
    deploy_mapping_smooth_quant_scale(mapping, scale)
    for actual, expected in zip(outputs(mapping, inputs), before, strict=True):
        assert_bytes(actual, expected)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_direct_cast_matches_fp32_intermediate_for_all_valid_bf16_scales(dtype):
    scale = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    scale = scale[torch.isfinite(scale) & (scale > 0)]
    assert_bytes(scale.to(dtype), scale.float().to(dtype))
