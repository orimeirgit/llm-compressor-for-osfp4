# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Preserve optimizer bytes while adopting the paper's mathematical names."""

import pytest
import torch
from compressed_tensors.quantization import preset_name_to_scheme

from llmcompressor.modifiers.osfp4.optimization import rtn, sic
from llmcompressor.modifiers.osfp4.optimization._activation_groups import (
    assemble_activation_quantization_groups,
)
from llmcompressor.observers import Observer

from . import _paper_reference as reference

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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["rtn", "sic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_blocks", [1, 3])
@pytest.mark.parametrize("weight_only", [False, True])
def test_optimizer_results_match_pre_rename_bytes(
    device, mode, dtype, num_blocks, weight_only
):
    generator = torch.Generator().manual_seed(131)
    n = num_blocks * 16
    # Strided weights and activations exercise layout conversions without changing math.
    W = torch.randn(5, 2 * n, generator=generator).to(device=device, dtype=dtype)[
        :, ::2
    ]
    rows = torch.randn(79, 2 * n, generator=generator).to(dtype)[:, ::2]
    rows[0, 0] = -0.0
    batches = list(rows.split([11, 0, 31, 37]))
    original_weight = W.clone()
    original_batches = [batch.clone() for batch in batches]
    sigma_x = (rows.float().T @ rows.float() / rows.shape[0]).to(device)
    original_sigma_x = sigma_x.clone()
    sigma_x_squared = rows.float().square().sum(0) / rows.shape[0]
    sample = None if weight_only else batches

    def execute(optimizer):
        args = preset_name_to_scheme(
            "NVFP4A16" if weight_only else "NVFP4", []
        ).weights.model_copy(
            update={
                "observer": "osfp4",
                "observer_kwargs": {
                    "mode": mode,
                    "num_iters": 2,
                    "lr": 0.01,
                },
            },
            deep=True,
        )
        observer = Observer.load_from_registry("osfp4", base_name="weight", args=args)
        kwargs = {"observer": observer, "alpha_dtype": dtype}
        if mode == "sic":
            return optimizer(W, sample, sigma_x, weight_dtype=dtype, **kwargs)
        return optimizer(W, sample, sigma_x_squared=sigma_x_squared, **kwargs)

    expected = execute(getattr(reference, f"optimize_{mode}"))
    actual = execute(rtn.optimize_rtn if mode == "rtn" else sic.optimize_sic)
    assert_bytes(actual.alpha, expected.alpha_star)
    assert_bytes(actual.gamma_w_final, expected.gamma_w_star)
    assert_bytes(actual.weight_zero_point, expected.weight_zero_point)
    if mode == "sic":
        assert_bytes(actual.quantized_weight, expected.quantized_weight)
    assert_bytes(W, original_weight)
    assert_bytes(sigma_x, original_sigma_x)
    for batch, original in zip(batches, original_batches, strict=True):
        assert_bytes(batch, original)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_renamed_gram_accumulation_and_block_assembly_match_reference(dtype):
    generator = torch.Generator().manual_seed(51)
    rows = torch.randn(43, 96, generator=generator).to(dtype)[:, ::2]
    batches = list(rows.split([0, 12, 31, 0]))
    layer = torch.nn.Linear(48, 5).to(dtype)
    actual = sic.make_empty_hessian(layer)
    expected = reference.make_empty_hessian(layer)
    assert_bytes(actual, expected)
    for batch in batches:
        assert sic.accumulate_hessian(batch, layer, actual) is actual
        reference.accumulate_hessian(batch, layer, expected)
        assert_bytes(actual, expected)
    assert_bytes(
        assemble_activation_quantization_groups(batches, 3, torch.device("cpu")),
        reference.assemble_activation_quantization_groups(
            batches, 3, torch.device("cpu")
        ),
    )
