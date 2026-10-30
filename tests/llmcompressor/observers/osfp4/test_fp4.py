# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

import pytest
import torch
from compressed_tensors.compressors.nvfp4.helpers import (
    pack_fp4_to_uint8,
    unpack_fp4_from_uint8,
)
from compressed_tensors.quantization import fake_quantize, preset_name_to_scheme
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA

from llmcompressor.observers.osfp4.scale_selection import (  # noqa: E501
    _get_e4m3_scale_grid,
)
from tests.llmcompressor.observers.osfp4._naming_reference.fp4 import (
    quantize_e2m1,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_local_e2m1_matches_compressed_tensors_at_all_boundaries(dtype):
    boundaries = torch.tensor(
        [0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=dtype,
    )
    lower = torch.nextafter(boundaries, torch.full_like(boundaries, -torch.inf))
    upper = torch.nextafter(boundaries, torch.full_like(boundaries, torch.inf))
    values = torch.cat((lower, boundaries, upper, -lower, -boundaries, -upper))
    values = torch.cat((values, torch.tensor([0.0, -0.0], dtype=dtype)))
    original = values.clone()

    actual = quantize_e2m1(values)
    expected = FP4_E2M1_DATA.cast_to_fp4(values.clone())
    assert torch.equal(values, original)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_local_e2m1_matches_random_finite_inputs(dtype):
    generator = torch.Generator().manual_seed(23)
    values = (torch.randn(4096, generator=generator) * 10).to(dtype)
    actual = quantize_e2m1(values)
    expected = FP4_E2M1_DATA.cast_to_fp4(values.clone())
    assert torch.equal(actual, expected)


def test_local_e2m1_matches_canonical_nvfp4_fake_quantize():
    values = torch.tensor(
        [
            -8.0,
            -5.0,
            -3.5,
            -2.5,
            -1.75,
            -1.25,
            -0.75,
            -0.25,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            8.0,
        ],
        dtype=torch.float32,
    ).unsqueeze(0)
    args = preset_name_to_scheme("NVFP4", ["Linear"]).weights
    scale = torch.ones((1, 1), dtype=torch.float32)
    zero_point = torch.zeros((1, 1), dtype=args.zp_dtype)

    actual = fake_quantize(
        values,
        scale,
        zero_point,
        args,
        global_scale=torch.ones(1),
    )
    expected = quantize_e2m1(values)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_local_e2m1_matches_for_every_e4m3_scale(dtype):
    scales = _get_e4m3_scale_grid("cpu", torch.float32).to(dtype).unsqueeze(1)
    codes = torch.tensor(
        [
            -6.0,
            -5.0,
            -3.5,
            -2.5,
            -1.75,
            -1.25,
            -0.75,
            -0.25,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            6.0,
        ],
        dtype=dtype,
    ).unsqueeze(0)
    normalized = (codes * scales) / scales
    actual = quantize_e2m1(normalized)
    expected = FP4_E2M1_DATA.cast_to_fp4(normalized.clone())
    assert torch.equal(actual, expected)


def test_nvfp4_pack_preserves_codes_and_signed_zero_bits():
    codes = torch.tensor(
        [[0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 6.0, -6.0]],
        dtype=torch.float32,
    )
    packed = pack_fp4_to_uint8(codes)
    unpacked = unpack_fp4_from_uint8(packed, 1, codes.shape[1], dtype=codes.dtype)

    assert torch.equal(unpacked, codes)
    assert torch.equal(torch.signbit(unpacked), torch.signbit(codes))
    assert int(packed[0, 0]) == 0x80


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.get_device_module("cuda").is_available(),
                reason="CUDA unavailable",
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_library_quantizer_matches_frozen_bytes_for_every_finite_16bit_input(
    dtype, device
):
    from ._naming_reference.fp4 import quantize_e2m1 as reference

    values = (
        torch.arange(65536, dtype=torch.int32).to(torch.int16).view(dtype).to(device)
    )
    values = values[torch.isfinite(values)]
    expected = reference(values)
    actual = FP4_E2M1_DATA.cast_to_fp4(values)
    assert_bytes(actual, expected)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.get_device_module("cuda").is_available(),
                reason="CUDA unavailable",
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("strided", [False, True])
def test_library_quantizer_matches_frozen_bytes_at_boundaries_and_finite_values(
    dtype, device, strided
):
    from ._naming_reference.fp4 import quantize_e2m1 as reference

    g = torch.Generator().manual_seed(239)
    random_bits = torch.randint(
        -(2**31), 2**31, (8192,), generator=g, dtype=torch.int32
    )
    random_values = random_bits.view(torch.float32).to(dtype)
    boundaries = torch.tensor([0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=dtype)
    edges = torch.cat(
        (
            boundaries,
            torch.nextafter(boundaries, torch.full_like(boundaries, -torch.inf)),
            torch.nextafter(boundaries, torch.full_like(boundaries, torch.inf)),
        )
    )
    values = torch.cat(
        (
            random_values,
            edges,
            -edges,
            torch.tensor([0.0, -0.0, torch.inf, -torch.inf, torch.nan], dtype=dtype),
        )
    ).to(device)
    values = values[torch.isfinite(values)]
    if strided:
        values = torch.stack((values, values), dim=1)[:, 0]
        assert not values.is_contiguous()
    expected = reference(values)
    assert_bytes(FP4_E2M1_DATA.cast_to_fp4(values), expected)
    empty = values[:0]
    assert_bytes(FP4_E2M1_DATA.cast_to_fp4(empty), reference(empty))
