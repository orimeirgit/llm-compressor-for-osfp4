# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Block-chunked scale optimization must match the whole-tensor optimizer."""

import pytest
import torch

from llmcompressor.observers.osfp4 import observer as osfp4_observer
from llmcompressor.observers.osfp4.loss import (
    JointLossCoefficients,
    build_joint_loss_coefficients,
    build_weight_loss_coefficients,
    compute_joint_loss,
    compute_weight_loss,
)
from llmcompressor.observers.osfp4.scale_optimization import (
    create_log_scale_parameters,
    initialize_scale_values_from_absmax,
    run_chunked_optimizer,
    run_optimizer,
    select_blocks_per_chunk,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert torch.equal(
        actual.detach().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def make_blocks(L=7, m=5, k=11, device="cpu"):
    generator = torch.Generator().manual_seed(1234)
    W = torch.randn(L, m, 16, generator=generator) * 0.02
    X = torch.randn(L, 16, k, generator=generator)
    # Exact zeros and values outside the LUT domain on both sides.
    X[0, 0, :3] = 0.0
    X[1, 2, :2] = 1e-6
    X[2, 3, :2] = 1e4
    W[3, 0, :4] = 0.0
    W[4, 1, 0] = 5.0
    metric = torch.rand(L, 16, generator=generator) + 0.1
    return W.to(device), X.to(device), metric.to(device)


def optimize(W, X, metric, *, steps, blocks_per_chunk):
    """Run Adam with the whole-tensor runner (None) or the chunked runner."""
    L = W.shape[0]
    initial = initialize_scale_values_from_absmax(
        W, X, blocks_per_chunk=blocks_per_chunk
    )
    log_scales = create_log_scale_parameters(initial)
    if X is None:
        coefficients = build_weight_loss_coefficients(W, weight_metric=metric)
    else:
        coefficients = build_joint_loss_coefficients(W, X, weight_metric=metric)

    def loss(blocks, scales):
        if X is None:
            return compute_weight_loss(
                W[blocks],
                scales.log_alpha.exp(),
                scales.log_gamma_w.exp(),
                coefficients[blocks],
            ).sum()
        return compute_joint_loss(
            W[blocks],
            X[blocks],
            scales.log_alpha.exp(),
            scales.log_gamma_w.exp(),
            scales.log_gamma_x.exp(),
            JointLossCoefficients(
                coefficients.coeff_w[blocks], coefficients.coeff_x[blocks]
            ),
        ).sum()

    if blocks_per_chunk is None:
        run_optimizer(
            log_scales, lambda: loss(slice(None), log_scales), steps=steps, lr=0.12
        )
    else:
        run_chunked_optimizer(
            log_scales,
            loss,
            steps=steps,
            lr=0.12,
            num_blocks=L,
            blocks_per_chunk=blocks_per_chunk,
        )
    return [p.detach().clone() for p in log_scales if p is not None]


@pytest.mark.parametrize("weight_only", [False, True])
@pytest.mark.parametrize("blocks_per_chunk", [1, 2, 3, 7, 100])
def test_chunked_adam_matches_whole_tensor_bytes_on_cpu(weight_only, blocks_per_chunk):
    W, X, metric = make_blocks()
    X = None if weight_only else X

    expected = optimize(W, X, metric, steps=4, blocks_per_chunk=None)
    actual = optimize(W, X, metric, steps=4, blocks_per_chunk=blocks_per_chunk)

    for a, e in zip(actual, expected):
        assert_bytes(a, e)


@pytest.mark.parametrize("weight_only", [False, True])
@pytest.mark.parametrize("blocks_per_chunk", [1, 3, 7])
def test_chunked_initialization_matches_bytes(weight_only, blocks_per_chunk):
    W, X, _ = make_blocks()
    X = None if weight_only else X

    expected = initialize_scale_values_from_absmax(W, X)
    actual = initialize_scale_values_from_absmax(
        W, X, blocks_per_chunk=blocks_per_chunk
    )

    for a, e in zip(actual, expected):
        if e is None:
            assert a is None
        else:
            assert_bytes(a, e)


@pytest.mark.parametrize("weight_only", [False, True])
def test_observer_optimization_does_not_depend_on_chunking_on_cpu(
    monkeypatch, weight_only
):
    W, X, metric = make_blocks()

    def run():
        if weight_only:
            return osfp4_observer._optimize_weight_scales(W, metric, steps=3, lr=0.12)
        return osfp4_observer._optimize_joint_scales(
            W, X, weight_metric=metric, steps=3, lr=0.12
        )

    expected = run()  # small problem: one chunk
    monkeypatch.setattr(
        osfp4_observer, "select_blocks_per_chunk", lambda m, k, num_blocks: 2
    )
    actual = run()

    for a, e in zip(actual, expected):
        if e is None:
            assert a is None
        else:
            assert_bytes(a, e)


def test_chunked_optimizer_clears_gradients_and_rejects_empty_chunks():
    W, X, metric = make_blocks()
    log_scales = create_log_scale_parameters(initialize_scale_values_from_absmax(W, X))
    coefficients = build_joint_loss_coefficients(W, X, weight_metric=metric)

    def loss(blocks, scales):
        return compute_joint_loss(
            W[blocks],
            X[blocks],
            scales.log_alpha.exp(),
            scales.log_gamma_w.exp(),
            scales.log_gamma_x.exp(),
            JointLossCoefficients(
                coefficients.coeff_w[blocks], coefficients.coeff_x[blocks]
            ),
        ).sum()

    run_chunked_optimizer(
        log_scales, loss, steps=1, lr=0.1, num_blocks=W.shape[0], blocks_per_chunk=3
    )
    assert all(p.grad is None for p in log_scales)
    with pytest.raises(ValueError, match="positive"):
        run_chunked_optimizer(
            log_scales, loss, steps=1, lr=0.1, num_blocks=W.shape[0], blocks_per_chunk=0
        )


def test_blocks_per_chunk_depends_only_on_shape():
    # Small problems keep a single chunk, matching the whole-tensor path.
    assert select_blocks_per_chunk(m=5, k=11, num_blocks=7) == 7
    # Llama-3.1-8B shapes: qkv, gate_up, and down_proj with auto subsampling.
    assert select_blocks_per_chunk(m=6144, k=8192, num_blocks=256) == 16
    assert select_blocks_per_chunk(m=28672, k=8192, num_blocks=256) == 16
    assert select_blocks_per_chunk(m=4096, k=28672, num_blocks=896) == 56
    # Weight-only gate_up is still large enough to split.
    assert select_blocks_per_chunk(m=28672, k=0, num_blocks=256) == 16
    # Never fewer than one block per chunk.
    assert select_blocks_per_chunk(m=10**6, k=10**6, num_blocks=3) == 1


@CUDA
def test_chunked_adam_is_deterministic_and_close_to_whole_tensor_on_cuda():
    W, X, metric = make_blocks(L=64, m=512, k=1024, device="cuda")

    expected = optimize(W, X, metric, steps=3, blocks_per_chunk=None)
    first = optimize(W, X, metric, steps=3, blocks_per_chunk=8)
    second = optimize(W, X, metric, steps=3, blocks_per_chunk=8)

    for a, b, e in zip(first, second, expected):
        assert_bytes(a, b)
        torch.testing.assert_close(a, e, rtol=1e-3, atol=1e-3)


@CUDA
def test_joint_scale_optimization_peak_memory_is_about_three_copies():
    device = torch.device("cuda")
    L, m, k = 256, 4096, 16384  # n = 4096; whole-tensor peak is about 15 copies
    generator = torch.Generator().manual_seed(0)
    W = (torch.randn(L, m, 16, generator=generator) * 0.02).to(device)
    X = torch.randn(L, 16, k, generator=generator).to(device)
    metric = torch.rand(L, 16, generator=generator).to(device)
    block_bytes = (W.numel() + X.numel()) * 4

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    osfp4_observer._optimize_joint_scales(W, X, weight_metric=metric, steps=2, lr=0.12)
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - baseline

    # W_blocks and X_blocks already exist (copy 1); coefficients add copy 2 and the
    # chunk graph about one more.
    assert peak_extra <= 2.5 * block_bytes, peak_extra / block_bytes
