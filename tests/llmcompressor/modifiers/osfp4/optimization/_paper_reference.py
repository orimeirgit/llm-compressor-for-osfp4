# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Frozen optimization from cefa7cf6, before the paper-notation rename.

Numerical bodies below are copied from optimization/{rtn,sic,_activation_groups}.py.
The observer adapter at the end translates field names only. Do not update these
algorithms to match production changes.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch.nn import Linear

from llmcompressor.observers.osfp4 import (
    OptimizedQuantizationGroupScales as CurrentOptimizedScales,
)
from llmcompressor.observers.osfp4 import OSFP4Observer
from tests.llmcompressor.observers.osfp4._naming_reference.fp4 import quantize_e2m1

_QUANTIZATION_GROUP_SIZE = 16
SIC_PRECISION = torch.float32


class OptimizedQuantizationGroupScales(NamedTuple):
    alpha_star: torch.Tensor
    gamma_w: torch.Tensor


def assemble_activation_quantization_groups(
    cached_activation_batches: list[torch.Tensor] | tuple[torch.Tensor, ...],
    quantization_group_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Return contiguous FP32 activation groups ``[G, 16, samples]``."""
    sample_count = sum(
        batch.numel() // (quantization_group_count * _QUANTIZATION_GROUP_SIZE)
        for batch in cached_activation_batches
    )
    groups = torch.empty(
        quantization_group_count,
        _QUANTIZATION_GROUP_SIZE,
        sample_count,
        device=device,
        dtype=torch.float32,
    )
    start = 0
    for activations in cached_activation_batches:
        converted = (
            activations.detach()
            .to(
                device=device,
                dtype=torch.float32,
                non_blocking=(
                    activations.device.type == "cpu"
                    and device.type == "cuda"
                    and activations.is_pinned()
                ),
            )
            .reshape(-1, quantization_group_count, _QUANTIZATION_GROUP_SIZE)
            .permute(1, 2, 0)
        )
        end = start + converted.shape[-1]
        groups[:, :, start:end].copy_(converted)
        start = end
        # Release this transfer before allocating the next batch's FP32 buffer.
        del converted
    return groups


class RTNQuantizationResult(NamedTuple):
    """Final RTN alpha and checkpoint-layout weight qparams for one mapping."""

    alpha_star: torch.Tensor
    gamma_w_star: torch.Tensor
    weight_zero_point: torch.Tensor


def optimize_rtn(
    W: torch.Tensor,
    cached_activation_batches: list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
    *,
    observer: OSFP4Observer,
    alpha_dtype: torch.dtype,
    sigma_x_squared: torch.Tensor,
) -> RTNQuantizationResult:
    """Optimize RTN qparams for one validated ``W[rows, columns]`` mapping."""
    # Prepare weight groups and the normalized activation-energy metric.
    rows, columns = W.shape
    groups = columns // _QUANTIZATION_GROUP_SIZE
    weight_groups = (
        W.detach()
        .reshape(rows, groups, _QUANTIZATION_GROUP_SIZE)
        .permute(1, 0, 2)
        .contiguous()
    )
    metric = (
        sigma_x_squared.detach()
        .to(device=W.device, dtype=torch.float32)
        .reshape(groups, _QUANTIZATION_GROUP_SIZE)
    )
    activation_groups = None
    if cached_activation_batches is not None:
        activation_groups = assemble_activation_quantization_groups(
            cached_activation_batches,
            groups,
            W.device,
        )
    optimized = observer.optimize_quantization_group_scales(
        weight_groups,
        activation_quantization_groups=activation_groups,
        weight_metric=metric,
        alpha_dtype=alpha_dtype,
    )
    del activation_groups
    scales, zero_points = observer.select_weight_qparams(
        weight_groups,
        optimized,
        weight_metric=metric,
    )

    # Convert selected qparams to checkpoint layout.
    alpha_star = optimized.alpha_star.reshape(columns)
    gamma_w_star = scales.squeeze(-1).transpose(0, 1).contiguous()
    weight_zero_point = zero_points.squeeze(-1).transpose(0, 1).contiguous()

    return RTNQuantizationResult(alpha_star, gamma_w_star, weight_zero_point)


def make_empty_hessian(
    module: Linear,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Allocate an FP32 raw input Gram matrix for an OSFP4 Linear module."""
    device = module.weight.device if device is None else device
    columns = module.weight.shape[1]
    return torch.zeros(
        (columns, columns),
        device=device,
        dtype=SIC_PRECISION,
    )


def accumulate_hessian(
    inp: torch.Tensor,
    module: Linear,
    hessian: torch.Tensor,
) -> torch.Tensor:
    """Accumulate the raw FP32 activation Gram matrix ``X.T @ X``."""
    flat = inp.to(device=hessian.device, dtype=SIC_PRECISION).reshape(-1, inp.shape[-1])
    hessian.addmm_(flat.transpose(0, 1), flat)
    return hessian


class SICQuantizationResult(NamedTuple):
    """Final SIC qparams and post-alpha quantized weights for one mapping."""

    alpha_star: torch.Tensor
    gamma_w_star: torch.Tensor
    weight_zero_point: torch.Tensor
    quantized_weight: torch.Tensor


def optimize_sic(
    W: torch.Tensor,
    cached_activation_batches: list[torch.Tensor] | tuple[torch.Tensor, ...] | None,
    hessian: torch.Tensor,
    *,
    observer: OSFP4Observer,
    alpha_dtype: torch.dtype,
    weight_dtype: torch.dtype,
    percdamp: float = 0.01,
) -> SICQuantizationResult:
    """Optimize qparams and run right-to-left SIC for one validated mapping."""
    # Factor the damped second moment without mutating the cached Hessian.
    num_rows, num_columns = W.shape
    W = W.detach().to(dtype=torch.float32)
    H = hessian.detach().to(device=W.device, dtype=torch.float32).clone()
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(H.shape[0], device=H.device)
    H[diag, diag] += damp
    U = torch.linalg.cholesky(H, upper=True)
    diag_U = torch.diag(U)
    del H

    # Optimize original-weight groups together, defer E4M3 selection to the sweep.
    groups = num_columns // _QUANTIZATION_GROUP_SIZE
    weight_groups = (
        W.reshape(num_rows, groups, _QUANTIZATION_GROUP_SIZE)
        .permute(1, 0, 2)
        .contiguous()
    )
    metric = diag_U.square().reshape(
        groups,
        _QUANTIZATION_GROUP_SIZE,
    )
    activation_groups = None
    if cached_activation_batches is not None:
        activation_groups = assemble_activation_quantization_groups(
            cached_activation_batches, groups, W.device
        )
    optimized = observer.optimize_quantization_group_scales(
        weight_groups,
        activation_quantization_groups=activation_groups,
        weight_metric=metric,
        alpha_dtype=alpha_dtype,
    )
    del weight_groups, activation_groups

    # Store each block's selected qparams in checkpoint layout.
    alpha_values = optimized.alpha_star.reshape(num_columns)
    scale_values = W.new_empty((num_rows, groups))
    zero_point_values = torch.empty_like(scale_values)

    # Quantize right-to-left and propagate the alpha-restored residual.
    Y = U @ W.transpose(0, 1)
    Q = torch.zeros_like(W)
    for block_idx in range(groups - 1, -1, -1):
        block_start = block_idx * _QUANTIZATION_GROUP_SIZE
        block_end = block_start + _QUANTIZATION_GROUP_SIZE

        U_local = U[block_start:block_end, block_start:block_end]
        U_diag_block = diag_U[block_start:block_end]
        U_upper_strict = torch.triu(U_local, diagonal=1)
        W_local = W[:, block_start:block_end]
        unwanted_cross_terms = U_upper_strict @ W_local.transpose(0, 1)
        Y_local = Y[block_start:block_end, :]
        T_eff = (
            ((Y_local - unwanted_cross_terms) / U_diag_block[:, None])
            .transpose(0, 1)
            .unsqueeze(0)
        )
        block_optimized = OptimizedQuantizationGroupScales(
            optimized.alpha_star[block_idx : block_idx + 1],
            optimized.gamma_w[block_idx : block_idx + 1],
        )
        scales, zero_points = observer.select_weight_qparams(
            T_eff,
            block_optimized,
            weight_metric=metric[block_idx : block_idx + 1],
        )
        scale_1d = scales.view(-1)
        scale_values[:, block_idx] = scale_1d
        zero_point_values[:, block_idx] = zero_points.view(-1)
        opt_alpha_block = optimized.alpha_star[block_idx]

        for ii in range(block_end - 1, block_start - 1, -1):
            alpha_i = opt_alpha_block[ii - block_start]
            q_i = quantize_e2m1((Y[ii, :] / U[ii, ii]) / (alpha_i * scale_1d))
            Q[:, ii] = (q_i * scale_1d).to(dtype=weight_dtype).float()
            Y[:ii, :] -= U[:ii, ii].unsqueeze(1) * (alpha_i * Q[:, ii]).unsqueeze(0)

    return SICQuantizationResult(
        alpha_star=alpha_values,
        gamma_w_star=scale_values,
        weight_zero_point=zero_point_values,
        quantized_weight=Q,
    )


# Use the frozen observer so end-to-end comparisons include observer renames.
class _ObserverWithLegacyNames:
    def __init__(self, observer):
        self.observer = observer

    def optimize_quantization_group_scales(self, *args, **kwargs):
        from tests.llmcompressor.observers.osfp4._naming_reference.observer import (
            OSFP4Observer as FrozenObserver,
        )

        result = FrozenObserver.optimize_quantization_group_scales(
            self.observer, *args, **kwargs
        )
        return OptimizedQuantizationGroupScales(result.alpha, result.gamma_w)

    def select_weight_qparams(self, weight, optimized, *, weight_metric):
        from tests.llmcompressor.observers.osfp4._naming_reference.observer import (
            OSFP4Observer as FrozenObserver,
        )

        return FrozenObserver.select_weight_qparams(
            self.observer,
            weight,
            CurrentOptimizedScales(optimized.alpha_star, optimized.gamma_w),
            weight_metric=weight_metric,
        )


_baseline_rtn = optimize_rtn
_baseline_sic = optimize_sic


def optimize_rtn(*args, observer, **kwargs):
    return _baseline_rtn(*args, observer=_ObserverWithLegacyNames(observer), **kwargs)


def optimize_sic(*args, observer, **kwargs):
    return _baseline_sic(*args, observer=_ObserverWithLegacyNames(observer), **kwargs)
