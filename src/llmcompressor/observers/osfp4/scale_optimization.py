# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

from collections.abc import Callable
from typing import NamedTuple

import torch
from compressed_tensors.quantization.quant_args import FP4_E2M1_DATA

__all__ = [
    "QuantizationGroupScaleValues",
    "LogScaleParameters",
    "create_log_scale_parameters",
    "initialize_scale_values_from_absmax",
    "materialize_scale_values",
    "canonicalize_log_scale_parameters",
    "run_optimizer",
]

_FP8_MAX = 448.0
_FP8_SAFETY_FACTOR = 2.0


# Scale representations


class QuantizationGroupScaleValues(NamedTuple):
    """Positive scales: alpha ``[L, 16]``, gamma_w ``[L, m, 1]``,
    and optional gamma_x ``[L, 1, k]``."""

    alpha: torch.Tensor
    gamma_w: torch.Tensor
    gamma_x: torch.Tensor | None


class LogScaleParameters(NamedTuple):
    """Adam parameters for ``QuantizationGroupScaleValues`` in log space."""

    log_alpha: torch.Tensor
    log_gamma_w: torch.Tensor
    log_gamma_x: torch.Tensor | None


# Initialization


def initialize_scale_values_from_absmax(
    W_blocks: torch.Tensor,
    X_blocks: torch.Tensor | None,
) -> QuantizationGroupScaleValues:
    """Initialize alpha, gamma_w, and optional gamma_x from block absolute maxima."""

    def normalize(
        values: torch.Tensor,
        target_absmax: float,
        eps: float = 1e-12,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize each block and return its divisor."""
        divisors = torch.clamp(
            values.abs().amax(dim=(1, 2), keepdim=True) / target_absmax,
            min=eps,
        )
        return values / divisors, divisors

    def absmax_divisor(
        values: torch.Tensor,
        reduction_dim: int,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Return FP4 absolute-maximum divisors along one dimension."""
        maxima = torch.clamp(
            values.abs().amax(dim=reduction_dim, keepdim=True),
            min=eps,
        )
        return maxima / FP4_E2M1_DATA.max

    L, _, block_width = W_blocks.shape
    alpha = torch.ones(
        L,
        block_width,
        device=W_blocks.device,
        dtype=W_blocks.dtype,
    )
    target_absmax = FP4_E2M1_DATA.max * _FP8_MAX / _FP8_SAFETY_FACTOR
    W_blocks_normalized, weight_divisor = normalize(
        W_blocks,
        target_absmax,
    )
    gamma_w = weight_divisor * absmax_divisor(
        W_blocks_normalized / alpha.view(L, 1, -1),
        reduction_dim=2,
    )
    gamma_x = None
    if X_blocks is not None:
        X_blocks_normalized, activation_divisor = normalize(
            X_blocks,
            target_absmax,
        )
        gamma_x = activation_divisor * absmax_divisor(
            X_blocks_normalized * alpha.view(L, -1, 1),
            reduction_dim=1,
        )
    return QuantizationGroupScaleValues(
        alpha=alpha,
        gamma_w=gamma_w,
        gamma_x=gamma_x,
    )


def create_log_scale_parameters(
    scales: QuantizationGroupScaleValues,
) -> LogScaleParameters:
    """Create trainable log tensors from initialized scales."""
    return LogScaleParameters(
        scales.alpha.log().detach().clone().requires_grad_(True),
        scales.gamma_w.log().detach().clone().requires_grad_(True),
        None
        if scales.gamma_x is None
        else scales.gamma_x.log().detach().clone().requires_grad_(True),
    )


# Adam


def run_optimizer(
    log_scales: LogScaleParameters,
    loss_evaluator: Callable[[], torch.Tensor],
    *,
    steps: int,
    lr: float,
) -> None:
    """Optimize log scales."""
    optimizer = torch.optim.Adam(
        [parameter for parameter in log_scales if parameter is not None],
        lr=lr,
    )
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad()
            loss = loss_evaluator()
            loss.backward()
            optimizer.step()


# Canonicalization


def canonicalize_log_scale_parameters(log_scales: LogScaleParameters) -> None:
    """Center log_gamma_w and fold its mean into log_alpha in place."""
    with torch.no_grad():
        mean_log_gamma_w = log_scales.log_gamma_w.mean(dim=(1, 2), keepdim=True)
        log_scales.log_alpha.add_(mean_log_gamma_w.squeeze(-1))
        log_scales.log_gamma_w.sub_(mean_log_gamma_w)
        if log_scales.log_gamma_x is not None:
            log_scales.log_gamma_x.add_(mean_log_gamma_w)


# Materialization


def materialize_scale_values(
    log_scales: LogScaleParameters,
) -> QuantizationGroupScaleValues:
    """Return detached positive scales without mutating the optimized parameters."""
    return QuantizationGroupScaleValues(
        alpha=log_scales.log_alpha.detach().exp(),
        gamma_w=log_scales.log_gamma_w.detach().exp(),
        gamma_x=(
            None
            if log_scales.log_gamma_x is None
            else log_scales.log_gamma_x.detach().exp()
        ),
    )
