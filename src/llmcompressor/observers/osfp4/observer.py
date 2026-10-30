# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

from typing import NamedTuple

import torch
from loguru import logger

from llmcompressor.observers.base import Observer

from .loss import (
    build_joint_loss_coefficients,
    build_weight_loss_coefficients,
    compute_joint_loss,
    compute_weight_loss,
)
from .scale_optimization import (
    QuantizationGroupScaleValues,
    canonicalize_log_scale_parameters,
    create_log_scale_parameters,
    initialize_scale_values_from_absmax,
    materialize_scale_values,
    run_optimizer,
)
from .scale_selection import (
    _search_e4m3_gamma_star,
)

# Scale optimization


class OptimizedQuantizationGroupScales(NamedTuple):
    """Deployment-rounded alpha ``[L, 16]`` and continuous gamma_w ``[L, m, 1]``."""

    alpha: torch.Tensor
    gamma_w: torch.Tensor


def _optimize_joint_scales(
    W_blocks: torch.Tensor,
    X_blocks: torch.Tensor,
    *,
    weight_metric: torch.Tensor,
    steps: int,
    lr: float,
) -> QuantizationGroupScaleValues:
    """Learn alpha, gamma_w, and gamma_x from W_blocks, X_blocks, and weight_metric."""
    W_blocks = W_blocks.detach().to(dtype=torch.float32)
    X_blocks = X_blocks.detach().to(
        device=W_blocks.device,
        dtype=torch.float32,
    )
    weight_metric = weight_metric.detach().to(
        device=W_blocks.device,
        dtype=torch.float32,
    )
    initial_scales = initialize_scale_values_from_absmax(W_blocks, X_blocks)
    coefficients = build_joint_loss_coefficients(
        W_blocks,
        X_blocks,
        weight_metric=weight_metric,
    )
    log_scales = create_log_scale_parameters(initial_scales)

    def loss() -> torch.Tensor:
        """Evaluate the summed joint weight and activation loss."""
        return compute_joint_loss(
            W_blocks,
            X_blocks,
            log_scales.log_alpha.exp(),
            log_scales.log_gamma_w.exp(),
            log_scales.log_gamma_x.exp(),  # type: ignore[union-attr]
            coefficients,
        ).sum()

    run_optimizer(log_scales, loss, steps=steps, lr=lr)
    canonicalize_log_scale_parameters(log_scales)
    return materialize_scale_values(log_scales)


def _optimize_weight_scales(
    W_blocks: torch.Tensor,
    weight_metric: torch.Tensor,
    *,
    steps: int,
    lr: float,
) -> QuantizationGroupScaleValues:
    """Learn A16 alpha and gamma_w from W_blocks and weight_metric."""
    W_blocks = W_blocks.detach().to(dtype=torch.float32)
    weight_metric = weight_metric.detach().to(
        device=W_blocks.device,
        dtype=torch.float32,
    )
    initial_scales = initialize_scale_values_from_absmax(W_blocks, None)
    coefficients = build_weight_loss_coefficients(
        W_blocks,
        weight_metric=weight_metric,
    )
    log_scales = create_log_scale_parameters(initial_scales)

    def loss() -> torch.Tensor:
        """Evaluate the summed weight-only loss."""
        return compute_weight_loss(
            W_blocks,
            log_scales.log_alpha.exp(),
            log_scales.log_gamma_w.exp(),
            coefficients,
        ).sum()

    run_optimizer(log_scales, loss, steps=steps, lr=lr)
    canonicalize_log_scale_parameters(log_scales)
    return materialize_scale_values(log_scales)


# Observer


@Observer.register("osfp4")
class OSFP4Observer(Observer):
    """Optimize alpha and return NVFP4 block qparams."""

    _act_sync_dict = {}

    # Calibration statistics

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        """Update per-observation minimum and maximum statistics."""
        self.min_vals = torch.amin(observed, dim=(0, -1))
        self.max_vals = torch.amax(observed, dim=(0, -1))

    # Optimization and deployment rounding

    def optimize_quantization_group_scales(
        self,
        W_blocks: torch.Tensor,
        *,
        X_blocks: torch.Tensor | None = None,
        weight_metric: torch.Tensor,
        alpha_dtype: torch.dtype,
    ) -> OptimizedQuantizationGroupScales:
        """Optimize continuous scales for L independent OSFP4 blocks."""
        W_blocks = W_blocks.detach().to(dtype=torch.float32)
        weight_metric = weight_metric.detach().to(
            device=W_blocks.device,
            dtype=torch.float32,
        )
        optimizer_config = self.args.observer_kwargs
        if X_blocks is None:
            learned_scales = _optimize_weight_scales(
                W_blocks,
                weight_metric,
                steps=optimizer_config["num_iters"],
                lr=optimizer_config["lr"],
            )
        else:
            learned_scales = _optimize_joint_scales(
                W_blocks,
                X_blocks,
                weight_metric=weight_metric,
                steps=optimizer_config["num_iters"],
                lr=optimizer_config["lr"],
            )

        alpha = learned_scales.alpha.to(dtype=alpha_dtype).float()
        scales_valid = (
            torch.isfinite(alpha).all()
            & torch.all(alpha > 0)
            & torch.isfinite(learned_scales.gamma_w).all()
            & torch.all(learned_scales.gamma_w > 0)
        )
        if learned_scales.gamma_x is not None:
            scales_valid = (
                scales_valid
                & torch.isfinite(learned_scales.gamma_x).all()
                & torch.all(learned_scales.gamma_x > 0)
            )
        if not scales_valid:
            raise ValueError("OSFP4 observer produced invalid scales")

        return OptimizedQuantizationGroupScales(alpha, learned_scales.gamma_w)

    # Checkpoint scale selection

    def select_weight_qparams(
        self,
        W_target_blocks: torch.Tensor,
        optimized_scales: OptimizedQuantizationGroupScales,
        *,
        weight_metric: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select E4M3 gamma_w_final ``[L, m, 1]`` and corresponding zero points."""
        gamma_w_final, invalid_rows, fallback_used = _search_e4m3_gamma_star(
            W_target_blocks,
            optimized_scales.alpha,
            optimized_scales.gamma_w,
            weight_metric,
        )

        # Read both device flags together at the existing validation boundary.
        has_invalid_rows, used_fallback = torch.stack(
            (invalid_rows.any(), fallback_used)
        ).tolist()
        if used_fallback:
            logger.warning(
                "OSFP4: empty scale candidate set; applied boundary fallback."
            )
        if has_invalid_rows:
            raise ValueError("OSFP4 observer produced invalid checkpoint scales")

        zero_points = torch.zeros_like(gamma_w_final, dtype=self.args.zp_dtype)
        return gamma_w_final, zero_points
