# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Compute OSFP4 quantization results for one mapping."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from .calibration_cache import OSFP4CalibrationCache
from .optimization import (
    RTNQuantizationResult,
    SICQuantizationResult,
    optimize_rtn,
    optimize_sic,
)

if TYPE_CHECKING:
    from .base import OSFP4Mapping

__all__ = ["quantize_mapping"]


def quantize_mapping(
    mapping: OSFP4Mapping,
    cache: OSFP4CalibrationCache,
    *,
    mode: Literal["rtn", "sic"],
    dampening_frac: float,
    weight_only: bool,
    optimization_input_batches: tuple[torch.Tensor, ...] | None = None,
) -> RTNQuantizationResult | SICQuantizationResult:
    """Optimize one mapping without installing weights or smoothing."""
    layers = mapping.balance_layers
    weight_dtype = layers[0].weight.dtype
    device = layers[0].weight.device
    weight = torch.cat(
        [
            layer.weight.detach().to(device=device, dtype=torch.float32)
            for layer in layers
        ],
        dim=0,
    )
    cache.wait(mapping.mapping_name, weight.device)
    alpha_dtype = torch.bfloat16 if mapping.requires_runtime_smoothing else weight_dtype
    observer = layers[0].weight_observer
    input_batches = (
        None if weight_only else cache.cached_activations[mapping.mapping_name]
    )
    optimizer_batches = (
        input_batches
        if optimization_input_batches is None
        else optimization_input_batches
    )
    token_count = cache.token_count[mapping.mapping_name]
    if mode == "sic":
        result = optimize_sic(
            weight,
            optimizer_batches,
            cache.hessian[mapping.mapping_name].to(weight.device) / token_count,
            observer=observer,
            alpha_dtype=alpha_dtype,
            weight_dtype=weight_dtype,
            percdamp=dampening_frac,
        )
    else:
        sigma_x_squared = (
            cache.sigma_x_squared[mapping.mapping_name].to(weight.device) / token_count
        )
        result = optimize_rtn(
            weight,
            optimizer_batches,
            observer=observer,
            alpha_dtype=alpha_dtype,
            sigma_x_squared=sigma_x_squared,
        )

    return result
