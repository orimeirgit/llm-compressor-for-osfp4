# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Deterministic activation-vector subsampling for OSFP4 optimization."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch

__all__ = [
    "ActivationSubsample",
    "subsample_activations",
]

_ACTIVATION_SUBSAMPLE_SEED = 42


@dataclass(frozen=True, slots=True)
class ActivationSubsample:
    """Optimization-only activation batches and their provenance."""

    batches: tuple[torch.Tensor, ...] | None
    provenance: dict[str, int | str]


def _gather_rows(
    batches: Sequence[torch.Tensor],
    indices: torch.Tensor,
    *,
    pin_memory: bool,
) -> torch.Tensor:
    """Collect the selected activation rows from multiple batches into one tensor."""
    first = batches[0]
    sampled = torch.empty(
        (indices.numel(), first.shape[1]),
        dtype=first.dtype,
        device="cpu",
        pin_memory=pin_memory,
    )
    output_start = 0
    batch_start = 0
    for batch in batches:
        batch_end = batch_start + batch.shape[0]
        begin = int(torch.searchsorted(indices, batch_start, side="left").item())
        end = int(torch.searchsorted(indices, batch_end, side="left").item())
        if begin != end:
            local_indices = indices[begin:end] - batch_start
            output_end = output_start + local_indices.numel()
            sampled[output_start:output_end].copy_(batch.index_select(0, local_indices))
            output_start = output_end
        batch_start = batch_end
    return sampled


def subsample_activations(
    cached_activation_batches: Sequence[torch.Tensor],
    requested_size: int,
    *,
    output_rows: int,
) -> ActivationSubsample:
    """Select complete activation rows without changing global RNG state."""
    batches = cached_activation_batches
    dtype = batches[0].dtype
    total_tokens = 0
    all_pinned = True

    for batch in batches:
        if batch.dtype != dtype:
            raise ValueError("cached activation batches must share one dtype")
        total_tokens += batch.shape[0]
        all_pinned = all_pinned and batch.is_pinned()

    size = min(total_tokens, requested_size)

    if size == total_tokens:
        indices = torch.arange(total_tokens, device="cpu", dtype=torch.int64)
        selected = None  # Reuse all original batches without copying.
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_ACTIVATION_SUBSAMPLE_SEED)
        indices = torch.randperm(
            total_tokens, generator=generator, device="cpu", dtype=torch.int64
        )[:size]
        indices = indices.sort().values
        selected = (_gather_rows(batches, indices, pin_memory=all_pinned),)

    return ActivationSubsample(
        batches=selected,
        provenance={
            "policy": "fixed",
            "seed": _ACTIVATION_SUBSAMPLE_SEED,
            "m": output_rows,
            "k": size,
            "k1": total_tokens,
            "index_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
        },
    )
