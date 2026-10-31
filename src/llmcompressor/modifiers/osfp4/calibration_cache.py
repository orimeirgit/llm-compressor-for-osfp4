# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch.nn import Linear

from .activation_subsampling import (
    ActivationSubsample,
    _build_subsample_provenance,
    _select_activation_indices,
)
from .optimization.sic import accumulate_hessian, make_empty_hessian

__all__ = ["OSFP4CalibrationCache"]


@dataclass
class OSFP4CalibrationCache:
    """Own mapping inputs, raw statistic sums, token counts, and capture state."""

    offload_hessians: bool = False

    # Total tokens used to select activation subsampling indices before capture.
    total_num_tokens: dict[str, int] = field(default_factory=dict)
    sampled_indices: dict[str, torch.Tensor] = field(default_factory=dict)
    token_count: dict[str, int] = field(default_factory=dict)

    # CUDA queues for accumulating statistics and copying activations to CPU.
    capture_streams: dict[torch.device, torch.Stream] = field(default_factory=dict)
    # Completion markers used to wait until captured data is ready to use.
    capture_events: dict[str, dict[torch.device, torch.Event]] = field(default_factory=dict)

    # Sum of token outer products; divide by token_count for Sigma_X (SIC).
    hessian: dict[str, torch.Tensor] = field(default_factory=dict)
    # Sum of squared activations per channel; divide by token_count for sigma_X² (RTN).
    sigma_x_squared: dict[str, torch.Tensor] = field(default_factory=dict)
    # Used to update the input global scale after smoothing.
    input_channel_absmax: dict[str, torch.Tensor] = field(default_factory=dict)

    cached_activations: dict[str, list[torch.Tensor]] = field(default_factory=dict)

    def validate_sampling(self, mapping_name: str) -> None:
        """Check token and retained-row counts match the streaming sampling plan."""
        if mapping_name not in self.sampled_indices:
            return
        observed = self.token_count.get(mapping_name, 0)
        expected = self.total_num_tokens[mapping_name]
        if observed != expected:
            raise ValueError(
                f"OSFP4 {mapping_name}: expected {expected} calibration tokens, got {observed}"
            )
        retained = sum(len(batch) for batch in self.cached_activations.get(mapping_name, ()))
        if retained != self.sampled_indices[mapping_name].numel():
            raise ValueError(f"OSFP4 {mapping_name}: incorrect retained activation count")

    def prepare_sampled_activations(self, mapping_name: str, output_rows: int) -> ActivationSubsample:
        """Wait, validate, and combine already sampled rows for optimization.

        Return the subset and sampling metadata, or no subset if all tokens were kept.
        """
        self.wait(mapping_name, torch.device("cpu"))
        self.validate_sampling(mapping_name)
        indices = self.sampled_indices[mapping_name]
        total = self.total_num_tokens[mapping_name]
        batches = self.cached_activations[mapping_name]
        provenance = _build_subsample_provenance(indices, total, output_rows)
        if indices.numel() >= total:
            return ActivationSubsample(None, provenance)

        sampled = batches[0]
        if len(batches) > 1 or not sampled.is_contiguous():
            sampled = torch.empty(
                (indices.numel(), sampled.shape[1]),
                dtype=sampled.dtype,
                device="cpu",
                pin_memory=all(batch.is_pinned() for batch in batches),
            )
            torch.cat(batches, out=sampled)
        self.cached_activations[mapping_name] = [sampled]
        return ActivationSubsample((sampled,), provenance)

    def clear_all(self) -> None:
        """Clear all cached calibration data and capture streams."""
        self.cached_activations.clear()
        self.capture_events.clear()
        self.hessian.clear()
        self.sigma_x_squared.clear()
        self.token_count.clear()
        self.capture_streams.clear()
        self.sampled_indices.clear()
        self.total_num_tokens.clear()
        self.input_channel_absmax.clear()

    def clear_mapping(self, mapping_name: str) -> None:
        """Clear cached calibration data for one mapping."""
        self.cached_activations.pop(mapping_name, None)
        self.capture_events.pop(mapping_name, None)
        self.hessian.pop(mapping_name, None)
        self.sigma_x_squared.pop(mapping_name, None)
        self.token_count.pop(mapping_name, None)
        self.sampled_indices.pop(mapping_name, None)
        self.total_num_tokens.pop(mapping_name, None)
        self.input_channel_absmax.pop(mapping_name, None)

    def has_observation(self, mapping_name: str) -> bool:
        """Return whether a mapping has captured samples or inputs."""
        return self.token_count.get(mapping_name, 0) > 0 or bool(
            self.cached_activations.get(mapping_name)
        )

    def _get_capture_stream(self, device: torch.device) -> torch.Stream:
        """Return the reusable CUDA capture stream for a device."""
        stream = self.capture_streams.get(device)
        if stream is None:
            stream = torch.get_device_module(device).Stream(device=device)
            self.capture_streams[device] = stream
        return stream

    def make_capture_hook(
        self,
        mapping_name: str,
        *,
        capture_hessian: bool,
        cache_inputs: bool = True,
        capture_sigma_x_squared: bool = False,
        total_num_tokens: int | None = None,
        subsample_size: int | None = None,
    ) -> Callable[[Linear, tuple[torch.Tensor, ...]], None]:
        """Create a hook that captures the requested mapping statistics."""
        streaming = (
            cache_inputs
            and total_num_tokens is not None
            and subsample_size is not None
        )
        if streaming and (total_num_tokens <= 0 or subsample_size <= 0):
            raise ValueError(
                "expected tokens and activation sample size must be positive"
            )

        def update_statistics(
            layer: Linear,
            inputs: torch.Tensor,
            flat: torch.Tensor,
        ) -> None:
            """Accumulate enabled statistics from one captured activation batch."""
            if capture_hessian:
                if mapping_name not in self.hessian:
                    device: torch.device | str = (
                        "cpu" if self.offload_hessians else layer.weight.device
                    )
                    self.hessian[mapping_name] = make_empty_hessian(layer, device=device)
                hessian = self.hessian[mapping_name]
                if self.offload_hessians:
                    hessian = hessian.to(device=layer.weight.device)
                    self.hessian[mapping_name] = hessian
                hessian = accumulate_hessian(inputs, layer, hessian)
                self.hessian[mapping_name] = hessian
                if self.offload_hessians:
                    self.hessian[mapping_name] = hessian.to(device="cpu")
            if capture_sigma_x_squared:
                sigma_x_squared = flat.float().square().sum(dim=0)
                if mapping_name not in self.sigma_x_squared:
                    self.sigma_x_squared[mapping_name] = torch.zeros_like(sigma_x_squared)
                self.sigma_x_squared[mapping_name].add_(sigma_x_squared)
            if streaming and flat.shape[0]:
                maximum = flat.float().abs().amax(dim=0)
                previous = self.input_channel_absmax.get(mapping_name)
                self.input_channel_absmax[mapping_name] = (
                    maximum if previous is None else torch.maximum(previous, maximum)
                )

        def select_batch_indices(flat: torch.Tensor, start: int) -> torch.Tensor | None:
            """Locate selected activation rows within the current batch."""
            if not streaming:
                return None
            if mapping_name not in self.sampled_indices:
                self.sampled_indices[mapping_name] = _select_activation_indices(
                    total_num_tokens, subsample_size
                )
                self.total_num_tokens[mapping_name] = total_num_tokens
            if self.token_count[mapping_name] > total_num_tokens:
                self.validate_sampling(mapping_name)
            previous = self.cached_activations.get(mapping_name)
            if previous and previous[0].dtype != flat.dtype:
                raise ValueError("cached activation batches must share one dtype")
            selected = self.sampled_indices[mapping_name]
            if selected.numel() < total_num_tokens:
                lo, hi = torch.searchsorted(
                    selected, torch.tensor([start, start + flat.shape[0]])
                ).tolist()
                return selected[lo:hi] - start
            return None

        def capture_cuda(
            layer: Linear,
            inputs: torch.Tensor,
            flat: torch.Tensor,
            indices: torch.Tensor | None,
            retain: bool,
        ) -> torch.Tensor | None:
            """Accumulate statistics and copy retained rows on the capture stream."""
            rows = flat.shape[0] if indices is None else indices.numel()
            cached_activation = None
            if retain:
                cached_activation = (
                    torch.empty_like(flat, device="cpu", pin_memory=True)
                    if indices is None
                    else torch.empty(
                        (rows, flat.shape[1]),
                        dtype=flat.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                )
            compute_stream = torch.get_device_module(flat.device).current_stream(
                flat.device
            )
            source_ready = torch.get_device_module(flat.device).Event()
            source_ready.record(compute_stream)
            capture_stream = self._get_capture_stream(flat.device)
            with torch.get_device_module(flat.device).stream(capture_stream):
                capture_stream.wait_event(source_ready)
                update_statistics(layer, inputs, flat)
                if cached_activation is not None:
                    source = (
                        flat
                        if indices is None
                        else flat.index_select(0, indices.to(flat.device))
                    )
                    cached_activation.copy_(source, non_blocking=True)
                ready_event = torch.get_device_module(flat.device).Event()
                ready_event.record(capture_stream)
            # reshape may copy, so protect both the input and its flat view.
            inputs.record_stream(capture_stream)
            flat.record_stream(capture_stream)
            # One ordered capture stream per device: its newest event covers
            # all earlier captures for this mapping on that device.
            self.capture_events.setdefault(mapping_name, {})[flat.device] = ready_event
            return cached_activation

        def capture(layer: Linear, args: tuple[torch.Tensor, ...]) -> None:
            """Capture one layer input batch and schedule any CUDA copies."""
            inputs = args[0].detach()
            flat = inputs.reshape(-1, inputs.shape[-1])
            start = self.token_count.get(mapping_name, 0)
            self.token_count[mapping_name] = start + flat.shape[0]
            indices = select_batch_indices(flat, start)
            rows = flat.shape[0] if indices is None else indices.numel()
            retain = cache_inputs and (indices is None or rows > 0)

            if flat.device.type == "cuda":
                cached_activation = capture_cuda(layer, inputs, flat, indices, retain)
            else:
                update_statistics(layer, inputs, flat)
                cached_activation = None
                if retain:
                    cached_activation = (
                        flat.to(device="cpu", copy=True)
                        if indices is None
                        else flat.index_select(0, indices.to(flat.device)).cpu()
                    )
            if cached_activation is not None:
                self.cached_activations.setdefault(mapping_name, []).append(cached_activation)

        return capture

    def wait(self, mapping_name: str, device: torch.device) -> None:
        """Wait for outstanding capture events on the consuming device."""
        capture_events = self.capture_events.get(mapping_name, {}).values()
        if device.type == "cuda":
            stream = torch.get_device_module(device).current_stream(device)
            for event in capture_events:
                stream.wait_event(event)
            return
        for event in capture_events:
            event.synchronize()
