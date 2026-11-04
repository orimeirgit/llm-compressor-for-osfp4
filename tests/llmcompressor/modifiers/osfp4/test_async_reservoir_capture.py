# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Bounded staging must preserve reservoir writes and all-token statistics."""

from types import SimpleNamespace

import pytest
import torch

from llmcompressor.modifiers.osfp4.activation_subsampling import ActivationReservoir
from llmcompressor.modifiers.osfp4.calibration_cache import (
    OSFP4CalibrationCache,
    _PendingReservoirCapture,
    _ReservoirCaptureQueue,
)

from .test_reservoir_capture import reference_indices
from .test_streaming_capture import assert_bytes


class FakeEvent:
    def __init__(self, ready=False):
        self.ready = ready
        self.waits = 0

    def query(self):
        return self.ready

    def synchronize(self):
        self.waits += 1
        self.ready = True


def enqueue(queue, value, *, name="expert", ready=False):
    buffer = queue.acquire(2, torch.float32, pin_memory=False)
    assert buffer is not None
    buffer[:2].fill_(value)
    event = FakeEvent(ready)
    queue.pending.append(
        _PendingReservoirCapture(
            name,
            torch.tensor([0]),
            1,
            buffer,
            buffer[:2].view(1, 2),
            event,
        )
    )
    return buffer, event


def test_queue_commits_fifo_and_reuses_only_completed_buffers():
    committed = []
    queue = _ReservoirCaptureQueue(lambda entry: committed.append(entry.rows.clone()))
    first_buffer, first = enqueue(queue, 1)
    second_buffer, second = enqueue(queue, 2, ready=True)
    queue.poll()
    assert not committed  # A ready later write cannot overtake an earlier write.
    assert first.waits == second.waits == 0
    assert first_buffer.data_ptr() != second_buffer.data_ptr()
    first.ready = True
    queue.poll()
    assert [rows[0, 0].item() for rows in committed] == [1, 2]
    assert first.waits == second.waits == 0
    reused = queue.acquire(2, torch.float32, pin_memory=False)
    assert reused.data_ptr() in {first_buffer.data_ptr(), second_buffer.data_ptr()}
    assert queue.allocated_bytes == 16


@pytest.mark.parametrize("max_bytes,max_entries", [(16, 64), (64, 2)])
def test_queue_backpressure_is_bounded(max_bytes, max_entries):
    committed = []
    queue = _ReservoirCaptureQueue(
        lambda entry: committed.append(entry.rows[0, 0].item()),
        max_bytes=max_bytes,
        max_entries=max_entries,
    )
    _, first = enqueue(queue, 1)
    _, second = enqueue(queue, 2)
    enqueue(queue, 3)
    assert first.waits == 1 and second.waits == 0
    assert committed == [1]
    assert len(queue.pending) <= max_entries
    assert queue.allocated_bytes <= max_bytes
    queue.drain()
    assert committed == [1, 2, 3]


def test_queue_evicts_incompatible_idle_buffers_and_drains_oversize():
    queue = _ReservoirCaptureQueue(lambda _: None, max_bytes=16)
    enqueue(queue, 1, ready=True)
    queue.poll()
    replacement = queue.acquire(2, torch.float64, pin_memory=False)
    assert replacement.dtype == torch.float64
    assert queue.allocated_bytes == 16
    event = FakeEvent()
    queue.pending.append(
        _PendingReservoirCapture(
            "expert",
            torch.tensor([0]),
            1,
            replacement,
            replacement.view(1, 2),
            event,
        )
    )
    assert queue.acquire(5, torch.float32, pin_memory=False) is None
    assert event.waits == 1
    assert not queue.pending
    assert queue.allocated_bytes <= 16


def test_cache_commits_captured_length_and_drains_statistics_only_events():
    cache = OSFP4CalibrationCache()
    reservoir = cache.reservoirs["expert"] = ActivationReservoir(3)
    reservoir.select_batch(0, 3)  # Sampling has already advanced beyond first copy.
    cache.sampled_indices["expert"] = torch.arange(3)
    cache.token_count["expert"] = 3
    queue = _ReservoirCaptureQueue(cache._commit_reservoir_capture)
    cache.reservoir_queues[torch.device("cuda:0")] = queue
    buffer, first = enqueue(queue, 1)
    first.ready = True
    queue.poll()
    assert cache.cached_activations["expert"][0].shape == (1, 2)
    # Reuse its buffer, but preserve the final batch's distinct slot locations.
    staging = queue.acquire(4, torch.float32, pin_memory=False)
    staging.fill_(2)
    second = FakeEvent()
    queue.pending.append(
        _PendingReservoirCapture(
            "expert",
            torch.tensor([1, 2]),
            3,
            staging,
            staging.view(2, 2),
            second,
        )
    )
    statistics_only = FakeEvent()
    cache.capture_events["expert"] = {torch.device("cuda:0"): statistics_only}
    cache.validate_sampling("expert")
    assert second.waits == statistics_only.waits == 1
    assert_bytes(
        cache.cached_activations["expert"][0],
        torch.tensor([[1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]),
    )
    cache.clear_all()
    assert not cache.reservoir_queues
    assert not cache.cached_activations


def test_mapping_drain_preserves_other_pending_experts():
    committed = []
    queue = _ReservoirCaptureQueue(lambda entry: committed.append(entry.mapping_name))
    enqueue(queue, 1, name="first")
    enqueue(queue, 2, name="second")
    enqueue(queue, 3, name="first")
    _, last = enqueue(queue, 4, name="third")
    queue.drain("first")
    assert committed == ["first", "second", "first"]
    assert last.waits == 0
    assert len(queue.pending) == 1


def test_cuda_consumer_wait_also_commits_cpu_reservoir(monkeypatch):
    cache = OSFP4CalibrationCache()
    device = torch.device("cuda:0")
    reservoir = cache.reservoirs["expert"] = ActivationReservoir(1)
    reservoir.select_batch(0, 1)
    queue = _ReservoirCaptureQueue(cache._commit_reservoir_capture)
    cache.reservoir_queues[device] = queue
    _, copy_event = enqueue(queue, 3)
    statistics_event = FakeEvent()
    cache.capture_events["expert"] = {device: statistics_event}
    device_waits = []
    consumer = SimpleNamespace(wait_event=device_waits.append)
    cuda = SimpleNamespace(current_stream=lambda _: consumer)
    monkeypatch.setattr(torch, "get_device_module", lambda _: cuda)
    cache.wait_for_capture("expert", device)
    assert copy_event.waits == 1
    assert statistics_event.waits == 0
    assert device_waits == [statistics_event]
    assert_bytes(cache.cached_activations["expert"][0], torch.full((1, 2), 3.0))


def test_cpu_reservoir_keeps_index_storage_and_synchronous_rows():
    cache = OSFP4CalibrationCache()
    layer = torch.nn.Linear(4, 2)
    capture = cache.make_capture_hook(
        "expert",
        optimization_mode="sic",
        activation_subsample_size=7,
        reservoir_sampling=True,
    )
    rows = torch.arange(80).float().view(20, 4)
    allocation = None
    for batch in rows.split(3):
        capture(layer, (batch,))
        indices = cache.sampled_indices["expert"]
        if allocation is None:
            allocation = indices.untyped_storage().data_ptr()
        assert indices.untyped_storage().data_ptr() == allocation
        assert_bytes(cache.cached_activations["expert"][0], rows[indices])
    assert not cache.reservoir_queues
    torch.testing.assert_close(cache.hessian["expert"], rows.T @ rows)


def test_failed_capture_waits_before_recycling_staging(monkeypatch):
    cache = OSFP4CalibrationCache()
    device = torch.device("cpu")
    queue = _ReservoirCaptureQueue(cache._commit_reservoir_capture)
    buffer = queue.acquire(8, torch.float32, pin_memory=False)
    queue.pool.append(buffer)
    cache.reservoir_queues[device] = queue
    stream = FakeEvent()
    cache.capture_streams[device] = stream

    def fail(*args, **kwargs):
        assert not queue.pool
        assert stream.waits == 0
        raise RuntimeError("capture failed")

    monkeypatch.setattr(cache, "_capture_calibration_batch_cuda", fail)
    rows = torch.ones(2, 4)
    with pytest.raises(RuntimeError, match="capture failed"):
        cache._capture_reservoir_batch_cuda(
            "expert",
            torch.nn.Linear(4, 2),
            rows,
            rows,
            torch.arange(2),
            torch.arange(2),
            optimization_mode="rtn",
        )
    assert stream.waits == 1
    assert len(queue.pool) == 1
    assert queue.pool[0].data_ptr() == buffer.data_ptr()
    assert queue.allocated_bytes == 32


def test_queued_upload_retains_indices_and_requests_nonblocking_copy(monkeypatch):
    cache = OSFP4CalibrationCache()
    device = torch.device("cpu")
    queue = _ReservoirCaptureQueue(cache._commit_reservoir_capture)
    queue.pool.append(queue.acquire(8, torch.float32, pin_memory=False))
    cache.reservoir_queues[device] = queue
    reservoir = cache.reservoirs["expert"] = ActivationReservoir(2)
    reservoir.select_batch(0, 2)
    event = FakeEvent()

    def capture(*args, **kwargs):
        assert kwargs["non_blocking_indices"] is True
        cache.capture_events["expert"] = {device: event}
        staging = kwargs["staging_buffer"]
        staging.copy_(args[3])
        return staging

    monkeypatch.setattr(cache, "_capture_calibration_batch_cuda", capture)
    rows = torch.ones(2, 4)
    indices = torch.arange(2)
    cache._capture_reservoir_batch_cuda(
        "expert",
        torch.nn.Linear(4, 2),
        rows,
        rows,
        indices,
        indices,
        optimization_mode="rtn",
    )
    assert queue.pending[0].source_indices is indices
    assert event.waits == 0
    cache.wait_for_capture("expert", torch.device("cpu"))
    assert_bytes(cache.cached_activations["expert"][0], rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["rtn", "sic"])
@pytest.mark.parametrize("budget", [32, 1024 * 1024])
@pytest.mark.parametrize("offload_hessians", [False, True])
def test_cuda_queued_capture_matches_samples_and_all_row_statistics(
    mode, budget, offload_hessians, monkeypatch
):
    uploads = []
    original_to = torch.Tensor.to

    def audited_to(tensor, *args, **kwargs):
        if (
            tensor.device.type == "cpu"
            and tensor.dtype == torch.int64
            and args
            and getattr(args[0], "type", None) == "cuda"
        ):
            uploads.append(kwargs.get("non_blocking", False))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", audited_to)
    device = torch.device("cuda:0")
    cache = OSFP4CalibrationCache(offload_hessians=offload_hessians)
    reference = OSFP4CalibrationCache(offload_hessians=offload_hessians)
    cache.reservoir_queues[device] = _ReservoirCaptureQueue(
        cache._commit_reservoir_capture,
        max_bytes=budget,
        max_entries=2,
    )
    layer = torch.nn.Linear(16, 4, device=device, dtype=torch.bfloat16)
    hooks = {
        name: cache.make_capture_hook(
            name,
            optimization_mode=mode,
            activation_subsample_size=7,
            reservoir_sampling=True,
        )
        for name in ("a", "b")
    }
    reference_hooks = {
        name: reference.make_capture_hook(name, optimization_mode=mode)
        for name in hooks
    }
    rows = torch.arange(64, device=device).to(torch.bfloat16)[:, None].expand(-1, 16)
    for batch in rows.split(5):
        for name, offset in (("a", 0), ("b", 64)):
            value = batch + offset
            hooks[name](layer, (value,))
            reference_hooks[name](layer, (value,))
            queue = cache.reservoir_queues[device]
            assert queue.allocated_bytes <= budget
            assert len(queue.pending) <= 2
    for name, offset in (("a", 0), ("b", 64)):
        selection = cache.prepare_sampled_activations(name, 4)
        reference.wait_for_capture(name, torch.device("cpu"))
        assert_bytes(
            selection.batches[0], (rows + offset).cpu()[reference_indices(64, 7)]
        )
        statistic = "hessian" if mode == "sic" else "sigma_x_squared_sum"
        assert_bytes(
            getattr(cache, statistic)[name], getattr(reference, statistic)[name]
        )
        assert_bytes(
            cache.input_channel_absmax[name], (rows + offset).float().abs().amax(0)
        )
    cache.clear_all()
    assert not cache.reservoir_queues
    assert uploads and all(uploads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_cuda_delayed_producer_and_ephemeral_inputs_preserve_capture(mode):
    device = torch.device("cuda:0")
    cache = OSFP4CalibrationCache()
    layer = torch.nn.Linear(16, 4, device=device, dtype=torch.bfloat16)
    capture = cache.make_capture_hook(
        "expert",
        optimization_mode=mode,
        activation_subsample_size=7,
        reservoir_sampling=True,
    )
    # Build the reference independently on CPU. Reading back producer data here
    # would synchronize it and conceal a missing producer/capture dependency.
    expected = torch.arange(64).to(torch.bfloat16)[:, None].expand(-1, 16)
    producer = torch.cuda.Stream(device=device)
    with torch.cuda.stream(producer):
        for start in range(0, 64, 4):
            if hasattr(torch.cuda, "_sleep"):
                torch.cuda._sleep(1_000_000)
            value = (
                torch.arange(start, start + 4, device=device, dtype=torch.float32)
                .to(torch.bfloat16)[:, None]
                .expand(-1, 16)
                .clone()
            )
            capture(layer, (value,))
            del value
            # Encourage immediate reuse of the producer's freed allocations.
            # record_stream must retain capture inputs until the other stream
            # has finished its statistics and selected-row copy.
            for _ in range(4):
                torch.empty((4, 16), device=device, dtype=torch.bfloat16).fill_(-99)
    selection = cache.prepare_sampled_activations("expert", 4)
    assert_bytes(selection.batches[0], expected[reference_indices(64, 7)])
    if mode == "sic":
        assert_bytes(
            cache.hessian["expert"].cpu(), expected.float().T @ expected.float()
        )
    else:
        assert_bytes(
            cache.sigma_x_squared_sum["expert"].cpu(), expected.float().square().sum(0)
        )
    assert_bytes(
        cache.input_channel_absmax["expert"].cpu(), expected.float().abs().amax(0)
    )
    cache.clear_all()
