# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Unknown-length expert streams retain bounded, reproducible activation samples."""

import random
from collections import Counter
from itertools import combinations, product
from math import comb, factorial

import pytest
import torch

from llmcompressor.modifiers.osfp4.activation_subsampling import ActivationReservoir
from llmcompressor.modifiers.osfp4.calibration_cache import OSFP4CalibrationCache

from .test_streaming_capture import DEVICES, assert_bytes


def reference_indices(num_rows, cap):
    """Straight row-by-row Algorithm R, independent of batch replacement merging."""
    rng = random.Random(42)
    selected = list(range(min(num_rows, cap)))
    for index in range(cap, num_rows):
        slot = rng.randrange(index + 1)
        if slot < cap:
            selected[slot] = index
    return torch.tensor(sorted(selected), dtype=torch.int64)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["rtn", "sic"])
@pytest.mark.parametrize("cap", [1, 7, 300])
@pytest.mark.parametrize("batch_size", [1, 19, 257])
def test_reservoir_matches_reference_and_all_row_statistics(
    device, mode, cap, batch_size
):
    # Unique, exactly representable BF16 rows make row provenance unambiguous.
    rows = torch.arange(257).float()[:, None].expand(-1, 16).clone()
    rows[:, 1] = -rows[:, 1]
    rows = rows.to(dtype=torch.bfloat16, device=device)
    layer = torch.nn.Linear(16, 4).to(device=device, dtype=torch.bfloat16)
    cache = OSFP4CalibrationCache(offload_hessians=True)
    full_cache = OSFP4CalibrationCache(offload_hessians=True)
    capture = cache.make_capture_hook(
        "expert",
        optimization_mode=mode,
        activation_subsample_size=cap,
        reservoir_sampling=True,
    )
    capture_all = full_cache.make_capture_hook("expert", optimization_mode=mode)
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    allocation = None
    for batch in rows.split(batch_size):
        capture(layer, (batch,))
        capture_all(layer, (batch,))
        cache.validate_sampling("expert")
        cached = cache.cached_activations["expert"]
        assert len(cached) == 1
        assert len(cached[0]) == min(cap, cache.token_count["expert"])
        # A sliced view must not hide an allocation proportional to tokens seen.
        storage = cached[0].untyped_storage()
        assert storage.nbytes() == cap * 16 * rows.element_size()
        if allocation is None:
            allocation = storage.data_ptr()
        assert storage.data_ptr() == allocation
    capture(layer, (rows[:0],))
    assert random.getstate() == python_rng
    assert_bytes(torch.get_rng_state(), torch_rng)
    assert cache.token_count["expert"] == 257
    cache.wait_for_capture("expert", torch.device("cpu"))
    full_cache.wait_for_capture("expert", torch.device("cpu"))
    statistic = "hessian" if mode == "sic" else "sigma_x_squared_sum"
    assert_bytes(
        getattr(cache, statistic)["expert"], getattr(full_cache, statistic)["expert"]
    )
    assert_bytes(cache.input_channel_absmax["expert"], rows.float().abs().amax(0))
    expected_indices = reference_indices(257, cap)
    result = cache.prepare_sampled_activations("expert", output_rows=4)
    assert_bytes(result.batches[0], rows.cpu()[expected_indices])
    assert result.provenance["algorithm"] == "reservoir"
    assert result.provenance["k"] == min(cap, 257)
    assert result.provenance["k1"] == 257
    repeated = cache.prepare_sampled_activations("expert", output_rows=4)
    assert repeated.provenance == result.provenance
    assert_bytes(repeated.batches[0], result.batches[0])
    cache.clear_mapping("expert")
    assert not cache.reservoirs
    assert not cache.cached_activations
    assert not cache.sampled_indices
    assert not cache.input_channel_absmax
    assert not cache.has_calibration_data("expert")


@pytest.mark.parametrize("cap", [1, 17, 4096])
def test_storage_stops_growing_at_cap(cap):
    cache = OSFP4CalibrationCache()
    layer = torch.nn.Linear(4, 4)
    capture = cache.make_capture_hook(
        "expert",
        optimization_mode="rtn",
        activation_subsample_size=cap,
        reservoir_sampling=True,
    )
    batch = torch.arange(128 * 4).reshape(128, 4).float()
    for _ in range(128):
        capture(layer, (batch,))
        assert len(cache.cached_activations["expert"][0]) <= cap
        assert (
            cache.cached_activations["expert"][0].untyped_storage().nbytes()
            == cap * 4 * 4
        )
    assert cache.token_count["expert"] == 16384
    assert len(cache.reservoirs["expert"].indices) == cap
    cache.clear_all()
    assert not cache.reservoirs
    assert not cache.cached_activations


def test_expert_samples_are_independent_of_interleaving_and_batch_boundaries():
    rows = torch.arange(128 * 4).reshape(128, 4).float()
    layer = torch.nn.Linear(4, 4)
    cache = OSFP4CalibrationCache()
    captures = {
        name: cache.make_capture_hook(
            name,
            optimization_mode="rtn",
            activation_subsample_size=7,
            reservoir_sampling=True,
        )
        for name in ("expert0", "expert1", "unused")
    }
    captures["unused"](layer, (rows[:0],))
    assert not cache.has_calibration_data("unused")
    for batch in rows.split(3):
        captures["expert0"](layer, (batch,))
        captures["expert1"](layer, (batch + 1000,))
    # A fresh capture with one large batch must select exactly the same rows.
    other = OSFP4CalibrationCache()
    capture = other.make_capture_hook(
        "expert",
        optimization_mode="rtn",
        activation_subsample_size=7,
        reservoir_sampling=True,
    )
    capture(layer, (rows,))
    expected = other.prepare_sampled_activations("expert", 4)
    for name, offset in (("expert0", 0), ("expert1", 1000)):
        actual = cache.prepare_sampled_activations(name, 4)
        assert actual.provenance == expected.provenance
        assert_bytes(actual.batches[0], expected.batches[0] + offset)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"activation_subsample_size": None},
        {"activation_subsample_size": 0},
        {"activation_subsample_size": 3, "cache_activations": False},
        {"activation_subsample_size": 3, "total_num_tokens": 10},
    ],
)
def test_invalid_reservoir_setup(kwargs):
    with pytest.raises(ValueError, match="Reservoir capture"):
        OSFP4CalibrationCache().make_capture_hook(
            "expert",
            optimization_mode="rtn",
            reservoir_sampling=True,
            **kwargs,
        )


@pytest.mark.parametrize("cap", [1, 2, 3])
@pytest.mark.parametrize("batch_size", [1, 2, 6])
def test_algorithm_r_uniform_over_every_possible_draw_sequence(cap, batch_size):
    """Exhaustively check equal subset probabilities, without statistical tolerances."""
    num_rows = 6
    subset_counts = Counter()

    class ScriptedDraws:
        def __init__(self, choices):
            self.draws = iter(enumerate(choices, start=cap + 1))

        def randrange(self, stop):
            expected_stop, choice = next(self.draws)
            assert stop == expected_stop
            return choice

    # Every sequence has equal probability: 1 / ((K+1) * ... * N).
    for choices in product(*(range(n) for n in range(cap + 1, num_rows + 1))):
        reservoir = ActivationReservoir(cap)
        reservoir.generator = ScriptedDraws(choices)
        for offset in range(0, num_rows, batch_size):
            reservoir.select_batch(offset, min(batch_size, num_rows - offset))
        subset_counts[tuple(sorted(reservoir.indices))] += 1

    num_sequences = factorial(num_rows) // factorial(cap)
    expected_count = num_sequences // comb(num_rows, cap)
    assert subset_counts == Counter(
        {subset: expected_count for subset in combinations(range(num_rows), cap)}
    )
