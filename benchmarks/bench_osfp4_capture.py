# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Replay routed expert inputs through OSFP4 capture, without a model forward.

This measures capture overhead only. Synthetic inputs and routing are prepared
outside the timer and reused; results are not model throughput or quality scores.
Use separate processes with --source to compare two LLMC checkouts.
"""

import argparse
import functools
import hashlib
import importlib
import json
import math
import platform
import resource
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--mode", choices=("rtn", "sic"), default="rtn")
    parser.add_argument("--offload-hessians", action="store_true")
    parser.add_argument("--experts", type=positive_int, default=128)
    parser.add_argument("--top-k", type=positive_int, default=8)
    parser.add_argument("--input-width", type=positive_int, default=2048)
    parser.add_argument("--intermediate-width", type=positive_int, default=768)
    parser.add_argument("--sequence-length", type=positive_int, default=2048)
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--batches", type=positive_int, default=64)
    parser.add_argument("--repeat", type=positive_int, default=3)
    parser.add_argument(
        "--cap", type=positive_int, help="default: twice each input width"
    )
    parser.add_argument("--threads", type=positive_int, help="CPU intra-op threads")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--revision-label", help="provenance label for an archived source"
    )
    parser.add_argument("--phase", choices=("cold", "full", "both"), default="both")
    parser.add_argument(
        "--profile", type=Path, help="trace prefix; separate profiled runs"
    )
    parser.add_argument("--profile-batches", type=positive_int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k > args.experts:
        parser.error("--top-k cannot exceed --experts")
    if not (args.source / "src/llmcompressor").is_dir():
        parser.error("--source must be an LLMC checkout with src/llmcompressor")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error(
            "CUDA unavailable; --device cpu checks the harness, not CUDA speed"
        )
    return args


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def git_metadata(source):
    if not (source / ".git").exists():
        return {"revision": None, "dirty": None}
    return {
        "revision": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(source), "status", "--porcelain"], text=True
            ).strip()
        ),
    }


def build_workload(args, device):
    """Use valid top-k assignments; omit routing and input generation from timing."""
    generator = torch.Generator().manual_seed(args.seed)
    tokens = args.batch_size * args.sequence_length
    assignments = (
        torch.rand(tokens, args.experts, generator=generator)
        .topk(args.top_k, dim=1)
        .indices
    )
    counts = torch.bincount(assignments.flatten(), minlength=args.experts).tolist()
    workload = []
    for expert, count in enumerate(counts):
        for projection, width in (
            ("gate_up", args.input_width),
            ("down", args.intermediate_width),
        ):
            values = torch.randn(count, width, generator=generator).to(
                device=device, dtype=torch.bfloat16
            )
            # The cache needs the input width and weight device, not real weights.
            # No forward is executed and no expert matmul is included in timing.
            layer = torch.nn.Linear(
                width, 1, bias=False, device=device, dtype=torch.bfloat16
            )
            workload.append(
                (f"experts.{expert}.{projection}", layer, values, args.cap or 2 * width)
            )
    synchronize(device)
    return workload


@contextmanager
def profile_ranges(cache_module, sampling_module):
    """Instrumentation is confined to the separate profiled run."""
    targets = [
        (cache_module.OSFP4CalibrationCache, name)
        for name in (
            "_accumulate_calibration_statistics",
            "_capture_calibration_batch_cuda",
            "_capture_reservoir_batch_cuda",
            "_commit_reservoir_capture",
            "_update_activation_reservoir",
            "wait_for_capture",
        )
    ] + [(sampling_module.ActivationReservoir, "select_batch")]
    queue = getattr(cache_module, "_ReservoirCaptureQueue", None)
    if queue is not None:
        targets += [
            (queue, name) for name in ("poll", "drain", "acquire", "_commit_oldest")
        ]
    with ExitStack() as stack:
        for owner, name in targets:
            original = getattr(owner, name, None)
            if original is None:
                continue

            def instrument(fn, label):
                @functools.wraps(fn)
                def wrapped(*args, **kwargs):
                    with torch.profiler.record_function(label):
                        return fn(*args, **kwargs)

                return wrapped

            stack.enter_context(
                patch.object(
                    owner, name, instrument(original, f"osfp4.{owner.__name__}.{name}")
                )
            )
        yield


def make_cache(args, cache_class, workload):
    cache = cache_class(offload_hessians=args.offload_hessians)
    hooks = [
        cache.make_capture_hook(
            name,
            optimization_mode=args.mode,
            activation_subsample_size=cap,
            reservoir_sampling=True,
        )
        for name, _, _, cap in workload
    ]
    return cache, hooks


def replay(hooks, workload, batches):
    for _ in range(batches):
        for hook, (_, layer, values, _) in zip(hooks, workload):
            hook(layer, (values,))


def drain(cache, workload, device):
    for name, _, _, _ in workload:
        cache.wait_for_capture(name, torch.device("cpu"))
    synchronize(device)


def run_once(args, cache_module, sampling_module, workload, device, phase, trace=None):
    cache, hooks = make_cache(args, cache_module.OSFP4CalibrationCache, workload)
    # Prefill with the same stream instead of manufacturing reservoir metadata.
    prefill = (
        max(
            math.ceil(cap / len(values))
            for _, _, values, cap in workload
            if len(values)
        )
        if phase == "full"
        else 0
    )
    try:
        replay(hooks, workload, prefill)
        drain(cache, workload, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        batches = args.profile_batches if trace else args.batches
        with ExitStack() as stack:
            profiler = None
            if trace:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                stack.enter_context(profile_ranges(cache_module, sampling_module))
                profiler = stack.enter_context(
                    torch.profiler.profile(
                        activities=activities, record_shapes=True, profile_memory=True
                    )
                )
            start = time.perf_counter()
            replay(hooks, workload, batches)
            # Outstanding copies and commits belong to capture time, not cleanup.
            drain(cache, workload, device)
            elapsed = time.perf_counter() - start
        if profiler:
            trace.parent.mkdir(parents=True, exist_ok=True)
            profiler.export_chrome_trace(str(trace))
        sample_hash = hashlib.sha256()
        retained = 0
        for name, _, values, cap in workload:
            cache.validate_sampling(name)
            expected = (prefill + batches) * len(values)
            assert cache.token_count.get(name, 0) == expected
            if expected:
                indices = cache.sampled_indices[name]
                assert indices.numel() == min(cap, expected)
                sample_hash.update(name.encode())
                sample_hash.update(indices.cpu().numpy().tobytes())
                retained += indices.numel()
        result = {
            "seconds": elapsed,
            "source_tokens_per_second": batches
            * args.batch_size
            * args.sequence_length
            / elapsed,
            "prefill_batches": prefill,
            "timed_batches": batches,
            "retained_activation_rows": retained,
            "sample_index_sha256": sample_hash.hexdigest(),
            # Linux uses KiB; macOS uses bytes. This is cumulative process HWM.
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        }
        if device.type == "cuda":
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(
                device
            )
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        return result
    finally:
        cache.clear_all()
        synchronize(device)


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    source = args.source.resolve()
    sys.path.insert(0, str(source / "src"))
    cache_module = importlib.import_module(
        "llmcompressor.modifiers.osfp4.calibration_cache"
    )
    sampling_module = importlib.import_module(
        "llmcompressor.modifiers.osfp4.activation_subsampling"
    )
    loaded = Path(cache_module.__file__).resolve()
    if not loaded.is_relative_to(source / "src"):
        raise RuntimeError(f"Wrong source imported: {loaded}")
    device = torch.device(args.device)
    workload = build_workload(args, device)
    # Warm kernels on a disposable cache; measured cold runs start empty.
    warm_cache, warm_hooks = make_cache(
        args, cache_module.OSFP4CalibrationCache, workload
    )
    try:
        replay(warm_hooks, workload, 2)
        drain(warm_cache, workload, device)
    finally:
        warm_cache.clear_all()
    phases = ("cold", "full") if args.phase == "both" else (args.phase,)
    report = {
        "kind": "synthetic_osfp4_capture_only",
        "source": str(source),
        **git_metadata(source),
        "revision_label": args.revision_label,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "threads": torch.get_num_threads(),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "active_mappings": sum(bool(len(values)) for _, _, values, _ in workload),
        "maximum_reservoir_bytes": sum(
            cap * values.shape[1] * values.element_size()
            for _, _, values, cap in workload
            if len(values)
        ),
        "maximum_hessian_bytes": sum(
            values.shape[1] ** 2 * 4 for _, _, values, _ in workload if len(values)
        )
        if args.mode == "sic"
        else 0,
        "phases": {},
    }
    for phase in phases:
        runs = [
            run_once(args, cache_module, sampling_module, workload, device, phase)
            for _ in range(args.repeat)
        ]
        report["phases"][phase] = {
            "runs": runs,
            "median_seconds": statistics.median(r["seconds"] for r in runs),
            "median_source_tokens_per_second": statistics.median(
                r["source_tokens_per_second"] for r in runs
            ),
        }
    # Keep profiler initialization and tracing out of every unprofiled phase.
    if args.profile:
        for phase in phases:
            trace = args.profile.with_name(f"{args.profile.name}-{phase}.json")
            run_once(
                args, cache_module, sampling_module, workload, device, phase, trace
            )
            report["phases"][phase]["profile_trace"] = str(trace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
