# OSFP4 capture performance

`bench_osfp4_capture.py` measures the cost of capturing routed expert activations.
It does **not** execute expert forwards, routing kernels, attention, scale
optimization, or error propagation. Its tokens/second are capture-only throughput,
not full-model calibration throughput or a quality score.

The script generates valid top-k expert assignments on CPU, then creates two input
tensors per expert: one for the shared gate/up mapping and one for down. These
tensors are placed on the selected device before timing and reused for every
batch. Counts vary between experts, but routing and values repeat between batches;
this deliberate replay isolates capture costs and is not representative training
data. Attention and shared-expert capture are omitted.

`cold` starts each measured repetition with empty caches. It includes first-use
allocation, filling, and subsequent replacement. `full` prefills every active
expert to its cap outside the timer, drains work, then measures replacement. Both
phases drain pending transfers and CPU commits **inside** the measurement before
stopping the timer. The existing reservoir algorithm and seed are unchanged.

## Compare two revisions

Use an isolated environment with the normal LLMC dependencies and the companion
compressed-tensors fork. Keep the old and new LLMC revisions in separate checkouts.
Run each command in a new process so module imports and CPU thread settings do not
leak between revisions. Do not reinstall dependencies in a running experiment's
environment or launch this on its GPU while it is calibrating.

```bash
python benchmarks/bench_osfp4_capture.py \
  --source /path/to/baseline-llmc --device cuda --mode rtn \
  --experts 128 --top-k 8 --input-width 2048 --intermediate-width 768 \
  --sequence-length 2048 --batch-size 1 --batches 64 --repeat 3 \
  --output /tmp/capture-baseline-rtn.json

python benchmarks/bench_osfp4_capture.py \
  --source /path/to/updated-llmc --device cuda --mode rtn \
  --experts 128 --top-k 8 --input-width 2048 --intermediate-width 768 \
  --sequence-length 2048 --batch-size 1 --batches 64 --repeat 3 \
  --output /tmp/capture-updated-rtn.json
```

Replace widths and expert counts with the actual model's dimensions. By default,
each mapping retains at most twice its input width, matching `auto`. `--cap N`
sets the same explicit cap for both projection groups. The default 128-expert
workload can require several GB of host reservoir memory, plus CUDA inputs,
statistics, and staging buffers. Size the workload for the machine before running.

Repeat the comparison with `--mode sic`, and separately with
`--mode sic --offload-hessians` if that is the production configuration. Keep the
source workload, device, library versions, seed, thread count, and arguments equal
between paired runs. Check the JSON revision, dirty status, and imported source
path. A dirty revision needs its exact patch recorded alongside the report.
An archived checkout without Git metadata is also supported: provide
`--revision-label <baseline-commit>` and preserve its source archive. Its reported
Git revision and dirty status are `null`, not inferred from the label.

Each report includes three-run medians by default, individual times, source-token
throughput, retained rows, and a digest of per-mapping sample indices. Digests must
match between revisions for identical workloads and phases. They do not validate
activation values or statistics; the regression tests provide those checks.
CUDA peaks include persistent inputs and statistics. CPU peak RSS is a cumulative
process high-water mark, not a per-phase allocation measurement. The reported
reservoir/Hessian byte estimates exclude temporary buffers and framework overhead.

For CPU harness validation only:

```bash
python benchmarks/bench_osfp4_capture.py --device cpu --threads 1 \
  --experts 4 --top-k 2 --input-width 16 --intermediate-width 8 \
  --sequence-length 16 --cap 7 --batches 3 --repeat 1 \
  --output /tmp/capture-cpu-check.json
```

A passing CPU run does not verify CUDA overlap, device memory safety, or GPU speed.

## Diagnose CPU work and synchronization

Add `--profile /tmp/capture-trace --profile-batches 4`. This performs **additional**
profiled runs after the unprofiled measurements and writes separate cold/full
Chrome traces. Instrumentation wraps sampling, statistics, copy scheduling,
reservoir updates, waits, and queue operations when those methods exist in the
selected revision. PyTorch's operator and CUDA event records provide the detailed
copy, allocation, and tensor-operation timeline. The trace timings are not used
in reported throughput medians.

Look for immediate event synchronizations after each reservoir capture, repeated
pinned allocations, CPU `index_copy_` work, and Python sampling ranges. A queue
should move waits to capacity pressure or the final drain; the final drain must
remain included in elapsed time. Some synchronization can remain in routing,
index transfer, and Hessian offloading. Do not infer total speedup from GPU
utilization or the number of synchronization calls alone.

Run additional fresh processes with `--threads 1`, `--threads 2`, and
`--threads 4`, compared with the existing default. This changes only the benchmark
process. Sweep `--batch-size 1`, `2`, `4`, and `8` while adjusting `--batches` to
keep the total number of source tokens fixed. These sweeps are separate
experiments; do not change production defaults or compare sample-index hashes
across different batch sizes/routing streams.

## Verify a real decoder layer before claiming success

The synthetic replay establishes whether capture itself improved. A separate
GPU diagnostic must also replay identical inputs through a real decoder layer
with the normal routing and capture hooks enabled. Use the same checkpoint,
dataset order, 2048-token samples, batch size, precision, reservoir cap, and
`moe_calibrate_all_experts=False` setting for both revisions. Profile a few early
and already-full batches, then run three unprofiled timing repetitions.

Report collection time, optimization time, and the post-quantization propagation
pass separately. Record total layer time, tokens/second, peak host/device memory,
expert token counts, and sampling provenance. For `propagate_error=True`, the
propagation pass is real work and must be included in a projected full-run time.
Compare sampled activation bytes and all-token statistics using the regression
tests, then check actual quantization quality with the existing evaluation
protocol. Do not substitute the synthetic cache rate for measured layer time.

Keep initialization timing separate: the compressed-tensors diagonal-target
benchmark measures the name-resolution improvement independently. If capture gets
faster but full-layer time does not, use the real-layer trace to identify the
remaining expert, routing, transfer, or optimization cost. No first-K sampling,
changed reservoir algorithm, or changed production batch size is part of this
performance experiment.
