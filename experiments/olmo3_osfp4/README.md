# Olmo-3-7B-Instruct OSFP4 benchmark harness

Calibrates four OSFP4 checkpoints of `allenai/Olmo-3-7B-Instruct` (revision `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`) and evaluates them next to BF16 on WikiText-2 plus 8 tasks: MMLU-CoT, GSM8K, HellaSwag, WinoGrande, ARC-Challenge, GPQA Diamond, AIME 2025, IFBench.

| Method | Checkpoint | Scheme | Optimization |
|---|---|---|---|
| `bf16` | base model | none | none |
| `osfp4_rtn_w4a16` | `nvfp4a16-rtn` | NVFP4A16 | RTN |
| `osfp4_sic_w4a16` | `nvfp4a16-sic` | NVFP4A16 | SIC |
| `osfp4_rtn_w4a4` | `nvfp4-rtn` | NVFP4 | RTN |
| `osfp4_sic_w4a4` | `nvfp4-sic` | NVFP4 | SIC |

The harness needs this branch's Olmo-3 fix (shared runtime transforms for q/k/v and gate/up) and `compressed-tensors-for-osfp4` at `30d017e` (branch `osfp4-diagonal-transform`). Task variants, batch sizes, caps and sampling settings are in `benchmark_protocol.json`.

## Status

Paused on 2026-11-08 after 7 of 42 evaluation stages, to move to a faster GPU. Results so far and the Olmo-3 implementation findings are in `results/RESULTS.md`. Steps to continue on another machine are in `results/RESUME_ON_NEW_GPU.md`.

## Expected layout

Paths are hard-coded. Put this directory at `/workspace/osfp4-olmo3-harness`, by copying or symlinking.

| Path | Contents |
|---|---|
| `/workspace/osfp4-olmo3-harness` | this directory |
| `/workspace/osfp4-runs/olmo3-7b-instruct` | checkpoints, results, logs, lm-eval request cache |
| `/workspace/osfp4-cache` | `HF_HOME`: the model at the pinned revision, datasets, HF token |
| `/workspace/venvs/osfp4` | calibration environment (`requirements/requirements-osfp4.txt`) |
| `/workspace/venvs/osfp4-eval` | evaluation environment (`requirements/requirements-osfp4-eval.txt`) |
| `/workspace/llm-compressor-olmo3`, `/workspace/compressed-tensors-for-osfp4` | editable installs referenced by both environments |
| `/workspace/third_party/IFBench` | `allenai/IFBench` at `cd189a9`, the official verifier and `data/IFBench_test.jsonl` |

## Files

| File | Role |
|---|---|
| `calibrate.py` | one OSFP4 calibration. Accepted only if every quantized layer is bit-identical after save and reload. |
| `osfp4_loader.py` | loads a checkpoint together with its `smooth_quant_input` transforms. Always use it: transformers 5.21 ignores `transform_config`. |
| `eval_lmeval.py`, `eval_generate.py`, `eval_wikitext.py` | one stage each. Generation stages save every finished batch and resume from there. |
| `run_queue.py` | serial queue over the whole matrix; skips completed stages |
| `run_queue_parallel.py` | the same stages run concurrently: `OSFP4_SLOTS` jobs at once (default 2), at most `OSFP4_MAX_HEAVY` GPQA/AIME/IFBench stages at once (default 1) |
| `aggregate.py` | writes `benchmark-results.json` and `.csv` (scores, deltas vs BF16, cap hits) |
| `probe_*`, `diag*`, `check_tiny_roundtrip.py`, `diagnose_rtn_reload.py` | one-off investigations from setup |

## Run or resume

```bash
/workspace/venvs/osfp4/bin/python -u /workspace/osfp4-olmo3-harness/run_queue_parallel.py 2>&1 \
  | tee -a /workspace/osfp4-runs/olmo3-7b-instruct/benchmarks-progress.log
```

Run it in tmux. The queue holds `/tmp/osfp4-olmo3.lock` and refuses to start a second copy.

To drain the queue, `touch /tmp/osfp4-olmo3.drain`. Stages that are already running finish. Every new stage exits with code 75 before writing anything, so the queue lists it as failed, and a later rerun runs it again. Remove the file to run normally.

## Rebuilding the environments

`requirements/` holds `uv pip freeze` output from the original host. Notes for rebuilding from it:

- `torch==2.14.1+cu132` needs `--extra-index-url https://download.pytorch.org/whl/cu132` and a driver that supports CUDA 13.
- The editable entries point at the two repositories above.
- `compressed-tensors-for-osfp4` was installed with `SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2a3+osfp4.30d017e`.
