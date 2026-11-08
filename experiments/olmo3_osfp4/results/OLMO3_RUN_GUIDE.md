# Olmo-3-7B-Instruct OSFP4 run guide

> **Paused 2026-11-08 07:44 UTC** to move to a faster GPU after 7 of 42 evaluation stages. Results so far are in `RESULTS.md`, and the steps to continue on another machine are in `RESUME_ON_NEW_GPU.md`. Resume with `run_queue_parallel.py` (see that file). It replaces the serial `run_queue.py` command below.

Host: 1× RTX PRO 6000 Blackwell (GPU 0). **`/workspace` is not a persistent volume**: a recycle or destroy deletes everything below, including checkpoints and results. Stop/start keeps it.

## Follow the run

```bash
tmux attach -t osfp4-olmo3
```

Windows: **benchmarks** (the queue), **progress** (log tail), **calib**/**smoke**/**tests**/**setup** (finished setup jobs). Ctrl+b w picks a window; Ctrl+b d detaches.

```bash
watch -n 10 cat /workspace/osfp4-runs/olmo3-7b-instruct/benchmarks-status.json
tail -n 40 -F /workspace/osfp4-runs/olmo3-7b-instruct/benchmarks-progress.log
```

## What runs

| Setting | Value |
|---|---|
| Model | `allenai/Olmo-3-7B-Instruct` @ `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` |
| OSFP4 code | `/workspace/llm-compressor-olmo3`, local branch `osfp4-olmo3-shared-runtime` (base `osfp4-chunked-exact-alpha-grad` @ `6a650621`, Olmo-3 fix uncommitted) |
| compressed-tensors | `/workspace/compressed-tensors-for-osfp4`, branch `osfp4-diagonal-transform` @ `30d017e` |
| Harness | `/workspace/osfp4-olmo3-harness` (protocol in `benchmark_protocol.json`) |
| Calibration | 512 `HuggingFaceH4/ultrachat_200k` train_sft conversations, chat template, ≤2048 tokens, shuffle seed 42 (589,423 tokens) |
| Run dir | `/workspace/osfp4-runs/olmo3-7b-instruct` |

Order: 4 calibrations (`nvfp4a16-rtn`, `nvfp4a16-sic`, `nvfp4-rtn`, `nvfp4-sic`), then for `bf16`, `osfp4_rtn_w4a16`, `osfp4_sic_w4a16`, `osfp4_rtn_w4a4`, `osfp4_sic_w4a4`: WikiText-2, MMLU-CoT, GSM8K, HellaSwag, WinoGrande, ARC-Challenge, GPQA Diamond, AIME 2025, IFBench. Batch sizes, caps and sampling follow the Mistral RUN_GUIDE; task variants are recorded in the protocol file.

A calibration is accepted only when every quantized layer's effective weight, block scales, input global scale and transform scale are bit-identical after save → reload (`checkpoints/<name>-calibration.json`, `reload_parity`).

## Hugging Face token (needed for GPQA Diamond)

GPQA (`Idavidrein/gpqa`) is gated. Accept its terms on the Hub with your account, then in a tmux window:

```bash
HF_HOME=/workspace/osfp4-cache /workspace/venvs/osfp4-eval/bin/hf auth login
HF_HOME=/workspace/osfp4-cache /workspace/venvs/osfp4-eval/bin/hf auth whoami
```

Paste the token at the prompt (not on the command line). Without it the three GPQA stages fail and the rest continue; rerun the queue afterwards to fill them in.

## Stop and resume

Stop: select **benchmarks**, press Ctrl+C once, wait for the prompt, then check nothing remains:

```bash
pgrep -af 'run_queue.py|calibrate.py|eval_lmeval.py|eval_generate.py|eval_wikitext.py'
```

Resume (same command as the first start; completed stages are skipped):

```bash
/workspace/venvs/osfp4/bin/python -u /workspace/osfp4-olmo3-harness/run_queue.py 2>&1 | tee -a /workspace/osfp4-runs/olmo3-7b-instruct/benchmarks-progress.log
```

The queue holds `/tmp/osfp4-olmo3.lock` for the whole matrix and refuses to start a second copy.

## Results

| Path (under the run dir) | Contents |
|---|---|
| `benchmarks-selected9-olmo3-v1/<method>/lmeval/<task>/` | `results.json`, `samples.jsonl`, full lm-eval output |
| `benchmarks-selected9-olmo3-v1/<method>/generate/<task>/` | prompts, `responses.jsonl`, per-sample scores, `results.json` |
| `<checkpoint>-wikitext2.json`, `bf16-wikitext2.json` | perplexity |
| `benchmark-results.json` / `.csv` | final matrix, deltas in pp vs BF16 (written after all stages) |
| `benchmark-logs/<method>-<stage>.log` | per-stage logs |

Complete when `benchmarks-status.json` shows `benchmark_matrix_complete` / `awaiting_user_review_before_next_model` and the log prints `OLMO3_BENCHMARK_MATRIX_COMPLETE; STOPPED_BEFORE_ANY_NEXT_MODEL`.

## Do not

- Evaluate an OSFP4 checkpoint with plain `from_pretrained` or lm-eval `pretrained=<path>`: transformers 5.21 ignores `transform_config`, dropping all 224 `smooth_quant_input` scales. Use `osfp4_loader.load_model`.
- Run inference on the in-memory model right after `oneshot`: modules are CPU-offloaded and RTN weight fake-quantization is silently skipped. Reload the saved checkpoint.
