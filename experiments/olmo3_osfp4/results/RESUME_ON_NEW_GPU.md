# Resuming the Olmo-3 OSFP4 run on a new GPU

The run paused on 2026-11-08 at 07:44 UTC, on Vast instance `59597359` (1× RTX PRO 6000 Blackwell, 96 GB). Results so far are in `RESULTS.md`. Every path below is hard-coded in the harness, so keep them exactly.

## What is left (35 stages)

| Method | Remaining stages |
|---|---|
| `bf16` | AIME 2025, IFBench |
| `osfp4_rtn_w4a16` | HellaSwag, WinoGrande, ARC-Challenge, GPQA Diamond, AIME 2025, IFBench |
| `osfp4_sic_w4a16`, `osfp4_rtn_w4a4`, `osfp4_sic_w4a4` | all 9 (WikiText-2 plus 8 tasks) |

On the old GPU this was about 5 days of work. The bottleneck is the long-generation stages: GPQA, AIME and IFBench, batch 8 with an 8192-token cap.

## 1. Pick the machine

- **CUDA 13 driver.** The environments use `torch 2.14.1+cu132`, so the driver must support CUDA ≥ 13.0. Check `vast-capabilities | jq .hardware.gpu.cuda.driver_max_cuda`.
- **Disk ≥ 80 GB**: checkpoints 20 GB, base model 15 GB, environments 12 GB, datasets, plus headroom.
- **GPU memory ≥ 96 GB** keeps the current settings working. With ≥ 141 GB (H200, B200) you can try `OSFP4_MAX_HEAVY=2`, which runs two long-generation stages at once (untested).
- **One GPU.** The queue uses `cuda:0` only, so extra GPUs would sit idle without a small change to the queue.

## 2. Bring the data

Copy the run folder, without `smoke/`, to `/workspace/osfp4-runs/olmo3-7b-instruct/`. That is 20 GB of checkpoints plus 226 MB of results, logs and lm-eval request cache.

```bash
# from your computer, after downloading it to ./olmo3-osfp4/
ssh -p NEW_PORT root@NEW_IP mkdir -p /workspace/osfp4-runs/olmo3-7b-instruct
rsync -avP -e "ssh -p NEW_PORT" ./olmo3-osfp4/ root@NEW_IP:/workspace/osfp4-runs/olmo3-7b-instruct/

# or instance to instance, while the old one still exists
vastai copy C.59597359:/workspace/osfp4-runs/ C.NEW_ID:/workspace/osfp4-runs/
```

Keep each `checkpoints/<name>-calibration.json` next to its checkpoint folder. The queue silently skips any method whose calibration record is missing.

## 3. Code

```bash
cd /workspace
git clone -b osfp4-olmo3-shared-runtime https://github.com/orimeirgit/llm-compressor-for-osfp4.git llm-compressor-olmo3
git clone -b osfp4-diagonal-transform https://github.com/orimeirgit/compressed-tensors-for-osfp4.git
git -C compressed-tensors-for-osfp4 checkout 30d017eec16e3d365be66017cbc1a57782ff7bcb
mkdir -p third_party && git clone https://github.com/allenai/IFBench.git third_party/IFBench
git -C third_party/IFBench checkout cd189a9c2889526d8af901d2103fd3e1ffcea6df
ln -s /workspace/llm-compressor-olmo3/experiments/olmo3_osfp4 /workspace/osfp4-olmo3-harness
```

## 4. Environments

Rebuild them from the frozen package lists. Both lists resolved cleanly in a dry run on 2026-11-08.

```bash
R=/workspace/osfp4-olmo3-harness/requirements
IDX="--extra-index-url https://download.pytorch.org/whl/cu132 --index-strategy unsafe-best-match"
for v in osfp4 osfp4-eval; do
  uv venv --python 3.12 /workspace/venvs/$v
  grep -v '^-e ' $R/requirements-$v.txt > /tmp/req-$v.txt
  uv pip install --python /workspace/venvs/$v/bin/python -r /tmp/req-$v.txt $IDX
  SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2a3+osfp4.30d017e \
    uv pip install --python /workspace/venvs/$v/bin/python --no-deps -e /workspace/compressed-tensors-for-osfp4
done
uv pip install --python /workspace/venvs/osfp4/bin/python --no-deps -e /workspace/llm-compressor-olmo3
/workspace/venvs/osfp4-eval/bin/python -m nltk.downloader stopwords averaged_perceptron_tagger_eng punkt punkt_tab
```

`osfp4` is used for calibration and the queue; `osfp4-eval` runs every evaluation. The NLTK data is for the IFBench verifier.

If the new instance uses the same Vast PyTorch image, you can copy `/workspace/venvs/` (12 GB) instead of rebuilding. The editable installs point at the paths from step 3.

## 5. Model and Hugging Face token

```bash
export HF_HOME=/workspace/osfp4-cache
/workspace/venvs/osfp4-eval/bin/hf auth login    # paste the token at the prompt; GPQA is gated
/workspace/venvs/osfp4-eval/bin/hf download allenai/Olmo-3-7B-Instruct --revision 6e5971d9eba42665f5bd5a0fcf047f299ce1dccc
```

Datasets download on first use.

## 6. Check the new GPU reproduces the old numbers

The two checks below take about a minute each. The second one also exercises the OSFP4 loader and its 224 transforms.

```bash
cd /workspace/osfp4-olmo3-harness; export HF_HOME=/workspace/osfp4-cache
/workspace/venvs/osfp4-eval/bin/python eval_wikitext.py --model-path /workspace/osfp4-cache/hub/models--allenai--Olmo-3-7B-Instruct/snapshots/6e5971d9eba42665f5bd5a0fcf047f299ce1dccc --out /tmp/check-bf16-wikitext2.json
/workspace/venvs/osfp4-eval/bin/python eval_wikitext.py --model-path /workspace/osfp4-runs/olmo3-7b-instruct/checkpoints/nvfp4a16-rtn --out /tmp/check-w4a16rtn-wikitext2.json
grep perplexity /tmp/check-*-wikitext2.json
```

On the old GPU the results were BF16 14.006394 and RTN W4A16 14.002858. Agreement to about 3 decimals means the hardware change doesn't matter. A larger gap means the kernels differ, so look into it before mixing results from the two machines.

For a deeper check, rerun BF16 GSM8K with a fresh `--cache` directory and compare per-sample answers with the stored `samples.jsonl`. Reusing the old cache would just replay the stored answers.

## 7. Resume

```bash
tmux new -s osfp4-olmo3 -n benchmarks
/workspace/venvs/osfp4/bin/python -u /workspace/osfp4-olmo3-harness/run_queue_parallel.py 2>&1 \
  | tee -a /workspace/osfp4-runs/olmo3-7b-instruct/benchmarks-progress.log
# with >= 141 GB of GPU memory, optionally prefix: OSFP4_MAX_HEAVY=2
```

The queue skips completed stages and holds `/tmp/osfp4-olmo3.lock`. To wind it down, `touch /tmp/osfp4-olmo3.drain`: running stages finish and nothing new starts.

Watch progress with `cat /workspace/osfp4-runs/olmo3-7b-instruct/benchmarks-status.json` or the `benchmarks-progress.log`.

When the first IFBench stage finishes, check its `results.json` and `official_scorer.log`. That scoring path has not run end to end yet.

The matrix is done when the log prints `OLMO3_BENCHMARK_MATRIX_COMPLETE; STOPPED_BEFORE_ANY_NEXT_MODEL`. Stop there for review before starting any other model.
