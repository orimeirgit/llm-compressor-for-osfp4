# Olmo-3-7B-Instruct × OSFP4: results so far

**Status (2026-11-08 07:44 UTC): paused.** 7 of 42 evaluation stages are complete. The run is moving to a faster GPU, and `RESUME_ON_NEW_GPU.md` lists the 35 remaining stages and how to continue them. All four OSFP4 checkpoints are calibrated and passed the save → reload bit-parity check, so only evaluation is left.

Setup: `allenai/Olmo-3-7B-Instruct` at revision `6e5971d9`. OSFP4 comes from branch `osfp4-olmo3-shared-runtime` (`dcd8f533`), with `compressed-tensors-for-osfp4` at `30d017e`. Everything ran on 1× RTX PRO 6000 Blackwell (96 GB) with transformers 5.21, lm-eval 0.4.13 and LightEval 0.13.0. Task settings are in `benchmark_protocol.json`.

## Scores

| Benchmark | Setting | Metric | BF16 | OSFP4 RTN W4A16 | Δ vs BF16 |
|---|---|---|---:|---:|---:|
| MMLU-CoT | 0-shot chat, `mmlu_cot_llama`, 14,042 questions, 1024-token cap | exact match (strict) | 66.22 | 66.42 | +0.21 pp (p = 0.55) |
| GSM8K | 8-shot multi-turn chat, `gsm8k_cot_llama`, 1,319 problems, 1024-token cap | exact match (strict) | 90.37 | 90.83 | +0.45 pp (p = 0.53) |
| HellaSwag | 0-shot | acc_norm | 75.98 | not run | |
| WinoGrande | 0-shot | acc | 65.82 | not run | |
| ARC-Challenge | 0-shot | acc_norm | 54.01 | not run | |
| GPQA Diamond | 198 questions × 3 seeds, T = 1.0, top-p 0.95, top-k 64, 8192-token cap | accuracy, mean of seeds | 44.11 ± 1.56 | not run | |
| AIME 2025 | 30 problems × 8 seeds, 8192-token cap | accuracy, mean of seeds | not run | not run | |
| IFBench | 300 prompts, greedy, 8192-token cap | prompt-level loose | not run | not run | |
| WikiText-2 | raw test split, non-overlapping 2048-token windows | perplexity ↓ | 14.006 | 14.003 | −0.004 |

GPQA per seed: 45.45, 41.92, 44.95. p-values are exact McNemar tests on paired per-question correctness. SIC W4A16, RTN W4A4 and SIC W4A4 have no evaluations yet.

Responses that hit the cap:

| Benchmark | BF16 | RTN W4A16 |
|---|---:|---:|
| MMLU-CoT | 206 (1.5%) | 245 (1.7%) |
| GSM8K | 38 (2.9%) | 26 (2.0%) |
| GPQA Diamond | 19 of 594 (3.2%) | |

## What the finished stages show

1. **RTN W4A16 is statistically tied with BF16 on everything measured so far.** Quantization changes many individual answers: 15.9% of MMLU-CoT questions and 4.7% of GSM8K problems flip between right and wrong. The flips go both ways almost equally (MMLU-CoT 1,133 gained vs 1,104 lost, GSM8K 34 vs 28), and perplexity is unchanged.
2. **The FP8 block-scale floor doesn't cost RTN W4A16 anything measurable.** OSFP4 fixes the per-tensor weight global scale at 1. Olmo-3's q/k normalization lets q/k weights shrink, so 1.16% of all weight groups have their E4M3 block scale clamped at 2^-9. In `q_proj` of layers 4–6 and `k_proj` of layer 27 the share is 20–25%. Whether this matters for SIC, or once activations are also 4-bit (W4A4), is still open.
3. **The headline comparisons are still missing:** SIC vs RTN, and anything at W4A4, plus all GPQA/AIME/IFBench deltas.

## Caveats

- The Mistral harness was not available on this host, so the harness, the task variants and the calibration set were rebuilt here. Calibration used 512 UltraChat-200k conversations, ≤ 2048 tokens each, shuffle seed 42. Batch sizes, caps and sampling settings follow the Mistral RUN_GUIDE. Compare these deltas with the Mistral ones only where the task variants match.
- Greedy tasks ran once. GPQA averages 3 sampled seeds, with a ±1.6 pp spread between seeds.
- From 2026-11-07 22:04 UTC, two stages shared the GPU at once. A 128-prompt probe produced identical outputs shared and alone.
- The remaining stages will run on different hardware. Before resuming, `RESUME_ON_NEW_GPU.md` has a WikiText-2 check on BF16 and RTN W4A16 that shows whether the new GPU reproduces these numbers.
- The IFBench scoring path has not run end to end yet. Check the first IFBench `results.json` and `official_scorer.log`.

## Problems found applying OSFP4 to Olmo-3

| # | Problem | Effect | Status |
|---|---|---|---|
| 1 | Olmo-3 is post-norm: no norm sits in front of q/k/v or gate/up. The registry's default mapping (`post_attention_layernorm` → gate/up) is wrong, and the plausible-looking `post_feedforward_layernorm(i)` → q/k/v(i+1) is also wrong. | Folding scales into those norms changes the model: 49% relative logit error in an fp64 equivalence test, before any quantization. | **Fixed** in `dcd8f533`: q/k/v share one runtime input transform and gate/up another (exact in fp64). The v→o and up→down folds were already exact. |
| 2 | transformers 5.21 ignores `transform_config` when loading a compressed-tensors checkpoint. | Plain `from_pretrained` or lm-eval `pretrained=<path>` silently drops all 224 `smooth_quant_input` scales. | **Worked around**: `osfp4_loader.py` applies them and checks every one loads. |
| 3 | The model's `generation_config` sets `max_new_tokens=32768`. | It overrides the benchmark caps. | **Worked around** in the loader. Caps are passed explicitly. |
| 4 | Right after `oneshot`, modules are CPU-offloaded and RTN weight fake-quantization is skipped. | In-memory inference after calibration is wrong. The saved checkpoints are correct. | **Avoided**: evaluation always reloads, and `calibrate.py` requires bit-identical reload. |
| 5 | Weight global scale fixed at 1, combined with q/k normalization. | E4M3 block-scale floor clamp (finding 2 above). | **Not fixed**, so the method stays identical to the Mistral run. A per-tensor power-of-two global scale would fix it. |
| 6 | Fake quantization in HF `generate`. | W4A16 is about 1.5× and W4A4 about 2.4× slower than BF16. | Speed only, no effect on scores. |
| 7 | 391 OSFP4 tests fail at branch head `6a650621` and the two commits before it (byte-parity tests). | Pre-existing, unrelated to Olmo-3. | Not addressed. The 8 new Olmo-3 tests pass. |

## Files

| Path (run directory) | Contents |
|---|---|
| `benchmark-results.json`, `.csv` | scores, deltas vs BF16, cap hits, missing stages |
| `benchmarks-selected9-olmo3-v1/<method>/{lmeval,generate}/<task>/` | per-sample outputs and `results.json` for each stage |
| `bf16-wikitext2.json`, `nvfp4a16-rtn-wikitext2.json` | perplexity |
| `checkpoints/<name>-calibration.json` | calibration settings and the reload-parity record |
| `architecture-probe.json` | fp64 fold-equivalence tests and activation statistics |
