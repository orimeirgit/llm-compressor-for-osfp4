#!/bin/bash
# Solo vs concurrent throughput on 128 MMLU-CoT prompts at batch 16; outputs saved for identity checks.
set -uo pipefail
R=/workspace/osfp4-runs/olmo3-7b-instruct; O=$R/speed-probe/parallel; mkdir -p $O
PY=/workspace/venvs/osfp4-eval/bin/python; P=/workspace/osfp4-olmo3-harness/probe_batch_throughput.py
BF16=/workspace/osfp4-cache/hub/models--allenai--Olmo-3-7B-Instruct/snapshots/6e5971d9eba42665f5bd5a0fcf047f299ce1dccc
W4A4=$R/checkpoints/nvfp4-sic
export HF_HOME=/workspace/osfp4-cache TOKENIZERS_PARALLELISM=false
stamp() { date +%s.%N; }
t=$(stamp); $PY -u $P $W4A4 solo-w4a4 $O/solo-w4a4.json 16; echo "WALL solo-w4a4 $(echo "$(stamp)-$t" | bc)"
t=$(stamp); $PY -u $P $BF16 solo-bf16 $O/solo-bf16.json 16; echo "WALL solo-bf16 $(echo "$(stamp)-$t" | bc)"
t=$(stamp); $PY -u $P $W4A4 pairA-w4a4-1 $O/pairA-1.json 16 & $PY -u $P $W4A4 pairA-w4a4-2 $O/pairA-2.json 16 & wait; echo "WALL pairA-w4a4+w4a4 $(echo "$(stamp)-$t" | bc)"
t=$(stamp); $PY -u $P $BF16 pairB-bf16 $O/pairB-bf16.json 16 & $PY -u $P $W4A4 pairB-w4a4 $O/pairB-w4a4.json 16 & wait; echo "WALL pairB-bf16+w4a4 $(echo "$(stamp)-$t" | bc)"
echo PARALLEL_PROBE_DONE
