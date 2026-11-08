#!/bin/bash
# Tiny-limit smoke of every benchmark adapter on one checkpoint. Outputs stay under
# <run>/smoke/eval-<label>/ and are never mixed with full results.
set -uo pipefail
CKPT=$1; LABEL=$2
H=/workspace/osfp4-olmo3-harness
R=/workspace/osfp4-runs/olmo3-7b-instruct
PY=/workspace/venvs/osfp4-eval/bin/python
O=$R/smoke/eval-$LABEL
mkdir -p $O
export HF_HOME=/workspace/osfp4-cache TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
status=0
run() { echo "=== $*"; "$@" || { echo "SMOKE_STAGE_FAILED: $*"; status=1; }; }
run $PY -u $H/eval_wikitext.py --model-path $CKPT --out $O/wikitext2.json --limit-windows 4
for t in mmlu_cot gsm8k hellaswag winogrande arc_challenge; do
  lim=4; [ $t = mmlu_cot ] && lim=1
  run $PY -u $H/eval_lmeval.py --model-path $CKPT --task $t --protocol $H/benchmark_protocol.json \
    --out $O/$t --cache $O/lmeval-cache/$t --limit $lim
done
for t in aime2025 ifbench gpqa_diamond; do
  lim=2; [ $t = ifbench ] && lim=8
  run $PY -u $H/eval_generate.py --model-path $CKPT --task $t --protocol $H/benchmark_protocol.json \
    --out $O/$t --limit $lim
done
echo "SMOKE_DONE label=$LABEL status=$status"
