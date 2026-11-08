"""Throughput and prediction stability vs batch size on real MMLU-CoT prompts.

Greedy, 1024-token cap, stops on <|im_end|>/<|endoftext|>, chat template, the same
prompt text lm-eval's mmlu_cot_llama builds. Reports wall time, generated tokens/s,
peak memory, and how many outputs differ from batch 16.
"""
import json
import sys
import time
from pathlib import Path

import torch
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from osfp4_loader import load_model  # noqa: E402

ckpt, label, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
batch_sizes = [int(b) for b in sys.argv[4].split(",")]
N = 128

TEMPLATE = ("Given the following question and four candidate answers (A, B, C and D), choose the best answer.\n\n"
            "Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\n\n- For simple problems:\nDirectly provide the answer with minimal explanation.\n\n"
            "- For complex problems:\nUse this step-by-step format:\n## Step 1: [Concise description]\n[Brief explanation]\n"
            "## Step 2: [Concise description]\n[Brief explanation]\n\nRegardless of the approach, always conclude with:\n"
            "The best answer is [the_answer_letter].\nwhere the [the_answer_letter] is one of A, B, C or D.\n\nLet's think step by step.")
from datasets import concatenate_datasets, get_dataset_config_names  # noqa: E402

subjects = [c for c in get_dataset_config_names("hails/mmlu_no_train") if c != "all"]
rows = concatenate_datasets([load_dataset("hails/mmlu_no_train", s, split="test") for s in subjects])
rows = rows.shuffle(seed=0).select(range(N))
model, tok, info = load_model(ckpt)
tok.padding_side = "left"
prompts = [tok.apply_chat_template([{"role": "user", "content": TEMPLATE.format(
    q=r["question"].strip(), a=r["choices"][0], b=r["choices"][1], c=r["choices"][2], d=r["choices"][3])}],
    tokenize=False, add_generation_prompt=True) for r in rows]
# lm-eval orders generate requests by descending context length; mimic that.
order = sorted(range(N), key=lambda i: -len(tok(prompts[i], add_special_tokens=False)["input_ids"]))
stops = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
results = {}
outputs = {}
for bs in batch_sizes:
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    texts, gen_tokens = {}, 0
    torch.cuda.synchronize(); t0 = time.time()
    for s in range(0, N, bs):
        idx = order[s:s + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=1024, do_sample=False, eos_token_id=stops, pad_token_id=tok.pad_token_id)
        for i, toks in zip(idx, out[:, enc["input_ids"].shape[1]:].tolist()):
            cut = next((k for k, t in enumerate(toks) if t in stops), len(toks))
            texts[i] = tok.decode(toks[:cut], skip_special_tokens=True)
            gen_tokens += cut
    torch.cuda.synchronize(); wall = time.time() - t0
    outputs[bs] = texts
    results[bs] = {"wall_s": round(wall, 1), "gen_tok_per_s": round(gen_tokens / wall, 1),
                   "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
    if bs != batch_sizes[0]:
        diff = sum(outputs[bs][i] != outputs[batch_sizes[0]][i] for i in range(N))
        results[bs]["outputs_differing_from_bs%d" % batch_sizes[0]] = diff
    print(f"[speed] {label} bs={bs}: {results[bs]}", flush=True)
base = results[batch_sizes[0]]["wall_s"]
for bs in batch_sizes:
    results[bs]["speedup_vs_bs%d" % batch_sizes[0]] = round(base / results[bs]["wall_s"], 2)
Path(out_path).write_text(json.dumps({"label": label, "checkpoint": ckpt, "n_prompts": N, "results": results,
                                      "texts": {str(bs): [outputs[bs][i] for i in range(N)] for bs in batch_sizes}}, indent=2))
print(f"[speed] {label} summary: " + ", ".join(f"bs{bs} x{results[bs]['speedup_vs_bs%d' % batch_sizes[0]]}" for bs in batch_sizes), flush=True)
