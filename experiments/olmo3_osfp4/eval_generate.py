"""Batched generation benchmarks: GPQA Diamond and AIME 2025 (LightEval prompts and
metrics) and IFBench (official allenai/IFBench verifier).

Responses are appended per completed batch to responses.jsonl, so an interrupted
run resumes from the first unfinished batch with the original batch grouping.
"""

from __future__ import annotations

import os as _os
if _os.path.exists("/tmp/osfp4-olmo3.drain"):  # queue drain switch: start nothing new
    print("DRAINED: /tmp/osfp4-olmo3.drain exists; stage not started", flush=True)
    raise SystemExit(75)

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from osfp4_loader import load_model  # noqa: E402

IM_END, ENDOFTEXT = "<|im_end|>", "<|endoftext|>"
IFBENCH_REPO = Path("/workspace/third_party/IFBench")


def build_items(task: str, protocol: dict) -> list[dict]:
    """Return prompt items, identical for every method (prompts are saved and hashed)."""
    from datasets import load_dataset

    if task == "gpqa_diamond":
        from lighteval.tasks.tasks.gpqa import gpqa_instruct_prompt

        rows = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        random.seed(protocol["gpqa_choice_order_seed"])  # LightEval shuffles with global random
        items = []
        for i, row in enumerate(rows):
            doc = gpqa_instruct_prompt(row, "gpqa:diamond")
            items.append({"id": i, "query": doc.query, "gold_index": doc.gold_index,
                          "choices": list(doc.choices)})
        return items
    if task == "aime2025":
        from lighteval.tasks.tasks.aime import aime_prompt

        rows = load_dataset("yentinglin/aime_2025", "default", split="train")
        return [{"id": i, "query": aime_prompt(row, "aime25").query, "answer": str(row["answer"])}
                for i, row in enumerate(rows)]
    if task == "ifbench":
        rows = [json.loads(line) for line in (IFBENCH_REPO / "data/IFBench_test.jsonl").read_text().splitlines() if line.strip()]
        return [{"id": i, "query": row["prompt"], "key": row.get("key")} for i, row in enumerate(rows)]
    raise ValueError(task)


def score(task: str, items: list[dict], responses: dict, seeds: list[int], out: Path) -> dict:
    if task == "ifbench":
        resp_path = out / "ifbench_responses.jsonl"
        with resp_path.open("w") as f:
            for item in items:
                f.write(json.dumps({"prompt": item["query"], "response": responses[(seeds[0], item["id"])]["text"]}) + "\n")
        scored = out / "official"
        scored.mkdir(exist_ok=True)
        cmd = [sys.executable, "-m", "run_eval", f"--input_data={IFBENCH_REPO / 'data/IFBench_test.jsonl'}",
               f"--input_response_data={resp_path}", f"--output_dir={scored}"]
        proc = subprocess.run(cmd, cwd=IFBENCH_REPO, capture_output=True, text=True)
        (out / "official_scorer.log").write_text(proc.stdout + proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"IFBench scorer failed: {proc.stderr[-2000:]}")
        metrics = {}
        for mode in ("strict", "loose"):
            rows = [json.loads(l) for l in (scored / f"eval_results_{mode}.jsonl").read_text().splitlines() if l.strip()]
            metrics[f"prompt_level_{mode}_acc"] = sum(r["follow_all_instructions"] for r in rows) / len(rows)
            flags = [x for r in rows for x in r["follow_instruction_list"]]
            metrics[f"instruction_level_{mode}_acc"] = sum(flags) / len(flags)
        metrics["score"] = metrics["prompt_level_loose_acc"]
        metrics["n_prompts"] = len(items)
        return metrics

    from lighteval.metrics.metrics import Metrics
    from lighteval.models.model_output import ModelResponse
    from lighteval.tasks.requests import Doc

    if task == "aime2025":
        metric, key = Metrics.pass_at_k_math(sample_params={"k": 1}), "pass@k:k=1"
    else:
        metric, key = Metrics.gpqa_instruct_pass_at_k(sample_params={"k": 1}), "gpqa_pass@k:k=1"
    per_seed, per_sample = {}, []
    for seed in seeds:
        values = []
        for item in items:
            if task == "aime2025":
                doc = Doc(task_name="aime25", query=item["query"], choices=[item["answer"]], gold_index=0)
            else:
                doc = Doc(task_name="gpqa:diamond", query=item["query"], choices=item["choices"], gold_index=item["gold_index"])
            text = responses[(seed, item["id"])]["text"]
            value = float(metric.compute_sample(doc=doc, model_response=ModelResponse(text=[text]))[key])
            values.append(value)
            per_sample.append({"seed": seed, "id": item["id"], "correct": value,
                               "hit_cap": responses[(seed, item["id"])]["hit_cap"]})
        per_seed[str(seed)] = sum(values) / len(values)
    (out / "per_sample_scores.jsonl").write_text("\n".join(json.dumps(r) for r in per_sample) + "\n")
    accs = list(per_seed.values())
    return {"score": statistics.mean(accs), "per_seed": per_seed,
            "std_across_seeds": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
            "n_items": len(items), "n_seeds": len(seeds)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--task", choices=["gpqa_diamond", "aime2025", "ifbench"], required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None, help="smoke only: first N items, first seed")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()

    protocol = json.loads(Path(args.protocol).read_text())
    cfg = protocol["tasks"][args.task]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = build_items(args.task, protocol)
    seeds = list(cfg["seeds"])
    if args.limit:
        items, seeds = items[: args.limit], seeds[:1]
    prompts_blob = json.dumps(items, sort_keys=True).encode()
    prompts_sha = hashlib.sha256(prompts_blob).hexdigest()
    prompts_file = out / "prompts.json"
    if prompts_file.exists() and hashlib.sha256(prompts_file.read_bytes()).hexdigest() != prompts_sha:
        raise SystemExit(f"{prompts_file} differs from freshly built prompts; refusing to mix")
    prompts_file.write_bytes(prompts_blob)
    if args.prepare_only:
        print(f"GENERATION_PREPARED task={args.task} items={len(items)} seeds={len(seeds)} prompts_sha256={prompts_sha}")
        return

    model, tokenizer, info = load_model(args.model_path)
    tokenizer.padding_side = "left"
    stop_ids = [tokenizer.convert_tokens_to_ids(IM_END), tokenizer.convert_tokens_to_ids(ENDOFTEXT)]
    gen = dict(max_new_tokens=cfg["max_new_tokens"], eos_token_id=stop_ids,
               pad_token_id=tokenizer.pad_token_id)
    if cfg["do_sample"]:
        gen.update(do_sample=True, temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"])
    else:
        gen.update(do_sample=False, temperature=None, top_p=None, top_k=None)

    resp_file = out / "responses.jsonl"
    responses = {}
    if resp_file.exists():
        for line in resp_file.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                responses[(r["seed"], r["id"])] = r
    bs = cfg["batch_size"]
    total = len(seeds) * len(items)
    started = time.time()
    for seed in seeds:
        for start in range(0, len(items), bs):
            batch = items[start:start + bs]
            if all((seed, it["id"]) in responses for it in batch):
                continue
            texts = [tokenizer.apply_chat_template([{"role": "user", "content": it["query"]}],
                                                   tokenize=False, add_generation_prompt=True) for it in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
            # Seed per (seed, batch) so a resumed batch regenerates with the same RNG stream.
            torch.manual_seed(seed * 100003 + start)
            with torch.no_grad():
                generated = model.generate(**enc, **gen)
            new_tokens = generated[:, enc["input_ids"].shape[1]:]
            with resp_file.open("a") as f:
                for it, toks in zip(batch, new_tokens):
                    toks = toks.tolist()
                    ended = any(t in stop_ids for t in toks)
                    cut = next((i for i, t in enumerate(toks) if t in stop_ids), len(toks))
                    rec = {"seed": seed, "id": it["id"], "batch_start": start,
                           "text": tokenizer.decode(toks[:cut], skip_special_tokens=True),
                           "n_tokens": cut, "hit_cap": not ended}
                    responses[(seed, it["id"])] = rec
                    f.write(json.dumps(rec) + "\n")
            done = len(responses)
            rate = (time.time() - started) / max(1, done)
            print(f"[{args.task}] seed={seed} batch={start // bs + 1}/{-(-len(items) // bs)} "
                  f"done={done}/{total} cap_hits={sum(r['hit_cap'] for r in responses.values())}", flush=True)
    metrics = score(args.task, items, responses, seeds, out)
    metrics.update({"task": args.task, "model_path": args.model_path, "transforms_loaded": info["transforms"],
                    "generation": {k: v for k, v in gen.items() if k not in ("pad_token_id",)},
                    "prompts_sha256": prompts_sha, "limit": args.limit,
                    "cap_hits": sum(r["hit_cap"] for r in responses.values()),
                    "mean_response_tokens": statistics.mean(r["n_tokens"] for r in responses.values())})
    (out / "results.json").write_text(json.dumps(metrics, indent=2))
    print(f"GENERATION_COMPLETE task={args.task} score={metrics['score']:.4f}")


if __name__ == "__main__":
    main()
