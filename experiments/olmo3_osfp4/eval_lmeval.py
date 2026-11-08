"""Run one lm-eval task on a BF16 or OSFP4 checkpoint loaded with its transforms.

The model is passed to lm-eval as an in-memory object: lm-eval's own
``pretrained=<path>`` loading would drop the OSFP4 runtime transforms.
"""

from __future__ import annotations

import os as _os
if _os.path.exists("/tmp/osfp4-olmo3.drain"):  # queue drain switch: start nothing new
    print("DRAINED: /tmp/osfp4-olmo3.drain exists; stage not started", flush=True)
    raise SystemExit(75)

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from osfp4_loader import load_model  # noqa: E402

STOPS = ["<|im_end|>", "<|endoftext|>"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", required=True, help="lm-eval SQLite request cache prefix")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    cfg = json.loads(Path(args.protocol).read_text())["tasks"][args.task]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    model, tokenizer, info = load_model(args.model_path)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=cfg["batch_size"], max_length=4096)
    # Applied by lm-eval to generate_until tasks only. Olmo-3 ends turns with
    # <|im_end|>; HFLM always also stops on tokenizer.eos_token (<|endoftext|>).
    gen_kwargs = (f"until={STOPS[0]},max_gen_toks={cfg['max_gen_toks']},do_sample=False"
                  if "max_gen_toks" in cfg else None)

    results = evaluator.simple_evaluate(
        model=lm, tasks=[cfg["lmeval_task"]], task_manager=TaskManager(), gen_kwargs=gen_kwargs,
        num_fewshot=cfg.get("num_fewshot"),
        apply_chat_template=cfg["chat"],
        fewshot_as_multiturn=cfg.get("fewshot_as_multiturn", False),
        use_cache=args.cache, limit=args.limit, log_samples=True,
        random_seed=0, numpy_random_seed=1234, torch_random_seed=1234, fewshot_random_seed=1234,
    )
    samples = results.pop("samples", {})
    with (out / "samples.jsonl").open("w") as f:
        for task_name, rows in samples.items():
            for row in rows:
                f.write(json.dumps({"task": task_name, **row}, default=str) + "\n")
    metric_name = cfg["metric"]
    task_name = cfg["lmeval_task"]
    agg = results["results"].get(task_name, {})
    score = agg.get(metric_name)
    if score is None:
        raise SystemExit(f"metric {metric_name} missing for {task_name}: {sorted(agg)}")
    summary = {"task": args.task, "lmeval_task": task_name, "metric": metric_name, "score": score, "gen_kwargs": gen_kwargs,
               "model_path": args.model_path, "transforms_loaded": info["transforms"],
               "limit": args.limit, "n_samples": sum(len(v) for v in samples.values()),
               "config": cfg}
    if cfg["chat"]:
        caps = [len(tokenizer(r["resps"][0][0])["input_ids"]) >= cfg["max_gen_toks"] - 1
                for rows in samples.values() for r in rows if r.get("resps")]
        summary["cap_hits"] = int(sum(caps))
    (out / "results_full.json").write_text(json.dumps(results, indent=2, default=str))
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"LMEVAL_COMPLETE task={args.task} score={score:.4f}")


if __name__ == "__main__":
    main()
