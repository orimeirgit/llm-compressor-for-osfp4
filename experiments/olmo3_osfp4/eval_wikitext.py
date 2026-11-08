"""WikiText-2 (raw, test) token perplexity with non-overlapping 2048-token windows."""

from __future__ import annotations

import os as _os
if _os.path.exists("/tmp/osfp4-olmo3.drain"):  # queue drain switch: start nothing new
    print("DRAINED: /tmp/osfp4-olmo3.drain exists; stage not started", flush=True)
    raise SystemExit(75)

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from osfp4_loader import load_model  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--limit-windows", type=int, default=None)
    args = p.parse_args()

    from datasets import load_dataset

    model, tokenizer, info = load_model(args.model_path)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    windows = ids.numel() // args.seq_len
    if args.limit_windows:
        windows = min(windows, args.limit_windows)
    nll, count = 0.0, 0
    with torch.no_grad():
        for w in range(windows):
            chunk = ids[w * args.seq_len:(w + 1) * args.seq_len].unsqueeze(0).to("cuda:0")
            logits = model(chunk).logits.float()
            loss = torch.nn.functional.cross_entropy(logits[0, :-1], chunk[0, 1:], reduction="sum")
            nll += loss.item()
            count += chunk.shape[1] - 1
    ppl = math.exp(nll / count)
    result = {"task": "wikitext2", "perplexity": ppl, "windows": windows, "tokens_scored": count,
              "seq_len": args.seq_len, "model_path": args.model_path, "transforms_loaded": info["transforms"],
              "limit_windows": args.limit_windows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"WIKITEXT_COMPLETE ppl={ppl:.4f} windows={windows}")


if __name__ == "__main__":
    main()
