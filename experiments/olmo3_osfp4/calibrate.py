"""Calibrate one OSFP4 configuration for Olmo-3-7B-Instruct and validate the checkpoint.

Writes <out>/ (compressed-tensors checkpoint) and <out>-calibration.json.
The checkpoint is written to <out>.partial first and renamed only after the
reload-parity check passes, so a directory without the suffix is always valid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from compressed_tensors.offload import disable_offloading
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.osfp4 import OSFP4Modifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from osfp4_loader import load_model  # noqa: E402

TRANSFORM = "smooth_quant_input"


def sha(tensor: torch.Tensor) -> str:
    t = tensor.detach().contiguous().cpu()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()[:24]


def quantized_layer_fingerprint(model) -> dict[str, dict[str, str]]:
    """Hash effective (fake-quantized) weights and qparams of every quantized Linear.

    Run inside disable_offloading(): offloaded modules return a fresh copy on every
    attribute access, so the weight QDQ in quantized_forward would otherwise be lost.
    """
    prints = {}
    with torch.no_grad(), disable_offloading():
        for name, module in model.named_modules():
            scheme = getattr(module, "quantization_scheme", None)
            if scheme is None or scheme.weights is None or not hasattr(module, "weight"):
                continue
            weight = module.weight.data.to("cuda:0")
            effective = forward_quantize(module, weight, "weight", scheme.weights)
            entry = {"weight_effective": sha(effective.to(torch.bfloat16)),
                     "weight_scale": sha(module.weight_scale)}
            if hasattr(module, "input_global_scale"):
                entry["input_global_scale"] = sha(module.input_global_scale.float())
            if hasattr(module, TRANSFORM):
                entry["transform"] = sha(getattr(module, TRANSFORM).weight)
            prints[name] = entry
    return prints


def build_calibration_set(tokenizer, num_samples: int, max_len: int, seed: int):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=seed).select(range(num_samples))

    def encode(row):
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        return tokenizer(text, max_length=max_len, truncation=True, add_special_tokens=False)

    return ds.map(encode, remove_columns=ds.column_names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--scheme", choices=["NVFP4", "NVFP4A16"], required=True)
    p.add_argument("--mode", choices=["rtn", "sic"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    manifest_path = Path(str(out) + "-calibration.json")
    if out.exists() and manifest_path.exists() and json.loads(manifest_path.read_text()).get("status") == "passed":
        print(f"CALIBRATION_ALREADY_COMPLETE {out}")
        return
    partial = Path(str(out) + ".partial")
    shutil.rmtree(partial, ignore_errors=True)

    started = time.time()
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    calibration = build_calibration_set(tokenizer, args.num_samples, args.max_len, args.seed)
    tokens = sum(len(x) for x in calibration["input_ids"])
    print(f"calibration samples={len(calibration)} tokens={tokens}", flush=True)

    modifier = OSFP4Modifier(scheme=args.scheme, optimization_mode=args.mode, ignore=["lm_head"])
    oneshot(model=model, processor=tokenizer, dataset=calibration, recipe=[modifier],
            num_calibration_samples=args.num_samples, max_seq_length=args.max_len,
            pipeline="sequential", sequential_targets=["Olmo3DecoderLayer"])
    calibrated = time.time()

    before = quantized_layer_fingerprint(model)
    runtime_layers = sum("transform" in v for v in before.values())
    model.save_pretrained(partial, save_compressed=True)
    tokenizer.save_pretrained(partial)
    del model
    torch.cuda.empty_cache()

    reloaded, _, info = load_model(partial, device="cuda:0")
    with torch.no_grad():
        reloaded(torch.tensor([[100257, 9906]], device="cuda:0"))  # decompress packed weights
    after = quantized_layer_fingerprint(reloaded)
    mismatches = sorted(
        f"{name}.{key}" for name in before for key in before[name]
        if after.get(name, {}).get(key) != before[name][key]
    ) + sorted(set(after) - set(before))
    status = "passed" if not mismatches and info["transforms"] == runtime_layers else "failed"

    repo = Path(sys.modules["llmcompressor"].__file__).resolve().parents[2]
    diff = subprocess.run(["git", "-C", str(repo), "diff", "HEAD"], capture_output=True, text=True).stdout
    manifest = {
        "status": status,
        "model": args.model, "revision": args.revision,
        "scheme": args.scheme, "optimization_mode": args.mode,
        "modifier": modifier.model_dump(exclude={"resolved_config"}, mode="json") if hasattr(modifier, "model_dump") else str(modifier),
        "calibration": {"dataset": "HuggingFaceH4/ultrachat_200k", "split": "train_sft",
                        "shuffle_seed": args.seed, "num_samples": len(calibration),
                        "max_seq_length": args.max_len, "tokens": tokens,
                        "chat_template": True, "add_special_tokens": False},
        "pipeline": {"name": "sequential", "sequential_targets": ["Olmo3DecoderLayer"]},
        "quantized_layers": len(before), "runtime_transform_layers": runtime_layers,
        "reload_parity": {"checked_tensors": sum(len(v) for v in before.values()),
                          "transforms_loaded": info["transforms"], "mismatches": mismatches[:50]},
        "runtime": {"repo": str(repo), "branch": subprocess.run(["git", "-C", str(repo), "branch", "--show-current"], capture_output=True, text=True).stdout.strip(),
                    "commit": subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                    "uncommitted_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                    "torch": torch.__version__,
                    "transformers": sys.modules["transformers"].__version__,
                    "compressed_tensors": sys.modules["compressed_tensors"].__version__},
        "seconds": {"calibration": round(calibrated - started, 1), "total": round(time.time() - started, 1)},
        "activation_subsampling_records": len(getattr(modifier, "_activation_subsampling_records", {})),
    }
    (Path(str(out) + "-runtime.diff")).write_text(diff)
    if status == "passed":
        shutil.rmtree(out, ignore_errors=True)
        os.replace(partial, out)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("status", "quantized_layers", "runtime_transform_layers", "reload_parity", "seconds")}, indent=2))
    if status != "passed":
        raise SystemExit(f"CALIBRATION_FAILED {out}: reload parity mismatches")
    print(f"CALIBRATION_PASSED {out}")


if __name__ == "__main__":
    main()
