# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Calibrate OSFP4 or its NVFP4 baseline and record time and CUDA peak memory."""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from llmcompressor import oneshot
from llmcompressor.modifiers import OSFP4Modifier
from llmcompressor.modifiers.quantization import QuantizationModifier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=["osfp4", "baseline"], default="osfp4")
    parser.add_argument("--scheme", choices=["NVFP4", "NVFP4A16"], default="NVFP4")
    parser.add_argument("--optimization-mode", choices=["rtn", "sic"], default="sic")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--dataset", default="perfectblend")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-tasks", nargs="+", help="Optional lm-eval task names")
    args = parser.parse_args()
    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    options = dict(scheme=args.scheme, targets=["Linear"], ignore=["lm_head"])
    recipe = (
        OSFP4Modifier(
            **options, optimization_mode=args.optimization_mode, steps=args.steps
        )
        if args.method == "osfp4"
        else QuantizationModifier(**options)
    )
    accelerator = torch.accelerator.current_accelerator()
    cuda_devices = (
        list(range(torch.accelerator.device_count()))
        if accelerator is not None
        and accelerator.type == "cuda"
        and torch.accelerator.is_available()
        else []
    )
    for device in cuda_devices:
        torch.accelerator.synchronize(device)
        torch.accelerator.reset_peak_memory_stats(device)
    started = time.perf_counter()
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=args.dataset,
        recipe=recipe,
        num_calibration_samples=args.samples,
        max_seq_length=args.max_seq_length,
        pipeline="sequential",
    )
    for device in cuda_devices:
        torch.accelerator.synchronize(device)
    metrics = {
        "arguments": vars(args),
        "oneshot_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": {
            str(device): torch.accelerator.max_memory_allocated(device)
            for device in cuda_devices
        },
        "versions": {
            name: version(name)
            for name in ("torch", "transformers", "llmcompressor", "compressed-tensors")
        },
    }
    if args.eval_tasks:
        # Evaluate the live model: its diagonal hooks survive oneshot finalization.
        from lm_eval import simple_evaluate
        from lm_eval.models.huggingface import HFLM

        metrics["evaluation"] = simple_evaluate(
            model=HFLM(pretrained=model, tokenizer=tokenizer, batch_size=1),
            tasks=args.eval_tasks,
        )
    model.save_pretrained(args.output, save_compressed=True)
    tokenizer.save_pretrained(args.output)
    (args.output / "calibration_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
