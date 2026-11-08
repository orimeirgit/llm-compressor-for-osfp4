"""Tiny Olmo3: OSFP4 calibrate -> save compressed -> reload; compare logits.

Shows (1) plain from_pretrained drops the transforms, (2) osfp4_loader restores them.
"""
import sys
import tempfile

import torch
from transformers import AutoTokenizer, Olmo3Config, Olmo3ForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.osfp4 import OSFP4Modifier
from transformers import AutoModelForCausalLM

sys.path.insert(0, sys.argv[1])
from osfp4_loader import load_model  # noqa: E402

tokenizer_path = sys.argv[2]
results = {}
for scheme in ("NVFP4", "NVFP4A16"):
    for mode in ("rtn", "sic"):
        torch.manual_seed(0)
        config = Olmo3Config(vocab_size=100278, hidden_size=64, intermediate_size=128,
                             num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4, head_dim=16,
                             sliding_window=8, layer_types=["sliding_attention"] * 3 + ["full_attention"],
                             pad_token_id=100277, use_cache=False)
        model = Olmo3ForCausalLM(config).to(torch.bfloat16).eval()
        base_dir = tempfile.mkdtemp()
        model.save_pretrained(base_dir)
        model.config.name_or_path = base_dir  # oneshot re-reads the on-disk config
        from datasets import Dataset
        ids = torch.randint(0, 1000, (8, 32), generator=torch.Generator().manual_seed(1))
        ds = Dataset.from_dict({"input_ids": ids.tolist(), "attention_mask": torch.ones_like(ids).tolist()})
        oneshot(model=model, processor=AutoTokenizer.from_pretrained(tokenizer_path), dataset=ds, recipe=[OSFP4Modifier(scheme=scheme, optimization_mode=mode,
                ignore=["lm_head"], steps=2)], num_calibration_samples=8, pipeline="sequential",
                sequential_targets=["Olmo3DecoderLayer"])
        probe = torch.randint(0, 1000, (2, 24), generator=torch.Generator().manual_seed(2))
        from compressed_tensors.offload import disable_offloading
        with torch.no_grad():
            # After oneshot every module is CPU-offloaded; each `self.weight` access
            # onloads a fresh copy, so quantized_forward's patched weight QDQ is lost
            # (RTN would run with source weights). Keep tensors onloaded instead.
            stale = model(probe.cuda()).logits.float()
            with disable_offloading():
                ref = model(probe.cuda()).logits.float()
        offload_gap = (stale - ref).abs().max().item()
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp, save_compressed=True)
            AutoTokenizer.from_pretrained(tokenizer_path).save_pretrained(tmp)
            plain = AutoModelForCausalLM.from_pretrained(tmp, dtype=torch.bfloat16, device_map="cuda:0").eval()
            with torch.no_grad():
                plain_err = (plain(probe.cuda()).logits.float() - ref).abs().max().item()
            n_plain = sum(1 for n, _ in plain.named_parameters() if "smooth_quant_input" in n)
            del plain
            loaded, _, info = load_model(tmp)
            with torch.no_grad():
                fixed_err = (loaded(probe.cuda()).logits.float() - ref).abs().max().item()
        results[(scheme, mode)] = (n_plain, plain_err, info["transforms"], fixed_err, ref.abs().max().item(), offload_gap)
for k, (n_plain, plain_err, n_fixed, fixed_err, scale, gap) in results.items():
    print(f"[roundtrip] {k}: plain from_pretrained transforms={n_plain} max|dlogit|={plain_err:.4f} | "
          f"osfp4_loader transforms={n_fixed} max|dlogit|={fixed_err:.6f} (logit scale {scale:.3f}) | "
          f"in-memory offloaded-vs-onloaded gap={gap:.4f}")
