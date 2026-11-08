"""Is weight QDQ on decompressed OSFP4 weights an exact no-op, and what does it cost?"""
import sys
import time
from pathlib import Path

import torch
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.quantization.lifecycle.forward import forward_quantize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from osfp4_loader import load_model  # noqa: E402

model, tok, info = load_model(sys.argv[1])
with torch.no_grad():
    model(torch.tensor([[100257, 9906]], device="cuda:0"))  # trigger decompression
quantized = [(n, m) for n, m in model.named_modules()
             if getattr(m, "quantization_scheme", None) is not None and m.quantization_scheme.weights is not None]
statuses = {str(m.quantization_status) for _, m in quantized}
not_idempotent = []
with torch.no_grad():
    for n, m in quantized:
        w = m.weight.data
        if not torch.equal(forward_quantize(m, w, "weight", m.quantization_scheme.weights), w):
            not_idempotent.append(n)
print(f"[qdq] layers={len(quantized)} statuses={statuses} non_idempotent={len(not_idempotent)} {not_idempotent[:3]}")

tok.padding_side = "left"
prompts = [tok.apply_chat_template([{"role": "user", "content": f"Count from 1 to {20 + i}."}], tokenize=False,
                                   add_generation_prompt=True) for i in range(16)]
enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")


def run():
    torch.cuda.synchronize(); t = time.time()
    with torch.no_grad():
        logits = model(**enc).logits.float()
        out = model.generate(**enc, max_new_tokens=64, min_new_tokens=64, do_sample=False,
                             eos_token_id=None, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    return logits, out, time.time() - t


logits_a, out_a, t_a = run()
for _, m in quantized:
    m.quantization_status = QuantizationStatus.COMPRESSED  # skip weight QDQ; input QDQ unchanged
logits_b, out_b, t_b = run()
print(f"[qdq] with weight QDQ: {t_a:.1f}s | skipped: {t_b:.1f}s | logits bit-identical: {torch.equal(logits_a, logits_b)} "
      f"| generated tokens identical: {torch.equal(out_a, out_b)}")
