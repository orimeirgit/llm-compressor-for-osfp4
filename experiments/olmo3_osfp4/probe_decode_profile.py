"""Profile decode steps of a loaded checkpoint (relative breakdown; GPU may be shared)."""
import sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from osfp4_loader import load_model
model, tok, info = load_model(sys.argv[1])
tok.padding_side = "left"
enc = tok([tok.apply_chat_template([{"role": "user", "content": "Write a long story."}], tokenize=False, add_generation_prompt=True)] * 16,
          return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
gen = dict(max_new_tokens=48, min_new_tokens=48, do_sample=False, eos_token_id=None, pad_token_id=tok.pad_token_id)
with torch.no_grad():
    model.generate(**enc, **gen)
    torch.cuda.synchronize(); t = time.time(); model.generate(**enc, **gen); torch.cuda.synchronize()
    step = (time.time() - t) / 48
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model.generate(**enc, **{**gen, "max_new_tokens": 8, "min_new_tokens": 8})
ka = prof.key_averages()
cuda_total = sum(e.self_device_time_total for e in ka) / 1e3 / 8
launches = sum(e.count for e in ka if e.device_type.name == "CUDA") / 8
print(f"[prof] {sys.argv[2]}: {step*1000:.0f} ms/step at batch 16 | GPU busy {cuda_total:.0f} ms/step | kernels/step {launches:.0f} | transforms {info['transforms']}")
print(ka.table(sort_by="self_device_time_total", row_limit=6).splitlines()[3:9] and "\n".join(ka.table(sort_by="self_device_time_total", row_limit=6).splitlines()[:10]))
