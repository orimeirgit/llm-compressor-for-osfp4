"""Locate where a reloaded OSFP4 checkpoint diverges from the in-memory model."""
import sys
import tempfile

import torch
from datasets import Dataset
from transformers import AutoTokenizer, Olmo3Config, Olmo3ForCausalLM

from llmcompressor import oneshot
from llmcompressor.modifiers.osfp4 import OSFP4Modifier

sys.path.insert(0, sys.argv[1])
from osfp4_loader import load_model  # noqa: E402

tok = AutoTokenizer.from_pretrained(sys.argv[2])
scheme, mode = sys.argv[3], sys.argv[4]
torch.manual_seed(0)
config = Olmo3Config(vocab_size=100278, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=4, head_dim=16, sliding_window=8,
                     layer_types=["sliding_attention", "full_attention"], pad_token_id=100277, use_cache=False)
model = Olmo3ForCausalLM(config).to(torch.bfloat16).eval()
base = tempfile.mkdtemp()
model.save_pretrained(base)
model.config.name_or_path = base
ids = torch.randint(0, 1000, (8, 32), generator=torch.Generator().manual_seed(1))
ds = Dataset.from_dict({"input_ids": ids.tolist(), "attention_mask": torch.ones_like(ids).tolist()})
oneshot(model=model, processor=tok, dataset=ds, num_calibration_samples=8, pipeline="sequential",
        sequential_targets=["Olmo3DecoderLayer"],
        recipe=[OSFP4Modifier(scheme=scheme, optimization_mode=mode, ignore=["lm_head"], steps=2)])

probe = torch.randint(0, 1000, (2, 12), generator=torch.Generator().manual_seed(2))


def run(m):
    store, handles = {}, []
    for name, module in m.named_modules():
        if isinstance(module, torch.nn.Linear) or name.endswith("norm"):
            def hook(mod, args, output, name=name):
                store[name] = (args[0].detach().float().cpu(), output.detach().float().cpu())
            handles.append(module.register_forward_hook(hook))
    with torch.no_grad():
        logits = m(probe.to(m.model.embed_tokens.weight.device)).logits.float().cpu()
    for h in handles:
        h.remove()
    return logits, store


from compressed_tensors.offload import disable_offloading
with disable_offloading():
    mem_logits, mem_store = run(model)
mem_gs = {n: m.input_global_scale.detach().float().cpu().clone() for n, m in model.named_modules() if hasattr(m, 'input_global_scale')}
print('[diag] mem attn impl:', model.config._attn_implementation)
out = tempfile.mkdtemp()
model.save_pretrained(out, save_compressed=True)
tok.save_pretrained(out)
loaded, _, info = load_model(out, device=sys.argv[5] if len(sys.argv) > 5 else "cpu")
new_logits, new_store = run(loaded)
print('[diag] new attn impl:', loaded.config._attn_implementation)
for n, m in loaded.named_modules():
    if n in mem_gs:
        g = m.input_global_scale.detach().float().cpu()
        if not torch.equal(g, mem_gs[n]):
            print(f'[diag]   input_global_scale differs at {n}: mem={mem_gs[n].tolist()} new={g.tolist()}')
            break
else:
    print('[diag]   input_global_scale all equal' if mem_gs else '[diag]   no input_global_scale')
print(f"[diag] {scheme}-{mode}: transforms={info['transforms']} logits max|d|={(new_logits - mem_logits).abs().max().item():.4e}")
for name in mem_store:
    xin_m, out_m = mem_store[name]
    xin_n, out_n = new_store[name]
    din, dout = (xin_n - xin_m).abs().max().item(), (out_n - out_m).abs().max().item()
    if din > 0 or dout > 0:
        print(f"[diag]   first divergence at {name}: max|d input|={din:.3e} max|d output|={dout:.3e}")
        break
else:
    print("[diag]   no per-module divergence")
