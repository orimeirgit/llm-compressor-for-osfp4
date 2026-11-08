"""Olmo-3 architecture probes for OSFP4.

Part 1 (tiny random Olmo3): which SmoothQuant folds are exact equivalences?
Part 2 (real checkpoint): input statistics seen by each projection on a few
calibration samples, and what they imply for NVFP4 activation scales and SIC
dampening.
"""
import json
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)
model_path = sys.argv[1]
out_path = sys.argv[2]
report = {}


# ---------------------------------------------------------------- part 1
def tiny_olmo3():
    cfg = AutoConfig.from_pretrained(model_path)
    cfg.hidden_size, cfg.intermediate_size = 64, 128
    cfg.num_attention_heads = cfg.num_key_value_heads = 4
    cfg.head_dim = 16
    cfg.num_hidden_layers = 4
    cfg.layer_types = ["sliding_attention"] * 3 + ["full_attention"]
    cfg.vocab_size = 256
    cfg.pad_token_id = 0
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float64).eval()
    for name, p in model.named_parameters():  # non-trivial norm weights
        if "norm" in name:
            p.data.uniform_(0.5, 1.5)
    return model


def fold_into_norm(norm, linears, s):
    norm.weight.data *= s
    for lin in linears:
        lin.weight.data /= s.view(1, -1)


def fold_into_linear(prev, linears, s):
    prev.weight.data *= s.view(-1, 1)
    if prev.bias is not None:
        prev.bias.data *= s
    for lin in linears:
        lin.weight.data /= s.view(1, -1)


def runtime_scale(lin, s):
    lin.weight.data /= s.view(1, -1)
    lin.register_forward_pre_hook(lambda m, a: (a[0] * s,) + a[1:])


ids = torch.randint(0, 256, (2, 24))
cases = {
    "default_mapping: post_attention_layernorm -> gate/up (what the registry falls back to)":
        lambda L: fold_into_norm(L.post_attention_layernorm, [L.mlp.gate_proj, L.mlp.up_proj],
                                 torch.rand(64, dtype=torch.float64) + 0.5),
    "post_feedforward_layernorm(i) -> q/k/v(i+1) (looks plausible, also wrong)":
        None,
    "v_proj -> o_proj fold": lambda L: fold_into_linear(
        L.self_attn.v_proj, [L.self_attn.o_proj], torch.rand(64, dtype=torch.float64) + 0.5),
    "up_proj -> down_proj fold": lambda L: fold_into_linear(
        L.mlp.up_proj, [L.mlp.down_proj], torch.rand(128, dtype=torch.float64) + 0.5),
    "runtime input transform on q,k,v,gate,up,o,down (fixed branch)": "runtime",
}
part1 = {}
for label, fn in cases.items():
    model = tiny_olmo3()
    with torch.no_grad():
        ref = model(ids).logits
    layers = model.model.layers
    if fn is None:
        for i in range(len(layers) - 1):
            s = torch.rand(64, dtype=torch.float64) + 0.5
            nxt = layers[i + 1].self_attn
            fold_into_norm(layers[i].post_feedforward_layernorm, [nxt.q_proj, nxt.k_proj, nxt.v_proj], s)
    elif fn == "runtime":
        for L in layers:
            for lin in (L.self_attn.q_proj, L.self_attn.k_proj, L.self_attn.v_proj,
                        L.self_attn.o_proj, L.mlp.gate_proj, L.mlp.up_proj):
                runtime_scale(lin, torch.rand(lin.in_features, dtype=torch.float64) + 0.5)
            runtime_scale(L.mlp.down_proj, torch.rand(128, dtype=torch.float64) + 0.5)
    else:
        for L in layers:
            fn(L)
    with torch.no_grad():
        out = model(ids).logits
    rel = ((out - ref).norm() / ref.norm()).item()
    part1[label] = {"relative_logit_error": rel, "exact": rel < 1e-10}
    print(f"[fold] {rel:.3e}  {label}")
report["fold_equivalence_fp64"] = part1


# ---------------------------------------------------------------- part 2
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0").eval()
from datasets import load_dataset  # noqa: E402

ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=42).select(range(16))
texts = [tok.apply_chat_template(r["messages"], tokenize=False) for r in ds]
enc = [tok(t, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False) for t in texts]

layer_ids = [0, 1, 4, 8, 16, 24, 31]
stats = {}
rows = {}


def hook(name):
    def fn(mod, args):
        x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
        rows.setdefault(name, []).append(x[1:].cpu())  # drop BOS-position row
        rows.setdefault(name + "#withbos", []).append(x[:1].cpu())
    return fn


handles = []
for i in layer_ids:
    L = model.model.layers[i]
    for short, mod in {"q_proj": L.self_attn.q_proj, "o_proj": L.self_attn.o_proj,
                       "gate_proj": L.mlp.gate_proj, "down_proj": L.mlp.down_proj}.items():
        handles.append(mod.register_forward_pre_hook(hook(f"layers.{i}.{short}")))
with torch.no_grad():
    for e in enc:
        model(**{k: v.cuda() for k, v in e.items()})
for h in handles:
    h.remove()

FP4_MAX, FP8_MAX, FP8_MIN_SUB = 6.0, 448.0, 2.0 ** -9
for name in [n for n in rows if "#" not in n]:
    X = torch.cat(rows[name])
    first = torch.cat(rows[name + "#withbos"])
    absmax = X.abs().amax().item()
    absmax_with_first = max(absmax, first.abs().amax().item())
    ch_absmax = X.abs().amax(0)
    ch_rms = X.square().mean(0).sqrt()
    # NVFP4 activations: per-tensor global scale from calibration absmax, then per-16 FP8 block scales.
    global_scale = FP4_MAX * FP8_MAX / absmax_with_first
    g = X.reshape(X.shape[0], -1, 16).abs().amax(-1)          # group absmax
    block_scale = g / FP4_MAX * global_scale                  # value stored in E4M3
    underflow = (block_scale < FP8_MIN_SUB).float().mean().item()
    subnormal = (block_scale < 2.0 ** -6).float().mean().item()
    # SIC dampening: damp = 0.01 * mean(diag(Sigma_X)); compare to the median channel.
    diag = X.square().mean(0)
    damp = 0.01 * diag.mean()
    stats[name] = {
        "tokens": X.shape[0],
        "row_rms_median": X.square().mean(1).sqrt().median().item(),
        "absmax_excl_first_token": absmax,
        "absmax_incl_first_token": absmax_with_first,
        "top_channel_absmax": ch_absmax.topk(3).values.tolist(),
        "channel_absmax_max_over_median": (ch_absmax.max() / ch_absmax.median()).item(),
        "channel_rms_max_over_median": (ch_rms.max() / ch_rms.median()).item(),
        "fp8_block_scale_underflow_frac": underflow,
        "fp8_block_scale_subnormal_frac": subnormal,
        "sic_damp_over_median_diag": (damp / diag.median()).item(),
    }
    s = stats[name]
    print(f"[act] {name:22s} rms~{s['row_rms_median']:8.3f} absmax={s['absmax_incl_first_token']:9.1f} "
          f"ch_max/med={s['channel_absmax_max_over_median']:8.1f} underflow={underflow:.4f} "
          f"subnormal={subnormal:.4f} damp/med_diag={s['sic_damp_over_median_diag']:.3f}")
report["activation_stats"] = stats
json.dump(report, open(out_path, "w"), indent=2)
print("wrote", out_path)
