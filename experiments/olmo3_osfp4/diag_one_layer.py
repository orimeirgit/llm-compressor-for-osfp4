import sys, tempfile, torch
exec(open(sys.argv[5]).read().split("probe = torch.randint")[0])  # reuse calibration setup above
from compressed_tensors.quantization.lifecycle.forward import forward_quantize
q = model.model.layers[0].self_attn.q_proj
x = torch.randn(5, 64, dtype=torch.bfloat16)
print("[one] mem q_proj device:", q.weight.device, "status:", q.quantization_status)
with torch.no_grad():
    dev = q.weight.device
    out_mem = q(x.to(dev)).float().cpu()
    xt = q.smooth_quant_input(x.to(dev))
    w_eff = forward_quantize(q, q.weight.data, "weight", q.quantization_scheme.weights)
    manual = torch.nn.functional.linear(xt, w_eff).float().cpu()
    manual_cpu = torch.nn.functional.linear(xt.cpu(), w_eff.cpu()).float()
print("[one] module vs manual(same device):", (out_mem - manual).abs().max().item(),
      "| same-device vs cpu matmul:", (manual - manual_cpu).abs().max().item(),
      "| out scale:", out_mem.abs().max().item())
with torch.no_grad():
    F = torch.nn.functional.linear
    xd = x.to(dev); s = q.smooth_quant_input.weight
    cands = {
        "no transform, qdq(W)": F(xd, w_eff),
        "transform twice, qdq(W)": F(xd * s * s, w_eff),
        "transform, source W": F(xd * s, q.weight.data),
        "no transform, source W": F(xd, q.weight.data),
        "transform, qdq(W) [intended]": F(xd * s, w_eff),
    }
    for k, v in cands.items():
        print(f"[one]   {k:32s} max|d|={(out_mem - v.float().cpu()).abs().max().item():.3e}")
    print("[one] pre-hooks:", len(q._forward_pre_hooks), "| hooks:", len(q._forward_hooks),
          "| quantization_enabled:", getattr(q, "quantization_enabled", None))
import inspect
fwd = q.forward
print("[why] forward func:", getattr(fwd, "__func__", fwd), "| defined in:", inspect.getsourcefile(getattr(fwd, "__func__", fwd)))
print("[why] scheme.weights:", q.quantization_scheme.weights is not None, "| status < COMPRESSED:",
      q.quantization_status < __import__("compressed_tensors").quantization.QuantizationStatus.COMPRESSED)
print("[why] instance forward in __dict__:", "forward" in q.__dict__)
orig = getattr(q, "_original_forward_func", None)
print("[why2] wrapper code:", q.forward.__func__.__code__.co_filename.split("src/")[-1], q.forward.__func__.__code__.co_name)
print("[why2] _original_forward_func:", None if orig is None else (orig.__code__.co_filename.split("site-packages/")[-1].split("src/")[-1], orig.__code__.co_name))
print("[why2] wrapped chain:", [getattr(f, "__code__", None) and f.__code__.co_name for f in [getattr(q.forward.__func__, "__wrapped__", None)]])
a, b = q.weight, q.weight
print("[why3] offloaded:", type(q._parameters).__name__, "| q.weight is q.weight:", a is b, "| same storage:", a.data_ptr() == b.data_ptr())
from compressed_tensors.offload import remove_module_offload
remove_module_offload(q, onload_tensors=True)
with torch.no_grad():
    out_plain = q(x.to(q.weight.device)).float().cpu()
    xt2 = q.smooth_quant_input(x.to(q.weight.device))
    intended = torch.nn.functional.linear(xt2, forward_quantize(q, q.weight.data, "weight", q.quantization_scheme.weights)).float().cpu()
print("[why3] after removing offload: module vs intended max|d|:", (out_plain - intended).abs().max().item())
