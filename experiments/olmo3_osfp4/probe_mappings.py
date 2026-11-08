"""Resolve OSFP4 mappings for a model on the meta device and print them."""
import sys, collections, torch
from transformers import AutoConfig, AutoModelForCausalLM
from llmcompressor.modifiers.osfp4 import OSFP4Modifier
from llmcompressor.modifiers.transform.smoothquant.dynamic_mappings import get_layer_mappings_from_model

path = sys.argv[1]
cfg = AutoConfig.from_pretrained(path)
print({k: getattr(cfg, k, None) for k in ["architectures","hidden_size","intermediate_size","num_hidden_layers","num_attention_heads","num_key_value_heads","sliding_window","tie_word_embeddings","rope_scaling","max_position_embeddings","vocab_size","hidden_act"]})
print("layer_types:", collections.Counter(cfg.layer_types), cfg.layer_types[:8])
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
print("class:", model.__class__.__name__)
print("registry mappings:", get_layer_mappings_from_model(model))
print("layer-0 modules:", [n for n, _ in model.model.layers[0].named_modules() if n])
mod = OSFP4Modifier(scheme="NVFP4", targets=["Linear"], ignore=["lm_head"], optimization_mode="rtn")
mappings = mod._resolve_mappings(model)
for m in mappings:
    if ".layers.0." in m.mapping_name or m.mapping_name.endswith("layers.0"):
        print("  layer0 mapping:", m.mapping_name, "smooth=", type(m.smooth_layer).__name__ if m.smooth_layer is not None else None,
              "balance=", [type(b).__name__ + f"[{b.in_features}->{b.out_features}]" for b in m.balance_layers])
smooth = [m for m in mappings if m.smooth_layer is not None]
print("total mappings", len(mappings), "smooth-layer", len(smooth), "runtime", len(mappings)-len(smooth))
