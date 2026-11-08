"""Load BF16 or OSFP4 compressed-tensors checkpoints with their runtime transforms.

transformers' compressed-tensors quantizer applies ``quantization_config`` but not
``transform_config``. A plain ``from_pretrained`` therefore drops every
``*.smooth_quant_input.weight`` and evaluates weights that were already divided by
the smoothing scale without multiplying the inputs back: a silently wrong model.
This loader re-applies the transform config and loads every transform scale
strictly (no missing, no extra, exact dtype).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

TRANSFORM_SUFFIX = ".smooth_quant_input.weight"


def _transform_config(path: Path):
    config = json.loads((path / "config.json").read_text())
    quantization_config = config.get("quantization_config") or {}
    return quantization_config.get("transform_config")


def _read_transform_tensors(path: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    for shard in sorted(path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt") as handle:
            for key in handle.keys():
                if key.endswith(TRANSFORM_SUFFIX):
                    tensors[key] = handle.get_tensor(key)
    return tensors


def load_model(path: str | Path, device: str = "cuda:0"):
    """Return ``(model, tokenizer, info)``; ``info`` records the transform check."""
    path = Path(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, device_map=device
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    # Olmo-3's generation_config.json sets max_new_tokens=32768 and sampling defaults
    # (do_sample, temperature 0.6, top_p 0.95). transformers lets max_new_tokens win
    # over the max_length that lm-eval uses to enforce max_gen_toks, so the 1,024-token
    # cap would silently not apply. Neutralize the defaults; every harness passes its
    # own generation settings explicitly. EOS ids (<|im_end|>, <|endoftext|>) are kept.
    gc = model.generation_config
    gc.max_new_tokens, gc.max_length = None, None
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None
    info = {"path": str(path), "transforms": 0,
            "generation_config_eos": gc.eos_token_id}
    transform_config = _transform_config(path)
    saved = _read_transform_tensors(path)
    if transform_config is None:
        if saved:
            raise RuntimeError(f"{path}: transform tensors present without transform_config")
        return model, tokenizer, info

    from compressed_tensors.transform import TransformConfig, apply_transform_config
    from compressed_tensors.utils import update_offload_parameter

    apply_transform_config(model, TransformConfig.model_validate(transform_config))
    expected = {name for name, _ in model.named_parameters() if name.endswith(TRANSFORM_SUFFIX)}
    if expected != set(saved):
        raise RuntimeError(
            f"{path}: transform scales mismatch; missing={sorted(expected - set(saved))[:5]} "
            f"unexpected={sorted(set(saved) - expected)[:5]}"
        )
    for name, value in saved.items():
        module = model.get_submodule(name.removesuffix(".weight"))
        if value.dtype != module.weight.dtype or value.shape != module.weight.shape:
            raise RuntimeError(f"{name}: dtype/shape {value.dtype}{tuple(value.shape)} "
                               f"!= {module.weight.dtype}{tuple(module.weight.shape)}")
        update_offload_parameter(module, "weight", value)
        module._validated = False
    info["transforms"] = len(saved)
    return model, tokenizer, info
