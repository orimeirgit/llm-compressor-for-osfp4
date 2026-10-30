# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Real sequential calibration, reference parity, and compressed checkpoint reload."""

import json
from pathlib import Path

import pytest
import torch
from accelerate import init_empty_weights
from compressed_tensors.compressors import ModelCompressor
from compressed_tensors.quantization import (
    QuantizationConfig,
    apply_quantization_config,
)
from compressed_tensors.transform import TransformConfig, apply_transform_config
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from llmcompressor.args import DatasetArguments
from llmcompressor.core import create_session
from llmcompressor.modifiers.osfp4 import OSFP4Modifier
from llmcompressor.pipelines import CalibrationPipeline
from llmcompressor.pipelines.sequential import pipeline as sequential
from llmcompressor.transformers.compression.compressed_tensors_utils import (
    modify_save_pretrained,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scheme", ["NVFP4", "NVFP4A16"])
@pytest.mark.parametrize("mode", ["rtn", "sic"])
def test_sequential_reference_and_compressed_round_trip(
    scheme, mode, dtype, tmp_path, monkeypatch
):
    # Same seeded inputs and optimizer settings as the upstream OSFP4 fixtures.
    torch.manual_seed(42)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).to(dtype).eval()
    generator = torch.Generator().manual_seed(0)
    samples = [
        {"input_ids": torch.randint(0, 64, (1, 16), generator=generator)}
        for _ in range(4)
    ]
    loader = torch.utils.data.DataLoader(samples, batch_size=None)
    modifier = OSFP4Modifier(
        scheme=scheme,
        optimization_mode=mode,
        ignore=["lm_head"],
        steps=2,
        activation_subsample_size=16384,
    )
    monkeypatch.setattr(sequential, "get_main_device", lambda: torch.device("cpu"))
    with create_session() as session:
        session.initialize(
            model=model,
            recipe=[modifier],
            start=-1,
            calib_data=loader,
            sequential_targets=["LlamaDecoderLayer"],
        )
        pipeline = CalibrationPipeline.from_modifiers(
            session.lifecycle.recipe.modifiers
        )
        pipeline(
            model,
            loader,
            DatasetArguments(
                sequential_targets=["LlamaDecoderLayer"],
                sequential_offload_device="cpu",
                propagate_error=True,
            ),
        )
        with torch.no_grad():
            before_finalize = model(**samples[0]).logits
        session.finalize()
    with torch.no_grad():
        expected = model(**samples[0]).logits
    torch.testing.assert_close(expected, before_finalize, rtol=0, atol=0)

    if dtype == torch.float32:
        # The published fixtures use FP32; the NVFP4 packer reloads BF16 weights.
        reference = load_file(
            str(
                Path(__file__).parent
                / "fixtures"
                / f"{scheme.lower()}-{mode}.safetensors"
            )
        )
        # Only the storage name changes from the plugin contract to option A.
        reference = {
            name.replace(".smooth_quant_scale", ".osfp4_input.weight"): value
            for name, value in reference.items()
        }
        actual = model.state_dict()
        assert actual.keys() == reference.keys()
        for name, value in reference.items():
            assert actual[name].dtype == value.dtype, name
            assert torch.equal(
                actual[name].reshape(-1).view(torch.uint8),
                value.reshape(-1).view(torch.uint8),
            ), name

        return

    modify_save_pretrained(model)
    model.save_pretrained(tmp_path, save_compressed=True)
    saved_config = json.loads((tmp_path / "config.json").read_text())
    qdata = saved_config["quantization_config"]
    assert qdata["quant_method"] == "compressed-tensors"
    assert "osfp4_metadata" not in saved_config
    tconfig = TransformConfig.model_validate(qdata.pop("transform_config"))
    online = tconfig.config_groups["osfp4"]
    assert online.type == "diagonal"
    assert len(online.apply) == 1
    assert online.apply[0].location == "input"
    assert not online.apply[0].inverse
    saved_state = load_file(str(tmp_path / "model.safetensors"))
    assert any(key.endswith("weight_packed") for key in saved_state)
    for name in online.apply[0].targets:
        scale = saved_state[f"{name}.osfp4_input.weight"]
        assert scale.dtype == torch.bfloat16
        assert scale.ndim == 1
        assert (scale > 0).all() and torch.isfinite(scale).all()

    # Explicit transform-aware CT loading; weights already include balancing.
    qconfig = QuantizationConfig.model_validate(qdata)
    with init_empty_weights():
        restored = LlamaForCausalLM(config).to(dtype).eval()
        apply_quantization_config(restored, qconfig)
        apply_transform_config(restored, tconfig)
        compressor = ModelCompressor(
            quantization_config=qconfig, transform_config=tconfig
        )
        compressor.compress_model(restored)
    restored.load_state_dict(saved_state, assign=True, strict=True)
    with torch.no_grad():
        torch.testing.assert_close(
            restored(**samples[0]).logits, expected, rtol=0, atol=0
        )
        torch.testing.assert_close(model(**samples[0]).logits, expected, rtol=0, atol=0)

    # Older Transformers must fail instead of silently dropping transform_config.
    import inspect

    from transformers import CompressedTensorsConfig

    if "transform_config" not in inspect.signature(CompressedTensorsConfig).parameters:
        with pytest.raises(ValueError, match="loader does not handle diagonal"):
            AutoModelForCausalLM.from_pretrained(tmp_path, local_files_only=True)


def test_public_oneshot_entrypoint_with_default_pipeline(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from llmcompressor import oneshot

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    config._name_or_path = str(tmp_path / "source")
    config.save_pretrained(config._name_or_path)
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1})),
        unk_token="<unk>",
        eos_token="<eos>",
    )
    loader = torch.utils.data.DataLoader(
        [{"input_ids": torch.arange(8).reshape(1, -1)}], batch_size=None
    )
    monkeypatch.setattr(sequential, "get_main_device", lambda: torch.device("cpu"))
    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=loader,
        recipe=OSFP4Modifier(scheme="NVFP4", steps=1, ignore=["lm_head"]),
    )
    with torch.no_grad():
        assert torch.isfinite(
            model(input_ids=torch.arange(8).reshape(1, -1)).logits
        ).all()
    model.save_pretrained(tmp_path, save_compressed=True)
    metadata = json.loads((tmp_path / "config.json").read_text())["quantization_config"]
    assert metadata["transform_config"]["config_groups"]["osfp4"]["type"] == "diagonal"
