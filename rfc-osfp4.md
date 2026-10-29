# [RFC] Add OSFP4 to LLM Compressor

## Summary and scope

[OSFP4](https://arxiv.org/abs/2610.08231) jointly optimizes diagonal smoothing
and NVFP4 block scales. This RFC proposes upstreaming the existing
[OSFP4 modifier](https://github.com/neriahbd/OSFP4) into LLM Compressor (LLMC),
including calibration, checkpoint export, examples, and tests.

Proposed initial scope: dense models, `NVFP4` (W4A4) and `NVFP4A16` (W4A16),
RTN and Hessian-aware SIC optimization, and single-process sequential
calibration. OSFP4 would initially remain the only modifier in its recipe.
MoE, distributed calibration, and modifier composition are follow-ups.

The implementation selects **option A: a compressed-tensors diagonal transform**
for explicit input scales. This requires a companion compressed-tensors change.
Native vLLM serving remains a separate contribution; the existing plugin export
was an alternative that did not require new compressed-tensors functionality.

Follows up on [neriahbd/OSFP4#2](https://github.com/neriahbd/OSFP4/issues/2)
(cc @yiliu30).

## Smoothing behavior

Ignoring bias, the equivalent linear operation is `y = (W / s) (x * s)`,
where `s` scales input channels and divides the corresponding weight columns.
Projections sharing an input may share a smoothing vector.

The current implementation folds scales into preceding normalization layers
where its mappings allow this. Other targets, including `o_proj` and
`down_proj`, retain explicit input scales. Foldability depends on architecture
and scale constraints; AWQ supports additional predecessor mappings.

## LLMC integration

Add `OSFP4Modifier` using `Modifier` and `QuantizationMixin`, with an `osfp4`
observer. The proposed interface preserves the existing options:

```python
from llmcompressor import oneshot
from llmcompressor.modifiers import OSFP4Modifier

oneshot(
    model=model,
    dataset=calibration_dataset,
    recipe=OSFP4Modifier(
        scheme="NVFP4", optimization_mode="sic",
        targets=["Linear"], ignore=["lm_head"],
    ),
)
model.save_pretrained(output_dir, save_compressed=True)
```

Follow LLMC's [modifier](https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-modifier/)
and [observer](https://docs.vllm.ai/projects/llm-compressor/en/latest/developer-tutorials/add-observer/)
contracts:

| Stage | OSFP4 responsibility |
|---|---|
| Initialization | Validate options; resolve targets and existing SmoothQuant mappings; initialize quantization. |
| Calibration start | Register capture hooks for activation statistics and RTN/SIC inputs. |
| Sequential epoch end | Optimize observed mappings; deploy smoothing and weight parameters; recompute W4A4 activation parameters from smoothed inputs; release mapping caches. |
| Calibration end / finalization | Verify target coverage, freeze quantization, remove capture hooks, and preserve export state. |

The observer integration must preserve optimized weight parameters and respect
fused-observer global-scale requirements. Generic observation must not overwrite
the selected parameters. Unsupported pipelines and distributed calibration
must be rejected explicitly.

Explicit smoothing must remain active during sequential propagation and
supported post-`oneshot` evaluation, independently of temporary capture hooks.
Reload must restore that behavior or reject an unsupported checkpoint.

## Checkpoint contract

Use standard compressed-tensors NVFP4 weights and quantization parameters with
`quant_method="compressed-tensors"`, a `transform_config` declaring
`type="diagonal"`, and one `<layer>.osfp4_input.weight` vector per explicit target.
The companion factory applies an elementwise input multiply with independent
per-layer vectors. Export only the online input transform, since inverse weight
balancing has already been applied. The existing vLLM plugin does not consume
this contract.

LLMC must export each explicit scale with its exact target association through
the normal save path. Preserve the current semantics: a positive, finite BF16
vector of length `in_features`, used as the activation multiplier. Weight
balancing and activation calibration must use the deployed, rounded scale.
Saved weights already include inverse balancing; loading must not apply it again.

A transform-aware loader must install transforms before loading their vectors.
The companion compressed-tensors change rejects loaders that discard diagonal
metadata. Transformers 5.17.0 is such an unsupported loader; explicit CT loading
is covered by packed-checkpoint round-trip tests. Older consumers may still
ignore metadata, so coordinated consumer support is required before deployment.

## Validation

LLMC tests should cover RTN/SIC in both precision modes, folded and explicit
scales, sequential propagation, post-calibration evaluation, export/reload,
and unsupported configurations. Compare against the reference implementation
and reproduce reported accuracy results against NVFP4 baselines, recording
calibration time and peak memory. Add a runnable calibration example.

## Future vLLM work — separate contribution

Native vLLM support must implement the diagonal transform contract: load and
validate vectors, map targets to vLLM layers (including fused projections), and
apply smoothing exactly once without dense-matrix execution or `sqrt(n)`
normalization. W4A4 must scale before activation
quantization, including fused paths; W4A16 must scale before the weight-only
linear operation. Tensor-parallel loading must shard
input-channel scales for row-parallel layers and replicate them for
column-parallel layers.

Kernel fusion is a later optimization. Plugin deprecation follows compatible
vLLM releases and validation, after this transform contract is supported. Legacy plugin-checkpoint migration
is a separate compatibility decision.

## Questions for maintainers

1. Is the proposed initial LLMC scope acceptable?
2. Is the modifier/observer division appropriate for this joint optimization?
3. Is the selected diagonal-transform contract acceptable, with a companion
   compressed-tensors change and separate vLLM consumer implementation?
