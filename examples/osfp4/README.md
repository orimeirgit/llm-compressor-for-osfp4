# OSFP4 (option A: diagonal transforms)

OSFP4 jointly optimizes channel smoothing and NVFP4 block scales. This
integration supports dense models, `NVFP4` (W4A4), `NVFP4A16` (W4A16), RTN and
Hessian-aware SIC. Calibration is sequential and single-process; OSFP4 must
be the only modifier in the recipe. MoE and modifier composition are unsupported.

## Install the two development branches

This implementation needs the companion compressed-tensors branch
`codex/osfp4-diagonal-transform`. From this repository, after installing the
normal LLMC dependencies:

```bash
pip install -e .
pip install --no-deps -e ../compressed-tensors-for-osfp4
```

Install the companion last: LLMC's released dependency pin currently resolves
to compressed-tensors 0.19.0, which has no diagonal transform. Update that pin
to the first release containing this support before releasing OSFP4. The
modifier checks for support and reports a clear error if it is missing.

## Calibrate and compare

```bash
python examples/osfp4/calibrate.py --model meta-llama/Llama-3.1-8B-Instruct \
  --output Llama-3.1-8B-OSFP4 --scheme NVFP4 --optimization-mode sic
python examples/osfp4/calibrate.py --model meta-llama/Llama-3.1-8B-Instruct \
  --output Llama-3.1-8B-NVFP4 --scheme NVFP4 --method baseline
```

Use `--optimization-mode rtn` for OSFP4 RTN and `--scheme NVFP4A16` for weight-only
quantization. Both methods use the same seed, dataset, sample count, sequence
length, and sequential pipeline. `calibration_metrics.json` records settings,
package versions, wall time for `oneshot` (including data preparation), and peak
CUDA allocation on each device. It excludes model loading, evaluation, and export.
CPU runs have an empty CUDA-memory result.

With `lm-eval` installed, add `--eval-tasks wikitext` (or other tasks) to evaluate
the live model before saving. Full-model accuracy, calibration time and GPU
memory still need measurement; tiny-model tests do not establish paper accuracy.
For paper reproduction, match the model revisions and data/evaluation protocols
in the [original OSFP4 examples](https://github.com/neriahbd/OSFP4/tree/main/examples).

## Recipe and lifecycle

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
    pipeline="sequential",
)
model.save_pretrained(output_dir, save_compressed=True)
```

The modifier also loads through LLMC's YAML recipe factory. Defaults are
`steps=80`, `lr=0.12`, `dampening_frac=0.01`, `offload_hessians=False`, and
`activation_subsample_size="auto"` (twice the input width). Weight-only mode
does not cache activation rows. Input widths must be divisible by 16.

Scales fold into preceding normalization layers when the architecture mappings
cover all their consumers. Other layers receive a permanent input transform.
Optimization runs at sequential epoch boundaries; temporary capture hooks and
caches are cleared, while transforms remain active during propagation and
post-calibration evaluation. Explicit scales are rounded to BF16 before weight
balancing and activation calibration. Optimized weight qparams are preserved.

## Checkpoint and consumer contract

Exports use `quant_method="compressed-tensors"` and
`quantization_config.transform_config.config_groups.osfp4`, with `type="diagonal"`
and exact target names. Every explicit target stores
`<layer>.osfp4_input.weight`: a positive, finite BF16 vector of length
`in_features`. Equal-width layers have independent vectors. The operation is
`x * s` before activation quantization; weights already contain `W / s`.
Only the online input transform is exported, so reload never balances weights
again. No dense matrix, Hadamard normalization, or plugin metadata is used.

The companion CT factory supports strict transform-aware state loading, including
packed NVFP4 checkpoints. Tests reconstruct quantization and transform modules
before loading the state dict and verify identical BF16 logits after decompression.
See `tests/llmcompressor/modifiers/osfp4/test_checkpoint.py` for the complete
loading sequence, including preservation of nonpersistent model buffers.

Transformers 5.17.0 does not apply `transform_config`; with the companion CT
changes its `from_pretrained` path rejects these checkpoints. A future loader
must attach transforms before loading weights and validate all required scales.
Keep this CT version installed; older consumers may ignore the metadata.

**Serving is a separate vLLM contribution.** Stock vLLM and the existing
`vllm-osfp4` plugin do not support this new contract. vLLM needs vector loading,
target mapping (including fused projections), an elementwise multiply before
NVFP4 activation quantization, no `sqrt(n)` normalization, and tensor-parallel
input-channel sharding/replication. W4A16 needs the same multiply before its
linear operation. Plugin-format migration and kernel fusion are follow-ups.

## Validation

```bash
pytest tests/llmcompressor/modifiers/osfp4 tests/llmcompressor/observers/osfp4
# In the companion repository:
pytest tests/test_transform/factory/test_diagonal.py
```

Tests cover numerical kernels, both optimizers and schemes, sequential
propagation, folded/explicit smoothing, finalization, invalid options,
partial-deployment failures, compressed export/reload, and unsupported loaders.
Four seeded FP32 cases match the original implementation byte-for-byte after
renaming only the explicit scale tensor keys. CUDA tests run when available.
