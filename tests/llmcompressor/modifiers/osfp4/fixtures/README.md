# OSFP4 reference fixtures

Unmodified fixtures from `neriahbd/OSFP4` commit
`21e069ca80468fa9c44c82aa4e1378531a897c42`, `tests/golden/fixtures`.
Apache-2.0; see the repository NOTICE for attribution.

The source generator uses a seeded two-layer FP32 Llama, four 16-token
calibration samples, two optimizer steps, and all combinations of NVFP4/NVFP4A16
and RTN/SIC. `test_checkpoint.py` uses the same setup and compares every tensor
byte-for-byte, translating `.smooth_quant_scale` to `.osfp4_input.weight`.
The JSON files document the historical plugin metadata and are not the new
export format.
