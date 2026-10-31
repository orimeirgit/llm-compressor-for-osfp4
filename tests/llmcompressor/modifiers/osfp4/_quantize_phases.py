# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Compose production phases for mapping-level tests."""

from llmcompressor.modifiers.osfp4 import base, osfp4_quantize, smoothing


def run_phases(
    mapping,
    cache,
    *,
    optimization_mode,
    dampening_frac,
    weight_only,
    sampled_activations=None,
):
    result = osfp4_quantize.quantize_mapping(
        mapping,
        cache,
        optimization_mode=optimization_mode,
        dampening_frac=dampening_frac,
        weight_only=weight_only,
        sampled_activations=sampled_activations,
    )
    scale = smoothing.deploy_mapping_smooth_quant_scale(mapping, result.alpha)
    if not weight_only:
        smoothing.update_smoothed_input_global_scale(mapping, cache, scale)
    return base.prepare_mapping_parameters(mapping, result)
