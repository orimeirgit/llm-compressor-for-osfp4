# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

"""Byte-exact comparisons with the independent pre-rename observer snapshot."""

import pytest
import torch
from compressed_tensors.quantization import preset_name_to_scheme

from llmcompressor.observers.osfp4 import loss, lut, observer
from llmcompressor.observers.osfp4 import scale_optimization as optimization
from llmcompressor.observers.osfp4 import scale_selection as selection

from ._naming_reference import loss as old_loss
from ._naming_reference import lut as old_lut
from ._naming_reference import observer as old_observer
from ._naming_reference import scale_optimization as old_optimization
from ._naming_reference import scale_selection as old_selection

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.get_device_module("cuda").is_available(),
            reason="CUDA unavailable",
        ),
    ),
]


def assert_bytes(actual, expected):
    if expected is None:
        assert actual is None
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for new, old in zip(actual, expected, strict=True):
            assert_bytes(new, old)
    else:
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert torch.equal(
            actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        )


def inputs(device, blocks=3, dtype=torch.float32):
    generator = torch.Generator().manual_seed(147)
    # Noncontiguous tensors with multiple blocks and unequal m/k dimensions.
    W = torch.randn(blocks, 16, 5, generator=generator).transpose(1, 2)
    X = torch.randn(blocks, 7, 16, generator=generator).transpose(1, 2)
    W[0, 0, 0] = -0.0
    X[0, 0, 0] = 0.0
    metric = X.square().mean(dim=2)
    return tuple(t.to(device=device, dtype=dtype) for t in (W, X, metric))


def coefficients(module, W, X, metric):
    if X is None:
        return module.build_weight_loss_coefficients(W, weight_metric=metric)
    reference_kwargs = {"activation_reference": W} if module is old_loss else {}
    return module.build_joint_loss_coefficients(
        W, X, weight_metric=metric, **reference_kwargs
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("blocks", [1, 3])
@pytest.mark.parametrize("component", ["weight", "activation", "joint"])
def test_loss_coefficients_values_and_gradients_match_bytes(device, blocks, component):
    W, X, metric = inputs(device, blocks)

    def execute(module, scale_module):
        coeff = coefficients(module, W, X, metric)
        initial = scale_module.initialize_scale_values_from_absmax(W, X)
        alpha, gamma_w, gamma_x = (
            value.detach().clone().requires_grad_(True) for value in initial
        )
        if component == "weight":
            result = module.compute_weight_loss(W, alpha, gamma_w, coeff.coeff_w)
        elif component == "activation":
            keyword = "row_count" if module is old_loss else "m"
            result = module.compute_activation_loss(
                X, alpha, gamma_x, coeff.coeff_x, **{keyword: W.shape[1]}
            )
        else:
            result = module.compute_joint_loss(W, X, alpha, gamma_w, gamma_x, coeff)
        gradients = torch.autograd.grad(
            result.sum(), (alpha, gamma_w, gamma_x), allow_unused=True
        )
        return coeff, result, gradients

    assert_bytes(execute(loss, optimization), execute(old_loss, old_optimization))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("weight_only", [False, True])
@pytest.mark.parametrize("steps", [0, 1, 4])
def test_initialization_adam_steps_and_materialization_match_bytes(
    device, weight_only, steps
):
    W, X, metric = inputs(device)
    X = None if weight_only else X

    def execute(module, loss_module):
        initial = module.initialize_scale_values_from_absmax(W, X)
        parameters = module.create_log_scale_parameters(initial)
        snapshots = [
            initial,
            tuple(p.detach().clone() if p is not None else None for p in parameters),
        ]
        gradients = []
        handles = [
            p.register_hook(lambda grad: gradients.append(grad.clone()))
            for p in parameters
            if p is not None
        ]
        coeff = coefficients(loss_module, W, X, metric)

        def evaluate():
            if X is None:
                result = loss_module.compute_weight_loss(
                    W, parameters.log_alpha.exp(), parameters.log_gamma_w.exp(), coeff
                )
            else:
                result = loss_module.compute_joint_loss(
                    W,
                    X,
                    parameters.log_alpha.exp(),
                    parameters.log_gamma_w.exp(),
                    parameters.log_gamma_x.exp(),
                    coeff,
                )
            snapshots.append(
                (
                    result.detach().clone(),
                    tuple(
                        p.detach().clone() if p is not None else None
                        for p in parameters
                    ),
                )
            )
            return result.sum()

        module.run_optimizer(parameters, evaluate, steps=steps, lr=0.03)
        snapshots.append(
            tuple(p.detach().clone() if p is not None else None for p in parameters)
        )
        module.canonicalize_log_scale_parameters(parameters)
        snapshots.append(
            tuple(p.detach().clone() if p is not None else None for p in parameters)
        )
        snapshots.append(module.materialize_scale_values(parameters))
        for handle in handles:
            handle.remove()
        return snapshots, gradients

    assert_bytes(execute(optimization, loss), execute(old_optimization, old_loss))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("tile_rows", [1, 3, 1000])
@pytest.mark.parametrize("case", ["normal", "tie", "fallback", "invalid"])
def test_e4m3_selection_masks_and_fallback_match_bytes(
    device, tile_rows, case, monkeypatch
):
    W, _, metric = inputs(device)
    alpha = torch.linspace(0.6, 1.8, 48, device=device).reshape(3, 16)
    gamma = torch.linspace(0.1, 2.0, 15, device=device).reshape(3, 5, 1)
    if case == "tie":
        metric.zero_()
    elif case == "fallback":
        gamma.flatten()[0] = 1e-20
        gamma.flatten()[-1] = 1e20
    elif case == "invalid":
        gamma.flatten()[:4] = torch.tensor(
            [0.0, -1.0, float("nan"), float("inf")], device=device
        )
    for module in (selection, old_selection):
        monkeypatch.setattr(
            module, "_E4M3_SEARCH_PRIMARY_TENSOR_BYTES", tile_rows * 19 * 16 * 4
        )
    assert_bytes(
        selection._search_e4m3_gamma_star(W, alpha, gamma, metric),
        old_selection._search_e4m3_gamma_star(W, alpha, gamma, metric),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lut_tables_values_and_gradients_match_bytes(device, dtype):
    assert_bytes(lut._get_phi_lut(device, dtype), old_lut._get_phi_lut(device, dtype))
    assert_bytes(
        lut._get_lut_interpolation_data(device, dtype),
        old_lut._get_lut_interpolation_data(device, dtype),
    )
    values = torch.tensor(
        [
            -1000,
            -6,
            -0.0625,
            -0.0,
            0.0,
            0.01,
            0.0625,
            0.25,
            0.75,
            1.25,
            1.75,
            2.5,
            3.5,
            5.0,
            6.0,
            255.0,
            256.0,
            1000,
        ],
        dtype=dtype,
        device=device,
    )

    def execute(module):
        x = values.clone().requires_grad_(True)
        result = module.phi_lut(x)
        return result, torch.autograd.grad(result.sum(), x)

    assert_bytes(execute(lut), execute(old_lut))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("weight_only", [False, True])
@pytest.mark.parametrize("steps", [0, 3])
def test_observer_rounded_scales_and_qparams_match_bytes(
    device, dtype, weight_only, steps
):
    W, X, metric = inputs(device, dtype=dtype)
    args = preset_name_to_scheme("NVFP4", []).weights.model_copy(
        update={
            "observer": "osfp4",
            "observer_kwargs": {"num_iters": steps, "lr": 0.03},
        },
        deep=True,
    )

    def execute(module):
        instance = module.OSFP4Observer(base_name="weight", args=args)
        keyword = (
            "activation_quantization_groups" if module is old_observer else "X_blocks"
        )
        optimized = instance.optimize_quantization_group_scales(
            W,
            **{keyword: None if weight_only else X},
            weight_metric=metric,
            alpha_dtype=dtype,
        )
        return optimized, instance.select_weight_qparams(
            W, optimized, weight_metric=metric
        )

    assert_bytes(execute(observer), execute(old_observer))


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan"), float("inf")])
def test_observer_selection_errors_match(invalid):
    errors = []
    for module in (observer, old_observer):
        args = preset_name_to_scheme("NVFP4", []).weights
        instance = module.OSFP4Observer(base_name="weight", args=args)
        optimized = module.OptimizedQuantizationGroupScales(
            torch.ones(1, 16), torch.full((1, 2, 1), invalid)
        )
        with pytest.raises(ValueError) as error:
            instance.select_weight_qparams(
                torch.ones(1, 2, 16), optimized, weight_metric=torch.ones(1, 16)
            )
        errors.append(str(error.value))
    assert errors[0] == errors[1]
