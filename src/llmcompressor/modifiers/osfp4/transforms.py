# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the OSFP4 and vLLM projects

import torch
from compressed_tensors import TRANSFORM_CONFIG_NAME
from compressed_tensors.transform import (
    TransformArgs,
    TransformConfig,
    TransformFactory,
    TransformScheme,
    apply_transform_config,
)
from torch.nn import Module

TRANSFORM_NAME = "osfp4"
TRANSFORM_MODULE = "osfp4_input"


def initialize_diagonal_smoothing(model: Module, targets: list[str]) -> None:
    """Attach permanent input transforms; OSFP4 balances weights separately.

    Export only the online operation. The checkpoint weights already contain
    inverse balancing, so reconstructing transforms must never divide them again.
    """
    existing = getattr(model, TRANSFORM_CONFIG_NAME, None)
    if existing is not None and existing.config_groups:
        raise ValueError("OSFP4 does not yet compose with existing transforms")
    if not targets:
        return
    if "diagonal" not in TransformFactory.registered_names():
        raise ImportError(
            "OSFP4 requires compressed-tensors with diagonal transform support. "
            "Install the companion compressed-tensors branch; see examples/osfp4."
        )
    config = TransformConfig(
        config_groups={
            TRANSFORM_NAME: TransformScheme(
                type="diagonal",
                precision=torch.bfloat16,
                apply=[TransformArgs(targets=sorted(targets), location="input")],
            )
        }
    )
    apply_transform_config(model, config)
