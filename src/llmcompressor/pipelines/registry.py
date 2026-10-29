from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from compressed_tensors.registry import RegistryMixin, standardize_lookup_name
from loguru import logger
from torch.utils.data.dataloader import DataLoader

from llmcompressor.modifiers import Modifier

if TYPE_CHECKING:
    from llmcompressor.args.dataset_arguments import DatasetArguments

__all__ = ["CalibrationPipeline"]


class CalibrationPipeline(ABC, RegistryMixin):
    @staticmethod
    @abstractmethod
    def __call__(
        model: torch.nn.Module,
        dataloader: DataLoader,
        dataset_args: "DatasetArguments",
    ):
        raise NotImplementedError()

    @classmethod
    def from_modifiers(
        cls, modifiers: list[Modifier], user: str | None = None
    ) -> "CalibrationPipeline":
        """
        Infer which calibration pipeline to use based on the available modifiers and
        any user specifications

        :param modifiers: modifiers to apply to model
        :param user: pipeline name passed by user
        :return: CalibrationPipeline instance to be called with data (if not datafree)
        """
        user = standardize_lookup_name(user) if user else None
        from llmcompressor.modifiers.osfp4 import OSFP4Modifier

        if any(isinstance(modifier, OSFP4Modifier) for modifier in modifiers):
            if len(modifiers) != 1:
                raise ValueError("OSFP4 must be the only modifier in its recipe")
            if user not in (None, "sequential", "independent"):
                raise ValueError("OSFP4 requires the sequential calibration pipeline")
            # oneshot defaults to independent; OSFP4 needs sequential propagation.
            return cls.load_from_registry("sequential")
        inferred = standardize_lookup_name(cls._infer_pipeline(modifiers))
        independent = standardize_lookup_name("independent")

        if user == independent:
            inferred = independent

        if user is not None and user != inferred:
            logger.warning(
                f"Calibration pipeline is set to `{user}`, but it is recommended to "
                f"use `{inferred}`"
            )

        pipeline = user or inferred
        return cls.load_from_registry(pipeline)

    @staticmethod
    def _infer_pipeline(modifiers: list[Modifier]) -> str:
        if any(modifier.requires_calibration_data for modifier in modifiers):
            return "sequential"
        else:
            return "datafree"
