from typing import TYPE_CHECKING

from frontier.m2n_transfer.analytical_m2n_transfer_predictor import (
    AnalyticalM2NTransferPredictor,
)
from frontier.m2n_transfer.base_m2n_transfer_predictor import BaseM2NTransferPredictor
from frontier.types import M2NTransferType
from frontier.utils.base_registry import BaseRegistry

if TYPE_CHECKING:
    from frontier.config.m2n_transfer_config import BaseM2NTransferConfig


class M2NTransferPredictorRegistry(BaseRegistry):
    """Registry for M2N transfer predictors."""

    @classmethod
    def get_key_from_str(cls, key_str: str) -> M2NTransferType:
        return M2NTransferType.from_str(key_str)

    @classmethod
    def get(
        cls,
        predictor_type: M2NTransferType,
        config: "BaseM2NTransferConfig",
    ) -> BaseM2NTransferPredictor:
        if predictor_type not in cls._registry:
            raise ValueError(f"M2N transfer predictor type {predictor_type} is not registered")
        return cls._registry[predictor_type](config)


M2NTransferPredictorRegistry.register(
    M2NTransferType.ANALYTICAL, AnalyticalM2NTransferPredictor
)
