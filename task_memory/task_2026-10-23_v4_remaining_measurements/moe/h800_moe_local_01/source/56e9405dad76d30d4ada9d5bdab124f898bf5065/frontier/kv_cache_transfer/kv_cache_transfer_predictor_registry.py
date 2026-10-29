from typing import TYPE_CHECKING

from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import (
    AnalyticalKVCacheTransferPredictor,
)
from frontier.kv_cache_transfer.base_kv_cache_transfer_predictor import (
    BaseKVCacheTransferPredictor,
)
from frontier.types import KVCacheTransferType
from frontier.utils.base_registry import BaseRegistry

if TYPE_CHECKING:
    from frontier.config.kv_cache_transfer_config import BaseKVCacheTransferConfig


class KVCacheTransferPredictorRegistry(BaseRegistry):
    """Registry for KV cache transfer predictors."""

    @classmethod
    def get_key_from_str(cls, key_str: str) -> KVCacheTransferType:
        return KVCacheTransferType.from_str(key_str)

    @classmethod
    def get(
        cls,
        predictor_type: KVCacheTransferType,
        config: "BaseKVCacheTransferConfig",
    ) -> BaseKVCacheTransferPredictor:
        if predictor_type not in cls._registry:
            raise ValueError(
                f"KV cache transfer predictor type {predictor_type} is not registered"
            )
        return cls._registry[predictor_type](config)


KVCacheTransferPredictorRegistry.register(
    KVCacheTransferType.ANALYTICAL, AnalyticalKVCacheTransferPredictor
)
