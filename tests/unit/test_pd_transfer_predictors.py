"""pd-disaggregation transfer predictor contract tests."""

from types import SimpleNamespace

import pytest

from frontier.config.kv_cache_transfer_config import AnalyticalKVCacheTransferConfig
from frontier.config.m2n_transfer_config import AnalyticalM2NTransferConfig
from frontier.types import ClusterType


class _ModelConfig:
    num_layers = 2
    num_kv_heads = 4
    num_q_heads = 8
    embedding_dim = 64
    is_moe = False
    use_mla = False

    def get_head_dim(self) -> int:
        return 8

    def get_runtime_num_kv_heads(self) -> int:
        return self.num_kv_heads

    def get_runtime_head_size(self) -> int:
        return self.get_head_dim()


class _MLAModelConfig:
    """DeepSeek-V2 style latent-MLA model config stub.

    Dense-blind transfer sizing would use ``num_kv_heads * get_head_dim()``; the
    family-aware path must instead use the runtime latent layout
    (``get_runtime_num_kv_heads() == 1``, ``get_runtime_head_size() == 576``,
    ``kv_factor == 1``).
    """

    num_layers = 2
    num_kv_heads = 128
    num_q_heads = 128
    embedding_dim = 24576
    is_moe = False
    use_mla = True
    kv_lora_rank = 512
    qk_nope_head_dim = 128
    qk_rope_head_dim = 64
    qk_head_dim = 192
    v_head_dim = 128

    def get_head_dim(self) -> int:
        return 192

    def get_runtime_num_kv_heads(self) -> int:
        return 1

    def get_runtime_head_size(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim


class _ReplicaConfig:
    model_config = _ModelConfig()


class _MLAReplicaConfig:
    model_config = _MLAModelConfig()


class _Batch:
    def __init__(self) -> None:
        self.requests = [
            SimpleNamespace(num_prefill_tokens=3),
            SimpleNamespace(num_prefill_tokens=5),
        ]

    def get_effective_total_tokens_for_transfer(self, _source_cluster_type: ClusterType) -> int:
        return 6


def test_kv_cache_analytical_predictor_computes_size_time_and_registry_lookup() -> None:
    from frontier.kv_cache_transfer import (
        AnalyticalKVCacheTransferPredictor,
        KVCacheTransferPredictorRegistry,
    )
    from frontier.types import KVCacheTransferType

    config = AnalyticalKVCacheTransferConfig(
        network_bandwidth_gbps=80.0,
        network_latency_ms=0.25,
        kv_cache_dtype_size_bytes=2,
        enable_compression=True,
        compression_ratio=2.0,
    )
    predictor = AnalyticalKVCacheTransferPredictor(config)
    batch = _Batch()
    replica_config = _ReplicaConfig()

    kv_size = predictor.get_kv_cache_size(batch, replica_config)
    expected_size = 8 * 2 * 4 * 8 * 2 * 2
    assert kv_size == expected_size

    request_size = predictor.get_kv_cache_size_for_request(batch.requests[0], replica_config)
    assert request_size == 3 * 2 * 4 * 8 * 2 * 2

    transfer_ms = predictor.get_transfer_time(
        ClusterType.PREFILL,
        ClusterType.DECODE,
        batch,
        kv_size,
    )
    expected_effective_bytes = expected_size / 2.0
    expected_bandwidth_bytes_per_ms = (80.0 * 1e9) / (8 * 1000)
    expected_transfer_ms = 0.25 + expected_effective_bytes / expected_bandwidth_bytes_per_ms
    assert transfer_ms == pytest.approx(expected_transfer_ms)

    assert predictor.supports_latency_hiding() is False
    assert KVCacheTransferPredictorRegistry.get_key_from_str("analytical") is KVCacheTransferType.ANALYTICAL
    assert isinstance(
        KVCacheTransferPredictorRegistry.get(KVCacheTransferType.ANALYTICAL, config),
        AnalyticalKVCacheTransferPredictor,
    )


def test_kv_cache_analytical_predictor_sizes_latent_mla_cache() -> None:
    """MLA transfer sizing must follow the latent runtime layout, not dense heads.

    Dense-blind sizing would yield ``num_kv_heads(128) * get_head_dim(192) * 2``
    per token-layer; the family-aware path must instead use the latent layout
    (runtime kv heads = 1, runtime head size = 576, kv_factor = 1).
    """
    from frontier.kv_cache_transfer import AnalyticalKVCacheTransferPredictor

    config = AnalyticalKVCacheTransferConfig(
        network_bandwidth_gbps=80.0,
        network_latency_ms=0.25,
        kv_cache_dtype_size_bytes=2,
    )
    predictor = AnalyticalKVCacheTransferPredictor(config)
    batch = _Batch()
    replica_config = _MLAReplicaConfig()

    # total_tokens=8, num_layers=2, runtime_kv_heads=1, runtime_head_size=576,
    # kv_factor=1, dtype_size=2 -> 8 * 2 * 1 * 576 * 1 * 2 = 18432
    kv_size = predictor.get_kv_cache_size(batch, replica_config)
    assert kv_size == 8 * 2 * 1 * 576 * 1 * 2

    request_size = predictor.get_kv_cache_size_for_request(batch.requests[0], replica_config)
    assert request_size == 3 * 2 * 1 * 576 * 1 * 2


def test_m2n_analytical_predictor_computes_size_time_and_registry_lookup() -> None:
    from frontier.m2n_transfer import AnalyticalM2NTransferPredictor, M2NTransferPredictorRegistry
    from frontier.types import M2NTransferType

    config = AnalyticalM2NTransferConfig(
        memory_bandwidth_gbps=160.0,
        network_latency_ms=0.12,
        activation_dtype_size_bytes=2,
        enable_compression=True,
        compression_ratio=2.0,
        enable_p2p_optimization=True,
    )
    predictor = AnalyticalM2NTransferPredictor(config)
    batch = _Batch()
    replica_config = _ReplicaConfig()

    activation_size = predictor.get_activation_size(batch, replica_config, ClusterType.DECODE_ATTN)
    expected_size = 6 * 64 * 2
    assert activation_size == expected_size

    request_size = predictor.get_activation_size_for_request(
        SimpleNamespace(), replica_config, ClusterType.DECODE_FFN
    )
    assert request_size == 1 * 64 * 2

    transfer_ms = predictor.get_transfer_time(
        ClusterType.DECODE_ATTN,
        ClusterType.DECODE_FFN,
        batch,
        activation_size,
    )
    expected_effective_bytes = int(expected_size / 2.0)
    expected_bandwidth_bytes_per_ms = 160.0 * 125_000
    expected_transfer_ms = (0.12 + expected_effective_bytes / expected_bandwidth_bytes_per_ms) / 1.2
    assert transfer_ms == pytest.approx(expected_transfer_ms)

    assert M2NTransferPredictorRegistry.get_key_from_str("analytical") is M2NTransferType.ANALYTICAL
    assert isinstance(
        M2NTransferPredictorRegistry.get(M2NTransferType.ANALYTICAL, config),
        AnalyticalM2NTransferPredictor,
    )


def test_m2n_analytical_predictor_rejects_invalid_transfer_pairs() -> None:
    from frontier.m2n_transfer import AnalyticalM2NTransferPredictor

    predictor = AnalyticalM2NTransferPredictor(AnalyticalM2NTransferConfig())

    with pytest.raises(ValueError, match="DECODE_ATTN <-> DECODE_FFN"):
        predictor.get_transfer_time(
            ClusterType.PREFILL,
            ClusterType.DECODE,
            _Batch(),
            activation_size_bytes=64,
        )


def test_transfer_predictors_handle_zero_token_and_invalid_source_boundaries() -> None:
    from frontier.kv_cache_transfer import AnalyticalKVCacheTransferPredictor
    from frontier.m2n_transfer import AnalyticalM2NTransferPredictor

    kv_predictor = AnalyticalKVCacheTransferPredictor(
        AnalyticalKVCacheTransferConfig(network_bandwidth_gbps=100.0, network_latency_ms=0.1)
    )
    replica_config = _ReplicaConfig()
    zero_request = SimpleNamespace(num_prefill_tokens=0)
    assert kv_predictor.get_kv_cache_size_for_request(zero_request, replica_config) == 0
    assert kv_predictor.get_transfer_time(
        ClusterType.PREFILL,
        ClusterType.DECODE,
        None,
        kv_cache_size_bytes=0,
    ) == pytest.approx(0.1)

    m2n_predictor = AnalyticalM2NTransferPredictor(AnalyticalM2NTransferConfig())
    with pytest.raises(ValueError, match="Invalid source cluster type"):
        m2n_predictor.get_activation_size(_Batch(), replica_config, ClusterType.PREFILL)
