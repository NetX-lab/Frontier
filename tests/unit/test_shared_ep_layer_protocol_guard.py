from __future__ import annotations

from types import SimpleNamespace

from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import (
    RoundRobinClusterScheduler,
)
from frontier.types import ClusterType


class _MixedModelConfig:
    is_moe = True

    def is_moe_layer(self, layer_id: int) -> bool:
        return layer_id == 2


def _scheduler(cluster_type: ClusterType):
    scheduler = object.__new__(RoundRobinClusterScheduler)
    scheduler._cluster_type = cluster_type
    scheduler._config = SimpleNamespace(
        replica_config=SimpleNamespace(
            model_config=_MixedModelConfig(),
            attn_data_parallel_size=1,
        )
    )
    scheduler._predictor = SimpleNamespace(
        _prefill_routing_details={0: {2: {0: 1.0}}},
        _decode_routing_details={0: {2: {0: 1.0}}},
        _monolithic_routing_details={0: {2: {0: 1.0}}},
    )
    return scheduler


def test_monolithic_prefill_guard_only_admits_moe_layers() -> None:
    scheduler = _scheduler(ClusterType.MONOLITHIC)

    assert scheduler._uses_shared_prefill_ep_wave(None, 2) is True
    assert scheduler._uses_shared_prefill_ep_wave(None, 1) is False


def test_monolithic_decode_guard_only_admits_moe_layers() -> None:
    scheduler = _scheduler(ClusterType.MONOLITHIC)

    assert scheduler._uses_shared_decode_ep_wave(None, 2) is True
    assert scheduler._uses_shared_decode_ep_wave(None, 1) is False
