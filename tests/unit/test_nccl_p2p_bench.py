"""Message sizes and the Frontier mapping of the NCCL point-to-point benchmark.

The benchmark runs only on a two-GPU worker, so a wrong message size or unit
conversion would surface as a mis-set KV transfer model after the run; the
pure pieces that decide them are checked here.
"""

from __future__ import annotations

import pytest

from frontier.config.kv_cache_transfer_config import AnalyticalKVCacheTransferConfig
from frontier.kv_cache_transfer.analytical_kv_cache_transfer_predictor import (
    AnalyticalKVCacheTransferPredictor,
)
from frontier.types import ClusterType
from tests.comparison.calibration.nccl_p2p_bench import (
    fit_latency_bandwidth,
    frontier_transfer_mapping,
    request_messages,
    sweep_sizes,
)

MIB = 1024 * 1024


def test_request_messages_are_one_kv_slice_per_layer():
    assert request_messages("llama2_7b_dense_example", 2048) == (32, 32 * MIB)
    assert request_messages("Qwen3-30B-A3B-tiny", 2048) == (8, 4 * MIB)


def test_sweep_doubles_from_the_minimum_to_the_maximum():
    sizes = sweep_sizes(64 * 1024, 128 * MIB)

    assert sizes[0] == 64 * 1024 and sizes[-1] == 128 * MIB and len(sizes) == 12
    assert all(b == 2 * a for a, b in zip(sizes, sizes[1:]))
    with pytest.raises(ValueError):
        sweep_sizes(3000, 128 * MIB)


def test_fit_recovers_latency_and_bandwidth_in_frontier_units():
    # 200 Gbps is 25e9 bytes/s, i.e. 25e6 bytes per millisecond.
    ms_per_byte = 1 / 25e6
    points = [(size, 0.02 + size * ms_per_byte) for size in sweep_sizes(65536, 128 * MIB)]

    latency_ms, fitted = fit_latency_bandwidth(points)

    assert latency_ms == pytest.approx(0.02)
    assert fitted == pytest.approx(ms_per_byte)
    request_bytes = 32 * 32 * MIB
    mapping = frontier_transfer_mapping(request_bytes, 0.5 + request_bytes * fitted, fitted)
    assert mapping["network_bandwidth_gbps"] == pytest.approx(200.0)
    assert mapping["network_latency_ms"] == pytest.approx(0.5)
    # Frontier's analytical transfer model reproduces the measured request time.
    predictor = AnalyticalKVCacheTransferPredictor(AnalyticalKVCacheTransferConfig(**mapping))
    predicted_ms = predictor.get_transfer_time(
        ClusterType.PREFILL, ClusterType.DECODE, None, request_bytes
    )
    assert predicted_ms == pytest.approx(0.5 + request_bytes * fitted)
