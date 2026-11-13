"""D5 T1 uses the first emitted D output, preserving internal TTFT."""

import csv
import json
from types import SimpleNamespace

import pytest

from frontier.config import global_vars
from frontier.config.metrics_config import MetricsConfig
from frontier.entities import Request
from frontier.metrics.constants import RequestMetricsTimeDistributions
from frontier.metrics.metrics_store import MetricsStore
from frontier.types import ClusterType


def _pd_request(num_decode_tokens=3):
    request = Request(1.0, 4, num_decode_tokens)
    request.on_batch_schedule(2.0, ClusterType.PREFILL)
    request.on_batch_end(3.0, 4, ClusterType.PREFILL)
    request.on_kv_cache_transfer_start(3.5)
    request.on_kv_cache_transfer_complete(5.0, 1.5)
    return request


@pytest.mark.parametrize("mode", ["offline", "online"])
@pytest.mark.parametrize("num_decode_tokens", [1, 3])
def test_pd_client_ttft_ends_at_first_d_output(monkeypatch, mode, num_decode_tokens):
    monkeypatch.setattr(global_vars, "SIMULATION_MODE", mode)
    request = _pd_request(num_decode_tokens)
    assert request.ttft == 2.0
    assert request.client_ttft == 0.0

    request.on_batch_schedule(6.0, ClusterType.DECODE)
    request.on_batch_end(7.0, 1, ClusterType.DECODE)

    assert request.client_ttft == 6.0
    assert request.first_decode_token_completed_at == 7.0
    # This historical field is the second output, not the PD first output.
    assert request.decode_first_token_completed_at == 0.0
    assert request.ttft == 2.0
    assert request.prefill_completed_at == 3.0
    assert request.kv_cache_transfer_end_time < request.first_decode_token_completed_at
    if num_decode_tokens > 1:
        request.on_batch_end(8.0, 1, ClusterType.DECODE)
        assert request.decode_first_token_completed_at == 8.0
        assert request.client_ttft == 6.0


def test_colocated_first_output_and_historical_ttft_are_unchanged():
    request = Request(1.0, 4, 3)
    assert request.ttft == request.client_ttft == 0.0
    request.on_batch_schedule(2.0, ClusterType.MONOLITHIC)
    request.on_batch_end(3.0, 4, ClusterType.MONOLITHIC)
    assert request.ttft == request.client_ttft == 2.0
    assert request.first_decode_token_completed_at == request.prefill_completed_at == 3.0
    request.on_batch_end(4.0, 1, ClusterType.MONOLITHIC)
    assert request.ttft == request.client_ttft == 2.0
    assert request.decode_first_token_completed_at == 4.0


@pytest.mark.parametrize("sys_arch", ["pd-disaggregation", "co-location", "pd-af-disaggregation"])
def test_export_preserves_legacy_ttft_and_adds_pd_client_interval_only(tmp_path, sys_arch):
    config = MetricsConfig(
        output_dir=str(tmp_path), store_plots=False,
        enable_metrics_ground_truth_trace=True,
    )
    store = MetricsStore(SimpleNamespace(metrics_config=config, sys_arch=sys_arch), {})
    if sys_arch == "co-location":
        request = Request(1.0, 4, 3)
        request.on_batch_schedule(2.0, ClusterType.MONOLITHIC)
        request.on_batch_end(3.0, 4, ClusterType.MONOLITHIC)
        cluster_type = ClusterType.MONOLITHIC
    else:
        request = _pd_request()
        cluster_type = ClusterType.DECODE
        request.on_batch_schedule(6.0, cluster_type)
        request.on_batch_end(7.0, 1, cluster_type)
    request.on_batch_end(8.0, 1, cluster_type)
    request.on_batch_end(9.0, 1, cluster_type)
    store._on_request_end(9.0, request)
    # Completion reporting remains idempotent in disaggregated execution.
    store._on_request_end(9.0, request)
    store._store_request_metrics(str(tmp_path))

    with (tmp_path / "request_metrics.csv").open() as stream:
        [row] = list(csv.DictReader(stream))
    [truth] = [json.loads(line) for line in (tmp_path / "metrics_ground_truth.jsonl").read_text().splitlines()]
    assert float(row["ttft"]) == truth["ttft_ms"] == 2000.0
    assert truth["ttft_s"] == 2.0
    if sys_arch == "pd-disaggregation":
        assert float(row["client_ttft"]) == truth["client_ttft_ms"] == 6000.0
        assert truth["client_ttft_s"] == 6.0
        assert store._request_metrics_time_distributions[
            RequestMetricsTimeDistributions.CLIENT_TTFT
        ]._data_series == [(request.id, 6.0)]
    else:
        assert "client_ttft" not in row
        assert "client_ttft_ms" not in truth
        assert "client_ttft_s" not in truth
        assert RequestMetricsTimeDistributions.CLIENT_TTFT not in store._request_metrics_time_distributions


def test_real_pd_handoff_cannot_admit_d_before_transfer_completion(monkeypatch):
    from tests.unit.test_step5_structural_pd import (
        _Metrics, _harness, _request_owned_by_prefill, _start_event,
    )

    for name, value in (
        ("SYS_ARCH", "pd-disaggregation"), ("SIMULATION_MODE", "offline"),
        ("USE_CUDA_GRAPH", False), ("DECODE_CUDA_GRAPH_MODE", "none"),
        ("CUDAGRAPH_CAPTURE_SIZES", None), ("KV_CONNECTOR", "nixl"),
        ("IS_MOE", None), ("_IS_MOE_INITIALIZED", False),
        ("QUANTIZATION_MANAGER", None),
    ):
        monkeypatch.setattr(global_vars, name, value)
    source, target, scheduler, _ = _harness()
    request = _request_owned_by_prefill(source, 17)
    [end_event] = _start_event(request, source).handle_event(scheduler, _Metrics())

    assert request not in target._waiting_requests
    assert request.first_decode_token_completed_at == 0.0
    assert request.client_ttft == 0.0
    _, scheduled, _ = target._schedule_decode_waiting_requests(512)
    assert scheduled == []

    end_event.handle_event(scheduler, _Metrics())
    _, scheduled, _ = target._schedule_decode_waiting_requests(512)
    assert scheduled == [request]
    request.on_batch_schedule(end_event.time, ClusterType.DECODE)
    request.on_batch_end(end_event.time + 0.25, 1, ClusterType.DECODE)
    assert (
        request.prefill_completed_at <= request.kv_cache_transfer_start_time
        <= request.kv_cache_transfer_end_time == end_event.time
        < request.first_decode_token_completed_at
    )
    assert request.client_ttft == end_event.time + 0.25 - request.arrived_at
