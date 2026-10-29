"""File-backed transfer inputs; all timings here are CPU contract fixtures."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from frontier.attention.memory import get_attention_kv_transfer_payload
from frontier.entities import Request
from frontier.entities.batch import DummyForwardBatch
from frontier.kv_cache_transfer.nixl_read_cost import NixlReadCost
from frontier.model_architectures import get_model_architecture_profile
from frontier.types import ClusterType
from tests.unit.test_step5_measured_pd import _measured_cluster
from tests.unit.test_step5_structural_pd import (
    _GlobalScheduler, _Metrics, _start_event, capacity, simulation_globals, step_config,
)


class _ExecutionCosts(NixlReadCost):
    get_nixl_transfer_coverage = NixlReadCost.get_coverage
    get_nixl_transfer_time_ms = NixlReadCost.get_time_ms


@pytest.fixture
def document(step_config, capacity):
    payload = get_attention_kv_transfer_payload(
        step_config.model_config, capacity, prompt_tokens=6128,
        tensor_parallel_size=8, target_reserved_blocks=4096,
    )
    return {
        "schema": "nixl_read_cost.v1",
        "model_architecture_profile": get_model_architecture_profile(step_config.model_config).profile_id,
        "profile_identity": "CPU_CONTRACT_FIXTURE_NOT_NATIVE",
        "source_device": "h800", "target_device": "h800",
        "source_attention_tp": 8, "target_attention_tp": 8,
        "duration_scope": "read_to_gpu_ready",
        "cache_capacity": asdict(capacity),
        "measurements": [{
            "num_prefill_tokens": 6144, "num_computed_tokens": 6128,
            "transfer_bytes_per_rank": payload.transfer_bytes_per_rank,
            "time_ms": 3.5,
        }],
    }


def _write(tmp_path, document):
    path = tmp_path / "read_cost.json"
    path.write_text(json.dumps(document))
    return path


def _coverage(costs, step_config, capacity, *, prompt=6144):
    return costs.get_coverage(
        request=Request(0, prompt, 1024), source_replica_config=step_config,
        target_replica_config=step_config, cache_capacity=capacity,
        target_cost_provider=costs,
    )


@pytest.mark.parametrize("mode", ["offline", "online"])
def test_file_backed_cost_runs_fixed_request_pd_lifecycle(tmp_path, document, monkeypatch, mode):
    from frontier.config import global_vars

    monkeypatch.setattr(global_vars, "SIMULATION_MODE", mode)
    costs = _ExecutionCosts.from_file(_write(tmp_path, document))
    source_cluster = _measured_cluster(ClusterType.PREFILL, costs)
    target_cluster = _measured_cluster(ClusterType.DECODE, costs)
    scheduler = _GlobalScheduler(source_cluster, target_cluster)
    source = source_cluster.get_replica_scheduler(source_cluster._serving_replica_id, 0)
    request = Request(0, 6144, 1024)
    source.add_request(request)
    [batch] = source.on_schedule(0.1)
    batch.on_batch_end(0.2, ClusterType.PREFILL)
    source.on_batch_end(batch)
    metrics = _Metrics()
    [end] = _start_event(request, source).handle_event(scheduler, metrics)
    info = end._transfer_info
    target = target_cluster.get_replica_scheduler(info.target_replica_id, info.target_replica_local_id)
    assert info.transfer_time_ms == 3.5
    assert info.kv_cache_size_bytes == document["measurements"][0]["transfer_bytes_per_rank"]
    assert target.on_schedule(end.time - 0.001) == []
    end.handle_event(scheduler, metrics)

    def advance_idle_lanes(now):
        for lane in target._dp_group():
            if lane is target:
                continue
            for idle_batch in lane.on_schedule(now):
                assert isinstance(idle_batch, DummyForwardBatch)
                lane.on_dummy_forward_end(now + 0.005)

    for output_index in range(1024):
        now = end.time + 0.01 * (output_index + 1)
        advance_idle_lanes(now)
        batches = target.on_schedule(now)
        if not batches:
            assert target._waits_at_dp_sync
            advance_idle_lanes(now)
            batches = target.on_schedule(now)
        [batch] = batches
        assert batch.num_tokens == [16 if output_index == 0 else 1]
        assert batch.num_context_tokens == [6128 if output_index == 0 else 6143 + output_index]
        batch.on_batch_end(now + 0.005, ClusterType.DECODE)
        target.on_batch_end(batch)
        target_cluster.on_replica_batch_end(now + 0.005, info.target_replica_id, info.target_replica_local_id, batch)
    assert request.completed and request.num_emitted_decode_tokens == 1024
    assert request.num_processed_tokens == 7168
    assert source._allocation_map == target._allocation_map == {}
    assert len(metrics.starts) == len(metrics.ends) == 1


@pytest.mark.parametrize("field,value", [
    ("time_ms", 0), ("time_ms", -1), ("time_ms", float("nan")),
    ("time_ms", float("inf")), ("time_ms", True), ("time_ms", "3.5"),
    ("transfer_bytes_per_rank", 0), ("transfer_bytes_per_rank", 1.5),
    ("num_computed_tokens", 6144), ("num_computed_tokens", 6129),
    ("num_computed_tokens", True),
])
def test_bad_measurement_is_rejected_at_load(tmp_path, document, field, value):
    document["measurements"][0][field] = value
    with pytest.raises(ValueError):
        NixlReadCost.from_file(_write(tmp_path, document))


@pytest.mark.parametrize("field,value", [
    ("schema", "unknown"), ("duration_scope", "network_only"),
    ("profile_identity", ""), ("source_device", ""),
    ("source_attention_tp", 0), ("target_attention_tp", True),
    ("measurements", []),
])
def test_bad_header_is_rejected_at_load(tmp_path, document, field, value):
    document[field] = value
    with pytest.raises(ValueError):
        NixlReadCost.from_file(_write(tmp_path, document))


def test_duplicate_and_missing_records_rejected(tmp_path, document):
    document["measurements"].append(dict(document["measurements"][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        NixlReadCost.from_file(_write(tmp_path, document))
    del document["measurements"]
    with pytest.raises(ValueError, match="Incomplete"):
        NixlReadCost.from_file(_write(tmp_path, document))
    with pytest.raises(ValueError, match="Missing measured NIXL profile"):
        NixlReadCost.from_file(None)


@pytest.mark.parametrize("field,value", [
    ("model_architecture_profile", "another_model"),
    ("source_device", "h200"), ("target_device", "h200"),
    ("source_attention_tp", 4), ("target_attention_tp", 4),
])
def test_cross_profile_lookup_rejected(tmp_path, document, step_config, capacity, field, value):
    document[field] = value
    costs = NixlReadCost.from_file(_write(tmp_path, document))
    with pytest.raises(ValueError, match="mismatch"):
        _coverage(costs, step_config, capacity)


def test_layout_payload_and_scope_remain_bound(tmp_path, document, step_config, capacity):
    costs = NixlReadCost.from_file(_write(tmp_path, document))
    coverage = _coverage(costs, step_config, capacity)
    with pytest.raises(ValueError, match="prompt length"):
        _coverage(costs, step_config, capacity, prompt=6145)
    with pytest.raises(ValueError, match="mismatch"):
        _coverage(costs, step_config, replace(capacity, fixed_bytes_per_request_slot=1))
    size = document["measurements"][0]["transfer_bytes_per_rank"]
    with pytest.raises(ValueError, match="payload"):
        costs.get_time_ms(coverage=coverage, payload=SimpleNamespace(transfer_bytes_per_rank=size + 1))
    with pytest.raises(ValueError, match="belong"):
        costs.get_time_ms(coverage=replace(coverage, profile_identity="another"), payload=SimpleNamespace(transfer_bytes_per_rank=size))
    assert costs.get_time_ms(coverage=coverage, payload=SimpleNamespace(transfer_bytes_per_rank=size)) == 3.5
