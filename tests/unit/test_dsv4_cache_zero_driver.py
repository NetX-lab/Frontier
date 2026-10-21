"""Structural evidence validation rejects missing or misattributed cache work."""

from copy import deepcopy
from pathlib import Path
import runpy

import pytest


@pytest.fixture
def driver():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] /
                              "integration/run_dsv4_structural.py"))


def _records():
    identity = dict(cluster_type="MONOLITHIC", replica_id=0, replica_local_id=1)
    # A repeated assignment is preserved; unique reservation extent counts it once.
    decisions = [
        dict(event="grouped_cache_allocation", iteration_id=7, **identity,
             group_new_block_ids={"a": [3], "b": [7]}, zero_required_block_ids=[3, 7]),
        dict(event="grouped_cache_allocation", iteration_id=7, **identity,
             group_new_block_ids={"a": [3], "b": []}, zero_required_block_ids=[3]),
        dict(event="grouped_cache_batch", iteration_id=7, **identity, batch_id=10,
             zero_required_block_ids=[3, 3, 7], physical_block_size_bytes=1002240),
        dict(event="grouped_cache_allocation", iteration_id=8, **identity,
             group_new_block_ids={"a": [], "b": []}, zero_required_block_ids=[]),
        dict(event="grouped_cache_batch", iteration_id=8, **identity, batch_id=11,
             zero_required_block_ids=[], physical_block_size_bytes=1002240),
    ]
    stages = [dict(batch_id=batch, **identity) for batch in (10, 11)]
    traces = [dict(name="kv_cache_block_zero", batch_id=10, layer_id=0,
                   ts_start=1.0, duration_ms=0.01, type="COMPUTE", meta={
        "cost_source": "one synthetic dummy duration; native costs unqualified",
        "source_tensor_calls": [{
            "block_ids": [3, 3, 7], "block_assignment_count": 3, "unique_block_count": 2,
            "physical_block_size_bytes": 1002240, "physical_reservation_extent_bytes": 2004480,
            "native_store_traffic_bytes": "UNKNOWN", "native_h2d_cost": "UNKNOWN",
        }]}), dict(name="dsv4_embedding", batch_id=10, layer_id=0,
                   ts_start=1.00001, duration_ms=0.01, type="COMPUTE", meta={})]
    return decisions, stages, traces


def _validate(driver, records):
    return driver["validate_cache_zeroing"](*records, group_ids=("a", "b"),
                                           physical_block_size_bytes=1002240,
                                           dummy_execution_time_ms=0.01)


def test_structural_driver_matches_raw_assignment_order_and_unique_extent(driver):
    result = _validate(driver, _records())
    assert result["cache_zero_invocations"] == 1
    assert result["cache_zero_block_assignments"] == 3
    assert result["cache_zero_reservation_extent_bytes"] == 2004480


def test_structural_driver_uses_capacity_order_independently_of_json_object_order(driver):
    records = _records()
    for row in records[0]:
        if row["event"] == "grouped_cache_allocation":
            row["group_new_block_ids"] = dict(reversed(tuple(row["group_new_block_ids"].items())))
    assert _validate(driver, records)["cache_zero_block_assignments"] == 3


def test_structural_driver_accepts_existing_aggregated_layer_identity(driver):
    records = _records()
    zero = records[2][0]
    zero["layer_id"] = -1
    zero["meta"]["global_layer_ids"] = [0]
    zero["meta"]["source_tensor_calls"][0]["global_layer_id"] = 0
    assert _validate(driver, records)["cache_zero_invocations"] == 1


@pytest.mark.parametrize("mutation", [
    "missing_zero", "duplicate_zero", "reordered_ids", "null_id", "wrong_extent",
    "wrong_stride", "layer_scaled", "late_zero", "idle_zero", "later_step_zero",
    "wrong_lane", "missing_handoff", "wrong_iteration", "phase_zero", "claimed_native",
    "wrong_layer",
])
def test_structural_driver_rejects_corrupted_cache_evidence(driver, mutation):
    decisions, stages, traces = deepcopy(_records())
    zero = traces[0]
    call = zero["meta"]["source_tensor_calls"][0]
    if mutation == "missing_zero":
        traces.pop(0)
    elif mutation == "duplicate_zero":
        traces.append(deepcopy(zero))
    elif mutation == "reordered_ids":
        decisions[2]["zero_required_block_ids"] = [3, 7, 3]
    elif mutation == "null_id":
        decisions[0]["group_new_block_ids"]["a"] = [0]
        decisions[0]["zero_required_block_ids"] = [0, 7]
    elif mutation == "wrong_extent":
        call["physical_reservation_extent_bytes"] = 3006720
    elif mutation == "wrong_stride":
        decisions[2]["physical_block_size_bytes"] = 822272
    elif mutation == "layer_scaled":
        zero["duration_ms"] *= 43
    elif mutation == "late_zero":
        zero["ts_start"] = traces[1]["ts_start"]
    elif mutation == "idle_zero":
        zero["batch_id"] = 999
    elif mutation == "later_step_zero":
        traces.append({**deepcopy(zero), "batch_id": 11})
    elif mutation == "wrong_lane":
        stages[0]["replica_local_id"] = 0
    elif mutation == "missing_handoff":
        decisions.pop(2)
    elif mutation == "wrong_iteration":
        decisions[2]["iteration_id"] = 6
    elif mutation == "phase_zero":
        zero["meta"]["phase"] = "pre_dispatch"
    elif mutation == "claimed_native":
        call["native_store_traffic_bytes"] = 2004480
    elif mutation == "wrong_layer":
        zero["meta"]["global_layer_ids"] = [1]
    with pytest.raises(ValueError):
        _validate(driver, (decisions, stages, traces))


def test_structural_source_identity_includes_batch_and_parameter_contracts(driver):
    paths = driver["SOURCE_FILES"]
    assert "frontier/entities/batch.py" in paths
    assert "frontier/utils/param_counter.py" in paths
    identity = driver["source_identity"]()
    assert all(len(digest) == 64 for digest in identity.values())
