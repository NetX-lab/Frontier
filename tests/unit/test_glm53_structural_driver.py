"""Fault-injection checks for the manual GLM structural artifact validator."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


DRIVER_PATH = Path(__file__).resolve().parents[1] / "integration" / "run_glm53_structural.py"
SPEC = importlib.util.spec_from_file_location("glm53_structural_driver", DRIVER_PATH)
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


@pytest.fixture
def stages():
    return {
        10: {"replica_id": 0, "stage_id": 0, "replica_local_id": 0,
             "stage_start_ts": 0, "stage_end_ts": 1, "request_num_tokens": [32]},
        2: {"replica_id": 0, "stage_id": 0, "replica_local_id": 1,
            "stage_start_ts": .0002, "stage_end_ts": 1, "request_num_tokens": [32]},
    }


def lane_row(layer=3, ep=0, source=10):
    start = .01 + (layer - 3) * .003
    phases = {}
    phase_start = start
    for name, costs in driver.EP_PHASE_COSTS.items():
        duration = sum(costs.values())
        phases[name] = {"start_time_s": phase_start, "duration_ms": duration, "operators_ms": costs.copy()}
        phase_start += duration / 1000
    return {"record_type": "ep_wave_lane", "cluster": "MONOLITHIC", "replica_id": 0, "stage_id": 0,
            "global_layer_ids": [layer], "attention_family_id": "dsa_attention",
            "attention_variant_id": "full_index" if layer in driver.FULL_INDEX else "shared_index",
            "wave_start_time_s": start, "wave_end_time_s": phase_start, "source_batch_ids": [source],
            "ep_id": ep, "batch_id": 10000 + layer * 64 + ep, "phases": phases}


def cohort_rows(source=10):
    for layer in range(3, 78):
        for ep in range(64):
            yield lane_row(layer, ep, source)


def wave_state(physical=70, cohort=(0, 0, 1)):
    row = lane_row()
    key = (0, 0, 3, row["wave_start_time_s"], row["wave_end_time_s"], 10)
    return {key: {"cohort": cohort, "physical_tokens": physical, "lane_mask": driver.ALL_EP,
                  "batch_ids": [10000 + 3 * 64 + ep for ep in range(64)],
                  "gating_mask": 0, "routed_mask": 0, "routed_cost_mask": driver.ALL_EP, "routed_tokens": 0}}


def token_trace(waves, name="moe_gating_linear", ep=0, tokens=None):
    key, wave = next(iter(waves.items()))
    gating = name == "moe_gating_linear"
    if tokens is None:
        active = [lane for lane in range(64) if wave["routed_cost_mask"] & (1 << lane)]
        tokens = wave["physical_tokens"] if gating else wave["physical_tokens"] * 8 // len(active) + (
            active.index(ep) < wave["physical_tokens"] * 8 % len(active))
    return {"name": name, "cluster": "MONOLITHIC", "replica_id": 0, "layer_id": 3,
            "batch_id": wave["batch_ids"][ep], "duration_ms": .125 if gating else .25,
            "meta": {"source_batch_ids": [10], "global_layer_ids": [3], "stage_id": 0,
                     "wave_start_time_s": key[3], "wave_end_time_s": key[4], "ep_id": ep,
                     "attention_family_id": "dsa_attention", "attention_variant_id": "shared_index",
                     "num_layers": 1, "parallel_context": {"PP": 1, "ATTN_TP": 8, "ATTN_DP": 8,
                                                              "MOE_TP": 1, "MOE_EP": 64, "ROUTER_TOPK": 8},
                     "phase": "pre_dispatch" if gating else "routed_compute",
                     "effective_total_tokens_compute": tokens, "tokens_are_post_routing": not gating,
                     "tensor_shape": {"input": [tokens, 6144], "output": [tokens, 256 if gating else 2048]}}}


def add_token_traces(waves):
    wave = next(iter(waves.values()))
    for ep in range(64):
        for name in ("moe_gating_linear", "moe_grouped_gemm"):
            if name == "moe_grouped_gemm" and not wave["routed_cost_mask"] & (1 << ep):
                continue
            driver.check_ep_trace(token_trace(waves, name, ep), waves)


def test_real_cohort_joins_shared_end_and_nonminimum_representative(monkeypatch, stages, tmp_path):
    monkeypatch.setattr(driver, "rows", lambda _: cohort_rows())
    waves, stats = driver.check_waves(tmp_path, stages, "offline-none")
    assert stats["real_source_cohorts"] == 1
    assert stats["real_source_waves"] == 75
    assert stats["lane_records"] == 75 * 64
    assert {wave["physical_tokens"] for wave in waves.values()} == {70}


def test_source_workload_shape_under_configured_metadata(stages):
    eager, _ = driver.source_cohorts(stages, "offline-none")
    configured, _ = driver.source_cohorts(stages, "online-piecewise")
    assert eager[(0, 0, 1)]["physical_tokens"] == 70
    assert configured[(0, 0, 1)]["physical_tokens"] == 256
    assert any("Graph configuration only" in limit for limit in driver.LIMITS)


@pytest.mark.parametrize("mutation", ["all_dummy", "one_layer_dummy", "missing_lane", "wrong_cohort_bounds"])
def test_dummy_global_coverage_cannot_replace_actual_cohort_coverage(monkeypatch, stages, tmp_path, mutation):
    def mutated():
        for row in cohort_rows():
            if mutation == "all_dummy" or (mutation == "one_layer_dummy" and row["global_layer_ids"] == [7]):
                row["source_batch_ids"] = [999]
            if mutation == "missing_lane" and row["global_layer_ids"] == [7] and row["ep_id"] == 63:
                continue
            if mutation == "wrong_cohort_bounds":
                row["wave_end_time_s"] = 2
            yield row
    monkeypatch.setattr(driver, "rows", lambda _: mutated())
    with pytest.raises(ValueError, match="cohort|participants"):
        driver.check_waves(tmp_path, stages, "offline-none")


def test_ep_operator_omission_with_preserved_phase_total_rejected(monkeypatch, stages, tmp_path):
    row = lane_row()
    costs = row["phases"]["pre_dispatch"]["operators_ms"]
    costs["moe_shuffling"] += costs.pop("moe_gating_linear")
    monkeypatch.setattr(driver, "rows", lambda _: iter([row]))
    with pytest.raises(ValueError, match="EP phase scope keys or costs"):
        driver.check_waves(tmp_path, stages, "offline-none")


def test_empty_routed_lanes_keep_fixed_scopes_and_shared_barriers(monkeypatch, stages, tmp_path):
    for stage in stages.values():
        stage["request_num_tokens"] = [1]
    def with_empty_lanes():
        for row in cohort_rows():
            if row["ep_id"] >= 16:
                pre = row["phases"]["pre_dispatch"]
                pre["operators_ms"]["moe_shuffling"] = 0
                pre["duration_ms"] -= .25
                routed = row["phases"]["routed_compute"]
                routed["operators_ms"]["moe_grouped_gemm"] = routed["duration_ms"] = 0
            yield row
    monkeypatch.setattr(driver, "rows", lambda _: with_empty_lanes())
    waves, _ = driver.check_waves(tmp_path, stages, "offline-none")
    assert {wave["routed_cost_mask"] for wave in waves.values()} == {(1 << 16) - 1}
    one_wave = {next(iter(waves)): next(iter(waves.values()))}
    add_token_traces(one_wave)
    assert driver.check_wave_tokens(one_wave, stages)["real_cohort_routed_assignments"] == 64
    row = token_trace(one_wave, "moe_grouped_gemm", ep=16, tokens=1)
    with pytest.raises(ValueError, match="empty EP lane"):
        driver.check_ep_trace(row, one_wave)


@pytest.mark.parametrize("mutation", ["omit_zero", "move_outer_cost", "cpu_model_total", "scope"])
def test_stage_scope_mutations_rejected(mutation):
    row = {"execution_scope": "ATTN_DP_LANE", "execution_time": {
        "component_ledger_ms": driver.STAGE_COSTS.copy(), "total_time_ms": 128.25,
        "model_time_ms": 127, "diagnostic_total_time_ms": 128.25,
        "diagnostic_component_ledger_ms": {"pp_stage_boundary_handoff_time": 0}}}
    driver.check_stage_costs(row)
    if mutation == "omit_zero":
        row["execution_time"]["component_ledger_ms"].pop("attn_mla_v_up_proj_time")
    elif mutation == "move_outer_cost":
        costs = row["execution_time"]["component_ledger_ms"]
        costs["attention_pre_proj_time"] -= .25
        costs["attention_prefill_execution_time"] += .25
    elif mutation == "cpu_model_total":
        row["execution_time"]["model_time_ms"] += .25
    else:
        row["execution_scope"] = "FULL_MODEL"
    with pytest.raises(ValueError, match="scope|ownership"):
        driver.check_stage_costs(row)


def test_ep_token_conservation_and_missing_trace(stages):
    waves = wave_state()
    add_token_traces(waves)
    stats = driver.check_wave_tokens(waves, stages)
    assert stats["real_cohort_routed_assignments"] == 70 * 8
    next(iter(waves.values()))["routed_tokens"] -= 1
    with pytest.raises(ValueError, match="conservation"):
        driver.check_wave_tokens(waves, stages)
    next(iter(waves.values()))["gating_mask"] &= ~1
    with pytest.raises(ValueError, match="missing EP"):
        driver.check_wave_tokens(waves, stages)


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_source", "wrong_bound", "wrong_lane_batch",
                                    "post_routing", "wrong_tensor_shape", "wrong_width", "wrong_display_layer"])
def test_token_trace_join_and_shape_mutations_rejected(mutation):
    waves = wave_state()
    row = token_trace(waves)
    if mutation == "duplicate":
        driver.check_ep_trace(row, waves)
    elif mutation == "wrong_source":
        row["meta"]["source_batch_ids"] = [2]
    elif mutation == "wrong_bound":
        row["meta"]["wave_start_time_s"] += .001
    elif mutation == "wrong_lane_batch":
        row["batch_id"] += 1
    elif mutation == "post_routing":
        row["meta"]["tokens_are_post_routing"] = True
    elif mutation == "wrong_tensor_shape":
        row["meta"]["tensor_shape"]["input"][0] += 1
    elif mutation == "wrong_display_layer":
        row["layer_id"] = 4
    else:
        row = token_trace(waves, tokens=8)
    with pytest.raises(ValueError):
        driver.check_ep_trace(row, waves)


def test_ep_trace_canonical_layer_survives_display_expansion_quota():
    waves = wave_state()
    row = token_trace(waves)
    row["layer_id"] = -1
    driver.check_ep_trace(row, waves)
    assert next(iter(waves.values()))["gating_mask"] == 1


def test_nonrequest_wave_requires_dummy_shape_and_is_counted_inside_real_span(stages):
    waves = wave_state(physical=8, cohort=None)
    add_token_traces(waves)
    stats = driver.check_wave_tokens(waves, stages)
    assert stats["real_cohort_routed_assignments"] == 0
    assert stats["validated_dummy_shaped_waves"] == 1
    assert stats["dummy_shaped_wave_temporal_counts"]["within_real_cohort_span"] == 1
    unknown = wave_state(physical=70, cohort=None)
    add_token_traces(unknown)
    with pytest.raises(ValueError, match="unattributed"):
        driver.check_wave_tokens(unknown, stages)


@pytest.mark.parametrize("field", ["head", "dirty_tracked_diff_sha256", "untracked_runtime_file_sha256", "driver_sha256"])
def test_source_identity_changes_and_missing_fields_reject_pass(field):
    before = {"head": "a" * 40, "dirty_tracked_diff_sha256": "b" * 64,
              "untracked_runtime_file_sha256": {"frontier/new_op.py": "c" * 64}, "driver_sha256": "d" * 64}
    assert driver.check_source_identity(before, deepcopy(before))["stable"]
    after = deepcopy(before)
    after[field] = {} if field == "untracked_runtime_file_sha256" else "e" * len(after[field])
    with pytest.raises(ValueError, match="source changed"):
        driver.check_source_identity(before, after)
    del after[field]
    with pytest.raises(ValueError, match="missing source identity"):
        driver.check_source_identity(before, after)


def test_validate_source_identity_is_required_before_artifact_admission(tmp_path):
    with pytest.raises(ValueError, match="missing source identity"):
        driver.validate(tmp_path, "offline-none", tmp_path / "run.log", source_before={}, source_after={})


def test_positive_dense_trace_cannot_include_inactive_moe_layers(tmp_path, stages):
    row = {"name": "mlp_up_proj", "batch_id": 10, "layer_id": -1, "duration_ms": .75,
           "meta": {"attention_family_id": "dsa_attention", "attention_variant_id": "full_index",
                    "global_layer_ids": [0, 1, 2, 6]}}
    (tmp_path / "op_traces.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="positive trace contains inactive layer identities"):
        driver.check_traces(tmp_path, stages, {})
