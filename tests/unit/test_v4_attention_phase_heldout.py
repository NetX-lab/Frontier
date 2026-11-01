"""Forward attention qualification, canonical fold identity and complete coverage."""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError, deepseek_v4_tile_schedule_planners
from frontier.config.model_config import BaseModelConfig
from frontier.types import ClusterType
import v4_attention_phase_heldout as phase_heldout
from v4_attention_heldout import A6_MEDIAN_LIMIT, A6_P90_LIMIT, assign_units
from v4_attention_phase_heldout import (
    aggregate_forward_attention, attention_row_diagnostics, model_layer_counts,
    recorded_row_folds, statistics_of, summarize,
)


@pytest.fixture
def model():
    return BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")


def forward_rows(model, *, unit_id=0, mode="FULL", phase="decode", context=6144, fold=0):
    counts = model_layer_counts(model)
    rows = [{"row_index": unit_id * len(counts) + index, "case_id": f"case-{unit_id}", "fold": fold,
             "key": [variant, mode, phase, planner], "padded_rows": 1, "all_scopes_exact": True,
             "predicted_composed_us": 100.0, "measured_window_us": 100.0,
             "measured_composed_us": 1000.0}
            for index, (variant, planner) in enumerate(sorted(counts))]
    unit = {"unit_id": unit_id, "fold": fold, "row_indices": [row["row_index"] for row in rows],
            "row_count": len(rows), "case_ids": [f"case-{unit_id}"],
            "coordinate": {"graph_runtime_mode": mode, "phase": phase, "padded_rows": 1,
                           "dispatch_num_requests": 1.0, "prefill_seq_lens": [],
                           "prefill_kv_cache_sizes": [], "decode_kv_cache_sizes": [context]}}
    return rows, unit


def test_weights_use_model_specs_and_serving_planner_owner(model):
    variants = tuple(spec.variant_id for spec in model.get_layer_attention_specs())
    expected = list(zip(variants, deepseek_v4_tile_schedule_planners(variants)))
    counts = model_layer_counts(model)
    assert sum(counts.values()) == model.num_layers
    assert counts == {key: expected.count(key) for key in set(expected)}
    rows, unit = forward_rows(model)
    for index, row in enumerate(rows):
        row["predicted_composed_us"] = 90 + index * 7
        row["measured_window_us"] = 100 + index * 3
    result = aggregate_forward_attention(rows, model, [unit])
    actual = result["workloads"][0]
    predicted = sum(counts[(row["key"][0], row["key"][3])] * row["predicted_composed_us"] for row in rows)
    measured = sum(counts[(row["key"][0], row["key"][3])] * row["measured_window_us"] for row in rows)
    assert actual["predicted_attention_us"] == predicted
    assert actual["measured_attention_us"] == measured
    assert actual["signed_error_us"] == predicted - measured
    assert actual["signed_relative_error"] == pytest.approx((predicted - measured) / measured)
    assert result["layer_count"] == model.num_layers
    assert result["e2e_acceptance"] == "NOT_ASSESSED"


def test_model_layer_count_is_not_fixed_to_the_active_model():
    specs = tuple(SimpleNamespace(variant_id=variant) for variant in
                  ("uncompressed", "uncompressed", "csa", "hca", "csa", "hca"))
    model = SimpleNamespace(num_layers=len(specs), get_layer_attention_specs=lambda: specs)
    rows, unit = forward_rows(model)
    result = aggregate_forward_attention(rows, model, [unit])
    assert result["layer_count"] == 6
    assert result["workloads"][0]["measured_attention_us"] == 600.0


def test_direct_window_reference_decides_verdict_not_composed_medians(model):
    rows, unit = forward_rows(model)
    result = aggregate_forward_attention(rows, model, [unit])
    assert result["attention_verdict"] == "PASS"
    assert result["by_group"][0]["p90_absolute_relative_error"] == 0.0
    for row in rows:
        row["measured_composed_us"] = row["predicted_composed_us"]
        row["measured_window_us"] = 200.0
    result = aggregate_forward_attention(rows, model, [unit])
    assert result["attention_verdict"] == "FAIL"
    assert result["workloads"][0]["signed_relative_error"] == -0.5


def test_limits_remain_inclusive_median_five_percent_and_p90_ten_percent():
    assert A6_MEDIAN_LIMIT == 0.05 and A6_P90_LIMIT == 0.10
    errors = [0.0, 0.01, A6_MEDIAN_LIMIT, A6_P90_LIMIT, A6_P90_LIMIT]
    passed = statistics_of(errors, [-value for value in errors])
    assert passed["verdict"] == "PASS"
    assert passed["median_absolute_relative_error"] == A6_MEDIAN_LIMIT
    assert passed["p90_absolute_relative_error"] == A6_P90_LIMIT
    assert passed["signed_median_relative_error"] == -A6_MEDIAN_LIMIT
    assert statistics_of([A6_MEDIAN_LIMIT + 0.001])["verdict"] == "FAIL"
    errors = [0.0] * 9 + [0.5] * 2
    assert statistics_of(errors)["median_absolute_relative_error"] == 0.0
    assert statistics_of(errors)["verdict"] == "FAIL"
    assert statistics_of([])["verdict"] == "NOT_EVALUABLE"


def test_missing_partial_and_fully_skipped_workloads_are_not_admitted(model):
    rows, unit = forward_rows(model)
    _, fully_missing = forward_rows(model, unit_id=1, context=6176)
    result = aggregate_forward_attention(rows[:-1], model, [unit, fully_missing])
    assert result["attention_verdict"] == "NOT_EVALUABLE"
    assert result["complete_workloads"] == 0
    assert result["omitted_workloads"] == 2
    assert result["partially_evaluable_workloads"] == 1
    assert result["fully_unevaluable_workloads"] == 1
    assert result["by_group"][0]["verdict"] == "NOT_EVALUABLE"
    assert result["omissions"][0]["missing_layer_groups"] == [[rows[-1]["key"][0], rows[-1]["key"][3]]]
    assert result["omissions"][1]["missing_row_indices"] == fully_missing["row_indices"]


def test_omissions_do_not_pollute_complete_workload_error_population(model):
    rows, unit = forward_rows(model)
    _, omitted = forward_rows(model, unit_id=1, context=6176)
    result = aggregate_forward_attention(rows, model, [unit, omitted])
    assert result["by_group"][0]["count"] == 1
    assert result["by_group"][0]["total_workloads"] == 2
    assert result["by_group"][0]["omitted_workloads"] == 1
    assert result["attention_verdict"] == "PASS"
    assert result["verdict_population"] == "complete_evaluable_workloads_only"


def test_duplicate_group_and_unexpected_group_are_not_evaluable(model):
    rows, unit = forward_rows(model)
    duplicate = copy.deepcopy(rows[0])
    duplicate["row_index"] = max(unit["row_indices"]) + 1
    unit["row_indices"].append(duplicate["row_index"])
    unit["row_count"] += 1
    result = aggregate_forward_attention([*rows, duplicate], model, [unit])
    assert result["complete_workloads"] == 0
    assert "duplicate_layer_groups" in result["omissions"][0]["reasons"]
    duplicate["key"][0] = "unknown_variant"
    result = aggregate_forward_attention([*rows, duplicate], model, [unit])
    assert "unexpected_layer_groups" in result["omissions"][0]["reasons"]
    assert result["attention_verdict"] == "NOT_EVALUABLE"


def test_repeated_prediction_row_is_rejected(model):
    rows, unit = forward_rows(model)
    with pytest.raises(ValueError, match="more than once"):
        aggregate_forward_attention([*rows, rows[0]], model, [unit])


@pytest.mark.parametrize("mutate,match", [
    (lambda rows, unit: rows[0].update(fold=1), "recorded fold"),
    (lambda rows, unit: rows[0].update(row_index=99999), "absent"),
    (lambda rows, unit: rows[0].update(case_id="wrong_case"), "identity"),
    (lambda rows, unit: rows[0]["key"].__setitem__(1, "PIECEWISE"), "identity"),
])
def test_cross_fold_and_workload_identity_mismatches_are_rejected(model, mutate, match):
    rows, unit = forward_rows(model)
    mutate(rows, unit)
    with pytest.raises(ValueError, match=match):
        aggregate_forward_attention(rows, model, [unit])


def test_canonical_geometry_not_case_id_owns_workload_identity(model):
    rows, first = forward_rows(model)
    other_rows, second = forward_rows(model, unit_id=1, context=6176)
    second["case_ids"] = first["case_ids"]
    for row in other_rows:
        row["case_id"] = rows[0]["case_id"]
    result = aggregate_forward_attention([*rows, *other_rows], model, [first, second])
    assert result["complete_workloads"] == 2
    second["coordinate"] = first["coordinate"]
    with pytest.raises(ValueError, match="canonical identity"):
        aggregate_forward_attention([*rows, *other_rows], model, [first, second])


def test_verdict_groups_are_mode_and_phase_only(model):
    rows, units = [], []
    for unit_id, (mode, phase) in enumerate((("FULL", "decode"), ("PIECEWISE", "decode"),
                                           ("PIECEWISE", "first_tail"), ("NONE", "prefill"))):
        items, unit = forward_rows(model, unit_id=unit_id, mode=mode, phase=phase)
        rows.extend(items)
        units.append(unit)
    result = aggregate_forward_attention(rows, model, units)
    assert {tuple(group["key"]) for group in result["by_group"]} == {
        ("FULL", "decode"), ("PIECEWISE", "decode"), ("PIECEWISE", "first_tail"), ("NONE", "prefill")}
    assert all(group["count"] == 1 for group in result["by_group"])


@pytest.mark.parametrize("window", [0.0, float("nan")])
def test_invalid_layer_window_is_not_evaluable(model, window):
    rows, unit = forward_rows(model)
    rows[0]["measured_window_us"] = window
    result = aggregate_forward_attention(rows, model, [unit])
    assert result["attention_verdict"] == "NOT_EVALUABLE"
    assert result["omissions"][0]["invalid_timing_row_indices"] == [rows[0]["row_index"]]
    json.dumps(result, allow_nan=False)


def test_all_present_scope_and_phase_diagnostics_retain_signed_errors():
    scopes = ("dsv4_fused_qkv_projection", "dsv4_sparse_attention", "dsv4_output_projection")
    fields = {f"time_stats.{scope}.median": 0.1 for scope in scopes}
    fields.update({f"gap_stats.{scope}.median": 0.01 for scope in scopes})
    row = pd.Series({"operator_scopes": json.dumps(scopes), "attention_variant": "uncompressed",
                     "graph_runtime_mode": "FULL", "padded_rows": 1, "indexer_short_context": False,
                     "layer_window_stats.median": 0.6, **fields})
    prediction = SimpleNamespace(cost_ms={scope: 0.09 for scope in scopes},
                                 gap_ms={scope: -0.01 for scope in scopes})
    item = attention_row_diagnostics(row, prediction, 128)
    assert set(item["scope_relative_errors"]) == set(scopes)
    assert item["predicted_composed_us"] == pytest.approx(240.0)
    assert item["measured_composed_us"] == pytest.approx(330.0)
    assert item["measured_window_us"] == 600.0
    assert item["window_signed_relative_error"] == pytest.approx(-0.6)
    assert item["signed_relative_error"] == pytest.approx(240 / 330 - 1)
    summary = summarize([item])
    assert set(summary["scopes"]) == set(scopes)
    assert set(summary["scope_gap_errors_us"]) == set(scopes)
    assert all(value["signed_median"] == -20 for value in summary["scope_gap_errors_us"].values())
    skipped = summarize([], scopes)
    assert all(value["verdict"] == "NOT_EVALUABLE" for value in skipped["scopes"].values())


def test_recorded_folds_are_reused_and_validated_before_training():
    records = []
    for case in range(10):
        for variant in range(2):
            records.append({"case_id": f"case-{case}", "graph_runtime_mode": "FULL", "phase": "decode",
                            "padded_rows": 1, "dispatch_num_requests": 1,
                            "prefill_seq_lens": "[]", "prefill_kv_cache_sizes": "[]",
                            "decode_kv_cache_sizes": json.dumps([6144 + case]), "variant": variant})
    frame = pd.DataFrame(records)
    recorded = {"arguments": {"folds": 5, "seed": 1234},
                "attention": {"fold_assignment": assign_units(frame, 5, 1234)}}
    recorded = json.loads(json.dumps(recorded))
    before = copy.deepcopy(recorded)
    row_folds = recorded_row_folds(frame, recorded)
    assert recorded == before
    assert set(row_folds.values()) == set(range(5))
    assert all(row_folds[index] == unit["fold"] for unit in recorded["attention"]["fold_assignment"]
               for index in unit["row_indices"])
    assert all(row_folds[2 * case] == row_folds[2 * case + 1] for case in range(10))
    recorded["attention"]["fold_assignment"][0]["fold"] = (row_folds[0] + 1) % 5
    with pytest.raises(ValueError, match="canonical workload KFold"):
        recorded_row_folds(frame, recorded)


def test_main_preserves_folds_all_diagnostics_and_explicit_coverage(model, tmp_path, monkeypatch):
    scope = "dsv4_fused_qkv_projection"
    records = []
    for padded in (1, 2, 4, 8, 16):
        for variant, planner in sorted(model_layer_counts(model)):
            records.append({"case_id": f"dummy-{padded}", "graph_runtime_mode": "FULL", "phase": "dummy",
                            "padded_rows": padded, "dispatch_num_requests": padded,
                            "prefill_seq_lens": "[]", "prefill_kv_cache_sizes": "[]", "decode_kv_cache_sizes": "[]",
                            "attention_variant": variant, "plans_tile_schedule": planner,
                            "operator_scopes": json.dumps([scope]), "indexer_short_context": False,
                            f"time_stats.{scope}.median": 0.1, f"gap_stats.{scope}.median": 0.01,
                            "layer_window_stats.median": 0.11})
    frame = pd.DataFrame(records)
    units = assign_units(frame, 5, 1234)
    table = tmp_path / "attention.csv"
    table.write_text("synthetic input; the loader is replaced in this test\n")
    recorded = {"arguments": {"folds": 5, "seed": 1234}, "config": {"case": "synthetic"},
                "input_table": {"path": str(table), "sha256": hashlib.sha256(table.read_bytes()).hexdigest()},
                "attention": {"fold_assignment": units}}
    result_path, output_path = tmp_path / "a6.json", tmp_path / "phase.json"
    result_path.write_text(json.dumps(recorded))
    replica = SimpleNamespace(model_config=model)
    cluster = SimpleNamespace(replica_config=replica, replica_scheduler_config=SimpleNamespace(block_size=256))
    config = SimpleNamespace(get_clusters=lambda: {ClusterType.DECODE: cluster})
    monkeypatch.setattr(phase_heldout, "build_p2_config", lambda scratch: (config, recorded["config"]))
    monkeypatch.setattr(phase_heldout, "load_dsv4_attention_dataframe", lambda *args: frame)
    monkeypatch.setattr(phase_heldout, "manager_trainer", lambda cluster, cache: int(cache.name[5:]))
    observed = []

    def train(training, fold):
        expected_test = {index for unit in units if unit["fold"] == fold for index in unit["row_indices"]}
        assert set(training.index) == set(frame.index) - expected_test
        observed.append(fold)
        return {"fold": fold}

    class Predictor:
        multi_stream_gemm_token_threshold = 128

        def __init__(self, models):
            pass

        def scope_times(self, features, **kwargs):
            if features.values["padded_rows"] == 16:
                raise DeepSeekV4CoverageError("synthetic coverage miss")
            return SimpleNamespace(cost_ms={scope: 0.1}, gap_ms={scope: 0.01})

    monkeypatch.setattr(phase_heldout, "train_dsv4_attention_models", train)
    monkeypatch.setattr(phase_heldout, "DeepSeekV4AttentionPredictor", Predictor)
    monkeypatch.setattr(phase_heldout, "exact_lookup", lambda *args: True)
    monkeypatch.setattr(sys, "argv", ["v4_attention_phase_heldout.py", "--result", str(result_path),
                                     "--output", str(output_path), "--scratch-dir", str(tmp_path / "scratch")])
    phase_heldout.main()
    output = json.loads(output_path.read_text())
    assert observed == list(range(5))
    assert output["folds"] == {"folds": 5, "seed": 1234}
    assert output["classification"] == "A6_FORWARD_ATTENTION_QUALIFICATION"
    assert output["layer_phase_classification"] == "DIAGNOSTIC_ONLY"
    assert output["e2e_acceptance"] == "NOT_ASSESSED"
    assert output["attention"]["complete_workloads"] == 4
    assert output["attention"]["fully_unevaluable_workloads"] == 1
    assert output["attention"]["omitted_workloads"] == 1
    assert len(output["coverage_skips"]) == len(model_layer_counts(model))
    assert all(item["reason"] == "synthetic coverage miss" for item in output["coverage_skips"])
    assert set(output["overall"]["scopes"]) == {scope}
    assert all(scope in item["all"]["scopes"] for item in output["by_key"])


def test_empty_statistics_are_serializable_without_nan():
    result = statistics_of([None], [None])
    assert result["count"] == 0
    assert result["undefined_relative_error_count"] == 1
    assert result["verdict"] == "NOT_EVALUABLE"
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="finite"):
        statistics_of([np.inf])
