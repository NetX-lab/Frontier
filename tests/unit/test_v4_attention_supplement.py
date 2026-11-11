"""Writer-extreme planning, union checks and projected replay coverage on CPU."""
from collections import Counter
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.attention.dsv4_features import build_dsv4_features
from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.dsv4_predictor import load_dsv4_attention_dataframe
from frontier.profiling.attention.v4_attention_windows import (
    C1, assign_windows, max_writes, repeat_windows, required_windows, serving_classes,
)
from frontier.types import MeasurementType
from v4_attention_coverage import check_steps, plan_from_spec, prediction_steps
from v4_attention_replay_coverage import check_requests, model_ranges, projected_rows

ROOT = Path(__file__).resolve().parents[2]
H3O = ROOT / "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs"
SUPPLEMENT = H3O / "g1h_supplement"
TABLE = ROOT / "data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv"


def test_writer_extremes_add_only_the_agreed_categories_and_widths():
    grid = required_windows()
    baseline = []
    for width, own in serving_classes():
        k4, k128 = max_writes(own)
        for category, mode, tails in (("FULL", "FULL", 0), ("PIECEWISE", "PIECEWISE", 0),
                                     ("first_tail", "PIECEWISE", 1)):
            for profile in C1["profiles"]:
                example = next(w for w in grid if (w.category, w.width, w.own, w.profile)
                               == (category, width, own, profile))
                baseline.extend(replace(example, mix=tuple(max(n, tails) for n in mix))
                                for mix in ((0, 0), (k4, 0), (k128, k128)))
    baseline = set(baseline) | {w for w in grid if w.category not in ("FULL", "PIECEWISE", "first_tail")}
    added = set(grid) - baseline
    assert len(baseline) == 1041 and len(grid) == 1473
    assert Counter(w.category for w in added) == {"FULL": 144, "PIECEWISE": 144, "first_tail": 144}
    assert Counter(w.width for w in added) == {8: 72, 16: 216, 24: 144}
    assert all(w.own > 4 for w in added)


def test_fixed_repeat_grid_matches_both_saved_plans():
    repeats = repeat_windows()
    assert len(repeats) == len(C1["widths"]) * 3
    for category in ("FULL", "PIECEWISE", "first_tail"):
        assert [w.width for w in repeats if w.category == category] == list(C1["widths"])
    for w in repeats:
        assert w.own == max(own for width, own in serving_classes() if width == w.width)
        assert w.profile == "high"
        assert w.mix == (max(max_writes(w.own)[0], w.tails), w.tails)
    grid = json.loads((SUPPLEMENT / "coverage_grid.json").read_text())
    assert grid["added_windows"] == 432
    assert grid["repeat_windows"] == json.loads(json.dumps([w.to_json() for w in repeats]))
    assert Counter(w["purpose"] for w in grid["windows"]) == {"supplement": 432, "repeat": 18}


@pytest.fixture(scope="module")
def plans():
    return [plan_from_spec(H3O / "g1h/package_plan.json"), plan_from_spec(SUPPLEMENT / "package_plan.json")]


def test_every_supplement_step_is_assigned_to_its_own_target(plans):
    plan = plans[1]
    targets = json.loads((SUPPLEMENT / "coverage_grid.json").read_text())["windows"]
    wanted = {row["case_id"]: row for row in targets}
    assert len(plan.cases) == 450
    assert Counter(c.engine_index for c in plan.cases) == {0: 300, 1: 150}
    for step in prediction_steps(plan):
        assert any(json.loads(json.dumps(w.to_json())) == {k: wanted[step["case_id"]][k] for k in w.to_json()}
                   for w in assign_windows(step))
    assert all(case.warmup == 0 and case.measured == 1 for case in plan.cases)


def test_several_plans_cover_union_and_duplicate_case_ids_are_rejected(plans):
    records = [(f"prediction:{i}", ordinal, step) for i, plan in enumerate(plans)
               for ordinal, step in enumerate(prediction_steps(plan), 1)]
    report = check_steps(plans, records)
    assert report["grid_complete"] and report["prediction_matches"]
    assert report["window_counts"] == {"predicted": 1473}
    assert not report["outside_steps"]
    changed = dict(records[-1][2], engine_index=99)
    broken = check_steps(plans, [*records[:-1], ("prediction:bad", 1, changed)])
    assert broken["mismatches"]
    with pytest.raises(ValueError, match="Duplicate case id"):
        check_steps([plans[0], plans[0]], [])


@pytest.fixture(scope="module")
def projected(plans):
    model = BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")
    replica = SimpleNamespace(attn_tensor_parallel_size=8, device="h200", kv_cache_dtype="fp8_ds_mla")
    measured = load_dsv4_attention_dataframe(str(TABLE), model, replica, 256, MeasurementType.KERNEL_ONLY)
    repeats = json.loads((SUPPLEMENT / "package_plan.json").read_text())["repeat_case_ids"]
    predicted = projected_rows(plans[1], measured, model, excluded_case_ids=repeats)
    assert len(predicted) == 432 * 5
    return measured, predicted, model


def test_probe_c_writer_failure_is_closed_by_the_promoted_supplement(projected):
    measured, predicted, model = projected
    # Preserve the reported request's exact feature bounds. The census does not
    # retain batch IDs/request lists, so this is a feature regression for the
    # reported csa first-tail padded16/own11/writes8 coordinate.
    values = build_dsv4_features([1] * 11, [6143, 6243, 6343, 6443, 6543, 6643, 6743, 6844, 6944, 7044, 7151],
                                 [False] + [True] * 10, variant="csa", compress_ratio=4,
                                 sliding_window=model.sliding_window, index_topk=model.index_topk,
                                 padded_rows=16).values
    assert values["num_compressed_writes"] == 8
    record = dict(variant="csa", graph_runtime_mode="PIECEWISE", phase="first_tail",
                  plans_tile_schedule=True, scopes=["dsv4_compressor_fused_compress_store"], values=values, count=1)
    ranges = model_ranges(measured)
    # The supplement rows promoted into the table measure writer counts up to 16,
    # so the coordinate the probe reported outside the earlier 1..7 group range is
    # now measured rather than projected.
    group = ranges["dsv4_attention__dsv4_compressor_fused_compress_store__csa__PIECEWISE__first_tail__time"]
    assert group["16.0"]["num_compressed_writes"] == (1.0, 16.0)
    assert not check_requests([record], ranges)["misses"]
    union = check_requests([record], model_ranges(pd.concat([measured, predicted], ignore_index=True)))
    assert not union["misses"]


@pytest.mark.parametrize("change,column", [("model", "model_key"), ("rows", "padded_rows"),
                                            ("context", "max_context_len")])
def test_replay_checker_reports_each_outside_request_with_all_features(change, column):
    name = "dsv4_attention__scope__csa__PIECEWISE__first_tail__time"
    ranges = {name: {"16.0": {"num_compressed_writes": (1, 11), "max_context_len": (6143, 7166)}}}
    values = dict(padded_rows=16, num_compressed_writes=8, max_context_len=7151)
    record = dict(variant="csa", graph_runtime_mode="PIECEWISE", phase="first_tail", plans_tile_schedule=False,
                  scopes=["scope"], values=values, count=7)
    if change == "model": ranges = {}
    elif change == "rows": values["padded_rows"] = 12
    else: values["max_context_len"] = 7167
    result = check_requests([record], ranges)
    assert result["counts"]["missed_requests"] == 1 and result["counts"]["missed_calls"] == 7
    assert result["misses"][0]["request"]["values"] == values
    assert result["misses"][0]["violations"][0]["reasons"][0]["column"] == column


def test_prepare_reuses_g1h_p1h_builder_without_building_package(monkeypatch):
    path = SUPPLEMENT / "prepare_package.py"
    spec = importlib.util.spec_from_file_location("supplement_prepare_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = module.package_builder()
    assert builder.HERE == SUPPLEMENT and builder.ROOT == ROOT
    called = []
    monkeypatch.setattr(module, "package_builder", lambda: SimpleNamespace(main=lambda: called.append(True)))
    module.main()
    assert called == [True]
