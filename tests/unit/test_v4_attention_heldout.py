"""Complete-unit splitting, runtime admission and legally zero scope audit."""

import json

import pandas as pd
import pytest

from frontier.execution_time_predictor.dsv4_predictor import (
    DeepSeekV4AttentionPredictor, load_dsv4_attention_dataframe,
)
from frontier.types import MeasurementType
from test_dsa_predictor import _fit
from test_dsv4_attention_runtime import h200_replica, synthetic_replay
from test_dsv4_replay_rows import plan
from frontier.cc_backend.backends.measured_cc_backend import _Curve
from v4_attention_heldout import (
    A6_MEDIAN_LIMIT, A6_P90_LIMIT, assign_units, collective_heldout, error_stats, evaluate_attention,
    gate, row_features, runtime_scopes, workload_coordinate,
)


@pytest.fixture
def replay(tmp_path, plan):
    replica = h200_replica()
    model = replica.model_config
    path = synthetic_replay(tmp_path, plan)
    df = load_dsv4_attention_dataframe(str(path), model, replica, 256, MeasurementType.KERNEL_ONLY)
    return df, model, replica


def test_equal_geometry_cases_and_five_layers_stay_in_one_unit(replay):
    df, _, _ = replay
    duplicate = df.copy()
    duplicate["case_id"] += "_duplicate"
    for column in ("prefill_seq_lens", "prefill_kv_cache_sizes", "decode_kv_cache_sizes"):
        duplicate[column] = duplicate[column].map(lambda value: value if pd.isna(value) else json.dumps(json.loads(value), indent=1))
    rows = pd.concat([df, duplicate], ignore_index=True)
    units = assign_units(rows, folds=2, seed=1234)
    assert len(units) == 3
    assert {unit["row_count"] for unit in units} == {10}
    assert all(len(unit["case_ids"]) == 2 for unit in units)
    assert units == assign_units(rows, folds=2, seed=1234)
    assert sorted(index for unit in units for index in unit["row_indices"]) == list(rows.index)
    for unit in units:
        training = rows.drop(index=[i for other in units if other["fold"] == unit["fold"]
                                    for i in other["row_indices"]])
        assert workload_coordinate(rows.iloc[unit["row_indices"][0]]) not in {
            workload_coordinate(row) for _, row in training.iterrows()}


def test_fold_training_never_sees_heldout_units_and_rejections_are_counted(replay, monkeypatch):
    df, model, replica = replay
    units = assign_units(df, folds=2, seed=1234)
    observed_folds, requested = set(), []
    real = DeepSeekV4AttentionPredictor.scope_times

    def scope_times(self, features, **kwargs):
        requested.append((features.phase, kwargs["graph_runtime_mode"], tuple(kwargs["scopes"])))
        assert "dsv4_query_padding" not in kwargs["scopes"]
        assert "dsv4_attention_output_zero" not in kwargs["scopes"]
        return real(self, features, **kwargs)

    monkeypatch.setattr(DeepSeekV4AttentionPredictor, "scope_times", scope_times)

    def factory(fold):
        heldout = {workload_coordinate(df.loc[unit["row_indices"][0]]) for unit in units if unit["fold"] == fold}

        def train(**kwargs):
            observed_folds.add(fold)
            assert not heldout & {workload_coordinate(row) for _, row in kwargs["df"].iterrows()}
            return _fit(**kwargs)

        return train

    result = evaluate_attention(df, model, replica, folds=2, seed=1234, trainer_factory=factory)
    assert observed_folds == {0, 1}
    assert requested
    assert result["unit_count"] == 3
    assert result["legally_zero"]["count"] > 0
    assert result["legally_zero"]["mismatch_count"] == 0
    assert result["legally_zero"]["verdict"] == "PASS"
    assert all(fold["training_test_unit_overlap"] == 0 for fold in result["folds"])
    q = result["per_scope"]["dsv4_q_projection"]
    assert q["joint"]["heldout_count"] == len(df)
    assert q["joint"]["not_evaluable_count"] == len(df)
    assert q["joint"]["not_evaluable_by_column"] == {"model_key": len(df)}
    assert q["joint"]["kernel_time_relative_error"]["count"] == 0
    assert q["joint"]["verdict"] == "NOT_EVALUABLE"
    assert result["admission_only"]["prediction_mismatch_count"] == 0


def test_runtime_features_are_the_loader_columns_and_selection_matches(replay):
    df, model, replica = replay
    for _, row in df.iterrows():
        features = row_features(row)
        assert features.phase == row.phase
        assert features.values["padded_rows"] == row.padded_rows
        assert runtime_scopes(row, model, replica) == set(json.loads(row.operator_scopes))


def test_gate_uses_absolute_kernel_error_and_inclusive_design_thresholds():
    assert gate(dict(count=1, median_absolute=A6_MEDIAN_LIMIT, p90_absolute=A6_P90_LIMIT)) == "PASS"
    assert gate(dict(count=1, median_absolute=A6_MEDIAN_LIMIT, p90_absolute=A6_P90_LIMIT + 0.001)) == "FAIL"
    assert gate(error_stats([])) == "NOT_EVALUABLE"
    stats = error_stats([-0.02, -0.01, 0.03])
    assert stats["median_absolute"] == 0.02
    assert stats["signed_median"] == -0.01


def test_missing_measured_row_group_is_not_evaluable(replay):
    df, model, replica = replay
    decode = df[df.phase == "decode"].copy()
    other_size = decode.copy()
    other_size["case_id"] += "_larger"
    other_size["padded_rows"] = 8
    rows = pd.concat([decode, other_size], ignore_index=True)
    result = evaluate_attention(rows, model, replica, folds=2, seed=1234,
                                trainer_factory=lambda fold: _fit)
    q = result["per_scope"]["dsv4_q_projection"]["joint"]
    assert q["not_evaluable_by_column"] == {"padded_rows": 10}
    assert q["evaluable_count"] == 0
    assert result["legally_zero"]["mismatch_count"] == 0


def test_admitted_middle_workload_has_runtime_statistics_and_same_old_prediction(replay, tmp_path):
    df, model, replica = replay
    decode = df[df.phase == "decode"].copy()
    cases = []
    for offset in (0, 32, 64):
        case = decode.copy()
        case["case_id"] += f"_context_{offset}"
        case["decode_kv_cache_sizes"] = case.decode_kv_cache_sizes.map(
            lambda value: json.dumps([context + offset for context in json.loads(value)]))
        case["max_key_len"] += offset
        cases.append(case)
    path = tmp_path / "attention_graph_replay_kernel_only.csv"
    pd.concat(cases, ignore_index=True).to_csv(path, index=False)
    rows = load_dsv4_attention_dataframe(str(path), model, replica, 256, MeasurementType.KERNEL_ONLY)
    result = evaluate_attention(rows, model, replica, folds=3, seed=1234,
                                trainer_factory=lambda fold: _fit)
    q = result["per_scope"]["dsv4_q_projection"]
    assert q["joint"]["heldout_count"] == 15
    assert q["joint"]["evaluable_count"] == 5
    assert q["joint"]["not_evaluable_by_column"] == {"min_context_len": 10}
    assert q["joint"]["kernel_time_relative_error"]["median_absolute"] == 0.0
    assert q["joint"]["kernel_time_relative_error"]["p90_absolute"] == 0.0
    assert q["joint"]["verdict"] == "PASS"
    assert q["shared_admission"]["joint"] == q["shared_admission"]["per_column"]
    assert result["legally_zero"]["mismatch_count"] == 0
    assert result["admission_only"]["prediction_mismatch_count"] == 0


def test_collective_points_are_left_out_one_at_a_time():
    def held_out(sizes, times):
        return collective_heldout(dict(_Curve(sizes, times, ("call",)).describe(), key=("op", "tp", 8)))

    # A power law is exact under log-log interpolation; a bent interior point is not.
    assert held_out((1, 2, 4, 8), (1.0, 2.0, 4.0, 8.0))["verdict"] == "PASS"
    bent = held_out((1, 2, 4), (1.0, 3.0, 4.0))
    assert bent["heldout"]["count"] == 1 and bent["verdict"] == "FAIL"
    assert bent["heldout"]["median_absolute"] == pytest.approx(1 - 2.0 / 3.0)
    # End points are never held out: the runtime does not extrapolate.
    assert held_out((1, 4), (1.0, 4.0))["verdict"] == "NOT_EVALUABLE"


def test_family_report_uses_current_measured_collective_sources(tmp_path):
    from test_measured_cc_backend import _backend, _row
    from v4_attention_heldout import family_classification, source_reference
    from frontier.cc_backend.backends.measured_cc_backend import MeasuredCCBackend

    backend = _backend(tmp_path, [_row(size, float(size)) for size in (100, 200, 400)])
    families = family_classification(tmp_path / "summary.csv", backend)
    exact = next(row for row in families if row["family"].startswith("unequal"))
    assert exact["prediction"] == "exact_only"
    assert exact["source"] == source_reference(MeasuredCCBackend._predict_exact_sp)
    uniform = next(row for row in families if row["family"] == "uniform measured collectives")
    assert uniform["curves"][0]["heldout"]["count"] == 1
    assert uniform["curves"][0]["verdict"] == "PASS"
