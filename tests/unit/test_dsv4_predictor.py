"""CPU replay aggregate, fitting, signed-gap lookup and strict admission contracts."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.attention.dsv4_features import build_dsv4_features, build_dsv4_dummy_features
from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.dsv4_composition import compose_attention_phase
from frontier.execution_time_predictor.dsv4_predictor import (
    DeepSeekV4AttentionPredictor, TILE_PLANNER_SCOPE, TARGET_TIME, TARGET_GAP,
    dsv4_input_path, load_dsv4_attention_dataframe, train_dsv4_attention_models,
    dsv4_model_key, dsv4_model_name,
)
from frontier.profiling.attention import dsv4_attention_main as main
from frontier.profiling.attention.dsv4_replay_rows import build_dsv4_replay_rows
from frontier.types import MeasurementType
from test_dsa_predictor import _fit
from test_dsv4_attention_profiling import PLAN_ARGS
from test_dsv4_graph_bridge_check import bridge
from test_dsv4_replay_rows import MODE, OUT, Q, plan, saved_records


@pytest.fixture
def model():
    return BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")


@pytest.fixture
def replica():
    return SimpleNamespace(attn_tensor_parallel_size=8, device="h200", kv_cache_dtype="fp8_ds_mla")


def batch_features(row, model):
    if row["phase"] == "dummy":
        return build_dsv4_dummy_features(int(row["padded_rows"]))
    q = json.loads(row["prefill_seq_lens"])
    c = json.loads(row["prefill_kv_cache_sizes"])
    d = json.loads(row["decode_kv_cache_sizes"])
    return build_dsv4_features(q + [1] * len(d), c + d, [False] * len(q) + [True] * len(d),
                               variant=row["attention_variant"], compress_ratio=int(row["compress_ratio"]),
                               sliding_window=model.sliding_window, index_topk=model.index_topk,
                               padded_rows=int(row["padded_rows"]))


def options(row, threshold):
    return dict(variant=row["attention_variant"], graph_runtime_mode=row["graph_runtime_mode"],
                rows=int(row["padded_rows"]), multi_stream_gemm_token_threshold=threshold,
                indexer_short_context=bool(row["indexer_short_context"]))


def predict(predictor, row, model, **overrides):
    args = dict(variant=row["attention_variant"], graph_runtime_mode=row["graph_runtime_mode"],
                plans_tile_schedule=bool(row["plans_tile_schedule"]), scopes=json.loads(row["operator_scopes"]))
    args.update(overrides)
    return predictor.scope_times(batch_features(row, model), **args)


def aggregate(tmp_path, plan, specs=None):
    plan = replace(plan, kv_cache_dtype="fp8_ds_mla")
    scopes = dict.fromkeys((layer.global_layer_id for layer in plan.shape.layers), (Q, TILE_PLANNER_SCOPE, OUT))
    specs = [dict(spec, scopes=scopes) for spec in (specs or [{}])]
    record = saved_records(plan, specs)
    record = replace(record, engine=dict(record.engine, device_name="NVIDIA H200", kv_cache_dtype="fp8_ds_mla"))
    expected = build_dsv4_replay_rows(plan, [record])
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "dsv4_raw_e0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in (
        record.engine, *record.steps, *record.calls, *record.intervals)))
    main.main(["aggregate", *("fp8_ds_mla" if arg == "fp8" else arg for arg in PLAN_ARGS), *bridge.graph_mode_flags(MODE, (1, 2, 4, 8)),
               "--raw_dir", str(raw), "--output_root", str(tmp_path / "output"), "--device", "h200",
               "--source_revision", "source", "--model_config_revision", "config", "--allow_partial_engines"])
    path, = (tmp_path / "output").rglob("*.csv")
    assert len(expected) == len(pd.read_csv(path))
    return path


@pytest.fixture
def table(tmp_path, plan):
    return aggregate(tmp_path, plan)


def load(path, model, replica, measurement=MeasurementType.KERNEL_ONLY):
    return load_dsv4_attention_dataframe(str(path), model, replica, 256, measurement)


def trained(df):
    return DeepSeekV4AttentionPredictor(train_dsv4_attention_models(df, _fit),
                                      multi_stream_gemm_token_threshold=int(df.multi_stream_gemm_token_threshold.iloc[0]))


def test_records_aggregate_loader_training_prediction_and_composition_round_trip(tmp_path, plan, model, replica):
    phases = [dict(phase="decode"),
              dict(phase="first_tail", requests=[dict(num_scheduled_tokens=1, num_computed_tokens=63, prompt_len=64),
                                                  dict(num_scheduled_tokens=1, num_computed_tokens=80, prompt_len=64)]),
              dict(phase="prefill", requests=[dict(num_scheduled_tokens=4, num_computed_tokens=0, prompt_len=4)]),
              dict(phase="dummy", requests=[], tokens=4)]
    path = aggregate(tmp_path, plan, [dict(spec, mode=mode, case=f"{mode}_{spec['phase']}")
                                    for mode in ("FULL", "PIECEWISE", "NONE") for spec in phases])
    df = load(path, model, replica)
    predictor = trained(df)
    assert len(df) == 60
    for _, row in df.iterrows():
        result = predict(predictor, row, model)
        scopes = json.loads(row["operator_scopes"])
        assert set(result.cost_ms) == set(result.gap_ms) == set(scopes)
        own_times = {scope: row[f"time_stats.{scope}.median"] for scope in scopes}
        own_gaps = {scope: row[f"gap_stats.{scope}.median"] for scope in scopes}
        assert dict(result.cost_ms) == own_times and dict(result.gap_ms) == own_gaps
        assert result.gap_ms[OUT] < 0
        assert compose_attention_phase(result.cost_ms, result.gap_ms, **options(row, predictor.multi_stream_gemm_token_threshold)) == compose_attention_phase(own_times, own_gaps, **options(row, predictor.multi_stream_gemm_token_threshold))
        with pytest.raises(TypeError):
            result.gap_ms[OUT] = 0
        assert not hasattr(result, "kernel_counts")


def test_only_tile_planner_scope_splits_roles_and_names_separate_modes_phases(table, model, replica):
    df = load(table, model, replica)
    models = train_dsv4_attention_models(df, _fit)
    for target in (TARGET_TIME, TARGET_GAP):
        for planner in (False, True):
            assert dsv4_model_name(dsv4_model_key(TILE_PLANNER_SCOPE, "csa", "FULL", "decode", planner), target) in models
        pooled = dsv4_model_key(Q, "csa", "FULL", "decode", False)
        assert pooled == dsv4_model_key(Q, "csa", "FULL", "decode", True)
        assert dsv4_model_name(pooled, target) in models
    names = {dsv4_model_name(dsv4_model_key(scope, "csa", mode, phase, planner), target)
             for scope in (Q, TILE_PLANNER_SCOPE) for mode in ("FULL", "PIECEWISE", "NONE")
             for phase in ("decode", "first_tail", "prefill", "dummy")
             for planner in (False, True) for target in (TARGET_TIME, TARGET_GAP)}
    assert len(names) == 72


def test_training_excludes_absent_scopes_and_records_feature_and_geometry_bounds(table, model, replica):
    df = load(table, model, replica)
    index = df.index[0]
    scopes = json.loads(df.loc[index, "operator_scopes"])
    scopes.pop(OUT)
    df.loc[index, "operator_scopes"] = json.dumps(scopes)
    df.loc[index, f"time_stats.{OUT}.median"] = np.nan
    df.loc[index, f"gap_stats.{OUT}.median"] = np.nan
    calls = []
    def fit(**kwargs):
        assert kwargs["df"][kwargs["feature_cols"] + [kwargs["target_col"]]].notna().all().all()
        calls.append(kwargs)
        return _fit(**kwargs)
    models = train_dsv4_attention_models(df, fit)
    assert calls and models
    qmodel = models[dsv4_model_name(dsv4_model_key(Q, "csa", "FULL", "decode", False), TARGET_TIME)]
    assert qmodel._frontier_dsv4_bounds["padded_rows"] == (1, 1)
    assert qmodel._frontier_dsv4_bounds["max_context_len"] == (64, 64)


@pytest.mark.parametrize("column", ["padded_rows", "max_context_len"])
def test_lookup_checks_feature_and_geometry_bounds(table, model, replica, column):
    df = load(table, model, replica)
    row = df.iloc[0]
    feature = batch_features(row, model)
    feature = replace(feature, values={**feature.values, column: feature.values[column] + 100})
    with pytest.raises(DeepSeekV4CoverageError, match=column + ".*outside"):
        trained(df).scope_times(feature, variant=row.attention_variant, graph_runtime_mode="FULL",
                               plans_tile_schedule=True, scopes=[Q])


def test_missing_key_and_feature_schema_mismatch(table, model, replica):
    df = load(table, model, replica)
    row = df.iloc[0]
    predictor = trained(df)
    with pytest.raises(DeepSeekV4CoverageError, match="key.*PIECEWISE"):
        predict(predictor, row, model, graph_runtime_mode="PIECEWISE")
    for fitted in predictor._models.values():
        fitted._frontier_feature_names = ["wrong"]
    with pytest.raises(ValueError, match="feature schema mismatch"):
        predict(predictor, row, model)


def test_interpolation_uses_predict_inside_bounds(tmp_path, plan, model, replica):
    df = load(aggregate(tmp_path, plan, [dict(tokens=1), dict(tokens=4)]), model, replica)
    row = df.iloc[0].copy()
    row["padded_rows"] = 2
    result = predict(trained(df), row, model)
    assert set(result.cost_ms.values()) == set(result.gap_ms.values()) == {0.25}


@pytest.mark.parametrize("target,value,valid", [(TARGET_TIME, -0.1, False), (TARGET_TIME, 0, True),
                                                (TARGET_TIME, np.inf, False), (TARGET_GAP, np.nan, False),
                                                (TARGET_GAP, -0.2, True)])
def test_prediction_value_contract(table, model, replica, target, value, valid):
    df = load(table, model, replica)
    row = df.iloc[0]
    predictor = trained(df)
    key = dsv4_model_key(Q, row.attention_variant, "FULL", "decode", bool(row.plans_tile_schedule))
    predictor._models[dsv4_model_name(key, target)]._frontier_exact_lookup[(1.0,)] = value
    if valid:
        result = predict(predictor, row, model, scopes=[Q])
        assert (result.cost_ms if target == TARGET_TIME else result.gap_ms)[Q] == value
    else:
        with pytest.raises(ValueError, match="Invalid V4 " + target):
            predict(predictor, row, model, scopes=[Q])


def test_predictor_keeps_only_its_models_and_requires_some(table, model, replica):
    with pytest.raises(ValueError, match="requires trained"):
        DeepSeekV4AttentionPredictor({"attn_dsa_other": object()}, multi_stream_gemm_token_threshold=1024)
    df = load(table, model, replica)
    models = train_dsv4_attention_models(df, _fit)
    predictor = DeepSeekV4AttentionPredictor({**models, "attn_dsa_other": object()}, multi_stream_gemm_token_threshold=7)
    assert predictor._models == models and predictor.multi_stream_gemm_token_threshold == 7


@pytest.mark.parametrize("column,value", [
    ("schema_version", "dsv4_attention.v1"), ("padded_heads", 128),
    ("kv_cache_dtype", "bf16"), ("attention_kv_cache_dtype", "bf16"),
    ("compress_ratio", 7), ("phase", "prefill"),
    (f"gap_stats.{Q}.median", None), (f"time_stats.{Q}.median", 0),
    (f"time_stats.{Q}.median", np.inf), (f"gap_stats.{Q}.median", np.inf),
    (f"time_stats.{Q}.kernel_count", 1.5), ("plans_tile_schedule", "not_boolean"),
    ("multi_stream_gemm_token_threshold", "varied"),
    ("multi_stream_gemm_token_threshold", 0), ("multi_stream_gemm_token_threshold", 1.5),
    ("engine_cudagraph_mode", "PIECEWISE"),
    ("batch_size", 3), ("total_tokens", 3), ("max_key_len", 999),
    ("prefill_kv_cache_sizes", "[1]"), ("decode_kv_cache_sizes", "[1.5]"),
    ("decode_kv_cache_sizes", "bad_json"), ("attention_variant", "unknown"),
    ("operator_scopes", '{"not_a_scope":["not_a_scope"]}'),
    ("operator_scopes", '{"dsv4_q_projection":["other"]}'),
])
def test_loader_rejections_name_file_and_column(table, model, replica, column, value):
    df = pd.read_csv(table, float_precision="round_trip")
    if value is None:
        df = df.drop(columns=[column])
    elif value == "varied":
        df.loc[df.index[0], column] = 512
    else:
        df[column] = value
    df.to_csv(table, index=False)
    with pytest.raises(ValueError, match=column) as error:
        load(table, model, replica)
    assert str(table) in str(error.value)


def test_loader_rejects_device_and_cuda_event_measurement(table, model, replica):
    with pytest.raises(DeepSeekV4CoverageError, match="device_name") as error:
        load(table, model, SimpleNamespace(**{**vars(replica), "device": "h800"}))
    assert str(table) in str(error.value)
    with pytest.raises(DeepSeekV4CoverageError, match="measurement_type") as error:
        load(table, model, replica, MeasurementType.CUDA_EVENT)
    assert str(table) in str(error.value)


def test_loader_requires_nan_for_absent_family_scopes(table, model, replica):
    df = pd.read_csv(table, float_precision="round_trip")
    df["operator_scopes"] = df.operator_scopes.map(lambda value: json.dumps({Q: [Q], OUT: [OUT]}))
    df.to_csv(table, index=False)
    with pytest.raises(DeepSeekV4CoverageError, match="time_stats.*absent scope"):
        load(table, model, replica)


def test_input_mapping_and_aggregate_filename_agree(table):
    assert dsv4_input_path(str(table)) == str(table)
    assert dsv4_input_path(str(table.with_name("attention_kernel_only.csv"))) == str(table)
    with pytest.raises(ValueError, match="filename"):
        dsv4_input_path(str(table.with_name("attention.csv")))
