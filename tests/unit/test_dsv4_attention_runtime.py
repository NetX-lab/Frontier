"""V2 scope selection and sklearn-owned replay attention, without native imports."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
from sklearn.dummy import DummyRegressor

from frontier.config import (
    MetricsConfig, RandomForrestExecutionTimePredictorConfig, ReplicaConfig,
    VllmV1SchedulerConfig,
)
from frontier.entities.batch import DecodeCudaGraphMetadata, DummyForwardBatch
from frontier.execution_time_predictor.dsv4_composition import compose_attention_phase
from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor
from frontier.profiling.attention.dsv4_replay_rows import build_dsv4_replay_rows, DSV4_REPLAY_ROW_SCHEMA_VERSION
from frontier.profiling.attention.dsa_rows import write_rows_csv
from frontier.profiling.common.runtime_identity import RuntimeIdentity
from frontier.types import ClusterType, MeasurementType
from test_dsv4_attention_profiling import _metadata
from test_dsv4_attention_workload import model, replica, source_batch, workload
from test_dsv4_predictor import options
from test_dsv4_replay_rows import plan, saved_records


BASE_SCOPES = (
    "dsv4_fused_qkv_projection", "dsv4_fused_qkv_rmsnorm", "dsv4_q_projection",
    "dsv4_output_inv_rope_quant", "dsv4_grouped_output_projection", "dsv4_output_projection",
    "dsv4_fused_q_kv_prepare", "dsv4_sparse_attention",
)
COMPRESSOR_SCOPES = tuple("dsv4_compressor_" + suffix for suffix in (
    "projection", "save_partial_states", "fused_compress_store"))
CSA_SCOPES = ("dsv4_index_weights_projection", *("dsv4_index_compressor_" + suffix for suffix in (
    "projection", "save_partial_states", "fused_compress_store")))
INDEX_SCOPES = ("dsv4_index_query_projection", "dsv4_index_fused_query_prepare", "dsv4_index_topk")


def expected_scopes(variant, *, short=False, scores=True):
    scopes = set(BASE_SCOPES)
    if variant != "uncompressed":
        scopes.update(COMPRESSOR_SCOPES)
    if variant == "csa":
        scopes.update(CSA_SCOPES)
        scopes.update(("dsv4_index_short_context_fill",) if short else INDEX_SCOPES)
        if not short and scores:
            scopes.add("dsv4_index_scores")
    return scopes


@pytest.mark.parametrize("queries,contexts,decoding,mode,rows,short,scores", [
    ([1], [2050], [True], "NONE", 1, True, False),
    ([1], [2051], [True], "NONE", 1, False, True),
    ([2052], [0], [False], "NONE", 2052, False, True),
    ([1], [0], [False], "PIECEWISE", 1, False, True),
    ([2], [0], [False], "PIECEWISE", 2, False, False),
    ([2], [2], [False], "PIECEWISE", 4, False, True),
    ([2, 1], [0, 0], [False, True], "PIECEWISE", 4, False, True),
    ([4], [0], [False], "PIECEWISE", 4, False, True),
    ([1], [0], [True], "FULL", 4, False, True),
])
def test_csa_scope_rule_covers_short_long_and_score_branches(
    model, replica, queries, contexts, decoding, mode, rows, short, scores,
):
    batch = source_batch(queries, contexts, decoding)
    if mode != "NONE":
        batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
            "full_and_piecewise", mode, True, False, sum(queries), rows, len(queries), rows)
    result = workload(model, replica, batch, 2,
                      execution_mode="eager" if mode == "NONE" else "cuda_graph")
    assert set(result["operators"]) == expected_scopes("csa", short=short, scores=scores)
    assert result["indexer_short_context"] is short
    assert result["plans_tile_schedule"] is True
    follower = workload(model, replica, batch, 4,
                        execution_mode="eager" if mode == "NONE" else "cuda_graph")
    assert follower["plans_tile_schedule"] is False
    assert follower["operators"] == result["operators"]


@pytest.mark.parametrize("mode,rows,scores", [("NONE", 1, False), ("FULL", 4, True),
                                            ("PIECEWISE", 2, False), ("PIECEWISE", 4, True)])
def test_dummy_scope_sets_use_padding_geometry(model, replica, mode, rows, scores):
    batch = DummyForwardBatch(0, 0, 0)
    if mode != "NONE":
        batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
            "full_and_piecewise", mode, True, False, 1, rows, 1, rows)
    for layer, variant in ((0, "uncompressed"), (2, "csa"), (3, "hca")):
        result = workload(model, replica, batch, layer,
                          execution_mode="eager" if mode == "NONE" else "cuda_graph")
        assert set(result["operators"]) == expected_scopes(variant, short=mode == "NONE", scores=scores)
        assert result["num_tokens"] == rows


class ReplayOwner(SklearnExecutionTimePredictor):
    estimators = 0

    def _get_estimator(self):
        self.estimators += 1
        return DummyRegressor()

    def _get_grid_search_params(self):
        return {}


def h200_replica():
    return ReplicaConfig(
        model_name="deepseek-ai/DeepSeek-V4-Flash", device="h200", network_device="h200_dgx",
        attn_tensor_parallel_size=8, attn_dp=8, moe_tensor_parallel_size=1,
        moe_expert_parallel_size=64, total_expert_num=256, router_topk=6,
        attention_backend="FLASHMLA_SPARSE_DSV4", kv_cache_dtype="fp8", indexer_kv_cache_dtype="fp8",
        moe_routing_distribution_type="balanced",
    )


def replay_owner(tmp_path):
    return ReplayOwner(
        RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True, k_fold_cv_splits=2,
            num_training_job_threads=1), h200_replica(), VllmV1SchedulerConfig(block_size=256),
        MetricsConfig(cache_dir=str(tmp_path / "cache"), output_dir=str(tmp_path / "metrics")),
        cluster_type=ClusterType.MONOLITHIC,
    )


def row_batch(row):
    mode, rows = row["graph_runtime_mode"], int(row["padded_rows"])
    if row["phase"] == "dummy":
        batch = DummyForwardBatch(0, 0, 0)
    else:
        queries = json.loads(row["prefill_seq_lens"])
        contexts = json.loads(row["prefill_kv_cache_sizes"])
        decodes = json.loads(row["decode_kv_cache_sizes"])
        batch = source_batch(queries + [1] * len(decodes), contexts + decodes,
                             [False] * len(queries) + [True] * len(decodes))
    if mode != "NONE":
        batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
            "full_and_piecewise", mode, True, False, batch.total_num_tokens, rows,
            max(1, len(batch.requests)), rows,
        )
    return batch


def synthetic_replay(tmp_path, plan):
    plan = replace(plan, kv_cache_dtype="fp8_ds_mla")
    specs = []
    for phase, mode, rows in (("decode", "FULL", 4), ("dummy", "FULL", 4), ("dummy", "NONE", 1)):
        scopes = {}
        for layer in plan.shape.layers:
            names = expected_scopes(layer.attention_variant, short=mode == "NONE")
            # Put the signed output gap after every other scope.
            scopes[layer.global_layer_id] = tuple(sorted(names - {"dsv4_output_projection"})) + ("dsv4_output_projection",)
        spec = dict(phase=phase, mode=mode, tokens=rows, case=phase + mode, scopes=scopes)
        if phase == "dummy":
            spec["requests"] = []
            queries = [1] * rows if mode == "FULL" else [rows]
            spec["metadata"] = _metadata([dict(num_scheduled_tokens=q, num_computed_tokens=0, prompt_len=q)
                                          for q in queries])
        specs.append(spec)
    record = saved_records(plan, specs)
    # The fixture's fixed layer offsets leave unequal eager entry gaps. Keep
    # complete layers sequential so roles sharing a model have identical gaps.
    intervals = []
    for step in record.steps:
        step_rows = [row for row in record.intervals if row["step_index"] == step["step_index"]]
        unowned, = (row for row in step_rows if row["global_layer_id"] is None)
        intervals.append(unowned)
        previous_end = unowned["end_us"]
        for layer in plan.shape.layers:
            owned = [row for row in step_rows if row["global_layer_id"] == layer.global_layer_id]
            shift = previous_end + 3 - owned[0]["start_us"]
            intervals.extend(dict(row, start_us=row["start_us"] + shift, end_us=row["end_us"] + shift)
                             for row in owned)
            previous_end = owned[-1]["end_us"] + shift
    record = replace(record, intervals=tuple(intervals),
                     engine=dict(record.engine, device_name="NVIDIA H200", kv_cache_dtype="fp8_ds_mla"))
    identity = RuntimeIdentity(
        runtime_versions=record.engine["runtime_versions"], source_revision="synthetic",
        model_config_revision="fixture", model_config_sha256="fixture",
        device_name="NVIDIA H200", device_capability="sm90",
        schema_version=DSV4_REPLAY_ROW_SCHEMA_VERSION,
        quant_signature=h200_replica().model_config.get_quant_signature(),
        parallel_groups={"target_tensor_parallel": 8, "engine_tensor_parallel": 1},
        graph_mode=plan.limits.cudagraph_mode,
    )
    path = tmp_path / "attention_graph_replay_kernel_only.csv"
    write_rows_csv(build_dsv4_replay_rows(plan, [record]), identity, path)
    return path


def test_sklearn_owner_trains_single_rows_reuses_cache_and_prices_replay(plan, tmp_path, monkeypatch, caplog):
    from frontier.config.quantization_manager import QuantizationManager

    monkeypatch.setattr(QuantizationManager, "_instance", None)
    monkeypatch.setattr(QuantizationManager, "_initialized", False)
    owner = replay_owner(tmp_path)
    path = synthetic_replay(tmp_path, plan)
    owner._enable_dummy_mode = False
    owner._active_measurement_type = MeasurementType.KERNEL_ONLY
    owner._attention_input_file = str(path.with_name("attention_kernel_only.csv"))
    loaded = owner._load_attention_df(owner._attention_input_file)
    assert owner._replica_config.kv_cache_dtype == "fp8"
    owner._models = owner._train_attention_layer_models()
    hashes = {name: fitted._frontier_model_hash for name, fitted in owner._models.items()}
    estimator_count = owner.estimators
    assert estimator_count == len(owner._models) > 0
    assert "trained on its single row" in caplog.text
    files = {p: p.stat().st_mtime_ns for p in Path(owner._cache_dir).iterdir()}
    assert any(p.suffix == ".pkl" for p in files)
    owner._models = owner._train_attention_layer_models()
    assert owner.estimators == estimator_count
    assert hashes == {name: fitted._frontier_model_hash for name, fitted in owner._models.items()}
    assert files == {p: p.stat().st_mtime_ns for p in Path(owner._cache_dir).iterdir()}
    threshold = int(loaded.multi_stream_gemm_token_threshold.iloc[0])
    assert all(fitted._frontier_dsv4_multi_stream_gemm_token_threshold == threshold
               for fitted in owner._models.values())
    owner._predictions = owner._predict_for_attention_layer_models()
    assert set(owner._predictions) == {"dsv4_attention"}
    assert owner._predictions["dsv4_attention"].multi_stream_gemm_token_threshold == threshold
    seen = set()
    for _, row in loaded.iterrows():
        batch = row_batch(row)
        layer = json.loads(row.global_layer_ids)[0]
        with owner._counting_kernels() as counts:
            timing = owner._predict_attention_layer_time(batch, layer, ClusterType.MONOLITHIC).operator_times
        assert not counts.operators
        scopes = json.loads(row.operator_scopes)
        costs = {scope: row[f"time_stats.{scope}.median"] for scope in scopes}
        gaps = {scope: row[f"gap_stats.{scope}.median"] for scope in scopes}
        assert timing.op_times == costs
        assert dict(timing._dsv4_scope_gap_ms) == gaps
        assert set(timing._dsv4_workloads) == set(scopes)
        assert timing._dsv4_composition_inputs == options(row, threshold)
        assert compose_attention_phase(timing.op_times, timing._dsv4_scope_gap_ms,
                                       **timing._dsv4_composition_inputs) == compose_attention_phase(costs, gaps, **options(row, threshold))
        seen.add((row.phase, row.graph_runtime_mode))
    assert seen == {("decode", "FULL"), ("dummy", "FULL"), ("dummy", "NONE")}
