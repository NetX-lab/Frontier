"""CPU coverage for measured DSA loading, feature identity and runtime scopes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from frontier.attention.dsa_features import DSA_FEATURE_COLUMNS, DSACoverageError, build_dsa_features
from frontier.config.model_config import BaseModelConfig
from frontier.execution_time_predictor.dsa_predictor import (
    DSAPredictor, dsa_input_path, load_dsa_dataframe, train_dsa_models,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import _build_exact_feature_lookup
from frontier.types import MeasurementType


@pytest.fixture
def model():
    return BaseModelConfig.create_from_name("zai-org/GLM-5.3")


def _row(model, queries, contexts, variant="full_index", *, prefill=True):
    features = build_dsa_features(queries, contexts, model.index_topk)
    return {
        "schema_version": "dsa_attention.v1", "measurement_type": "KERNEL_ONLY",
        "model_architecture_profile": model.get_model_architecture_profile().profile_id,
        "quant_signature": model.get_quant_signature(),
        "profiling_precision": model.get_default_precision().name,
        "attention_backend": "FLASH_ATTN_MLA_SPARSE", "attention_impl": "FlashAttnMLASparseImpl",
        "attention_family": "dsa_attention", "attention_variant": variant,
        "n_embd": model.embedding_dim, "n_q_head": model.num_q_heads,
        "n_q_head_per_rank": model.num_q_heads // 8,
        "num_tensor_parallel_workers": 8, "block_size": 64,
        **{name: getattr(model, name) for name in (
            "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
            "v_head_dim", "index_topk", "index_n_heads", "index_head_dim",
        )},
        "prefill_seq_lens": json.dumps(queries if prefill else []),
        "prefill_kv_cache_sizes": json.dumps(contexts if prefill else []),
        "decode_kv_cache_sizes": json.dumps([] if prefill else contexts),
        "is_prefill": prefill, "max_key_len": features.values["max_key_len"],
        "time_stats.attn_dsa_indexer.median": 0.2 if variant == "full_index" else np.nan,
        "time_stats.attn_dsa_mla.median": 0.3,
        "time_stats.attn_dsa_indexer.kernel_count": 5 if variant == "full_index" else np.nan,
        "time_stats.attn_dsa_mla.kernel_count": 7,
    }


def _fit(**kwargs):
    columns = kwargs["feature_cols"]
    return SimpleNamespace(
        _frontier_feature_names=columns,
        _frontier_exact_lookup=_build_exact_feature_lookup(kwargs["df"], columns, kwargs["target_col"]),
        predict=lambda frame: np.repeat(0.25, len(frame)),
    )


def _load(tmp_path, model, rows, measurement=MeasurementType.KERNEL_ONLY):
    file = tmp_path / ("dsa_attention.csv" if measurement is MeasurementType.CUDA_EVENT else "dsa_attention_kernel_only.csv")
    pd.DataFrame(rows).to_csv(file, index=False)
    return load_dsa_dataframe(str(file), model, 8, 64, measurement)


def test_decode_and_prefill_features_follow_native_query_threshold():
    from frontier.attention.dsa_features import dsa_decode_query_limit

    assert dsa_decode_query_limit(16) == 128
    assert build_dsa_features([129], [0], 2048, decode_query_limit=dsa_decode_query_limit(16)).path == "prefill"
    assert build_dsa_features([1, 1], [2047, 2048], 2048).values["sum_selected_key_len"] == 4096
    assert build_dsa_features([256], [2048], 2048).path == "decode"
    assert build_dsa_features([257], [2048], 2048).path == "prefill"
    assert build_dsa_features([2], [2047], 2048).values["sum_selected_causal_key_pairs"] == 4096


def test_causal_features_clip_each_query_not_only_final_key_length():
    feature = build_dsa_features([3, 4], [1, 2], 4)
    assert feature.values["sum_causal_key_pairs"] == sum(c + i for q, c in [(3, 1), (4, 2)] for i in range(1, q + 1))
    assert feature.values["sum_selected_causal_key_pairs"] == sum(min(c + i, 4) for q, c in [(3, 1), (4, 2)] for i in range(1, q + 1))
    with pytest.raises(DSACoverageError, match="mixed"):
        build_dsa_features([1, 257], [128, 0], 2048)


def test_loader_only_opens_dsa_files_and_uses_actual_feature_lists(tmp_path, model):
    rows = [_row(model, [1, 1], [128, 2048], prefill=False)]
    df = _load(tmp_path, model, rows)
    # A malformed legacy table cannot be read by the DSA path.
    (tmp_path / "attention_kernel_only.csv").write_text("not,a,dsa,table\n")
    loaded = load_dsa_dataframe(str(tmp_path / "attention_kernel_only.csv"), model, 8, 64, MeasurementType.KERNEL_ONLY)
    pd.testing.assert_frame_equal(df, loaded)
    assert df.iloc[0].sum_key_len == 2178
    assert df.iloc[0].sum_selected_key_len == 2177
    assert dsa_input_path(str(tmp_path / "attention.csv")).endswith("dsa_attention.csv")


@pytest.mark.parametrize("column,value", [
    ("schema_version", "dense.v1"), ("measurement_type", "CUDA_EVENT"),
    ("quant_signature", "none"), ("model_architecture_profile", "generic"),
    ("profiling_precision", "FP8"), ("attention_family", "latent_mla"),
    ("block_size", 32), ("index_topk", 1), ("n_q_head_per_rank", 1),
])
def test_loader_rejects_incompatible_identity_or_shape(tmp_path, model, column, value):
    row = _row(model, [512], [0]); row[column] = value
    with pytest.raises(ValueError, match="DSA"):
        _load(tmp_path, model, [row])


def test_cuda_event_decode_host_stall_is_not_a_training_target(tmp_path, model):
    decode = _row(model, [1], [2048], prefill=False)
    decode["time_stats.attn_dsa_mla.median"] = 10000
    rows = [decode, _row(model, [512], [0])]
    for row in rows:
        row["measurement_type"] = "CUDA_EVENT"
    df = _load(tmp_path, model, rows, MeasurementType.CUDA_EVENT)
    assert list(df.dsa_workload) == ["prefill"]
    assert df["time_stats.attn_dsa_mla.median"].max() == 0.3


def test_phase_features_and_owner_only_indexer_use_exact_measured_rows(tmp_path, model):
    rows = [
        _row(model, queries, contexts, variant, prefill=prefill)
        for variant in ("full_index", "shared_index")
        for queries, contexts, prefill in [([1, 1], [128, 2048], False), ([64], [0], True), ([512], [0], True)]
    ]
    fitted = train_dsa_models(_load(tmp_path, model, rows), _fit)
    assert len(fitted) == 9
    for name, fitted_model in fitted.items():
        operator = "attn_dsa_indexer" if name.startswith("attn_dsa_indexer") else "attn_dsa_mla"
        path = "prefill" if name.endswith("_prefill") else "decode"
        assert tuple(fitted_model._frontier_feature_names) == DSA_FEATURE_COLUMNS[operator][path]
    predictor = DSAPredictor(fitted)
    features = build_dsa_features([1, 1], [128, 2048], model.index_topk)
    calls = []
    full = predictor.operator_times(features, "full_index", record_kernel_count=lambda *args: calls.append(args))
    shared = predictor.operator_times(features, "shared_index")
    assert full.total_time() == 0.5
    assert shared.total_time() == 0.3
    assert len(calls) == 2
    assert calls[0][2] == (2.0, 2178.0)


@pytest.mark.parametrize("queries,contexts", [([2], [0]), ([512, 512], [0, 0]), ([1], [999999])])
def test_uncovered_extend_multi_prefill_and_long_context_fail(tmp_path, model, queries, contexts):
    rows = [_row(model, [1], [128], prefill=False), _row(model, [64], [0]), _row(model, [512], [0])]
    predictor = DSAPredictor(train_dsa_models(_load(tmp_path, model, rows), _fit))
    with pytest.raises(DSACoverageError, match="coverage missing"):
        predictor.operator_times(build_dsa_features(queries, contexts, model.index_topk), "full_index")


def test_trained_data_is_required():
    with pytest.raises(ValueError, match="trained profiling data"):
        DSAPredictor({})


@pytest.mark.parametrize("variant", ["full_index", "shared_index"])
@pytest.mark.parametrize("capture_width", [1, 2, 4])
def test_regular_zero_context_measurements_cannot_qualify_full_dummy(
    tmp_path, model, variant, capture_width,
):
    rows = [_row(model, [1], [0], kind) for kind in ("full_index", "shared_index")]
    predictor = DSAPredictor(train_dsa_models(_load(tmp_path, model, rows), _fit))
    features = build_dsa_features([1], [0], model.index_topk)
    assert predictor.operator_times(features, variant).total_time() > 0
    with pytest.raises(DSACoverageError, match=f"capture width {capture_width}.*regular serving"):
        predictor.operator_times(features, variant, dummy_capture_width=capture_width)


@pytest.mark.parametrize("variant", ["full_index", "shared_index"])
def test_checked_in_one_token_context_band_never_expands_through_an_unmeasured_hole(model, variant):
    path = (Path(__file__).resolve().parents[2] /
            "data/profiling/compute/h800/zai-org/GLM-5.3/dsa_attention_kernel_only.csv")
    rows = load_dsa_dataframe(str(path), model, 8, 64, MeasurementType.KERNEL_ONLY)
    predictor = DSAPredictor(train_dsa_models(rows, _fit))
    single = rows[
        (rows["attention_variant"] == variant)
        & (rows["dsa_workload"] == "decode")
        & (rows["num_query_tokens"] == 1)
        & (rows["num_sequences"] == 1)
    ]
    measured_contexts = set(single["min_context_len"])
    # The original one-token grid starts at context66. New boundary coverage
    # must measure this band, not merely widen a model's rectangular bounds.
    for context in range(66):
        features = build_dsa_features([1], [context], model.index_topk)
        if context in measured_contexts:
            assert predictor.operator_times(features, variant).total_time() > 0
        else:
            with pytest.raises(DSACoverageError, match="coverage missing"):
                predictor.operator_times(features, variant)


@pytest.mark.parametrize("measurement", [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY])
def test_extended_rows_select_exact_features_and_interpolate_with_coverage(tmp_path, model, measurement):
    shapes = [
        ([2], [0]), ([256], [32768]), ([2] * 1024, [0] * 1024),
        ([256] * 32, [0] * 32), ([257, 257], [0, 0]),
        ([512] * 16, [12288] * 16), ([4096, 4096], [0, 0]),
        ([512, 512], [32256, 32256]), ([4096, 4096], [28672, 28672]),
        ([2, 2], [32768, 32768]), ([2] * 16, [12800] * 16),
    ]
    rows = [_row(model, q, c, variant) for variant in ("full_index", "shared_index") for q, c in shapes]
    for row in rows:
        row["measurement_type"] = measurement.value
    fitted = train_dsa_models(_load(tmp_path, model, rows, measurement), _fit)
    predictor = DSAPredictor(fitted)
    for queries, contexts in shapes:
        features = build_dsa_features(queries, contexts, model.index_topk)
        assert predictor.operator_times(features, "full_index").op_times == {
            "attn_dsa_indexer": 0.2, "attn_dsa_mla": 0.3,
        }
        assert predictor.operator_times(features, "shared_index").total_time() == 0.3
    # Non-round native chunks/contexts and heterogeneous calibration batches
    # use the existing fitted interpolation, not a synthetic measured row.
    for queries, contexts in [([192], [768]), ([64, 128], [384, 928]), ([768, 960], [512, 1024]),
                              ([64, 128], [32256, 32768])]:
        features = build_dsa_features(queries, contexts, model.index_topk)
        assert predictor.operator_times(features, "full_index").op_times == {
            "attn_dsa_indexer": 0.25, "attn_dsa_mla": 0.25,
        }
    for queries, contexts in [([2] * 1025, [0] * 1025), ([256], [33024]), ([512] * 17, [0] * 17)]:
        with pytest.raises(DSACoverageError, match="coverage missing"):
            predictor.operator_times(build_dsa_features(queries, contexts, model.index_topk), "full_index")


def test_attention_dp_dummy_forward_runs_only_outer_projection_bundle(model):
    from frontier.entities.batch import DummyForwardBatch
    from unittest.mock import ANY, Mock

    owner = SimpleNamespace(
        _model_config=model,
        _get_attention_layer_pre_proj_execution_time=lambda _: 0.1,
        _get_attention_layer_post_proj_execution_time=lambda _: 0.2,
        _get_attention_rope_execution_time=lambda _: 0.3,
        _get_attn_norm_layer_act_execution_time=Mock(return_value=0.4),
    )
    predictor = DSAPredictor({"attn_dsa_mla": SimpleNamespace()})
    timing = predictor.predict_attention_time(owner, DummyForwardBatch(0, 0, 0), 2)
    assert timing.operator_times.total_time() == 0
    assert timing.attention_layer_pre_proj_execution_time == 0.1
    assert timing.attention_layer_post_proj_execution_time == 0.2
    owner._get_attn_norm_layer_act_execution_time.assert_called_once_with(
        ANY, layer_id=2,
    )


def test_stage_metrics_retain_measured_scopes_without_dummy_reconstruction():
    from frontier.entities.time_components import AttentionOperatorTimes
    from frontier.scheduler.utils.dense_metrics import _restore_structural_dsa_attention

    layer = SimpleNamespace(attention_family_id="dsa_attention", attention_operator_times=AttentionOperatorTimes({
        "attn_dsa_indexer": 0.2, "attn_dsa_mla": 0.3,
    }))
    owner = SimpleNamespace(_enable_dummy_mode=False)
    assert _restore_structural_dsa_attention(layer, owner) is layer
    layer.attention_operator_times = None
    with pytest.raises(ValueError, match="executed attention scopes"):
        _restore_structural_dsa_attention(layer, owner)


def test_dsa_scopes_reuse_the_canonical_kernel_count_training(tmp_path, model):
    from frontier.execution_time_predictor.kernel_gap import KernelCountTraining

    training = KernelCountTraining()
    training._kernel_gap_ms = {"none": 0.1}
    training._active_measurement_type = MeasurementType.KERNEL_ONLY
    rows = _load(tmp_path, model, [_row(model, [1, 1], [128, 2048], prefill=False)])

    def fit(**kwargs):
        return training._paired_with_kernel_count_model(
            _fit(**kwargs), kwargs["model_name"], kwargs["df"], kwargs["feature_cols"], kwargs["target_col"],
            lambda name, df, columns, target: _fit(model_name=name, df=df, feature_cols=columns, target_col=target),
        )

    fitted = train_dsa_models(rows, fit)
    recorded = {}

    def record(name, estimator, key, columns):
        count = estimator._frontier_kernel_count_model
        recorded[count._frontier_operator_name] = count._frontier_exact_lookup[key]

    DSAPredictor(fitted).operator_times(build_dsa_features([1, 1], [128, 2048], model.index_topk),
                                      "full_index", record_kernel_count=record)
    assert recorded == {"attn_dsa_indexer": 5.0, "attn_dsa_mla": 7.0}


def test_dsa_metadata_registration_uses_only_the_child_reader(monkeypatch):
    from unittest.mock import Mock
    from frontier.attention.families import DSA_ATTENTION_FAMILY
    from frontier.attention.profiling_mapping import get_enabled_predictor_metric_names
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor

    predictor = SimpleNamespace()
    rows = pd.DataFrame({"measurement_type": ["CUDA_EVENT"]})
    predictor._load_attention_df = Mock(return_value=rows)
    predictor._get_profiling_metadata = Mock(return_value={})
    predictor._validate_active_measurement_type = Mock()
    predictor._register_profiling_metadata_for_ops = Mock()

    def reject_generic_read(*args, **kwargs):
        raise AssertionError("The generic metadata reader must not open DSA files")

    monkeypatch.setattr(pd, "read_csv", reject_generic_read)
    names = list(get_enabled_predictor_metric_names(DSA_ATTENTION_FAMILY))
    SklearnExecutionTimePredictor._register_profiling_metadata_from_file(predictor, "/profiles/attention.csv", names)
    predictor._load_attention_df.assert_called_once_with("/profiles/dsa_attention.csv")
    predictor._register_profiling_metadata_for_ops.assert_called_once_with(names, {}, "/profiles/dsa_attention.csv")


@pytest.fixture
def full_dummy_case(tmp_path, model, monkeypatch):
    """Entirely synthetic CPU input, never native qualification evidence."""
    import hashlib
    from frontier.config import ReplicaConfig, VllmV1SchedulerConfig, global_vars
    from frontier.execution_time_predictor.dsa_full_dummy_input import NATIVE_REVISION, SCHEMA

    replica = ReplicaConfig(model_name=model.get_name(), device="h800", network_device="h800_dgx",
        attn_tensor_parallel_size=8, attn_dp=2, moe_expert_parallel_size=16,
        kv_cache_dtype="auto", indexer_kv_cache_dtype="fp8")
    scheduler = VllmV1SchedulerConfig(block_size=64, num_blocks=128, num_blocks_mode="explicit",
        max_model_len=7168, max_tokens_in_batch=7168, batch_size_cap=8)
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_decode_only")
    monkeypatch.setattr(global_vars, "CUDAGRAPH_CAPTURE_SIZES", [1, 2, 4])
    sha = lambda raw: hashlib.sha256(raw).hexdigest()
    model_hash = sha((Path(__file__).resolve().parents[2] /
        "data/config/models/zai-org__GLM-5.3.json").read_bytes())
    versions = json.dumps({"vllm": "0.28.0", "torch": "synthetic-fixture"})
    shape = {"hidden_size": model.embedding_dim, "num_attention_heads": model.num_q_heads,
        "heads_per_rank": model.num_q_heads // 8, "target_tensor_parallel_size": 8,
        "dtype": model.torch_dtype, "model_architecture_profile": "glm_moe_dsa",
        **{name: getattr(model, name) for name in ("q_lora_rank", "kv_lora_rank", "qk_nope_head_dim",
            "qk_rope_head_dim", "v_head_dim", "index_topk", "index_n_heads", "index_head_dim")},
        "layers": [{"global_layer_id": i, "attention_variant": spec.variant_id,
                    "index_producer_layer": spec.index_producer_layer_id}
                   for i, spec in enumerate(model.get_layer_attention_specs())]}
    inventory, records = [], []
    for layer in shape["layers"]:
        i, variant = layer["global_layer_id"], layer["attention_variant"]
        mla, indexer = f"model.layers.{i}.self_attn.attn", f"model.layers.{i}.self_attn.indexer.k_cache"
        inventory.append({"global_layer_id": i, "mla_layer_name": mla,
                          "has_indexer": variant == "full_index", "skip_topk": variant == "shared_index"})
        for width in (1, 4):
            metadata = {mla: {"num_reqs": width, "num_actual_tokens": width, "max_query_len": 1,
                "max_seq_len": 7168, "block_size": 64, "topk_tokens": model.index_topk,
                "num_decodes": width, "num_prefills": 0, "num_decode_tokens": width, "prefill_max_seq_len": 0},
                indexer: {"max_seq_len": 7168, "num_decodes": width, "num_decode_tokens": width,
                          "num_prefills": 0, "num_prefill_tokens": 0}}
            witness = {"capture_width": width, "actual_query_tokens": 1, "actual_requests": 1,
                "cached_context_tokens": 0, "seq_lens": [1] + [0] * (width - 1),
                "query_start_loc": [0] + [1] * width, "slot_mappings": {"0": [-1] * width}, "metadata": metadata}
            for scope in (("attn_dsa_indexer", "attn_dsa_mla") if variant == "full_index" else ("attn_dsa_mla",)):
                for repeat in range(2):
                    records.append({"schema_version": SCHEMA, "measurement_type": "KERNEL_ONLY",
                        "execution_kind": "full_dummy", "graph_mode": "FULL", "global_layer_id": i,
                        "attention_variant": variant, "scope": scope, "repeat": repeat, "witness": witness,
                        "time_ms": 0.2 + repeat * 0.2, "kernel_count": 5 + repeat * 2})
    payload = {"schema_version": SCHEMA, "measurement_type": "KERNEL_ONLY", "gpu_execution_observed": True,
        "kernel_only_qualified": True, "consumer_admission": True, "native_revision": NATIVE_REVISION,
        "model_config_sha256": model_hash, "kv_cache_dtype": "auto", "capture_sizes": [1, 4],
        "resolved_capture_sizes": [1, 4], "plan": {"shape": shape, "hf_overrides": {},
            "limits": {"block_size": 64, "max_num_seqs": 8, "max_context_len": 7168, "decode_measured_steps": 2}},
        "identity": {"source_revision": NATIVE_REVISION, "schema_version": SCHEMA, "graph_mode": "FULL",
            "device_name": "NVIDIA H800", "device_capability": "sm90", "weight_source": "dummy",
            "quant_signature": model.get_quant_signature(), "model_config_sha256": model_hash,
            "runtime_versions": versions, "parallel_groups": {"engine_tensor_parallel": 8, "target_tensor_parallel": 8}},
        "inventory": {"attention_backend": "FLASH_ATTN_MLA_SPARSE", "attention_impl": "FlashAttnMLASparseImpl",
                      "layers": inventory}, "records": records}
    native = {"parallel_config": {"tensor_parallel_size": 8, "data_parallel_size": 2,
            "pipeline_parallel_size": 1, "enable_expert_parallel": True,
            "decode_context_parallel_size": 1, "prefill_context_parallel_size": 1},
        "cache_config": {"block_size": 64, "cache_dtype": "auto", "num_gpu_blocks": 128, "enable_prefix_caching": False},
        "compilation_config": {"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1, 2, 4]},
        "model_config": {"max_model_len": 7168, "dtype": model.torch_dtype, "quantization": "fp8"},
        "scheduler_config": {"max_num_seqs": 8,
            "max_num_batched_tokens": 7168, "enable_chunked_prefill": False}, "speculative_config": None}
    rows = [_row(model, [1], [0], variant) for variant in ("full_index", "shared_index") for _ in range(2)]
    for row in rows:
        row.update(runtime_versions=versions, device_name="NVIDIA H800", device_capability="sm90")
    serving = tmp_path / "dsa_attention_kernel_only.csv"
    pd.DataFrame(rows).to_csv(serving, index=False)
    path = tmp_path / "dsa_full_dummy.json"
    admission = {"schema_version": SCHEMA, "verdict": "PASS", "evidence": []}

    def write():
        path.write_text(json.dumps(payload))
        for key, name, raw in (("native_config", "native.json", json.dumps(native).encode()),
                               ("evidence", "qualification.txt", b"SYNTHETIC CPU FIXTURE; NOT NATIVE EVIDENCE")):
            (tmp_path / name).write_bytes(raw)
            reference = {"path": name, "sha256": sha(raw)}
            admission[key] = [reference] if key == "evidence" else reference
        admission["sidecar_sha256"] = sha(path.read_bytes())
        path.with_suffix(".admission.json").write_text(json.dumps(admission))

    def load():
        return load_dsa_dataframe(str(serving), model, 8, 64, MeasurementType.KERNEL_ONLY,
                                  replica_config=replica, scheduler_config=scheduler)

    write()
    return SimpleNamespace(payload=payload, native=native, admission=admission, path=path, serving=serving,
                           replica=replica, scheduler=scheduler, write=write, load=load)


@pytest.mark.parametrize("shared_manager", [False, True])
def test_full_dummy_real_loader_training_cache_and_runtime(full_dummy_case, tmp_path, monkeypatch, shared_manager):
    from sklearn.dummy import DummyRegressor
    from frontier.config import MetricsConfig, RandomForrestExecutionTimePredictorConfig, global_vars
    from frontier.config.quantization_manager import QuantizationManager
    from frontier.entities.batch import DecodeCudaGraphMetadata, DummyForwardBatch
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor
    from frontier.execution_time_predictor.profiling_dataframe_loaders import ProfilingDataFrameLoaders
    from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
    from frontier.types import ClusterType

    class Consumer(SklearnExecutionTimePredictor):
        estimators = 0

        def _get_estimator(self):
            self.estimators += 1
            return DummyRegressor()

        def _get_grid_search_params(self):
            return {}

    case = full_dummy_case
    monkeypatch.setattr(QuantizationManager, "_instance", None)
    monkeypatch.setattr(QuantizationManager, "_initialized", False)
    owner = Consumer(RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True, k_fold_cv_splits=2,
                     num_estimators=[1], max_depth=[2], min_samples_split=[2],
                     num_training_job_threads=1), case.replica, case.scheduler,
                     MetricsConfig(cache_dir=str(tmp_path / "cache")), cluster_type=ClusterType.MONOLITHIC)
    owner._enable_dummy_mode = False
    owner._active_measurement_type = MeasurementType.KERNEL_ONLY
    owner._kernel_gap_ms = {"cuda_graph": 0.1}
    owner._attention_input_file = str(case.serving)
    loaded = owner._load_attention_df(str(case.serving))
    alternate = ProfilingDataFrameLoaders._load_attention_df(owner, str(case.serving), case.replica, case.scheduler)
    pd.testing.assert_frame_equal(loaded, alternate)
    assert isinstance(loaded.attrs["full_dummy"], list)
    pd.concat([loaded, alternate])  # pandas metadata equality must remain scalar.
    train = owner._train_attention_layer_models
    if shared_manager:
        manager = ExecutionTimePredictionModelManager({}, MetricsConfig(cache_dir=owner._cache_dir))
        manager._active_measurement_type = owner._active_measurement_type
        manager._kernel_gap_ms = owner._kernel_gap_ms
        linear = pd.DataFrame([{"num_tokens": tokens, "num_tensor_parallel_workers": tp,
            "profiling_precision": case.replica.model_config.get_default_precision().name,
            "measurement_type": "KERNEL_ONLY"} for tp in (1, 8) for tokens in (1, 2)])
        for op in ("input_layernorm", "attn_pre_proj", "attn_post_proj", "attn_rope",
                   *case.replica.model_config.get_model_architecture_profile().predictor_attention_extra_ops):
            linear[f"time_stats.{op}.median"] = 0.1
            linear[f"time_stats.{op}.kernel_count"] = 1
        linear_file = tmp_path / "linear_op_kernel_only.csv"
        linear.to_csv(linear_file, index=False)
        train = lambda: manager._train_attn_models_for_cluster(
            ClusterType.DECODE, case.replica, owner._config, case.scheduler,
            str(linear_file), str(case.serving), set())
    owner._models = train()
    cold = {name: model._frontier_model_hash for name, model in owner._models.items()}
    assert owner.estimators > 0 or shared_manager
    cold_estimators = owner.estimators
    cached_files = {p: p.stat().st_mtime_ns for p in Path(owner._cache_dir).glob("*.pkl")}
    assert cached_files
    owner._models = train()
    assert cached_files == {p: p.stat().st_mtime_ns for p in Path(owner._cache_dir).glob("*.pkl")}
    assert owner.estimators == cold_estimators
    assert cold == {name: model._frontier_model_hash for name, model in owner._models.items()}
    if shared_manager:
        assert all(name in manager._models_by_precision_kernel_only["BF16"] for name in cold)
    owner._predictions = owner._predict_for_attention_layer_models()
    for name in ("_get_attention_layer_pre_proj_execution_time", "_get_attention_layer_post_proj_execution_time",
                 "_get_attention_rope_execution_time", "_get_attn_norm_layer_act_execution_time"):
        monkeypatch.setattr(owner, name, lambda *a, **kw: 0.0)  # Outer operators are outside this test's scopes.
    batch = DummyForwardBatch(0, 0, 0)
    for mode in ("FULL", "PIECEWISE"):
        batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata("full_decode_only", mode, True, False, 1, 4, 1, 4)
        for layer in (next(spec.global_layer_id for spec in case.replica.model_config.get_layer_attention_specs()
                           if spec.variant_id == variant) for variant in ("full_index", "shared_index")):
            with owner._counting_kernels() as counts:
                timing = owner._predict_attention_layer_time(batch, layer, ClusterType.MONOLITHIC)
            expected_scopes = 2 if case.replica.model_config.get_layer_attention_spec(layer).variant_id == "full_index" else 1
            assert timing.operator_times.total_time() == pytest.approx(0.3 * expected_scopes if mode == "FULL" else 0)
            assert sum(counts.operators.values()) == (6 * expected_scopes if mode == "FULL" else 0)
    batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata("full_decode_only", "FULL", True, False, 1, 2, 1, 2)
    with pytest.raises(DSACoverageError, match="capture width 2"):
        owner._predict_attention_layer_time(batch, 0, ClusterType.MONOLITHIC)
    case.payload["records"][0]["time_ms"] += 0.1
    case.write()
    changed = train()
    for name in cold:
        assert (cold[name] != changed[name]._frontier_model_hash) == name.endswith("full_dummy")

    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "piecewise")
    case.path.write_text("{")
    serving_only = owner._load_attention_df(str(case.serving))
    assert "full_dummy" not in serving_only.attrs
    owner._models = train()
    owner._predictions = owner._predict_for_attention_layer_models()
    ordinary = owner._predictions["dsa_attention"]
    features = build_dsa_features([1], [0], case.replica.model_config.index_topk)
    assert ordinary.operator_times(features, "full_index").total_time() == 0.5
    with pytest.raises(DSACoverageError, match="capture width 4"):
        ordinary.operator_times(features, "full_index", dummy_capture_width=4)


@pytest.mark.parametrize("fault", ["event", "flags", "source", "hardware", "tp", "shape", "cache", "compiled",
    "graph", "layout", "metadata", "indexer", "slots", "seq_lens", "missing", "duplicate", "layers", "repeat",
    "negative", "nan", "count", "owner", "model_hash", "receipt", "digest", "malformed", "evidence",
    "no_receipt", "runtime", "dtype", "backend", "mla", "false_kernel", "false_gpu", "quant", "prefix"])
def test_full_dummy_rejects_unqualified_or_incomplete_input(full_dummy_case, fault):
    case = full_dummy_case
    payload, native = case.payload, case.native
    row = payload["records"][0]
    if fault == "event": payload["measurement_type"] = "CUDA_EVENT"
    elif fault == "flags": payload["consumer_admission"] = False
    elif fault == "source": payload["native_revision"] = "other"
    elif fault == "hardware": case.replica.device = "h200"
    elif fault == "tp": payload["identity"]["parallel_groups"]["engine_tensor_parallel"] = 1
    elif fault == "shape": payload["plan"]["shape"]["hidden_size"] = 1
    elif fault == "cache": native["cache_config"]["num_gpu_blocks"] = 1
    elif fault == "compiled": native["compilation_config"]["mode"] = 0
    elif fault == "graph": native["compilation_config"]["cudagraph_capture_sizes"] = [1]
    elif fault == "layout": native["parallel_config"]["data_parallel_size"] = 1
    elif fault == "metadata": row["witness"]["metadata"] = {"nonempty": {}}
    elif fault == "indexer": row["witness"]["metadata"]["model.layers.0.self_attn.indexer.k_cache"]["num_decodes"] = 0
    elif fault == "slots": row["witness"]["slot_mappings"]["0"] = [0]
    elif fault == "seq_lens": row["witness"]["seq_lens"] = [2]
    elif fault == "missing": payload["records"].pop()
    elif fault == "duplicate": payload["records"].append(row)
    elif fault == "layers": payload["plan"]["shape"]["layers"].pop()
    elif fault == "repeat": row["repeat"] = 2
    elif fault == "negative": row["time_ms"] = -1
    elif fault == "nan": row["time_ms"] = float("nan")
    elif fault == "count": row["kernel_count"] = 1.5
    elif fault == "owner": row["attention_variant"] = "shared_index"
    elif fault == "model_hash": payload["model_config_sha256"] = "a" * 64
    elif fault == "receipt": case.admission["verdict"] = "FAIL"
    elif fault == "runtime": payload["identity"]["runtime_versions"] = '{"vllm":"0.28.0","torch":"other"}'
    elif fault == "dtype": native["model_config"]["dtype"] = "float16"
    elif fault == "backend": case.replica.attention_backend = "FLASHMLA_SPARSE"
    elif fault == "mla": row["witness"]["metadata"]["model.layers.0.self_attn.attn"]["num_actual_tokens"] = 2
    elif fault == "false_kernel": payload["kernel_only_qualified"] = False
    elif fault == "false_gpu": payload["gpu_execution_observed"] = False
    elif fault == "quant": native["model_config"]["quantization"] = None
    elif fault == "prefix": native["cache_config"]["enable_prefix_caching"] = True
    case.write()
    if fault == "digest": case.path.write_text(case.path.read_text() + " ")
    if fault == "malformed": case.path.write_text("{")
    if fault == "evidence": (case.path.parent / "qualification.txt").write_text("changed")
    if fault == "no_receipt": case.path.with_suffix(".admission.json").unlink()
    with pytest.raises(DSACoverageError, match="coverage missing"):
        case.load()


def test_existing_full_dummy_producer_plan_stays_unqualified(full_dummy_case):
    from frontier.profiling.attention.dsa_full_dummy import main
    from tests.unit.test_dsa_full_dummy_profiling import PLAN_ARGS

    case = full_dummy_case
    output = case.path.parent / "event-plan.json"
    main(["plan", *PLAN_ARGS, "--output", str(output)])
    case.path.write_bytes(output.read_bytes())
    with pytest.raises(DSACoverageError, match="gpu_execution_observed"):
        case.load()
    payload = json.loads(output.read_text())
    payload.update(gpu_execution_observed=True, kernel_only_qualified=True, consumer_admission=True)
    case.path.write_text(json.dumps(payload))
    with pytest.raises(DSACoverageError, match="timing method"):
        case.load()
