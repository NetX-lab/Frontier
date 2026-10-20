"""Structural DSA cost contracts; synthetic durations are not native evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import frontier.entities.time_components as time_components
from frontier.attention.families import DSA_ATTENTION_FAMILY
from frontier.attention.ops import (
    AttentionOperatorRole,
    AttentionOperatorSpec,
    AttentionPhase,
    ProjectionOwnership,
)
from frontier.attention.trace_mapping import get_attention_trace_op_times
from frontier.config.model_config import BaseModelConfig
from frontier.entities import ExecutionTime, StageExecutionTime
from frontier.metrics.constants import OperationMetrics
from frontier.metrics.data_series import DataSeries
from frontier.metrics.metrics_store import MetricsStore
from frontier.metrics.op_trace_utils import OpTraceContext, compute_op_trace_meta
from frontier.operators.spec import ResourceClass
from frontier.types import ClusterType, MeasurementType


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def enabled_scope_fixture(monkeypatch: pytest.MonkeyPatch):
    """Exercise canonical time plumbing without enabling production support."""
    phases = (AttentionPhase.PREFILL, AttentionPhase.DECODE, AttentionPhase.MIXED)
    family = replace(
        DSA_ATTENTION_FAMILY,
        dsa_frozen=False,
        execution_enabled=True,
        operators=(
            AttentionOperatorSpec(
                name="attn_dsa_indexer",
                role=AttentionOperatorRole.INDEXING,
                phases=phases,
                execution_time_attr="attn_dsa_indexer_time",
                resource_class=ResourceClass.COMP,
                projection_ownership=ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE,
            ),
            AttentionOperatorSpec(
                name="attn_dsa_mla",
                role=AttentionOperatorRole.ATTENTION,
                phases=phases,
                execution_time_attr="attn_dsa_mla_time",
                resource_class=ResourceClass.COMP,
                projection_ownership=ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE,
            ),
        ),
        disjoint_model_projection_attrs=(
            "attention_layer_pre_proj_execution_time",
            "attention_layer_post_proj_execution_time",
        ),
    )
    original = tuple(time_components.iter_execution_enabled_families(include_structural=True))
    monkeypatch.setattr(
        time_components, "iter_execution_enabled_families",
        lambda **_: tuple(
            family if item.family_id == family.family_id else item for item in original
        ),
    )
    time_components._canonical_operator_execution_time_attr_items.cache_clear()
    yield family
    time_components._canonical_operator_execution_time_attr_items.cache_clear()


def _layer(layer_id: int, *, indexer_ms: float = 5.0, **overrides) -> ExecutionTime:
    kwargs = dict(
        num_layers_per_pipeline_stage=1,
        global_layer_id=layer_id,
        attention_family_id="dsa_attention",
        attention_variant_id="full_index" if indexer_ms else "shared_index",
        attention_rope_execution_time=3.0,
        attention_kv_cache_save_execution_time=0.0,
        attention_decode_execution_time=0.0,
        attention_prefill_execution_time=0.0,
        attention_layer_pre_proj_execution_time=1.0,
        attention_layer_post_proj_execution_time=2.0,
        attn_norm_time=4.0,
        mlp_norm_time=0.0,
        add_time=0.0,
        tensor_parallel_communication_time=0.0,
        pipeline_parallel_communication_time=0.0,
        expert_parallel_communication_time=0.0,
        moe_gating_time=0.0,
        moe_shuffling_time=0.0,
        schedule_time=0.0,
        sampler_e2e_time=0.0,
        prepare_inputs_e2e_time=0.0,
        process_model_outputs_time=0.0,
        ray_comm_time=0.0,
        is_moe=False,
        op_times={"attn_dsa_indexer": indexer_ms, "attn_dsa_mla": 7.0},
    )
    kwargs.update(overrides)
    return ExecutionTime(**kwargs)


def test_inclusive_scopes_round_trip_without_double_count(enabled_scope_fixture):
    full = _layer(0, attn_dsa_indexer_time=500.0, attn_dsa_mla_time=700.0)
    shared = _layer(1, indexer_ms=0.0)
    stage = StageExecutionTime((full, shared))

    assert full.attn_dsa_indexer_time == 5.0
    assert full.attn_dsa_mla_time == 7.0
    assert full.attention_time_component.total_time() == 22.0
    assert full.attention_time == 22.0
    assert shared.attention_time == 17.0
    assert stage.attn_dsa_indexer_time == 5.0
    assert stage.attn_dsa_mla_time == 14.0
    assert stage.attention_time == stage.model_time_ms == 39.0
    assert stage.attention_time_component.total_time() == 39.0
    assert dict(stage.attention_operator_times.op_times) == {
        "attn_dsa_indexer": 5.0,
        "attn_dsa_mla": 14.0,
    }
    restored = _layer(0, op_times=dict(full.op_times))
    assert restored.attention_time == full.attention_time
    assert restored.op_times == full.op_times
    assert [(op.name, value) for op, value in get_attention_trace_op_times(
        shared, enabled_scope_fixture, skip_zero=False
    )] == [("attn_dsa_indexer", 0.0), ("attn_dsa_mla", 7.0)]
    assert [op.name for op, _ in get_attention_trace_op_times(
        shared, enabled_scope_fixture
    )] == ["attn_dsa_mla"]


def test_all_backbone_layers_keep_full_index_and_shared_costs(
    enabled_scope_fixture, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.chdir(REPO_ROOT)
    model = BaseModelConfig.create_from_name("zai-org/GLM-5.3")
    specs = model.get_layer_attention_specs()
    stage = StageExecutionTime(tuple(
        _layer(spec.global_layer_id, indexer_ms=(
            5.0 if spec.variant_id == "full_index" else 0.0
        )) for spec in specs
    ))

    assert stage.num_layers == 78
    assert stage.attn_dsa_indexer_time == 21 * 5.0
    assert stage.attn_dsa_mla_time == 78 * 7.0
    assert stage.attention_time == 78 * 10.0 + 21 * 5.0 + 78 * 7.0


@pytest.mark.parametrize("values", [
    None, {}, {"attn_dsa_mla": 7.0}, {"attn_dsa_indexer": 5.0},
])
def test_inclusive_trace_rejects_missing_indexer_timing(
    enabled_scope_fixture, values,
):
    layer = _layer(0, op_times=values)
    with pytest.raises(ValueError, match="missing structured attention operator"):
        get_attention_trace_op_times(layer, enabled_scope_fixture)


def test_inclusive_trace_rejects_extra_registered_attention_scope(
    enabled_scope_fixture,
):
    layer = _layer(0, op_times={
        "attn_dsa_indexer": 5.0, "attn_dsa_mla": 7.0, "attn_prefill": 2.0,
    })
    with pytest.raises(ValueError, match="Unexpected attention timing.*attn_prefill"):
        get_attention_trace_op_times(layer, enabled_scope_fixture)


def test_unknown_or_negative_scope_duration_fails(enabled_scope_fixture):
    with pytest.raises(ValueError, match="Unsupported ExecutionTime operator timing"):
        _layer(0, op_times={"attn_dsa_typo": 1.0})
    with pytest.raises(ValueError, match="Negative ExecutionTime operator timing"):
        _layer(0, op_times={"attn_dsa_indexer": -1.0, "attn_dsa_mla": 7.0})


@pytest.mark.parametrize("attr", [
    "attention_kv_cache_save_execution_time", "attention_prefill_execution_time",
    "attention_decode_execution_time", "attn_mla_kv_cache_save_time",
    "attn_mla_prefill_kv_up_proj_time", "attn_mla_prefill_time",
    "attn_mla_decode_q_latent_proj_time", "attn_mla_decode_time",
    "attn_mla_v_up_proj_time",
])
def test_inclusive_trace_rejects_legacy_child_charge(enabled_scope_fixture, attr):
    layer = _layer(0, **{attr: 0.1})
    with pytest.raises(ValueError, match="also charges dense/MLA child scopes"):
        get_attention_trace_op_times(layer, enabled_scope_fixture)


@pytest.fixture
def trace_context(monkeypatch: pytest.MonkeyPatch) -> OpTraceContext:
    monkeypatch.chdir(REPO_ROOT)
    return OpTraceContext(
        cluster_type=ClusterType.MONOLITHIC,
        model_config=BaseModelConfig.create_from_name("zai-org/GLM-5.3"),
        replica_config=SimpleNamespace(attn_tensor_parallel_size=8),
        total_tokens=16,
        effective_tokens_compute=16,
        effective_tokens_transfer=16,
        effective_tokens_rounded=16,
        tokens_are_post_routing=False,
    )


def test_indexer_trace_separates_inputs_from_replicated_internal_shapes(trace_context):
    meta = compute_op_trace_meta("attn_dsa_indexer", "COMPUTE", trace_context)
    assert meta["dtype"] == "BF16"
    assert meta["tensor_shape"] == {
        "hidden_states": [16, 6144], "qr": [16, 2048], "positions": [16],
    }
    assert meta["tensor_dtype"]["positions"] == "INT64"
    assert meta["tensor_size_bytes"]["positions"] == 16 * 8
    assert meta["tensor_size_bytes"]["qr"] == 16 * 2048 * 2
    assert meta["internal_tensor_shape"] == {
        "index_q_before_quant": [16, 32, 128],
        "index_k_before_quant": [16, 128],
        "selected_indices_active": [16, 2048],
    }
    assert meta["internal_tensor_dtype"]["selected_indices_active"] == "INT32"


def test_mla_trace_uses_local_query_heads_and_native_output_width(trace_context):
    meta = compute_op_trace_meta("attn_dsa_mla", "COMPUTE", trace_context)
    assert meta["tensor_shape"] == {
        "q": [16, 8, 256], "kv_c_normed": [16, 512],
        "k_pe": [16, 1, 64], "output": [16, 2048],
    }
    assert meta["dtype"] == "BF16"
    assert meta["tensor_size_bytes"]["output"] == 16 * 2048 * 2


def test_outer_trace_uses_actual_sparse_mla_projection_and_rope_shapes(trace_context):
    pre = compute_op_trace_meta("attn_pre_proj", "COMPUTE", trace_context)
    rope = compute_op_trace_meta("attn_rope", "COMPUTE", trace_context)
    post = compute_op_trace_meta("attn_post_proj", "COMPUTE", trace_context)
    assert pre["tensor_shape"] == {
        "input": [16, 6144], "q_c": [16, 2048],
        "kv_c_normed": [16, 512], "q": [16, 8, 256], "k_pe": [16, 1, 64],
    }
    assert rope["tensor_shape"]["q"] == [16, 8, 64]
    assert rope["tensor_shape"]["k"] == [16, 1, 64]
    assert post["tensor_shape"] == {"input": [16, 2048], "output": [16, 6144]}


def test_trace_requires_source_dimensions_and_derives_activation_dtype(trace_context):
    config = trace_context.model_config
    config.torch_dtype = "float32"
    meta = compute_op_trace_meta("attn_dsa_mla", "COMPUTE", trace_context)
    assert meta["dtype"] == "FP32"
    assert meta["tensor_size_bytes"]["output"] == 16 * 2048 * 4
    config.index_n_heads = None
    with pytest.raises(ValueError, match="requires positive int index_n_heads"):
        compute_op_trace_meta("attn_dsa_indexer", "COMPUTE", trace_context)


def test_legacy_csv_omits_absent_dsa_series(tmp_path):
    legacy = DataSeries("batch_id", "attn_prefill")
    legacy.put(0, 3.0)
    absent = [DataSeries("batch_id", metric.value) for metric in (
        OperationMetrics.ATTN_DSA_INDEXER, OperationMetrics.ATTN_DSA_MLA,
    )]
    store = MetricsStore.__new__(MetricsStore)
    store._config = SimpleNamespace(save_table_to_wandb=False)
    store._save_as_csv([legacy, *absent], "batch_id", str(tmp_path), "legacy")
    assert list(pd.read_csv(tmp_path / "legacy.csv").columns) == [
        "batch_id", "attn_prefill",
    ]


def test_production_family_requires_measured_data_at_prediction():
    assert DSA_ATTENTION_FAMILY.execution_enabled
    assert not DSA_ATTENTION_FAMILY.dsa_frozen
    DSA_ATTENTION_FAMILY.require_enabled_for_execution()


def test_measured_dsa_trace_uses_the_same_scopes_as_structural_trace():
    layer = _layer(0)
    measured = get_attention_trace_op_times(layer, DSA_ATTENTION_FAMILY)
    values = get_attention_trace_op_times(
        layer, DSA_ATTENTION_FAMILY, allow_structural=True,
    )
    assert measured == values
    assert [(op.name, value) for op, value in values] == [
        ("attn_dsa_indexer", 5.0), ("attn_dsa_mla", 7.0),
    ]


def test_inclusive_projection_ownership_requires_disjoint_declaration():
    family = replace(DSA_ATTENTION_FAMILY, disjoint_model_projection_attrs=())
    with pytest.raises(ValueError, match="Invalid attention projection ownership"):
        get_attention_trace_op_times(_layer(0), family, allow_structural=True)


@pytest.mark.parametrize("model_name", [
    "deepseek-ai/DeepSeek-V4-Flash",
])
def test_unmeasured_family_rejects_csv_loading_and_standalone_training(
    model_name, monkeypatch, tmp_path,
):
    from frontier.execution_time_predictor.profiling_dataframe_loaders import (
        ProfilingDataFrameLoaders,
    )
    from frontier.training.attention_trainer import AttentionTrainer

    monkeypatch.chdir(REPO_ROOT)
    model = BaseModelConfig.create_from_name(model_name)
    with pytest.raises(NotImplementedError, match="DSA|DeepSeek-V4"):
        ProfilingDataFrameLoaders()._load_attention_df(
            str(tmp_path / "missing.csv"), SimpleNamespace(model_config=model),
            SimpleNamespace(block_size=64),
        )
    with pytest.raises(NotImplementedError, match="DSA|DeepSeek-V4"):
        AttentionTrainer(
            str(tmp_path / "missing.csv"), str(tmp_path / "trained"),
            model_name, "h800",
        )
    assert not (tmp_path / "trained").exists()


def test_measured_dsa_needs_isolated_input_and_shared_training(tmp_path, monkeypatch):
    from frontier.execution_time_predictor.profiling_dataframe_loaders import ProfilingDataFrameLoaders
    from frontier.training.attention_trainer import AttentionTrainer

    monkeypatch.chdir(REPO_ROOT)
    model_name = "zai-org/GLM-5.3"
    model = BaseModelConfig.create_from_name(model_name)
    loader = ProfilingDataFrameLoaders()
    loader._active_measurement_type = MeasurementType.KERNEL_ONLY
    with pytest.raises(FileNotFoundError, match="dsa_attention_kernel_only"):
        loader._load_attention_df(str(tmp_path / "attention_kernel_only.csv"),
                                  SimpleNamespace(model_config=model, attn_tensor_parallel_size=8),
                                  SimpleNamespace(block_size=64))
    with pytest.raises(NotImplementedError, match="shared model manager"):
        AttentionTrainer(str(tmp_path / "attention.csv"), str(tmp_path / "trained"), model_name, "h800")
    assert not (tmp_path / "trained").exists()
