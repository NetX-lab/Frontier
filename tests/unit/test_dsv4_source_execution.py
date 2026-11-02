"""Pinned SP component contracts, without native or full-serving claims."""

from types import SimpleNamespace

import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.entities import Batch, Request
from frontier.entities.batch import DecodeCudaGraphMetadata, EPBatchGroup
from frontier.entities.time_components import MoETime
from frontier.attention.families import DEEPSEEK_V4_ATTENTION_FAMILY
from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.operators.families import DEEPSEEK_V4_MHC_FAMILY
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.types import ClusterType


class ComponentPredictor(SklearnMoEExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        raise NotImplementedError


@pytest.fixture
def predictor():
    result = ComponentPredictor.__new__(ComponentPredictor)
    result._model_config = BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")
    result._replica_config = SimpleNamespace(
        num_pipeline_stages=1, attn_tensor_parallel_size=8, attn_dp=8,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=64,
        attention_backend="FLASHMLA_SPARSE_DSV4",
        speculative_decoding_config=SimpleNamespace(enabled=False),
    )
    result._cluster_type = ClusterType.MONOLITHIC
    result._enable_dummy_mode = True
    result._dsv4_step_costs = None
    result._dummy_execution_time = 2.0
    result._cc_backend = RecordingCCBackend()
    return result


def source_batch(tokens=17):
    return Batch(
        replica_id=0, requests=[Request(0.0, tokens + 120, 4)],
        num_tokens=[tokens], num_context_tokens=[120],
        request_is_decoding=[False], is_moe=True,
    )


def component(predictor, batch, layer=0, rank=0, **overrides):
    context = dict(
        execution_mode="eager",
        prefix_caching=False, decode_context_parallel_size=1,
        prefill_context_parallel_size=1, enable_expert_parallel=True,
        moe_backend="auto", all2all_backend="allgather_reducescatter",
        skip_padding=True, input_padding_mask=None, selected_logits_rows=None,
    )
    context.update(overrides)
    return predictor._predict_dsv4_dummy_source_execution(
        batch, layer, rank, **context,
    )


def test_source_rank_distinguishes_physical_valid_and_unknown_routing(predictor):
    result = component(predictor, source_batch(), rank=5)
    assert result["status"] == "PARTIAL"
    assert result["source_rows"] == 17
    assert result["physical_local_rows"] == 3
    assert result["padded_rows"] == 24
    assert result["valid_local_rows"] == 2
    assert result["padding_mask_rows"] == 1
    assert result["routed_assignments"] is None
    assert isinstance(result["shared_ffn_time"], MoETime)
    assert result["shared_ffn_time"].total_time() == 6.0
    assert result["component_times"]["dsv4_mhc_first_pre"] == 2.0


@pytest.mark.parametrize("tokens,physical,padded,valid", [
    (1, 1, 8, (1, 0, 0, 0, 0, 0, 0, 0)),
    (17, 3, 24, (3, 3, 3, 3, 3, 2, 0, 0)),
])
def test_padded_ranks_retain_local_compute(predictor, tokens, physical, padded, valid):
    outputs = [component(predictor, source_batch(tokens), rank=r) for r in range(8)]
    assert tuple(row["valid_local_rows"] for row in outputs) == valid
    assert {row["physical_local_rows"] for row in outputs} == {physical}
    assert {row["padded_rows"] for row in outputs} == {padded}
    assert {row["shared_ffn_time"].total_time() for row in outputs} == {6.0}
    assert outputs[-1]["workloads"]["share_expert_up_proj"] == {
        "input_shape": (physical, 4096), "output_shape": (physical, 4096),
        "weight_tp": 1,
    }
    first = outputs[-1]["workloads"]["dsv4_mhc_first_pre"]
    assert first["input_shape"] == (physical, 4096)
    assert first["residual_shape"] == (physical, 4, 4096)
    assert first["post_mix_shape"] == (physical, 4, 1)
    assert first["res_mix_shape"] == (physical, 4, 4)


class RecordingCCBackend:
    def __init__(self):
        self.calls = []

    def predict_allreduce(self, **kwargs):
        self.calls.append(("allreduce", kwargs))
        return 7.0

    def predict_allgather(self, **kwargs):
        self.calls.append(("allgather", kwargs))
        return 3.0

    def predict_reduce_scatter(self, **kwargs):
        self.calls.append(("reduce_scatter", kwargs))
        return 5.0


def test_ordered_consumer_reuses_attention_and_real_cc_dispatch(predictor):
    predictor._cc_backend = backend = RecordingCCBackend()
    batch = source_batch()
    result = component(predictor, batch, layer=2, rank=7)
    assert tuple(step["scope"] for step in result["source_order"]) == (
        "dsv4_mhc_ffn_post_attn_pre", "dsv4_sp_attention_allgather",
        "attention", "dsv4_sp_attention_reduce_scatter",
        "dsv4_mhc_attn_post_ffn_pre", "share_expert_up_proj",
        "share_expert_act", "share_expert_down_proj",
    )
    assert [(kind, value["data_size_bytes"]) for kind, value in backend.calls] == [
        ("allgather", 24576), ("reduce_scatter", 139264),
    ]
    assert all(value["comm_domain"] == "ATTN_TP" for _, value in backend.calls)
    assert result["attention_time"].total_time() == 32.0
    assert result["selected_component_time_ms"] == 50.0
    assert result["workloads"]["dsv4_mhc_ffn_post_attn_pre"]["input_shape"] == (3, 4096)
    assert result["workloads"]["dsv4_mhc_attn_post_ffn_pre"]["output_shape"] == (3, 4096)
    assert batch.num_tokens == [17]
    assert batch.total_num_tokens == 17
    assert result["omitted_scopes"]
    assert result["routed_assignments"] is None


def test_full_model_endpoints_and_43_layer_boundary_conservation(predictor):
    batch = source_batch()
    outputs = [component(
        predictor, batch, layer=i, rank=5,
        selected_logits_rows=2 if i == 42 else None,
    ) for i in range(43)]
    names = [name for row in outputs for name in row["component_times"]]
    for name in (
        "dsv4_embedding", "dsv4_mhc_first_pre", "dsv4_mhc_final_post",
        "dsv4_sp_final_mhc_allgather", "dsv4_mhc_head", "dsv4_final_norm",
        "dsv4_lm_head",
    ):
        assert names.count(name) == 1
    assert names.count("dsv4_mhc_ffn_post_attn_pre") == 42
    assert names.count("dsv4_mhc_attn_post_ffn_pre") == 43
    assert names.count("dsv4_sp_attention_allgather") == 43
    assert names.count("dsv4_sp_attention_reduce_scatter") == 43
    assert names.count("share_expert_up_proj") == 43
    assert len(predictor._cc_backend.calls) == 87
    first = outputs[0]
    assert first["source_order"][0]["scope"] == "dsv4_embedding"
    assert first["workloads"]["dsv4_embedding"]["output_shape"] == (17, 4096)
    last = outputs[-1]
    assert tuple(step["scope"] for step in last["source_order"][-5:]) == (
        "dsv4_mhc_final_post", "dsv4_sp_final_mhc_allgather", "dsv4_mhc_head",
        "dsv4_final_norm", "dsv4_lm_head",
    )
    assert last["workloads"]["dsv4_mhc_final_post"]["output_shape"] == (3, 4, 4096)
    assert last["workloads"]["dsv4_mhc_head"]["input_shape"] == (17, 4, 4096)
    assert last["workloads"]["dsv4_final_norm"]["input_shape"] == (17, 4096)
    assert last["workloads"]["dsv4_lm_head"]["input_shape"] == (2, 4096)
    assert predictor._cc_backend.calls[-1][1]["data_size_bytes"] == 98304
    assert all(row["status"] == "PARTIAL" for row in outputs)
    assert all(row["routed_assignments"] is None for row in outputs)
    assert all(row["selected_component_time_ms"] == sum(
        step["time_ms"] for step in row["source_order"]
    ) for row in outputs)
    assert "embedding internal vocab reduction" in first["omitted_scopes"]
    assert "logits gathering and sampling" in last["omitted_scopes"]
    assert batch.num_tokens == [17]


def test_logit_rows_are_explicit_and_zero_omits_head(predictor):
    with pytest.raises(ValueError, match="selected_logits_rows"):
        component(predictor, source_batch(), layer=42)
    zero = component(predictor, source_batch(), layer=42, selected_logits_rows=0)
    assert "dsv4_lm_head" not in zero["component_times"]
    for invalid in (-1, True, 1.5, 18):
        with pytest.raises(ValueError, match="selected_logits_rows"):
            component(predictor, source_batch(), layer=42, selected_logits_rows=invalid)
    with pytest.raises(ValueError, match="last layer"):
        component(predictor, source_batch(), layer=3, selected_logits_rows=1)


@pytest.mark.parametrize("overrides,match", [
    ({"enable_expert_parallel": False}, "expert parallel"),
    ({"enable_expert_parallel": 1}, "expert parallel"),
    ({"moe_backend": "deep_gemm_mega_moe"}, "ordinary"),
    ({"moe_backend": "triton"}, "ordinary"),
    ({"all2all_backend": "deepep_high_throughput"}, "all2all"),
    ({"input_padding_mask": [False] * 17}, "input padding"),
    ({"skip_padding": 1}, "skip_padding"),
    ({"execution_mode": "capture"}, "eager"),
    ({"prefix_caching": True}, "prefix"),
    ({"decode_context_parallel_size": 2}, "DCP1"),
    ({"prefill_context_parallel_size": 2}, "PCP1"),
])
def test_explicit_native_scope_guards(predictor, overrides, match):
    with pytest.raises((ValueError, NotImplementedError), match=match):
        component(predictor, source_batch(), **overrides)


def test_skip_padding_does_not_remove_physical_compute_or_qualify_routing(predictor):
    masked = component(predictor, source_batch(), rank=7)
    unmasked = component(predictor, source_batch(), rank=7, skip_padding=False)
    assert masked["valid_local_rows"] == unmasked["valid_local_rows"] == 0
    assert masked["padding_mask_rows"] == 3
    assert unmasked["padding_mask_rows"] == 0
    assert masked["component_times"] == unmasked["component_times"]
    assert masked["routed_assignments"] is unmasked["routed_assignments"] is None


@pytest.mark.parametrize("field,value", [
    ("num_pipeline_stages", 2), ("attn_tensor_parallel_size", 1),
    ("attn_tensor_parallel_size", True), ("attn_dp", 1),
    ("moe_tensor_parallel_size", 2), ("moe_expert_parallel_size", 8),
    ("attention_backend", None), ("attention_backend", "FLASHINFER"),
])
def test_topology_uses_existing_sp_and_attention_guards(predictor, field, value):
    setattr(predictor._replica_config, field, value)
    with pytest.raises((ValueError, NotImplementedError)):
        component(predictor, source_batch())


@pytest.mark.parametrize("rank", [-1, 8, True, 1.5])
def test_rank_must_identify_a_source_tp_member(predictor, rank):
    with pytest.raises(ValueError, match="source TP rank"):
        component(predictor, source_batch(), rank=rank)


@pytest.mark.parametrize("field,value", [
    ("_num_tokens", []), ("_num_tokens", [True]), ("_num_tokens", [0]),
    ("_num_context_tokens", [-1]), ("_request_is_decoding", [True]),
    ("_total_num_tokens", 24),
])
def test_snapshot_validation_precedes_source_row_computation(predictor, field, value):
    batch = source_batch()
    setattr(batch, field, value)
    with pytest.raises(ValueError):
        component(predictor, batch)
    assert predictor._cc_backend.calls == []


def test_actual_eager_state_and_request_snapshot_are_preserved(predictor):
    batch = source_batch(1)
    batch.requests[0]._num_processed_tokens = 999
    batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
        "full_decode_only", "NONE", False, False, 1, 1, 1, 1,
    )
    result = component(predictor, batch, layer=3, rank=7)
    assert result["source_rows"] == 1
    assert result["attention_time"].total_time() == 22.0
    batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
        "full_decode_only", "FULL", True, False, 1, 8, 1, 8,
    )
    with pytest.raises(ValueError, match="graph"):
        component(predictor, batch)


def test_pd_spec_and_ep_destination_are_not_source_batches(predictor):
    for name, match in (
        ("spec_decode_metadata", "speculative"), ("afd_stage_metadata", "PD"),
        ("lane_workload", "source DP"),
    ):
        batch = source_batch()
        setattr(batch, name, object())
        with pytest.raises((ValueError, NotImplementedError), match=match):
            component(predictor, batch)
    with pytest.raises(ValueError, match="source DP"):
        component(predictor, EPBatchGroup.__new__(EPBatchGroup))
    predictor._replica_config.speculative_decoding_config.enabled = True
    with pytest.raises(NotImplementedError, match="speculative"):
        component(predictor, source_batch())


def test_dimensions_follow_config_and_weights_remain_replicated(predictor):
    predictor._model_config.embedding_dim = 2048
    predictor._model_config.share_expert_dim = 1024
    result = component(predictor, source_batch(), rank=7)
    assert result["workloads"]["share_expert_up_proj"] == {
        "input_shape": (3, 2048), "output_shape": (3, 2048), "weight_tp": 1,
    }
    assert result["workloads"]["share_expert_down_proj"]["input_shape"] == (3, 1024)
    assert result["workloads"]["dsv4_mhc_first_pre"]["residual_shape"] == (3, 4, 2048)
    assert predictor._cc_backend.calls[0][1]["data_size_bytes"] == 12288
    assert predictor._cc_backend.calls[1][1]["data_size_bytes"] == 69632


def test_structural_admission_keeps_normal_execution_and_training_closed(predictor):
    batch = source_batch()
    predictor._enable_dummy_mode = False
    with pytest.raises(DeepSeekV4CoverageError, match="dsv4_embedding"):
        component(predictor, batch)
    predictor._enable_dummy_mode = True
    result = predictor.predict_attention_layer_time(batch, 0, ClusterType.MONOLITHIC)
    assert "dsv4_sparse_attention" in result.operator_times.op_times
    assert DEEPSEEK_V4_ATTENTION_FAMILY.structural_execution_enabled is True
    for family in (DEEPSEEK_V4_ATTENTION_FAMILY, DEEPSEEK_V4_MHC_FAMILY):
        assert family.execution_enabled is False
        assert all(not op.predictor_target and not op.e2e_trace_target
                   for op in family.operators)
        assert all(op.profiling_target == (family is DEEPSEEK_V4_MHC_FAMILY)
                   for op in family.operators)


def test_component_requires_v4_and_valid_layer(predictor):
    for layer in (-1, 43, True):
        with pytest.raises((ValueError, IndexError)):
            component(predictor, source_batch(), layer=layer)
    predictor._model_config = BaseModelConfig.create_from_name("meta-llama/Llama-2-7b-hf")
    with pytest.raises(ValueError, match="profile"):
        component(predictor, source_batch())


def test_missing_cc_backend_preserves_moe_error_policy(predictor):
    predictor._cc_backend = None
    with pytest.raises(NotImplementedError, match="all-gather prediction not implemented"):
        component(predictor, source_batch())


def test_pd_cluster_is_rejected_without_a_phase_or_stage_result(predictor):
    predictor._cluster_type = ClusterType.PREFILL
    with pytest.raises((ValueError, NotImplementedError), match="co-location"):
        component(predictor, source_batch())
