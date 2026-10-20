"""Sequence-parallel GLM shared-expert row selection and token widths."""

from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.config.parallel_semantics import sequence_parallel_moe_token_widths
from frontier.entities import Batch, Request
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.moe_ep_workload import materialize_layer_ep_workload, build_contiguous_expert_ownership
from frontier.scheduler.utils.batch_builders import build_ep_lane_batch, build_virtual_global_batch
from frontier.scheduler.utils.ep_wave_inputs import prepare_ep_wave_inputs
from frontier.scheduler.utils.ep_wave import prepare_moe_wave_from_inputs
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.operators.spec import TensorParallelMode
from frontier.operators.typed_contracts import serialize_typed_operator_contracts
from frontier.types import ClusterType


class Predictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def glm_predictor(*, dp=4, pp=1):
    predictor = Predictor.__new__(Predictor)
    predictor._model_config = BaseModelConfig.create_from_name("zai-org/GLM-5.3")
    predictor._cluster_type = ClusterType.PREFILL
    predictor._replica_config = SimpleNamespace(
        attn_tensor_parallel_size=8, attn_dp=dp, num_pipeline_stages=pp,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=8 * dp,
    )
    return predictor


@pytest.mark.parametrize("dp,pp,expected_tp", [(4, 1, 1), (1, 1, 8), (4, 2, 8)])
def test_glm_shared_expert_contract(dp, pp, expected_tp):
    predictor = glm_predictor(dp=dp, pp=pp)
    contract = predictor._resolve_typed_linear_contract("share_expert_up_proj")
    assert contract.tensor_parallel_size == expected_tp
    assert contract.tensor_parallel_mode is (
        TensorParallelMode.REPLICATED if expected_tp == 1
        else TensorParallelMode.ATTENTION_TP
    )
    assert predictor._get_linear_op_tp_key("share_expert_up_proj") == expected_tp


def test_glm_shared_expert_csv_selects_replicated_sp_row(tmp_path):
    predictor = glm_predictor()
    selected = predictor._resolve_typed_linear_contract("share_expert_up_proj")
    rows = []
    for dp, time_ms in [(1, 8.0), (4, 1.0)]:
        contract = glm_predictor(dp=dp)._resolve_typed_linear_contract("share_expert_up_proj")
        rows.append({
            "num_tensor_parallel_workers": contract.tensor_parallel_size,
            "n_expanded_embd": contract.effective_ffn_width,
            "typed_operator_contracts": serialize_typed_operator_contracts({
                "share_expert_up_proj": contract.typed_metadata_identity(),
            }),
            "num_tokens": 4,
            "time_stats.share_expert_up_proj.median": time_ms,
        })
    path = tmp_path / "linear_op.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    manager = object.__new__(ExecutionTimePredictionModelManager)
    loaded = manager._load_linear_op_df(
        str(path), selected.tensor_parallel_size,
        layer_contract=selected, operator_name="share_expert_up_proj",
    )
    assert loaded["time_stats.share_expert_up_proj.median"].tolist() == [1.0]


def test_glm_dense_layers_use_dense_ffn_rows():
    predictor = glm_predictor()
    predictor._enable_dummy_mode = False
    predictor._supports_operation = lambda name: name.startswith("mlp_")
    predictor._get_mlp_layer_up_proj_execution_time = lambda _: 12.288
    predictor._get_mlp_layer_down_proj_execution_time = lambda _: 6.144
    predictor._get_mlp_layer_act_execution_time = lambda _: 1.536
    predictor._get_mlp_norm_layer_act_execution_time = lambda _: 0.1
    batch = Batch(0, [Request(0.0, 9, 0)], [9], is_moe=True)
    assert predictor._requires_dense_mlp_compute_models()
    assert "mlp_up_proj" in predictor._get_compute_model_names()
    timing = predictor.predict_mlp_layer_time(batch, 0, ClusterType.PREFILL)
    assert timing.mlp_layer_up_proj_execution_time == 12.288
    dense_contract = predictor._resolve_typed_linear_contract("mlp_up_proj")
    assert dense_contract.effective_ffn_width == predictor._model_config.dense_mlp_hidden_dim
    assert dense_contract.tensor_parallel_size == 8


@pytest.mark.parametrize("tokens,local", [(1, 1), (7, 1), (8, 1), (9, 2), (8192, 1024)])
def test_sequence_parallel_widths(tokens, local):
    for rank in range(8):
        widths = sequence_parallel_moe_token_widths([tokens], 8, rank)
        assert widths.local_tokens == (local,)
        assert widths.valid_tokens == (max(0, min(local, tokens - rank * local)),)
        assert widths.gathered_tokens == 8 * local


def test_sequence_parallel_widths_unequal_lanes():
    widths = sequence_parallel_moe_token_widths([1, 7, 9, 8192], 8)
    assert widths.local_tokens == (1, 1, 2, 1024)
    assert widths.gathered_tokens == 8224


@pytest.mark.parametrize("layer_id,expected", [(0, 9), (2, 9), (3, 9), (4, 2), (77, 2)])
def test_glm_input_norm_width_after_dense_boundary(layer_id, expected):
    predictor = glm_predictor()
    predictor._get_prediction_for_features = lambda _name, features, **_: features["num_tokens"]
    batch = Batch(0, [Request(0.0, 9, 0)], [9], is_moe=True)
    assert predictor._get_attn_norm_layer_act_execution_time(batch, layer_id) == expected


class MoEPredictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def test_glm_ep_wave_uses_local_and_gathered_rows():
    predictor = MoEPredictor.__new__(MoEPredictor)
    config = glm_predictor()
    predictor._model_config = config._model_config
    predictor._replica_config = config._replica_config
    predictor._cluster_type = ClusterType.PREFILL
    predictor._router_topk = 1
    predictor._moe_ep_size = 32
    predictor._predictions = {name: {} for name in [
        "moe_gating_linear", "moe_gating_routing_topk", "moe_grouped_gemm",
        "share_expert_up_proj", "share_expert_act", "share_expert_down_proj",
    ]}
    predictor._select_moe_gating_prediction_model_name = lambda name, _batch: name
    predictor._get_prediction_for_features = lambda _name, features, **_: features["num_tokens"]
    batches = {
        d: Batch(0, [Request(0.0, tokens, 0)], [tokens], is_moe=True)
        for d, tokens in enumerate([1, 7, 8, 9])
    }
    inputs = prepare_ep_wave_inputs(
        source_batches=batches, batch=batches[0], step_id_getter=lambda _: 0,
        aggregate_batch_builder=build_virtual_global_batch, cluster_type=ClusterType.PREFILL,
    )
    lanes = []

    def materialize(**kwargs):
        assert kwargs["routing_token_count"] == 40
        return materialize_layer_ep_workload(
            routing_ratios={expert: 1 / 32 for expert in range(32)},
            target_replica_id=0, global_layer_id=3, routing_token_count=40,
            router_topk=1, total_expert_num=32, moe_expert_parallel_size=32,
            expert_to_ep=build_contiguous_expert_ownership(32, 32),
        )

    from frontier.entities import EPBatchGroup

    def lane_builder(**kwargs):
        return build_ep_lane_batch(
            **kwargs, cluster_type=ClusterType.PREFILL,
            create_batch_group=lambda requests, tokens, replica, ep, time, ids, workload:
                EPBatchGroup(requests, tokens, replica, ep, time, ids, workload,
                             ClusterType.PREFILL, is_moe=True),
        )

    def predict(batch, *args, **kwargs):
        local = 2 if batch.ep_id // 8 == 3 else 1
        assert batch.moe_pre_routing_effective_total_tokens == local
        assert predictor._get_gating_linear_time(batch) == local
        assert predictor._get_gating_routing_topk_time(batch) == local
        assert predictor._get_mlp_norm_layer_act_execution_time(batch) == local
        assert predictor._get_share_expert_up_proj_execution_time(batch) == local
        assert predictor._get_share_expert_act_execution_time(batch) == local
        assert predictor._get_share_expert_down_proj_execution_time(batch) == local
        assert predictor._get_grouped_gemm_time(batch.lane_workload, batch=batch) == 40
        assert predictor._resolve_typed_linear_tp_size("share_expert_up_proj") == 1
        assert batch.get_effective_total_tokens_rounded(ClusterType.PREFILL) == local
        lanes.append(batch)

    predictor.predict_stage_execution_time = predict
    noop = lambda **_: None
    plan = prepare_moe_wave_from_inputs(
        wave_inputs=inputs, time=0.0, materialize_workload=materialize,
        trace_identity_builder=noop, conservation_logger=noop, predictor=predictor,
        lane_builder=lane_builder, phase_getter=lambda *_, **__: (0., 0., 0., 0., 0.),
        workload_logger=noop, barrier_logger=noop, wave_logger=noop,
        cluster_type=ClusterType.PREFILL, replica_id=0, stage_id=0, layer_id=3,
    )
    assert len(lanes) == 32
    assert plan.layer_workload.routing_token_count == 40
