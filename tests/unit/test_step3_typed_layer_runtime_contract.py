"""RED coverage for Step3's profile-owned typed runtime layer contract."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from frontier.config.model_config import BaseModelConfig
from frontier.entities.time_components import AttentionTime
from frontier.execution_time_predictor.sklearn_moe_execution_time_predictor import (
    SklearnMoEExecutionTimePredictor,
)
from frontier.types import ClusterType


class _DummyMoEPredictor(SklearnMoEExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def _step3_config() -> BaseModelConfig:
    return BaseModelConfig.create_from_name("step3-moe-noquant")


def test_step3_mixed_identity_free_aggregate_fails_fast() -> None:
    """A mixed dense/routed stage cannot use one scalar MoE classification."""

    config = _step3_config()
    predictor = object.__new__(_DummyMoEPredictor)
    predictor._enable_dummy_mode = True
    predictor._dummy_execution_time = 1.0
    predictor._num_layers_per_pipeline_stage = config.num_layers
    predictor._model_config = config
    predictor._moe_ep_size = 1
    predictor._router_topk = 1
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._replica_config = SimpleNamespace(
        num_pipeline_stages=1,
        attn_tensor_parallel_size=8,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        attn_dp=1,
    )
    batch = SimpleNamespace(
        id=1,
        size=1,
        num_tokens=2,
        total_num_tokens=2,
        num_prefill_tokens=2,
        num_decode_tokens=0,
        requests=[],
        is_idle=False,
    )

    with pytest.raises(ValueError, match="mixed.*layer identity"):
        predictor.predict_stage_execution_time(
            batch,
            stage_id=0,
            cluster_type=ClusterType.MONOLITHIC,
            num_layers=config.num_layers,
            layer_id=0,
            layer_ids=tuple(range(config.num_layers)),
            include_ffn=True,
            include_attention=True,
        )


def test_step3_dense_boundary_uses_standard_mlp_getters() -> None:
    """Step3 dense boundary layers use the standard dense MLP profile family."""

    predictor = object.__new__(_DummyMoEPredictor)
    predictor._enable_dummy_mode = False
    predictor._model_config = _step3_config()
    predictor._supports_operation = lambda _operation: True
    predictor._get_mlp_layer_up_proj_execution_time = MagicMock(return_value=20.0)
    predictor._get_mlp_layer_down_proj_execution_time = MagicMock(return_value=30.0)
    predictor._get_mlp_layer_act_execution_time = MagicMock(return_value=40.0)
    predictor._get_share_expert_up_proj_execution_time = MagicMock(return_value=2.0)
    predictor._get_share_expert_down_proj_execution_time = MagicMock(return_value=3.0)
    predictor._get_share_expert_act_execution_time = MagicMock(return_value=4.0)
    predictor._get_mlp_norm_layer_act_execution_time = MagicMock(return_value=1.0)

    batch = SimpleNamespace(id=1, total_num_tokens=2, requests=[])
    result = predictor.predict_mlp_layer_time(
        batch,
        layer_id=0,
        cluster_type=ClusterType.MONOLITHIC,
    )

    assert result.mlp_layer_up_proj_execution_time == pytest.approx(20.0)
    assert result.mlp_layer_down_proj_execution_time == pytest.approx(30.0)
    assert result.mlp_layer_act_execution_time == pytest.approx(40.0)
    predictor._get_share_expert_up_proj_execution_time.assert_not_called()
    predictor._get_share_expert_down_proj_execution_time.assert_not_called()
    predictor._get_share_expert_act_execution_time.assert_not_called()


def test_step3_layer_aware_mlp_path_distinguishes_dense_and_routed_layers() -> None:
    """The profile selects shared auxiliary rows only for routed layers."""

    predictor = object.__new__(_DummyMoEPredictor)
    predictor._model_config = _step3_config()

    assert predictor._use_shared_expert_mlp_path(layer_id=0) is False
    assert predictor._use_shared_expert_mlp_path(layer_id=4) is True


def test_step3_requires_standard_dense_mlp_models() -> None:
    """Mixed Step3 configs must load genuine dense MLP rows at dense width."""

    predictor = object.__new__(_DummyMoEPredictor)
    predictor._model_config = _step3_config()

    assert predictor._requires_dense_mlp_compute_models() is True


def _internal_step3_predictor() -> _DummyMoEPredictor:
    """Build the minimal state needed to exercise the internal FFN seam."""

    predictor = object.__new__(_DummyMoEPredictor)
    predictor._enable_dummy_mode = False
    predictor._cluster_type = ClusterType.MONOLITHIC
    predictor._num_layers_per_pipeline_stage = 1
    predictor._model_config = _step3_config()
    predictor._replica_config = SimpleNamespace(
        num_pipeline_stages=1,
        attn_tensor_parallel_size=1,
        moe_tensor_parallel_size=1,
        moe_expert_parallel_size=1,
        attn_dp=1,
    )
    predictor.predict_attention_layer_time = lambda **_kwargs: AttentionTime()
    predictor._get_add_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_mlp_norm_layer_act_execution_time = lambda _batch: 0.0
    predictor._get_schedule_time = lambda _batch: 0.0
    predictor._get_sampler_e2e_time = lambda _batch: 0.0
    predictor._get_prepare_inputs_e2e_time = lambda _batch: 0.0
    predictor._get_process_model_outputs_time = lambda _batch: 0.0
    predictor._get_ray_comm_time = lambda _batch: 0.0
    predictor._get_pp_producer_send_path_runtime_time = lambda *_args: 0.0
    predictor._get_pp_receiver_head_runtime_time = lambda *_args: 0.0
    predictor._get_pp_prefill_consumer_active_runtime_time = lambda *_args: 0.0
    predictor._get_pp_stage_boundary_handoff_time = lambda *_args: 0.0
    predictor._get_mtp_terminal_overshoot_time = lambda *_args, **_kwargs: 0.0
    predictor._should_include_spec_decode_proposer_overhead = lambda _batch: False
    return predictor


def _internal_batch() -> SimpleNamespace:
    """Provide the fields used by the predictor's validation and trace paths."""

    return SimpleNamespace(
        id=1,
        size=1,
        num_tokens=2,
        total_num_tokens=2,
        num_prefill_tokens=2,
        num_decode_tokens=0,
        requests=[],
        is_idle=False,
    )


def test_internal_step3_dense_boundary_rejects_moe_selector_before_input_resolution() -> None:
    """The private seam must not force a dense boundary into routed MoE work."""

    predictor = _internal_step3_predictor()
    predictor._resolve_moe_execution_inputs = MagicMock(
        side_effect=AssertionError("input resolution reached before selector guard")
    )

    with pytest.raises(ValueError, match="include_moe does not match"):
        predictor._get_execution_time_internal(
            batch=_internal_batch(),
            pipeline_stage=0,
            moe_tokens_input=1,
            include_moe=True,
            include_ffn=True,
            include_attention=False,
            layer_id=0,
        )

    predictor._resolve_moe_execution_inputs.assert_not_called()


def test_internal_step3_routed_shared_only_path_remains_legal() -> None:
    """A routed layer may request its shared auxiliary FFN without routing."""

    predictor = _internal_step3_predictor()
    predictor._get_share_expert_up_proj_execution_time = lambda _batch: 1.0
    predictor._get_share_expert_down_proj_execution_time = lambda _batch: 2.0
    predictor._get_share_expert_act_execution_time = lambda _batch: 3.0
    predictor._get_mlp_layer_up_proj_execution_time = lambda _batch: (
        (_ for _ in ()).throw(AssertionError("standard dense lookup reached"))
    )
    predictor._get_mlp_layer_down_proj_execution_time = lambda _batch: (
        (_ for _ in ()).throw(AssertionError("standard dense lookup reached"))
    )
    predictor._get_mlp_layer_act_execution_time = lambda _batch: (
        (_ for _ in ()).throw(AssertionError("standard dense lookup reached"))
    )

    execution_time = predictor._get_execution_time_internal(
        batch=_internal_batch(),
        pipeline_stage=0,
        include_moe=False,
        include_ffn=True,
        include_attention=False,
        layer_id=4,
    )

    assert execution_time._is_moe is False
    assert execution_time._mlp_layer_up_proj_execution_time == pytest.approx(1.0)
    assert execution_time._mlp_layer_down_proj_execution_time == pytest.approx(2.0)
    assert execution_time._mlp_layer_act_execution_time == pytest.approx(3.0)


def test_internal_step3_identity_free_mixed_aggregate_fails_before_input_resolution() -> None:
    """A mixed Step3 aggregate needs explicit global layer identities."""

    predictor = _internal_step3_predictor()
    predictor._resolve_moe_execution_inputs = MagicMock(
        side_effect=AssertionError("input resolution reached before aggregate guard")
    )

    with pytest.raises(ValueError, match="mixed-model FFN prediction requires explicit layer identity"):
        predictor._get_execution_time_internal(
            batch=_internal_batch(),
            pipeline_stage=0,
            include_moe=False,
            include_ffn=True,
            include_attention=False,
            num_layers=61,
            layer_id=0,
        )

    predictor._resolve_moe_execution_inputs.assert_not_called()
