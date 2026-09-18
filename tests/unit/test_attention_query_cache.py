"""Stage-local numerical reuse with model-owned physical layer identities."""
from __future__ import annotations

from dataclasses import dataclass
import inspect

import pytest

from predictor_cache_fixtures import CacheFixtureDisaggregationPredictor, CacheFixturePredictor
from frontier.entities import Batch, ExecutionTime, Request
from frontier.entities.time_components import AttentionOperatorTimes, AttentionTime
from frontier.types import ClusterType, MeasurementType


@dataclass
class _Batch:
    """Explicit query inputs consumed by the deterministic numerical fixture."""

    id: int = 1
    size: int = 1
    num_tokens: int = 8
    phase: str = "prefill"
    context: int = 16
    padded_tokens: int = 8
    state_init_mode: str = "zero"


class _Predictor(CacheFixturePredictor):
    def __init__(self):
        super().__init__(hybrid=True)
        self._enable_dummy_mode = False
        self.artifact_coefficient = 1.0
        self.calls = []
        self.layer_calls = []
        self.observed_caches = []

    def _select_measurement_type_for_batch(self, batch):
        return MeasurementType.CUDA_EVENT

    def _require_predictions_for_measurement_type(self, measurement_type, batch):
        pass

    def _activate_measurement_type(self, measurement_type):
        pass

    def _emit_cuda_graph_activation_records(self, *args):
        pass

    def _get_moe_tokens_input(self, batch, layer_id):
        self.layer_calls.append(layer_id)
        return layer_id + 1

    def predict_attention_layer_time(self, *, batch, layer_id, cluster_type):
        spec = self._model_config.get_layer_attention_spec(layer_id)
        self.calls.append(layer_id)
        family_factor = 2 if spec.family_id == "gated_delta_net" else 3
        value = self.artifact_coefficient * family_factor * (
            batch.context + batch.padded_tokens
            + (10 if batch.phase == "decode" else 0)
            + (20 if batch.state_init_mode == "resume" else 0)
        )
        op = ("gdn_core_prefill" if batch.phase == "prefill" else "gdn_core_decode") if family_factor == 2 else ("attn_prefill" if batch.phase == "prefill" else "attn_decode")
        return AttentionTime(operator_times=AttentionOperatorTimes({op: value}))

    def _get_execution_time_internal(self, batch, pipeline_stage, *, layer_id,
                                     attention_query_cache, moe_tokens_input, **kwargs):
        if not self.observed_caches or self.observed_caches[-1] is not attention_query_cache:
            self.observed_caches.append(attention_query_cache)
        attention = self._predict_attention_layer_time_with_query_cache(
            batch=batch, layer_id=layer_id, cluster_type=ClusterType.MONOLITHIC,
            cache=attention_query_cache,
        )
        return ExecutionTime(
            num_layers_per_pipeline_stage=1,
            attention_rope_execution_time=0, attention_kv_cache_save_execution_time=0,
            attention_decode_execution_time=0, attention_prefill_execution_time=0,
            attention_layer_pre_proj_execution_time=0, attention_layer_post_proj_execution_time=0,
            attn_norm_time=0, mlp_norm_time=0, add_time=0,
            tensor_parallel_communication_time=0, pipeline_parallel_communication_time=0,
            expert_parallel_communication_time=0, moe_gating_time=0, moe_shuffling_time=0,
            schedule_time=0, sampler_e2e_time=0, prepare_inputs_e2e_time=0,
            process_model_outputs_time=0, ray_comm_time=0, is_moe=True,
            moe_grouped_gemm_time=float(moe_tokens_input),
            attention_operator_times=attention.operator_times,
        )


def _stage(predictor, batch):
    return predictor.predict_stage_execution_time(
        batch, stage_id=0, cluster_type=ClusterType.MONOLITHIC, num_layers=8,
    )


def test_stage_cache_matches_uncached_numerics_and_preserves_layer_ownership():
    predictor = _Predictor()
    batch = _Batch()
    stage = _stage(predictor, batch)
    assert predictor.calls == [0, 3]
    assert predictor.layer_calls == list(range(8))
    assert predictor._attention_query_cache_hits == 6
    assert predictor._attention_query_cache_misses == 2

    uncached = [predictor._predict_attention_layer_time_with_query_cache(
        batch=batch, layer_id=layer_id, cluster_type=ClusterType.MONOLITHIC,
    ) for layer_id in range(8)]
    assert predictor.calls == [0, 3, *range(8)]
    assert predictor._attention_query_cache_hits == 6
    assert predictor._attention_query_cache_misses == 2
    assert set(predictor.observed_caches[0]) == {
        (spec.family_id, spec.variant_id)
        for spec in predictor._model_config.get_layer_attention_specs()
    }
    assert [item.total_time() for item in uncached] == [48, 48, 48, 72, 48, 48, 48, 72]
    assert stage.model_time_ms == sum(item.total_time() for item in uncached) + 36
    for layer_id, (layer, expected) in enumerate(zip(stage.layer_execution_times, uncached)):
        spec = predictor._model_config.get_layer_attention_spec(layer_id)
        assert layer.global_layer_id == layer_id
        assert (layer.attention_family_id, layer.attention_variant_id) == (spec.family_id, spec.variant_id)
        assert layer.attention_time == expected.total_time()
        assert layer.moe_grouped_gemm_time == layer_id + 1


def test_cached_attention_payloads_have_independent_mutable_operator_maps():
    predictor = _Predictor()
    cache = {}
    values = [predictor._predict_attention_layer_time_with_query_cache(
        batch=_Batch(), layer_id=layer, cluster_type=ClusterType.MONOLITHIC, cache=cache,
    ) for layer in (0, 1)]
    values[0].operator_times.op_times["gdn_core_prefill"] = 999
    values[0].attention_prefill_execution_time = 999
    third = predictor._predict_attention_layer_time_with_query_cache(
        batch=_Batch(), layer_id=2, cluster_type=ClusterType.MONOLITHIC, cache=cache,
    )
    assert values[1].total_time() == third.total_time() == 48
    assert values[1] is not third
    assert values[1].operator_times is not third.operator_times
    assert predictor.calls == [0]


@pytest.mark.parametrize("change,expected", [
    ("phase", 648), ("context", 486), ("padding", 612),
    ("state", 828), ("artifact", 900),
])
def test_new_stage_recomputes_changed_query_and_artifact(change, expected):
    predictor = _Predictor()
    batch = _Batch()
    before = _stage(predictor, batch)
    if change == "phase":
        batch.phase = "decode"
    elif change == "context":
        batch.context = 17
    elif change == "padding":
        batch.padded_tokens = 16
    elif change == "state":
        batch.state_init_mode = "resume"
    else:
        predictor.artifact_coefficient = 2
    after = _stage(predictor, batch)
    assert before.model_time_ms == 468
    assert after.model_time_ms == expected
    assert predictor.calls == [0, 3, 0, 3]
    assert predictor._attention_query_cache_misses == 4
    assert predictor.observed_caches[0] is not predictor.observed_caches[1]


def test_attention_cache_lifetime_is_one_public_stage_call():
    predictor = _Predictor()
    for context in range(32):
        _stage(predictor, _Batch(context=context))
    caches = predictor.observed_caches
    assert len(caches) == len({id(cache) for cache in caches}) == 32
    assert all(len(cache) == 2 for cache in caches)
    assert "_attention_query_cache" not in predictor.__dict__
    assert predictor._attention_query_cache_misses == 64
    assert predictor._attention_query_cache_hits == 192
    assert "attention_query_cache" not in inspect.signature(predictor.predict_stage_execution_time).parameters


@pytest.mark.parametrize("disaggregated", [False, True])
def test_stage_owned_lookups_run_once_while_each_layer_keeps_attention(
    monkeypatch, disaggregated,
):
    predictor = (
        CacheFixtureDisaggregationPredictor() if disaggregated else CacheFixturePredictor()
    )
    role = ClusterType.PREFILL if disaggregated else ClusterType.MONOLITHIC
    predictor._enable_dummy_mode = False
    batch = Batch(
        replica_id=0,
        requests=[Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=1)],
        num_tokens=[8], is_moe=True,
    )
    for name in (
        "_require_predictions_for_measurement_type", "_activate_measurement_type",
        "_emit_cuda_graph_activation_records",
    ):
        monkeypatch.setattr(predictor, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        predictor, "predict_attention_layer_time",
        lambda *args, **kwargs: AttentionTime(attention_prefill_execution_time=2.0),
    )
    calls = {}
    for name in (
        "_get_schedule_time", "_get_sampler_e2e_time", "_get_prepare_inputs_e2e_time",
        "_get_process_model_outputs_time", "_get_ray_comm_time",
        "_get_pp_producer_send_path_runtime_time", "_get_pp_receiver_head_runtime_time",
        "_get_pp_prefill_consumer_active_runtime_time", "_get_pp_stage_boundary_handoff_time",
    ):
        calls[name] = 0

        def lookup(*args, name=name, **kwargs):
            calls[name] += 1
            return 1.0

        monkeypatch.setattr(predictor, name, lookup)
    stage = predictor.predict_stage_execution_time(
        batch, stage_id=0, cluster_type=role, num_layers=8, include_ffn=False,
    )
    assert calls == dict.fromkeys(calls, 1)
    assert stage.attention_time == 16.0
    assert [layer.attention_time for layer in stage.layer_execution_times] == [2.0] * 8
    assert [layer.schedule_time for layer in stage.layer_execution_times] == [1.0] + [0.0] * 7


@pytest.mark.parametrize("dummy_mode", [False, True])
def test_direct_gdn_prediction_requires_loaded_artifact_before_dense_lookup(monkeypatch, dummy_mode):
    predictor = CacheFixturePredictor(hybrid=True)
    predictor._enable_dummy_mode = dummy_mode
    batch = Batch(
        replica_id=0,
        requests=[Request(arrived_at=0.0, num_prefill_tokens=8, num_decode_tokens=1)],
        num_tokens=[8], is_moe=True,
    )

    def unexpected_lookup(*args, **kwargs):
        raise AssertionError("Missing GDN artifacts must fail before any numerical lookup")

    for name in (
        "_get_attention_prefill_execution_time", "_get_attention_decode_execution_time",
        "_get_attn_norm_layer_act_execution_time",
    ):
        monkeypatch.setattr(predictor, name, unexpected_lookup)
    with pytest.raises(ValueError, match="requires loaded GDN artifacts"):
        predictor.predict_attention_layer_time(batch, 0, ClusterType.MONOLITHIC)
