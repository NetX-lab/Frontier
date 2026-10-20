"""DP dummy forwards inherit the coordinated runtime mode, not host timing."""

from types import SimpleNamespace

import pytest

from frontier.config import global_vars
from frontier.entities import Batch, DummyForwardBatch, Request
from frontier.entities.batch import DecodeCudaGraphMetadata
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.scheduler.replica_scheduler.vllm_v1_iteration_policy import IterationSchedulingPolicy
from frontier.scheduler.utils.dp_dummy_forward import (
    resolve_dummy_forward_inputs,
    start_dummy_forward,
)
from frontier.types import ClusterType, MeasurementType


def decode_batch(width, runtime="FULL"):
    requests = [Request(0.0, 512, 4) for _ in range(width)]
    for request in requests:
        request._is_prefill_complete = True
        request._num_processed_tokens = 512
    batch = Batch(0, requests, [1] * width, is_moe=True)
    batch.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
        config_mode="full_decode_only", runtime_mode=runtime,
        capture_hit=runtime != "NONE", is_mixed_batch=False,
        original_total_tokens=width, padded_total_tokens=width,
        original_decode_batch_size=width, padded_decode_batch_size=width,
    )
    return batch


class PricingPredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        raise NotImplementedError

    def _get_grid_search_params(self):
        raise NotImplementedError


def predictor(role=ClusterType.MONOLITHIC):
    owner = object.__new__(PricingPredictor)
    owner._cluster_type = role
    owner._event_measurement_type_for_replica = lambda: MeasurementType.CUDA_EVENT
    return owner


def test_dummy_graph_pricing_waits_for_the_peer_shape(monkeypatch):
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", "full_decode_only")
    calls = []
    def predict(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            forward_launch_time=0.0,
            get_single_layer_attention_scope_time=lambda: 100.0,
        )
    scheduler = SimpleNamespace(
        _cluster_type=ClusterType.MONOLITHIC,
        _predictor=SimpleNamespace(
            _num_layers_per_pipeline_stage=78,
            predict_stage_execution_time=predict,
        ),
        get_pipeline_stage_layer_bounds=lambda *a: (0, 78),
    )
    dummy = DummyForwardBatch(0, 0, 0)
    dummy._stage_owner_replica_local_id = 1
    events = start_dummy_forward(scheduler, 1.0, 0, 0, dummy)
    assert calls == [], "The first dummy layer cannot be priced before DP mode/width sync"
    assert events[0].time == 1.0


def test_dummy_decode_metadata_selects_kernel_only(monkeypatch):
    monkeypatch.setattr(global_vars, "SYS_ARCH", "co-location")
    dummy = DummyForwardBatch(0, 0, 0)
    dummy.decode_cuda_graph_metadata = decode_batch(4).decode_cuda_graph_metadata
    assert predictor()._select_measurement_type_for_batch(dummy) is MeasurementType.KERNEL_ONLY


@pytest.mark.parametrize("role", [ClusterType.MONOLITHIC, ClusterType.DECODE])
def test_dummy_without_graph_metadata_selects_event_rows(role, monkeypatch):
    monkeypatch.setattr(global_vars, "SYS_ARCH", "pd-disaggregation")
    dummy = DummyForwardBatch(0, 0, 0)
    assert predictor(role)._select_measurement_type_for_batch(dummy) is MeasurementType.CUDA_EVENT


@pytest.mark.parametrize("runtime,width", [("FULL", 4), ("PIECEWISE", 4), ("NONE", 1)])
def test_dummy_first_layer_uses_dp_mode_and_projection_width(runtime, width):
    real = decode_batch(4, runtime)
    dummy = DummyForwardBatch(0, 0, 0)
    dummy.decode_cuda_graph_metadata = decode_batch(1).decode_cuda_graph_metadata
    dummy._dummy_dispatch_pending = True
    dummy.stage_start_time = 1.0
    seen = []

    def predict(batch, *args, **kwargs):
        family = predictor()._select_measurement_type_for_batch(batch)
        seen.append((family, batch.get_effective_total_tokens_for_compute(ClusterType.MONOLITHIC)))
        return SimpleNamespace(
            forward_launch_time=0.0,
            get_single_layer_attention_scope_time=lambda: 2.0,
        )

    scheduler = SimpleNamespace(
        _cluster_type=ClusterType.MONOLITHIC,
        _predictor=SimpleNamespace(predict_stage_execution_time=predict),
    )
    room = {"batches": {0: real, 1: dummy}, "arrival_times": {0: 1.01, 1: 1.0}}
    resolve_dummy_forward_inputs(scheduler, 0, 0, room)
    family = MeasurementType.CUDA_EVENT if runtime == "NONE" else MeasurementType.KERNEL_ONLY
    assert seen == [(family, width)]
    assert room["arrival_times"][1] == pytest.approx(1.002)
    assert dummy.total_num_tokens == 1
    assert dummy.requests == []
    assert not dummy._dummy_dispatch_pending
    # Only the first layer negotiates. Later layers retain the same descriptor.
    resolve_dummy_forward_inputs(scheduler, 0, 1, room)
    assert len(seen) == 1


def test_an_eager_peer_disables_dummy_graph_even_with_other_graph_peers():
    dummy = DummyForwardBatch(0, 0, 0)
    dummy.decode_cuda_graph_metadata = decode_batch(1).decode_cuda_graph_metadata
    dummy._dummy_dispatch_pending = True
    dummy.stage_start_time = 0.0
    scheduler = SimpleNamespace(
        _cluster_type=ClusterType.MONOLITHIC,
        _predictor=SimpleNamespace(predict_stage_execution_time=lambda *a, **kw: SimpleNamespace(
            forward_launch_time=0.0, get_single_layer_attention_scope_time=lambda: 1.0,
        )),
    )
    room = {"batches": {0: decode_batch(4), 1: decode_batch(2, "NONE"), 2: dummy},
            "arrival_times": {0: 0.1, 1: 0.2, 2: 0.0}}
    resolve_dummy_forward_inputs(scheduler, 0, 0, room)
    assert dummy.decode_cuda_graph_metadata is None
    assert dummy.get_effective_total_tokens_for_compute(ClusterType.MONOLITHIC) == 1


@pytest.mark.parametrize("mode,runtime", [("none", None), ("full_decode_only", "FULL"),
                                          ("piecewise", "PIECEWISE")])
def test_dummy_dispatch_is_uniform_decode(mode, runtime, monkeypatch):
    monkeypatch.setattr(global_vars, "DECODE_CUDA_GRAPH_MODE", mode)
    monkeypatch.setattr(global_vars, "CUDAGRAPH_CAPTURE_SIZES", [1, 2, 4])
    policy = IterationSchedulingPolicy()
    policy._cluster_type = ClusterType.DECODE
    metadata = policy._build_decode_cuda_graph_metadata(DummyForwardBatch(0, 0, 0))
    assert (None if metadata is None else metadata.runtime_mode) == runtime
    if metadata:
        assert metadata.original_total_tokens == 1
        assert not metadata.is_mixed_batch


@pytest.mark.parametrize("runtime,scopes", [("NONE", 0), ("PIECEWISE", 0), ("FULL", 2)])
def test_full_dummy_attention_has_one_actual_zero_context_request(runtime, scopes, monkeypatch):
    from frontier.config.model_config import BaseModelConfig
    from frontier.execution_time_predictor.dsa_predictor import DSAPredictor
    from frontier.entities.time_components import AttentionOperatorTimes

    monkeypatch.setattr(global_vars, "KV_CONNECTOR", "nixl")
    owner = predictor()
    owner._model_config = BaseModelConfig.create_from_name("zai-org/GLM-5.3")
    owner._enable_dummy_mode = False
    owner._log_architecture_attention_shape = lambda _: None
    owner._supports_operation = lambda _: True
    owner._get_attention_layer_pre_proj_execution_time = lambda _: 0.1
    owner._get_attention_layer_post_proj_execution_time = lambda _: 0.2
    owner._get_attention_rope_execution_time = lambda _: 0.3
    owner._get_attn_norm_layer_act_execution_time = lambda *a, **kw: 0.4
    owner._record_kernel_count = lambda *a: None
    dsa = DSAPredictor({"attn_dsa_mla": SimpleNamespace()})
    seen = []

    def operator_times(features, variant, **kwargs):
        seen.append((features, variant))
        return AttentionOperatorTimes({"attn_dsa_indexer": 0.5, "attn_dsa_mla": 0.6})

    dsa.operator_times = operator_times
    owner._predictions = {"dsa_attention": dsa}
    dummy = DummyForwardBatch(0, 0, 0)
    dummy.decode_cuda_graph_metadata = decode_batch(4, runtime).decode_cuda_graph_metadata
    result = owner._predict_attention_layer_time(dummy, 0, ClusterType.MONOLITHIC)
    assert len(seen) == (1 if runtime == "FULL" else 0)
    assert sum(value > 0 for value in result.operator_times.op_times.values()) == scopes
    if seen:
        features, variant = seen[0]
        assert features.values["num_query_tokens"] == 1
        assert features.values["sum_key_len"] == 1
        assert variant == "full_index"
