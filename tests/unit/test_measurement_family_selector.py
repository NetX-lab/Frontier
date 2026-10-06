"""Unit tests for measurement-family selection and manager family views."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.config import global_vars
from frontier.config.device_sku_config import H800DeviceSKUConfig
from frontier.entities import EPBatchGroup
from frontier.execution_time_predictor.measurement_input_paths import (
    uses_two_stream_eager_pricing,
)
from frontier.moe_ep_workload import materialize_layer_ep_workload
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.scheduler.utils.batch_builders import build_ep_lane_batch
from frontier.types import ClusterType, MeasurementType


def _make_predictor(
    cluster_type: ClusterType,
    runtime_mode: str | None = "NONE",
    two_stream_eager_pricing: bool = False,
):
    """Build a selector-only predictor; ``runtime_mode=None`` reads the batch's graph mode."""
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class DummyPredictor(SklearnExecutionTimePredictor):
        def __init__(self):
            self._cluster_type = cluster_type
            self._replica_config = SimpleNamespace(device_config=H800DeviceSKUConfig())
            self._two_stream_eager_pricing = two_stream_eager_pricing
            if runtime_mode is not None:
                self._get_decode_cuda_graph_runtime_mode = lambda _batch: runtime_mode

        def _get_estimator(self):
            return None

        def _get_grid_search_params(self):
            return {}

    return DummyPredictor()


@pytest.fixture(autouse=True)
def _reset_cuda_graph_config():
    global_vars.set_global_vars("offline", "co-location")
    global_vars.set_cuda_graph_config(False, None, "none")
    yield
    global_vars.set_global_vars("offline", "co-location")
    global_vars.set_cuda_graph_config(False, None, "none")


def _build_scheduler(cluster_type: ClusterType) -> VLLMv1EngineReplicaScheduler:
    scheduler = object.__new__(VLLMv1EngineReplicaScheduler)
    scheduler._cluster_type = cluster_type
    scheduler._replica_id = 3
    scheduler._replica_is_moe = False
    scheduler._num_stages = 1
    scheduler._batch_creation_counter = 0
    scheduler._replica_local_id = 0
    scheduler._max_batch_size = 64
    scheduler._max_num_running_reqs = 64
    scheduler._spec_decode_enabled = True
    scheduler._spec_decode_config = SimpleNamespace(num_speculative_tokens=2)
    scheduler._build_spec_decode_batch_metadata = lambda _batch: None
    return scheduler


def _build_spec_request(request_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        id=request_id,
        is_prefill_complete=True,
        is_decoding=True,
        is_recomputing=False,
        spec_decode_enabled=True,
        current_thinking_round_index=0,
        num_restarts=0,
        execution_epoch=0,
        execution_signature=(0, 0, 0),
        current_decode_token_index=0,
        num_context_tokens=16,
        is_thinking_mode_enabled=False,
        thinking_home_cluster_type=None,
    )


def test_monolithic_pure_decode_uses_kernel_only_when_decode_graph_active() -> None:
    predictor = _make_predictor(ClusterType.MONOLITHIC, runtime_mode="PIECEWISE")
    batch = SimpleNamespace(num_prefill_tokens=0, num_decode_tokens=8)

    assert predictor._select_measurement_type_for_batch(batch) == MeasurementType.KERNEL_ONLY


def test_monolithic_prefill_and_true_mixed_use_eager() -> None:
    predictor = _make_predictor(ClusterType.MONOLITHIC, runtime_mode="FULL")

    prefill_batch = SimpleNamespace(num_prefill_tokens=16, num_decode_tokens=0)
    mixed_batch = SimpleNamespace(num_prefill_tokens=8, num_decode_tokens=8)

    assert predictor._select_measurement_type_for_batch(prefill_batch) == MeasurementType.CUDA_EVENT
    assert predictor._select_measurement_type_for_batch(mixed_batch) == MeasurementType.CUDA_EVENT


UNIFORM_ROUTING = {expert: 1.0 for expert in range(4)}


def _forward_step(
    num_prefill_tokens: int, num_decode_tokens: int, runtime_mode: str | None
) -> SimpleNamespace:
    return SimpleNamespace(
        id=1,
        global_id=1,
        replica_id=0,
        time=0.0,
        total_num_tokens=num_prefill_tokens + num_decode_tokens,
        num_prefill_tokens=num_prefill_tokens,
        num_decode_tokens=num_decode_tokens,
        decode_cuda_graph_metadata=(
            None if runtime_mode is None else SimpleNamespace(runtime_mode=runtime_mode)
        ),
    )


def _ep_lane(
    step: SimpleNamespace,
    *,
    ep_id: int,
    routing_ratios: dict[int, float],
    moe_expert_parallel_size: int,
) -> EPBatchGroup:
    workload = materialize_layer_ep_workload(
        routing_ratios=routing_ratios,
        target_replica_id=0,
        global_layer_id=0,
        routing_token_count=step.total_num_tokens,
        router_topk=2,
        total_expert_num=4,
        moe_expert_parallel_size=moe_expert_parallel_size,
        expert_to_ep={expert: expert * moe_expert_parallel_size // 4 for expert in range(4)},
    )
    return build_ep_lane_batch(
        source_batch=step,
        layer_id=0,
        ep_id=ep_id,
        layer_workload=workload,
        create_batch_group=lambda *args: EPBatchGroup(
            *args, cluster_type=ClusterType.MONOLITHIC, is_moe=True
        ),
        cluster_type=ClusterType.MONOLITHIC,
    )


def test_monolithic_ep_lane_of_graph_decode_step_uses_kernel_only() -> None:
    predictor = _make_predictor(ClusterType.MONOLITHIC, runtime_mode=None)
    lane = _ep_lane(
        _forward_step(0, 4, "FULL"),
        ep_id=0,
        routing_ratios=UNIFORM_ROUTING,
        moe_expert_parallel_size=1,
    )

    # The lane's routed tokens count as prefill; its forward step is pure decode.
    assert lane.num_prefill_tokens == 8
    assert predictor._select_measurement_type_for_batch(lane) == MeasurementType.KERNEL_ONLY
    assert predictor._should_strip_collective_sim_allreduce_launch_overhead(lane) is True


def test_monolithic_ep_lane_of_mixed_step_uses_eager() -> None:
    predictor = _make_predictor(ClusterType.MONOLITHIC, runtime_mode=None)
    lane = _ep_lane(
        _forward_step(8, 4, None),
        ep_id=0,
        routing_ratios=UNIFORM_ROUTING,
        moe_expert_parallel_size=1,
    )

    assert predictor._select_measurement_type_for_batch(lane) == MeasurementType.CUDA_EVENT


def test_monolithic_zero_token_ep_lane_of_graph_decode_step_uses_kernel_only() -> None:
    predictor = _make_predictor(ClusterType.MONOLITHIC, runtime_mode=None)
    lane = _ep_lane(
        _forward_step(0, 4, "FULL"),
        ep_id=1,
        routing_ratios={0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0},
        moe_expert_parallel_size=2,
    )

    assert lane.total_num_tokens == 0
    assert predictor._select_measurement_type_for_batch(lane) == MeasurementType.KERNEL_ONLY


def test_specialized_clusters_use_eager_when_cuda_graph_disabled() -> None:
    prefill_predictor = _make_predictor(ClusterType.PREFILL)
    decode_predictor = _make_predictor(ClusterType.DECODE)
    decode_attn_predictor = _make_predictor(ClusterType.DECODE_ATTN)
    decode_ffn_predictor = _make_predictor(ClusterType.DECODE_FFN)
    global_vars.set_cuda_graph_config(
        use_cuda_graph=False,
        cudagraph_capture_sizes=None,
        decode_cuda_graph_mode="none",
    )
    batch = SimpleNamespace(num_prefill_tokens=0, num_decode_tokens=4)

    assert prefill_predictor._select_measurement_type_for_batch(batch) == MeasurementType.CUDA_EVENT
    assert decode_predictor._select_measurement_type_for_batch(batch) == MeasurementType.CUDA_EVENT
    assert decode_attn_predictor._select_measurement_type_for_batch(batch) == MeasurementType.CUDA_EVENT
    assert decode_ffn_predictor._select_measurement_type_for_batch(batch) == MeasurementType.CUDA_EVENT


def test_pd_af_decode_clusters_use_reference_measurement_families_without_cuda_graph() -> None:
    global_vars.set_global_vars("offline", "pd-af-disaggregation")
    decode_attn_predictor = _make_predictor(ClusterType.DECODE_ATTN)
    decode_ffn_predictor = _make_predictor(ClusterType.DECODE_FFN)

    pure_decode_batch = SimpleNamespace(num_prefill_tokens=0, num_decode_tokens=4)
    mixed_batch = SimpleNamespace(num_prefill_tokens=8, num_decode_tokens=4)

    assert (
        decode_attn_predictor._select_measurement_type_for_batch(pure_decode_batch)
        == MeasurementType.KERNEL_ONLY
    )
    assert (
        decode_attn_predictor._select_measurement_type_for_batch(mixed_batch)
        == MeasurementType.CUDA_EVENT
    )
    assert (
        decode_ffn_predictor._select_measurement_type_for_batch(pure_decode_batch)
        == MeasurementType.KERNEL_ONLY
    )


def test_eager_baselines_do_not_enable_kernel_only_families() -> None:
    prefill_predictor = _make_predictor(ClusterType.PREFILL)
    decode_predictor = _make_predictor(ClusterType.DECODE)
    decode_attn_predictor = _make_predictor(ClusterType.DECODE_ATTN)
    decode_ffn_predictor = _make_predictor(ClusterType.DECODE_FFN)
    global_vars.set_cuda_graph_config(
        use_cuda_graph=False,
        cudagraph_capture_sizes=None,
        decode_cuda_graph_mode="none",
    )

    try:
        assert prefill_predictor._get_default_measurement_type_for_cluster() == MeasurementType.CUDA_EVENT
        assert decode_predictor._get_default_measurement_type_for_cluster() == MeasurementType.CUDA_EVENT
        assert decode_attn_predictor._get_default_measurement_type_for_cluster() == MeasurementType.CUDA_EVENT
        assert decode_ffn_predictor._get_default_measurement_type_for_cluster() == MeasurementType.CUDA_EVENT

        assert prefill_predictor._should_enable_measurement_family(MeasurementType.CUDA_EVENT) is True
        assert prefill_predictor._should_enable_measurement_family(MeasurementType.KERNEL_ONLY) is False

        assert decode_predictor._should_enable_measurement_family(MeasurementType.CUDA_EVENT) is True
        assert decode_predictor._should_enable_measurement_family(MeasurementType.KERNEL_ONLY) is False

        assert decode_attn_predictor._should_enable_measurement_family(MeasurementType.CUDA_EVENT) is True
        assert decode_attn_predictor._should_enable_measurement_family(MeasurementType.KERNEL_ONLY) is False

        assert decode_ffn_predictor._should_enable_measurement_family(MeasurementType.CUDA_EVENT) is True
        assert decode_ffn_predictor._should_enable_measurement_family(MeasurementType.KERNEL_ONLY) is False
    finally:
        global_vars.reset_global_vars()


def test_decode_cluster_uses_kernel_only_when_decode_cuda_graph_enabled() -> None:
    global_vars.set_cuda_graph_config(False, [1, 2, 4], "piecewise")
    decode_predictor = _make_predictor(ClusterType.DECODE)
    batch = SimpleNamespace(num_prefill_tokens=0, num_decode_tokens=4)

    assert decode_predictor._select_measurement_type_for_batch(batch) == MeasurementType.KERNEL_ONLY


def test_pd_af_decode_clusters_use_kernel_only_when_cuda_graph_enabled() -> None:
    global_vars.set_cuda_graph_config(True, [1, 2, 4], "none")
    decode_attn_predictor = _make_predictor(ClusterType.DECODE_ATTN)
    decode_ffn_predictor = _make_predictor(ClusterType.DECODE_FFN)
    batch = SimpleNamespace(num_prefill_tokens=0, num_decode_tokens=4)

    assert decode_attn_predictor._select_measurement_type_for_batch(batch) == MeasurementType.KERNEL_ONLY
    assert decode_ffn_predictor._select_measurement_type_for_batch(batch) == MeasurementType.KERNEL_ONLY

def test_cuda_graph_enabled_clusters_enable_kernel_only_families() -> None:
    decode_predictor = _make_predictor(ClusterType.DECODE)
    decode_attn_predictor = _make_predictor(ClusterType.DECODE_ATTN)
    decode_ffn_predictor = _make_predictor(ClusterType.DECODE_FFN)
    global_vars.set_cuda_graph_config(
        use_cuda_graph=True,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        decode_cuda_graph_mode="full_decode_only",
    )

    try:
        assert decode_predictor._get_default_measurement_type_for_cluster() == MeasurementType.KERNEL_ONLY
        assert decode_attn_predictor._get_default_measurement_type_for_cluster() == MeasurementType.KERNEL_ONLY
        assert decode_ffn_predictor._get_default_measurement_type_for_cluster() == MeasurementType.KERNEL_ONLY

        assert decode_predictor._should_enable_measurement_family(MeasurementType.CUDA_EVENT) is False
        assert decode_predictor._should_enable_measurement_family(MeasurementType.KERNEL_ONLY) is True

        assert decode_attn_predictor._should_enable_measurement_family(MeasurementType.CUDA_EVENT) is False
        assert decode_attn_predictor._should_enable_measurement_family(MeasurementType.KERNEL_ONLY) is True

        assert decode_ffn_predictor._should_enable_measurement_family(MeasurementType.CUDA_EVENT) is False
        assert decode_ffn_predictor._should_enable_measurement_family(MeasurementType.KERNEL_ONLY) is True
    finally:
        global_vars.reset_global_vars()


def test_monolithic_spec_mixed_batch_uses_eager_when_full_decode_only_cannot_dispatch() -> None:
    predictor = _make_predictor(ClusterType.MONOLITHIC, runtime_mode="NONE")
    scheduler = _build_scheduler(ClusterType.MONOLITHIC)
    global_vars.set_cuda_graph_config(
        use_cuda_graph=False,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        decode_cuda_graph_mode="full_decode_only",
    )

    try:
        batch = scheduler._create_batch(
            [_build_spec_request(1), _build_spec_request(2), _build_spec_request(3)],
            [3, 1, 2],
        )
        assert batch.decode_cuda_graph_metadata is None
        assert predictor._select_measurement_type_for_batch(batch) == MeasurementType.CUDA_EVENT
    finally:
        global_vars.reset_global_vars()


def test_monolithic_spec_batch_stays_eager_under_piecewise_when_spec_decode_disables_cuda_graph() -> None:
    predictor = _make_predictor(ClusterType.MONOLITHIC, runtime_mode="NONE")
    scheduler = _build_scheduler(ClusterType.MONOLITHIC)
    global_vars.set_cuda_graph_config(
        use_cuda_graph=False,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        decode_cuda_graph_mode="piecewise",
    )

    try:
        batch = scheduler._create_batch(
            [_build_spec_request(1), _build_spec_request(2), _build_spec_request(3)],
            [3, 1, 2],
        )
        assert batch.decode_cuda_graph_metadata is None
        assert predictor._select_measurement_type_for_batch(batch) == MeasurementType.CUDA_EVENT
    finally:
        global_vars.reset_global_vars()


def test_monolithic_spec_batch_uses_kernel_only_under_piecewise_with_diagnostic_opt_in() -> None:
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    predictor = _make_predictor(ClusterType.MONOLITHIC, runtime_mode="NONE")
    predictor._get_decode_cuda_graph_runtime_mode = (
        SklearnExecutionTimePredictor._get_decode_cuda_graph_runtime_mode.__get__(
            predictor, type(predictor)
        )
    )
    scheduler = _build_scheduler(ClusterType.MONOLITHIC)
    global_vars.set_cuda_graph_config(
        use_cuda_graph=False,
        cudagraph_capture_sizes=[1, 2, 4, 8, 16],
        decode_cuda_graph_mode="piecewise",
        allow_spec_decode_cuda_graph_diagnostic=True,
    )

    try:
        batch = scheduler._create_batch(
            [_build_spec_request(1), _build_spec_request(2), _build_spec_request(3)],
            [3, 1, 2],
        )
        assert batch.decode_cuda_graph_metadata is not None
        assert batch.decode_cuda_graph_metadata.runtime_mode == "PIECEWISE"
        assert predictor._select_measurement_type_for_batch(batch) == MeasurementType.KERNEL_ONLY
    finally:
        global_vars.reset_global_vars()


class _DummyModelConfig:
    def get_name(self) -> str:
        return "meta-llama/Llama-2-7b-hf"


def _make_manager():
    from frontier.execution_time_predictor.shared_prediction_model_manager import (
        ExecutionTimePredictionModelManager,
    )

    manager = ExecutionTimePredictionModelManager({}, SimpleNamespace(cache_dir="."))
    manager._all_dummy_mode = False
    manager._trained_models_eager = {"attn_prefill": object()}
    manager._trained_models_kernel_only = {"attn_decode": object()}

    replica_config = SimpleNamespace(
        device="a100",
        network_device="a100_pairwise_nvlink",
        model_config=_DummyModelConfig(),
    )
    predictor_config = SimpleNamespace(
        linear_op_input_file="compute/{DEVICE}/{MODEL}.csv",
        mlp_input_file="",
        atten_input_file="attention/{DEVICE}/{MODEL}.csv",
        moe_input_file="moe/{DEVICE}/{MODEL}.csv",
        all_reduce_input_file="network/{NETWORK_DEVICE}/all_reduce.csv",
        send_recv_input_file="network/{NETWORK_DEVICE}/send_recv.csv",
        cpu_overhead_input_file="cpu/{DEVICE}/{MODEL}.csv",
        cpu_overhead_kernel_only_input_file="cpu_kernel/{DEVICE}/{MODEL}.csv",
        pp_stage_boundary_input_file="other/{DEVICE}/{MODEL}/pp_stage_boundary.csv",
        pp_receiver_head_input_file="other/{DEVICE}/{MODEL}/pp_receiver_head.csv",
        pp_producer_send_path_input_file="other/{DEVICE}/{MODEL}/pp_producer_send_path.csv",
        pp_prefill_consumer_active_input_file="other/{DEVICE}/{MODEL}/pp_prefill_consumer_active.csv",
        linear_op_kernel_only_input_file="compute_kernel/{DEVICE}/{MODEL}.csv",
        atten_kernel_only_input_file="attention_kernel/{DEVICE}/{MODEL}.csv",
        moe_kernel_only_input_file="moe_kernel/{DEVICE}/{MODEL}.csv",
        kernel_gap_input_file="compute/{DEVICE}/{MODEL}/kernel_gap.csv",
    )
    cluster_config = SimpleNamespace(
        replica_config=replica_config,
        execution_time_predictor_config=predictor_config,
    )
    manager._cluster_configs = {
        ClusterType.PREFILL: cluster_config,
        ClusterType.DECODE: cluster_config,
        ClusterType.DECODE_ATTN: cluster_config,
        ClusterType.DECODE_FFN: cluster_config,
        ClusterType.MONOLITHIC: cluster_config,
    }
    return manager


def test_shared_manager_returns_family_grouped_models_by_cluster() -> None:
    manager = _make_manager()

    prefill_models = manager.get_models_for_cluster(ClusterType.PREFILL)
    decode_models = manager.get_models_for_cluster(ClusterType.DECODE)
    monolithic_models = manager.get_models_for_cluster(ClusterType.MONOLITHIC)

    assert set(prefill_models.keys()) == {"eager", "kernel_only"}
    assert set(prefill_models["eager"].keys()) == {"attn_prefill"}
    assert prefill_models["kernel_only"] == {}

    assert set(decode_models["eager"].keys()) == {"attn_prefill"}
    assert decode_models["kernel_only"] == {}

    assert set(monolithic_models["eager"].keys()) == {"attn_prefill"}
    assert monolithic_models["kernel_only"] == {}


def test_shared_manager_family_views_enable_kernel_only_only_when_graph_enabled() -> None:
    global_vars.set_cuda_graph_config(False, [1, 2, 4], "piecewise")
    manager = _make_manager()

    decode_models = manager.get_models_for_cluster(ClusterType.DECODE)
    monolithic_models = manager.get_models_for_cluster(ClusterType.MONOLITHIC)

    assert decode_models["eager"] == {}
    assert set(decode_models["kernel_only"].keys()) == {"attn_decode"}
    assert set(monolithic_models["eager"].keys()) == {"attn_prefill"}
    assert set(monolithic_models["kernel_only"].keys()) == {"attn_decode"}


def test_shared_manager_pd_af_measurement_types_match_reference_contract() -> None:
    manager = _make_manager()
    replica = manager._cluster_configs[ClusterType.DECODE_ATTN].replica_config
    global_vars.set_global_vars("offline", "pd-af-disaggregation")

    assert manager._get_measurement_types_for_cluster(ClusterType.DECODE_ATTN, replica) == [
        MeasurementType.CUDA_EVENT,
        MeasurementType.KERNEL_ONLY,
    ]
    assert manager._get_measurement_types_for_cluster(ClusterType.DECODE_FFN, replica) == [
        MeasurementType.KERNEL_ONLY
    ]
    models = manager.get_models_for_cluster(ClusterType.DECODE_ATTN)
    assert set(models["eager"]) == {"attn_prefill"}
    assert set(models["kernel_only"]) == {"attn_decode"}

    models = manager.get_models_for_cluster(ClusterType.DECODE_FFN)
    assert models["eager"] == {}
    assert set(models["kernel_only"]) == {"attn_decode"}

    global_vars.set_global_vars("offline", "co-location")
    global_vars.set_cuda_graph_config(True, [1, 2, 4], "none")

    assert manager._get_measurement_types_for_cluster(ClusterType.DECODE_ATTN, replica) == [
        MeasurementType.KERNEL_ONLY
    ]
    assert manager._get_measurement_types_for_cluster(ClusterType.DECODE_FFN, replica) == [
        MeasurementType.KERNEL_ONLY
    ]


def test_shared_manager_uses_active_family_cpu_overhead_path() -> None:
    manager = _make_manager()
    manager._active_measurement_type = MeasurementType.KERNEL_ONLY
    cluster_config = manager._cluster_configs[ClusterType.DECODE]

    input_files = manager._get_input_files_for_config(
        cluster_config.replica_config,
        cluster_config.execution_time_predictor_config,
    )

    assert input_files[4] == "cpu_kernel/a100/meta-llama/Llama-2-7b-hf.csv"


def test_independent_predictor_uses_active_family_cpu_overhead_path() -> None:
    cluster_config = _make_manager()._cluster_configs[ClusterType.MONOLITHIC]
    predictor = _make_predictor(ClusterType.MONOLITHIC, runtime_mode="FULL")
    predictor._config = cluster_config.execution_time_predictor_config
    predictor._replica_config = SimpleNamespace(
        **vars(cluster_config.replica_config), device_config=H800DeviceSKUConfig()
    )
    predictor._model_config = cluster_config.replica_config.model_config
    predictor._models_eager = predictor._models_kernel_only = {}
    predictor._predictions_eager = predictor._predictions_kernel_only = {}
    predictor._initialize_file_paths()

    decode_batch = SimpleNamespace(num_prefill_tokens=0, num_decode_tokens=8)
    predictor._activate_measurement_type(predictor._select_measurement_type_for_batch(decode_batch))
    assert predictor._cpu_overhead_input_file == "cpu_kernel/a100/meta-llama/Llama-2-7b-hf.csv"

    prefill_batch = SimpleNamespace(num_prefill_tokens=16, num_decode_tokens=0)
    predictor._activate_measurement_type(predictor._select_measurement_type_for_batch(prefill_batch))
    assert predictor._cpu_overhead_input_file == "cpu/a100/meta-llama/Llama-2-7b-hf.csv"


def test_shared_manager_returns_complete_training_file_paths() -> None:
    manager = _make_manager()

    training_file_paths = manager.get_training_file_paths(ClusterType.MONOLITHIC)

    assert training_file_paths == {
        "compute_input_file": "compute/a100/meta-llama/Llama-2-7b-hf.csv",
        "attention_input_file": "attention/a100/meta-llama/Llama-2-7b-hf.csv",
        "moe_input_file": "moe/a100/meta-llama/Llama-2-7b-hf.csv",
        "compute_device_event_input_file": "compute/a100/meta-llama/Llama-2-7b-hf_device_event.csv",
        "attention_device_event_input_file": "attention/a100/meta-llama/Llama-2-7b-hf_device_event.csv",
        "moe_device_event_input_file": "moe/a100/meta-llama/Llama-2-7b-hf_device_event.csv",
        "all_reduce_input_file": "network/a100_pairwise_nvlink/all_reduce.csv",
        "send_recv_input_file": "network/a100_pairwise_nvlink/send_recv.csv",
        "cpu_overhead_input_file": "cpu/a100/meta-llama/Llama-2-7b-hf.csv",
        "cpu_overhead_kernel_only_input_file": "cpu_kernel/a100/meta-llama/Llama-2-7b-hf.csv",
        "pp_stage_boundary_input_file": "other/a100/meta-llama/Llama-2-7b-hf/pp_stage_boundary.csv",
        "pp_receiver_head_input_file": "other/a100/meta-llama/Llama-2-7b-hf/pp_receiver_head.csv",
        "pp_producer_send_path_input_file": "other/a100/meta-llama/Llama-2-7b-hf/pp_producer_send_path.csv",
        "pp_prefill_consumer_active_input_file": "other/a100/meta-llama/Llama-2-7b-hf/pp_prefill_consumer_active.csv",
        "compute_kernel_only_input_file": "compute_kernel/a100/meta-llama/Llama-2-7b-hf.csv",
        "attention_kernel_only_input_file": "attention_kernel/a100/meta-llama/Llama-2-7b-hf.csv",
        "moe_kernel_only_input_file": "moe_kernel/a100/meta-llama/Llama-2-7b-hf.csv",
        "kernel_gap_input_file": "compute/a100/meta-llama/Llama-2-7b-hf/kernel_gap.csv",
    }


@pytest.mark.parametrize(
    ("measurement_type", "expected_compute", "expected_attention", "expected_moe"),
    [
        (
            MeasurementType.CUDA_EVENT,
            "compute/{DEVICE}/{MODEL}.csv",
            "attention/{DEVICE}/{MODEL}.csv",
            "moe/{DEVICE}/{MODEL}.csv",
        ),
        (
            MeasurementType.DEVICE_EVENT,
            "compute/{DEVICE}/{MODEL}_device_event.csv",
            "attention/{DEVICE}/{MODEL}_device_event.csv",
            "moe/{DEVICE}/{MODEL}_device_event.csv",
        ),
        (
            MeasurementType.KERNEL_ONLY,
            "compute_kernel/{DEVICE}/{MODEL}.csv",
            "attention_kernel/{DEVICE}/{MODEL}.csv",
            "moe_kernel/{DEVICE}/{MODEL}.csv",
        ),
    ],
)
def test_predictor_and_manager_share_measurement_path_contract(
    measurement_type: MeasurementType,
    expected_compute: str,
    expected_attention: str,
    expected_moe: str,
) -> None:
    manager = _make_manager()
    cluster_config = manager._cluster_configs[ClusterType.MONOLITHIC]
    manager._active_measurement_type = measurement_type
    manager_paths = manager._resolve_measurement_input_files_for_config(
        cluster_config.replica_config,
        cluster_config.execution_time_predictor_config,
        measurement_type,
    )

    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _PathProbePredictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            return None

        def _get_grid_search_params(self):
            return {}

    predictor = object.__new__(_PathProbePredictor)
    predictor._config = cluster_config.execution_time_predictor_config
    predictor._replica_config = cluster_config.replica_config
    predictor._model_config = cluster_config.replica_config.model_config
    predictor_paths = predictor._get_input_files(measurement_type)

    model = "meta-llama/Llama-2-7b-hf"
    assert manager_paths[0] == expected_compute.replace("{DEVICE}", "a100").replace(
        "{MODEL}", model
    )
    assert manager_paths[1] == expected_attention.replace(
        "{DEVICE}", "a100"
    ).replace("{MODEL}", model)
    assert manager_paths[5] == expected_moe.replace("{DEVICE}", "a100").replace(
        "{MODEL}", model
    )
    assert predictor_paths[0] == manager_paths[0]
    assert predictor_paths[1] == manager_paths[1]
    assert predictor_paths[2] == manager_paths[5]


def test_device_event_empty_fallback_stays_empty_across_entry_points() -> None:
    manager = _make_manager()
    config = manager._cluster_configs[ClusterType.MONOLITHIC].execution_time_predictor_config
    config.linear_op_input_file = ""
    config.mlp_input_file = ""
    config.atten_input_file = ""
    config.moe_input_file = ""
    replica = manager._cluster_configs[ClusterType.MONOLITHIC].replica_config

    manager_paths = manager._resolve_measurement_input_files_for_config(
        replica, config, MeasurementType.DEVICE_EVENT
    )
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _PathProbePredictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            return None

        def _get_grid_search_params(self):
            return {}

    predictor = object.__new__(_PathProbePredictor)
    predictor._config = config
    predictor._replica_config = replica
    predictor._model_config = replica.model_config
    predictor_paths = predictor._get_input_files(MeasurementType.DEVICE_EVENT)
    assert manager_paths[0] == manager_paths[1] == manager_paths[5] == ""
    assert predictor_paths[0] == predictor_paths[1] == predictor_paths[2] == ""


def test_device_event_path_contract_handles_extensionless_legacy_and_explicit_values() -> None:
    from frontier.execution_time_predictor.measurement_input_paths import (
        resolve_measurement_input_paths,
    )

    config = SimpleNamespace(
        linear_op_input_file="",
        mlp_input_file="legacy/linear",
        atten_input_file="attention/base",
        moe_input_file="",
        linear_op_device_event_input_file="explicit/{DEVICE}/linear.events",
        atten_device_event_input_file="",
        moe_device_event_input_file="explicit/{MODEL}/moe.events",
        all_reduce_input_file="net/{NETWORK_DEVICE}/all_reduce.csv",
        send_recv_input_file="",
        cpu_overhead_input_file="",
    )
    paths = resolve_measurement_input_paths(
        config,
        MeasurementType.DEVICE_EVENT,
        device="mi355x",
        model="model/name",
        network_device="xgmi",
    )
    assert paths.compute == "explicit/mi355x/linear.events"
    assert paths.attention == ""
    assert paths.moe == "explicit/model/name/moe.events"
    assert paths.all_reduce == "net/xgmi/all_reduce.csv"
    assert paths.send_recv == ""
    assert paths.cpu_overhead == ""


PLAIN_CPU_OVERHEAD_HEADER = "model_name,batch_size,sampler_e2e_mean,sampler_e2e_median\n"
PROBED_CPU_OVERHEAD_HEADER = (
    "model_name,batch_size,sampler_e2e_mean,sampler_e2e_median,"
    "forward_launch_mean,forward_launch_median\n"
)


def _stage_table(path, stages) -> str:
    """Write a forward_launch table with one row per pipeline stage in ``stages``."""
    path.write_text(
        PROBED_CPU_OVERHEAD_HEADER.replace("\n", ",pipeline_stage_id\n")
        + "".join(f"m,7,0.1,0.1,58.0,58.0,{stage}\n" for stage in stages)
    )
    return str(path)


def _predictor_config(**flags) -> SimpleNamespace:
    return SimpleNamespace(
        **{"enable_dummy_mode": False, "skip_cpu_overhead_modeling": False, **flags}
    )


def test_two_stream_eager_pricing_follows_forward_launch_columns(tmp_path) -> None:
    plain = tmp_path / "plain.csv"
    plain.write_text(PLAIN_CPU_OVERHEAD_HEADER)
    probed = _stage_table(tmp_path / "probed.csv", (0,))
    layout = {"sys_arch": "co-location", "num_pipeline_stages": 1}

    assert uses_two_stream_eager_pricing(_predictor_config(), str(probed), **layout) is True
    assert uses_two_stream_eager_pricing(_predictor_config(), str(plain), **layout) is False
    assert (
        uses_two_stream_eager_pricing(_predictor_config(), str(tmp_path / "absent.csv"), **layout)
        is False
    )
    assert (
        uses_two_stream_eager_pricing(_predictor_config(enable_dummy_mode=True), str(probed), **layout)
        is False
    )
    assert (
        uses_two_stream_eager_pricing(
            _predictor_config(skip_cpu_overhead_modeling=True), str(probed), **layout
        )
        is False
    )


def test_two_stream_eager_pricing_fails_fast_for_pd_af_and_tables_without_stages(tmp_path) -> None:
    probed = _stage_table(tmp_path / "probed.csv", (0,))
    unstaged = tmp_path / "unstaged.csv"
    unstaged.write_text(PROBED_CPU_OVERHEAD_HEADER)

    with pytest.raises(ValueError, match="PD-AF"):
        uses_two_stream_eager_pricing(
            _predictor_config(), probed, sys_arch="pd-af-disaggregation", num_pipeline_stages=1
        )
    with pytest.raises(ValueError, match="without pipeline_stage_id; republish"):
        uses_two_stream_eager_pricing(
            _predictor_config(), str(unstaged), sys_arch="co-location", num_pipeline_stages=1
        )


def test_two_stream_eager_pricing_takes_stage_tables_covering_every_stage(tmp_path) -> None:
    both = _stage_table(tmp_path / "both.csv", (0, 1))
    first = _stage_table(tmp_path / "first.csv", (0,))

    assert uses_two_stream_eager_pricing(
        _predictor_config(), both, sys_arch="co-location", num_pipeline_stages=2
    ) is True
    with pytest.raises(ValueError, match=r"stages \[0, 1\]; num_pipeline_stages=1"):
        uses_two_stream_eager_pricing(
            _predictor_config(), both, sys_arch="co-location", num_pipeline_stages=1
        )
    with pytest.raises(ValueError, match=r"stages \[0\]; num_pipeline_stages=2"):
        uses_two_stream_eager_pricing(
            _predictor_config(), first, sys_arch="co-location", num_pipeline_stages=2
        )
    assert uses_two_stream_eager_pricing(
        _predictor_config(), first, sys_arch="pd-disaggregation", num_pipeline_stages=1
    ) is True


def test_two_stream_enables_kernel_only_family_for_eager_step_roles() -> None:
    for cluster_type in (ClusterType.PREFILL, ClusterType.DECODE, ClusterType.MONOLITHIC):
        predictor = _make_predictor(cluster_type, two_stream_eager_pricing=True)

        assert predictor._should_enable_measurement_family(MeasurementType.CUDA_EVENT) is True
        assert predictor._should_enable_measurement_family(MeasurementType.KERNEL_ONLY) is True
        assert predictor._get_default_measurement_type_for_cluster() == MeasurementType.CUDA_EVENT


def _bind_step_pricing(predictor, cpu_overhead_ms: float) -> list:
    """Stub family activation and record the family each CPU-overhead lookup reads."""
    lookups = []
    predictor._config = SimpleNamespace(skip_cpu_overhead_modeling=False)
    predictor._require_predictions_for_measurement_type = lambda *_args: None
    predictor._activate_measurement_type = lambda measurement_type: setattr(
        predictor, "_active_measurement_type", measurement_type
    )

    def lookup(metric_name, _batch, _stage_id):
        lookups.append((metric_name, predictor._active_measurement_type))
        return cpu_overhead_ms

    predictor._lookup_cpu_overhead_prediction = lookup
    return lookups


def test_two_stream_eager_step_prices_kernel_only_operators_and_event_cpu_terms() -> None:
    global_vars.set_cuda_graph_config(False, [1, 2, 4, 8], "full_decode_only")
    predictor = _make_predictor(
        ClusterType.MONOLITHIC, runtime_mode=None, two_stream_eager_pricing=True
    )
    lookups = _bind_step_pricing(predictor, cpu_overhead_ms=4.0)

    eager_step = _forward_step(16, 4, None)
    assert predictor._activate_measurement_type_for_batch(eager_step) == MeasurementType.KERNEL_ONLY
    assert predictor._step_measurement_type == MeasurementType.CUDA_EVENT
    assert predictor._get_forward_launch_time(eager_step, 0) == 4.0
    assert predictor._get_sampler_e2e_time(eager_step, 0) == 4.0
    assert lookups == [
        ("forward_launch", MeasurementType.CUDA_EVENT),
        ("sampler_e2e", MeasurementType.CUDA_EVENT),
    ]
    assert predictor._active_measurement_type == MeasurementType.KERNEL_ONLY

    lookups.clear()
    graph_step = _forward_step(0, 4, "FULL")
    assert predictor._activate_measurement_type_for_batch(graph_step) == MeasurementType.KERNEL_ONLY
    assert predictor._step_measurement_type == MeasurementType.KERNEL_ONLY
    assert predictor._get_forward_launch_time(graph_step, 0) == 0.0
    assert predictor._get_sampler_e2e_time(graph_step, 0) == 4.0
    assert lookups == [("sampler_e2e", MeasurementType.KERNEL_ONLY)]


def test_eager_step_without_two_stream_keeps_event_operators_and_no_forward_launch() -> None:
    predictor = _make_predictor(ClusterType.MONOLITHIC, runtime_mode=None)
    lookups = _bind_step_pricing(predictor, cpu_overhead_ms=4.0)

    eager_step = _forward_step(16, 4, None)
    assert predictor._activate_measurement_type_for_batch(eager_step) == MeasurementType.CUDA_EVENT
    assert predictor._get_forward_launch_time(eager_step, 0) == 0.0
    assert predictor._get_sampler_e2e_time(eager_step, 0) == 4.0
    assert lookups == [("sampler_e2e", MeasurementType.CUDA_EVENT)]


STEP_FEATURE_NAMES = ["batch_size", "num_prefill_tokens", "num_decode_tokens"]


def test_stage_keyed_cpu_overhead_rows_price_each_stage() -> None:
    predictor = _make_predictor(ClusterType.MONOLITHIC)
    predictor._config = SimpleNamespace(skip_cpu_overhead_modeling=False)
    predictor._activate_measurement_type = lambda measurement_type: setattr(
        predictor, "_active_measurement_type", measurement_type
    )
    predictor._active_measurement_type = MeasurementType.KERNEL_ONLY
    predictor._step_measurement_type = MeasurementType.CUDA_EVENT
    step = (7.0, 0.0, 7.0)
    predictor._predictions = {
        "forward_launch": {
            "_on_demand_prediction": True,
            "_feature_names": [*STEP_FEATURE_NAMES, "pipeline_stage_id"],
            "_exact_lookup": {(*step, 0.0): 58.65, (*step, 1.0): 58.98},
        },
        "schedule": {
            "_on_demand_prediction": True,
            "_feature_names": STEP_FEATURE_NAMES,
            "_exact_lookup": {step: 0.2},
        },
    }
    batch = SimpleNamespace(size=7, num_prefill_tokens=0, num_decode_tokens=7)

    assert [predictor._get_forward_launch_time(batch, stage) for stage in (0, 1)] == [58.65, 58.98]
    # A table from a single-stage probe has no stage feature, so every stage reads the same row.
    assert [predictor._get_schedule_time(batch, stage) for stage in (0, 1)] == [0.2, 0.2]


class _CpuOverheadTableRead(Exception):
    pass


def test_kernel_only_family_trains_no_cpu_overhead_models_without_decode_graphs() -> None:
    predictor = _make_predictor(ClusterType.MONOLITHIC, two_stream_eager_pricing=True)
    predictor._config = SimpleNamespace(skip_cpu_overhead_modeling=False)
    predictor._cpu_overhead_input_file = "cpu_overheads.csv"

    def read_table(_path):
        raise _CpuOverheadTableRead

    predictor._load_cpu_overhead_df = read_table
    predictor._active_measurement_type = MeasurementType.KERNEL_ONLY
    assert predictor._train_cpu_overhead_models() == {}

    predictor._active_measurement_type = MeasurementType.CUDA_EVENT
    with pytest.raises(_CpuOverheadTableRead):
        predictor._train_cpu_overhead_models()

    # Decode graphs run steps in the kernel-only family, which then prices their CPU terms.
    global_vars.set_cuda_graph_config(False, [1, 2, 4], "full_decode_only")
    predictor._active_measurement_type = MeasurementType.KERNEL_ONLY
    with pytest.raises(_CpuOverheadTableRead):
        predictor._train_cpu_overhead_models()


def test_shared_manager_trains_and_exposes_kernel_only_for_two_stream_clusters() -> None:
    manager = _make_manager()
    manager._two_stream_eager_clusters = frozenset(
        {ClusterType.PREFILL, ClusterType.DECODE, ClusterType.MONOLITHIC}
    )
    replica = manager._cluster_configs[ClusterType.PREFILL].replica_config
    both_families = [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY]

    for cluster_type in (ClusterType.PREFILL, ClusterType.DECODE, ClusterType.MONOLITHIC):
        assert manager._get_measurement_types_for_cluster(cluster_type, replica) == both_families
        models = manager.get_models_for_cluster(cluster_type)
        assert set(models["eager"]) == {"attn_prefill"}
        assert set(models["kernel_only"]) == {"attn_decode"}

    # Graph decode steps never run eager, so a graph DECODE role keeps kernel-only only.
    global_vars.set_cuda_graph_config(False, [1, 2, 4], "piecewise")
    assert manager._get_measurement_types_for_cluster(ClusterType.DECODE, replica) == [
        MeasurementType.KERNEL_ONLY
    ]


def test_shared_manager_detects_two_stream_from_eager_cpu_overhead_table(tmp_path) -> None:
    manager = _make_manager()
    probed = _stage_table(tmp_path / "probed.csv", (0,))
    plain = tmp_path / "plain.csv"
    plain.write_text(PLAIN_CPU_OVERHEAD_HEADER)
    cluster_config = manager._cluster_configs[ClusterType.PREFILL]
    replica = SimpleNamespace(**vars(cluster_config.replica_config), num_pipeline_stages=1)

    def cluster_with(eager_cpu_file: str, kernel_only_cpu_file: str) -> SimpleNamespace:
        predictor_config = SimpleNamespace(
            **{
                **vars(cluster_config.execution_time_predictor_config),
                "cpu_overhead_input_file": eager_cpu_file,
                "cpu_overhead_kernel_only_input_file": kernel_only_cpu_file,
                "enable_dummy_mode": False,
                "skip_cpu_overhead_modeling": False,
            }
        )
        return SimpleNamespace(
            replica_config=replica, execution_time_predictor_config=predictor_config
        )

    assert manager._uses_two_stream_eager_pricing(cluster_with(str(probed), str(plain))) is True
    assert manager._uses_two_stream_eager_pricing(cluster_with(str(plain), str(probed))) is False
