"""Unit tests for measurement-family selection and manager family views."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.config import global_vars
from frontier.config.device_sku_config import H800DeviceSKUConfig
from frontier.scheduler.replica_scheduler.vllm_v1_engine_replica_scheduler import (
    VLLMv1EngineReplicaScheduler,
)
from frontier.types import ClusterType, MeasurementType


def _make_predictor(cluster_type: ClusterType, runtime_mode: str = "NONE"):
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class DummyPredictor(SklearnExecutionTimePredictor):
        def __init__(self):
            self._cluster_type = cluster_type
            self._replica_config = SimpleNamespace(device_config=H800DeviceSKUConfig())
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
