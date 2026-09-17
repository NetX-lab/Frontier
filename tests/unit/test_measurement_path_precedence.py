"""Profiling-path precedence across manager, predictor, and training boundaries."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier.config.config import RandomForrestExecutionTimePredictorConfig, ReplicaConfig
from frontier.execution_time_predictor.measurement_input_paths import (
    substitute_input_path,
    resolve_measurement_input_paths,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import SklearnExecutionTimePredictor
from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
from frontier.types import ClusterType, MeasurementType


class _PathProbePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


@pytest.mark.parametrize("path", ("", "gdn.csv", "{DEVICE}/{MODEL}/{DEVICE}/gdn.csv", "{NETWORK_DEVICE}/{DEVICE}/{MODEL}/gdn.csv"))
def test_compute_only_substitution_preserves_network_placeholder(path):
    expected = path.replace("{DEVICE}", "mi355x").replace("{MODEL}", "model/name")
    assert substitute_input_path(path, device="mi355x", model="model/name") == expected


@pytest.fixture
def paths():
    config = RandomForrestExecutionTimePredictorConfig()
    replica = ReplicaConfig(
        device="a100", network_device="a100_pairwise_nvlink",
        model_name="meta-llama/Llama-2-7b-hf",
    )
    predictor = object.__new__(_PathProbePredictor)
    predictor._config = config
    predictor._replica_config = replica
    predictor._model_config = replica.model_config
    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._cluster_configs = {
        ClusterType.MONOLITHIC: SimpleNamespace(
            replica_config=replica, execution_time_predictor_config=config,
        )
    }
    return config, replica, predictor, manager


def _canonical(config, replica, measurement_type):
    return resolve_measurement_input_paths(
        config, measurement_type, device=replica.device,
        model=replica.model_config.get_name(), network_device=replica.network_device,
    )


@pytest.mark.parametrize("alias", [False, True])
def test_missing_device_compute_preserves_explicit_attention_and_moe(paths, alias):
    config, replica, predictor, _ = paths
    attention_key = "attention_input_file_device_event" if alias else "attention_device_event_input_file"
    moe_key = "moe_input_file_device_event" if alias else "moe_device_event_input_file"
    overrides = {
        attention_key: "overrides/attention_device_event.csv",
        moe_key: "overrides/moe_device_event.csv",
    }

    predictor._initialize_file_paths(overrides)

    assert predictor._compute_input_file_device_event == _canonical(config, replica, MeasurementType.DEVICE_EVENT).compute
    assert predictor._attention_input_file_device_event == overrides[attention_key]
    assert predictor._moe_input_file_device_event == overrides[moe_key]


@pytest.mark.parametrize("measurement_type", list(MeasurementType))
def test_public_manager_and_predictor_preserve_canonical_family_paths(paths, measurement_type):
    config, replica, predictor, manager = paths
    canonical = _canonical(config, replica, measurement_type)
    public = manager.get_training_file_paths(ClusterType.MONOLITHIC)
    predictor_paths = predictor._get_input_files(measurement_type)
    manager_paths = manager._resolve_measurement_input_files_for_config(
        replica, config, measurement_type
    )
    suffix = {
        MeasurementType.CUDA_EVENT: "",
        MeasurementType.DEVICE_EVENT: "_device_event",
        MeasurementType.KERNEL_ONLY: "_kernel_only",
    }[measurement_type]

    assert public[f"compute{suffix}_input_file"] == canonical.compute
    assert public[f"attention{suffix}_input_file"] == canonical.attention
    assert public[f"moe{suffix}_input_file"] == canonical.moe
    assert predictor_paths == (
        canonical.compute, canonical.attention, canonical.moe,
        canonical.all_reduce, canonical.send_recv, canonical.cpu_overhead,
    )
    assert manager_paths == (
        canonical.compute, canonical.attention, canonical.all_reduce,
        canonical.send_recv, canonical.cpu_overhead, canonical.moe,
    )


def test_missing_override_fields_retain_configured_and_pp_paths(paths):
    _, _, predictor, manager = paths
    configured = manager.get_training_file_paths(ClusterType.MONOLITHIC)
    overrides = {"attention_device_event_input_file": "explicit/attention.events"}

    predictor._initialize_file_paths(overrides)

    assert predictor._compute_input_file_eager == configured["compute_input_file"]
    assert predictor._attention_input_file_device_event == overrides["attention_device_event_input_file"]
    for key in (
        "all_reduce_input_file", "send_recv_input_file", "cpu_overhead_input_file",
        "pp_stage_boundary_input_file", "pp_receiver_head_input_file",
        "pp_producer_send_path_input_file", "pp_prefill_consumer_active_input_file",
    ):
        assert getattr(predictor, f"_{key}") == configured[key]


def test_full_public_path_dictionary_keeps_pp_overrides(paths):
    _, _, predictor, manager = paths
    supplied = manager.get_training_file_paths(ClusterType.MONOLITHIC)
    pp_keys = (
        "pp_stage_boundary_input_file", "pp_receiver_head_input_file",
        "pp_producer_send_path_input_file", "pp_prefill_consumer_active_input_file",
    )
    for key in pp_keys:
        supplied[key] = f"explicit/{key}.csv"

    predictor._initialize_file_paths(supplied)

    for key in pp_keys:
        assert getattr(predictor, f"_{key}") == supplied[key]
    assert predictor._attention_input_file_device_event == supplied["attention_device_event_input_file"]


def test_empty_compute_attention_and_moe_do_not_invent_device_event_paths(paths):
    config, replica, predictor, manager = paths
    config.linear_op_input_file = config.mlp_input_file = ""
    config.atten_input_file = config.moe_input_file = ""
    canonical = _canonical(config, replica, MeasurementType.DEVICE_EVENT)
    public = manager.get_training_file_paths(ClusterType.MONOLITHIC)
    predictor._initialize_file_paths()

    assert (canonical.compute, canonical.attention, canonical.moe) == ("", "", "")
    assert predictor._compute_input_file_device_event == ""
    assert predictor._attention_input_file_device_event == ""
    assert predictor._moe_input_file_device_event == ""
    assert public["compute_device_event_input_file"] == ""
    assert public["attention_device_event_input_file"] == ""
    assert public["moe_device_event_input_file"] == ""


def test_legacy_linear_alias_and_template_substitution_remain_consistent(paths):
    config, replica, predictor, manager = paths
    config.linear_op_input_file = ""
    config.mlp_input_file = "profile/{DEVICE}/{MODEL}/{NETWORK_DEVICE}/linear"
    expected = "profile/a100/meta-llama/Llama-2-7b-hf/a100_pairwise_nvlink/linear_device_event"

    assert _canonical(config, replica, MeasurementType.DEVICE_EVENT).compute == expected
    assert manager.get_training_file_paths(ClusterType.MONOLITHIC)["compute_device_event_input_file"] == expected
    assert predictor._get_input_files(MeasurementType.DEVICE_EVENT)[0] == expected


@pytest.mark.parametrize("cpu_kernel", [None, "", "explicit/{DEVICE}/{MODEL}/cpu.kernel.csv"])
def test_kernel_cpu_path_distinguishes_empty_from_absent(paths, cpu_kernel):
    config, replica, _, manager = paths
    config.cpu_overhead_kernel_only_input_file = cpu_kernel
    canonical = _canonical(config, replica, MeasurementType.KERNEL_ONLY)
    expected = (
        _canonical(config, replica, MeasurementType.CUDA_EVENT).cpu_overhead
        if cpu_kernel is None
        else cpu_kernel.replace("{DEVICE}", replica.device).replace("{MODEL}", replica.model_config.get_name())
    )

    assert canonical.cpu_overhead == expected
    assert manager.get_training_file_paths(ClusterType.MONOLITHIC)["cpu_overhead_kernel_only_input_file"] == expected


def test_explicit_empty_device_override_is_preserved(paths):
    _, _, predictor, _ = paths

    predictor._initialize_file_paths({
        "compute_device_event_input_file": "",
        "attention_device_event_input_file": "",
        "moe_device_event_input_file": "",
    })

    assert predictor._compute_input_file_device_event == ""
    assert predictor._attention_input_file_device_event == ""
    assert predictor._moe_input_file_device_event == ""


def test_canonical_override_key_takes_precedence_over_historical_alias(paths):
    _, _, predictor, _ = paths

    predictor._initialize_file_paths({
        "attention_device_event_input_file": "",
        "attention_input_file_device_event": "legacy/attention.events",
    })

    assert predictor._attention_input_file_device_event == ""


@pytest.mark.parametrize("owner", ["predictor", "manager"])
def test_explicit_unknown_device_cannot_silently_select_cuda(paths, owner):
    _, _, predictor, manager = paths
    malformed = SimpleNamespace(device="not-a-registered-device")
    selector = predictor if owner == "predictor" else manager

    with pytest.raises(ValueError):
        selector._event_measurement_type_for_replica(malformed)


@pytest.mark.parametrize("device, expected", [
    ("a100", MeasurementType.CUDA_EVENT),
    ("mi355x", MeasurementType.DEVICE_EVENT),
])
def test_validated_device_metadata_selects_its_own_event_family(paths, device, expected):
    _, _, predictor, manager = paths
    replica = ReplicaConfig(
        device=device, model_name="meta-llama/Llama-2-7b-hf",
        network_device="a100_pairwise_nvlink",
    )

    assert predictor._event_measurement_type_for_replica(replica) == expected
    assert manager._event_measurement_type_for_replica(replica) == expected


@pytest.mark.parametrize("owner", ["predictor", "manager"])
def test_unknown_supplied_platform_cannot_select_cuda(paths, owner):
    _, replica, predictor, manager = paths
    malformed = SimpleNamespace(
        device=replica.device, device_config=SimpleNamespace(gpu_platform="unknown"),
    )
    selector = predictor if owner == "predictor" else manager

    with pytest.raises(ValueError):
        selector._event_measurement_type_for_replica(malformed)


@pytest.mark.parametrize("measurement_type", list(MeasurementType))
def test_standalone_attention_handoff_preserves_resolved_family_paths(paths, tmp_path, monkeypatch, measurement_type):
    from frontier.training import cli

    config, replica, _, manager = paths
    config.linear_op_input_file = str(tmp_path / "{DEVICE}" / "{MODEL}" / "linear.csv")
    config.atten_input_file = str(tmp_path / "{DEVICE}" / "{MODEL}" / "attention.csv")
    config.linear_op_kernel_only_input_file = str(tmp_path / "{DEVICE}" / "{MODEL}" / "linear.kernel.csv")
    config.atten_kernel_only_input_file = str(tmp_path / "{DEVICE}" / "{MODEL}" / "attention.kernel.csv")
    canonical = _canonical(config, replica, measurement_type)
    for name in (canonical.compute, canonical.attention):
        Path(name).parent.mkdir(parents=True, exist_ok=True)
        Path(name).touch()
    captured = Mock(return_value=SimpleNamespace(train=lambda: {}))
    monkeypatch.setattr(cli, "create_attention_trainer_from_model_config", captured)
    monkeypatch.setattr("sys.argv", [
        "frontier.training.cli", "attention",
        "--layer_dataset_path", canonical.attention,
        "--compute_dataset_path", canonical.compute,
        "--model_name", replica.model_name, "--device", replica.device,
        "--measurement_type", measurement_type.value,
        "--output_dir", str(tmp_path / "models"),
    ])

    assert cli.train_attention(cli.parse_args()) == 0
    supplied = captured.call_args.kwargs
    assert supplied["layer_dataset_path"] == canonical.attention
    assert supplied["compute_dataset_path"] == canonical.compute
    assert supplied["measurement_type"] == measurement_type.value
