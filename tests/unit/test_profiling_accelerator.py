from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from frontier.config.device_sku_config import BaseDeviceSKUConfig
from frontier.config.node_sku_config import BaseNodeSKUConfig
from frontier.profiling.common.accelerator import (
    accelerator_platform,
    get_available_gpu_ids,
    set_process_visible_device,
)
from frontier.types import DeviceSKUType, NodeSKUType


def _completed(command: list[str], stdout: str, returncode: int = 0):
    return subprocess.CompletedProcess(command, returncode, stdout, "")


def test_accelerator_platform_distinguishes_rocm_cuda_and_cpu() -> None:
    assert accelerator_platform(SimpleNamespace(version=SimpleNamespace(hip="7.2", cuda=None))) == "rocm"
    assert accelerator_platform(SimpleNamespace(version=SimpleNamespace(hip=None, cuda="12.8"))) == "cuda"
    assert accelerator_platform(SimpleNamespace(version=SimpleNamespace(hip=None, cuda=None))) == "cpu"


def test_rocm_visibility_precedes_cuda_compatibility_visibility() -> None:
    environ = {
        "ROCR_VISIBLE_DEVICES": "4,5,6",
        "CUDA_VISIBLE_DEVICES": "0,1",
    }
    assert get_available_gpu_ids(2, environ=environ) == [4, 5]


def test_discovery_uses_amd_smi_json_without_torch_initialization() -> None:
    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        calls.append(command)
        return _completed(command, json.dumps([{"gpu": 0}, {"gpu": 1}]))

    assert get_available_gpu_ids(2, environ={}, command_runner=runner) == [0, 1]
    assert calls == [["amd-smi", "list", "--json"]]


def test_discovery_falls_back_to_nvidia_smi() -> None:
    def runner(command, **_kwargs):
        if command[0] == "amd-smi":
            raise FileNotFoundError(command[0])
        return _completed(command, "0\n1\n")

    assert get_available_gpu_ids(2, environ={}, command_runner=runner) == [0, 1]


def test_discovery_rejects_non_integer_visibility_tokens() -> None:
    with pytest.raises(RuntimeError, match="integer GPU IDs"):
        get_available_gpu_ids(1, environ={"HIP_VISIBLE_DEVICES": "gpu-uuid"})


def test_discovery_reports_insufficient_visible_gpus() -> None:
    with pytest.raises(RuntimeError, match="Requested 2 GPUs but only found 1"):
        get_available_gpu_ids(2, environ={"CUDA_VISIBLE_DEVICES": "7"})


def test_set_process_visible_device_uses_native_rocm_variables() -> None:
    environ = {"CUDA_VISIBLE_DEVICES": "4"}
    torch_module = SimpleNamespace(version=SimpleNamespace(hip="7.2", cuda=None))

    description = set_process_visible_device(
        2,
        torch_module=torch_module,
        environ=environ,
    )

    assert description == "ROCR_VISIBLE_DEVICES=2"
    assert environ == {"ROCR_VISIBLE_DEVICES": "2"}


def test_set_process_visible_device_uses_cuda_variable() -> None:
    environ = {"ROCR_VISIBLE_DEVICES": "4", "HIP_VISIBLE_DEVICES": "4"}
    torch_module = SimpleNamespace(version=SimpleNamespace(hip=None, cuda="12.8"))

    description = set_process_visible_device(
        3,
        torch_module=torch_module,
        environ=environ,
    )

    assert description == "CUDA_VISIBLE_DEVICES=3"
    assert environ == {"CUDA_VISIBLE_DEVICES": "3"}


def test_mi355x_device_and_ubb_node_configs_are_registered() -> None:
    device = BaseDeviceSKUConfig.create_from_type(DeviceSKUType.MI355X)
    node = BaseNodeSKUConfig.create_from_type(NodeSKUType.MI355X_UBB)

    assert device.fp16_tflops == 2516
    assert device.total_memory_gb == 288
    assert node.device_sku_type is DeviceSKUType.MI355X
    assert node.num_devices_per_node == 8
