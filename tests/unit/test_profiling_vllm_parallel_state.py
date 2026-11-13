"""CPU check that profiling processes build vLLM layers with vLLM's parallel state.

vLLM weight parameters read vLLM's tensor-parallel group when they are built.
A profiling worker process starts without that group, so the linear-op FP8
binder and the MoE gate must form it first. The check runs in a fresh process
because the group is process-global state.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

FRESH_PROCESS_SCRIPT = """
import torch

# CPU stand-in for the worker's CUDA device.
torch.cuda.current_device = lambda: "cpu"

import vllm.model_executor.layers.quantization.fp8 as fp8
from vllm.distributed import get_tensor_model_parallel_world_size

import frontier.profiling.common.parallel_utils.tensor_parallel_layers as layers
from frontier.profiling.moe.moe_impl import MoEGatingNetwork


class KernelSelection(Exception):
    pass


def select_kernel(*, weight_shape, **_):
    # Kernel selection needs a GPU; reaching it means every weight parameter
    # and the block-shape validation were built.
    raise KernelSelection(tuple(weight_shape))


fp8.init_fp8_linear_kernel = select_kernel
layers.ColumnParallelLinear._get_precision = lambda self: layers.PrecisionType.FP8
layer = layers.ColumnParallelLinear(
    512, 2048, bias=False, fp8_weight_block_size=(128, 128), world_size=8,
    linear_metric_name="q_b_proj",
)
try:
    layers.build_fp8_linear_method_kernels(
        torch.nn.Sequential(layer), model_dtype=torch.bfloat16, model_type="glm_moe_dsa"
    )
except KernelSelection as selected:
    print("linear_kernel_selection", *selected.args[0])
gating = MoEGatingNetwork(hidden_dim=64, num_experts=8, router_topk=2)
print("gate", gating.linear_backend, *gating.gate.weight.shape)
print("vllm_tp_world_size", get_tensor_model_parallel_world_size())
"""


def test_profiling_layers_build_with_single_rank_vllm_parallel_state():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CPU check; GPU profiling runs build these layers on CUDA")
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("requires vLLM")
    if importlib.metadata.version("vllm").startswith("0.10."):
        pytest.skip("vLLM 0.10 runs block-FP8 GEMMs through its legacy custom op")
    env = {
        **os.environ,
        "VLLM_TARGET_DEVICE": "cpu",
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", FRESH_PROCESS_SCRIPT],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    lines = result.stdout.splitlines()
    # The TP=8 column shard of a 2048-wide output is 256 rows.
    assert "linear_kernel_selection 256 512" in lines
    assert "gate vllm_replicated_linear 8 64" in lines
    assert "vllm_tp_world_size 1" in lines
