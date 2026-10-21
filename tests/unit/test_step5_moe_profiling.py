"""Step-5-Preview routed-expert profiling: layer groups, router and expert kernels.

Step-5 routed layers differ from the single model-wide MoE setting: layers 3-87
run block-FP8 experts, layers 88-90 run BF16 experts, all of them use the
clamped ``swiglustep`` activation, and the router is an FP32 gate with biased
sigmoid top-k selection. These CPU tests check that the profiler measures each
group with that native configuration; none of them measures timing.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from frontier.config import PrecisionType
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.moe import main as moe_main
from frontier.profiling.moe.routed_layer_groups import (
    SWIGLUSTEP_ACTIVATION,
    RoutedLayerRouter,
    resolve_routed_layer_groups,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
STEP5 = "stepfun-ai/Step-5-Preview"
STEP5_ROUTER = RoutedLayerRouter(
    scoring_func="sigmoid",
    uses_selection_bias=True,
    renormalize=True,
    routed_scaling_factor=3.0,
    gate_precision=PrecisionType.FP32,
)


def test_step5_routed_layers_split_by_expert_precision():
    groups = resolve_routed_layer_groups(ModelConfig.from_model_name(STEP5))

    assert [group.layer_ids for group in groups] == [
        tuple(range(3, 88)),
        (88, 89, 90),
    ]
    assert [group.precision for group in groups] == [PrecisionType.FP8, PrecisionType.BF16]
    assert {group.activation for group in groups} == {SWIGLUSTEP_ACTIVATION}
    assert {group.router for group in groups} == {STEP5_ROUTER}
    assert groups[1].row_metadata() == {
        "moe_layer_ids": "88-90",
        "moe_routed_precision": "bf16",
        "moe_activation": "swiglustep",
    }


def test_models_without_layer_groups_keep_one_model_wide_setting():
    assert resolve_routed_layer_groups(ModelConfig.from_model_name("zai-org/GLM-5.3")) == ()


def test_dry_run_counts_each_step5_layer_group(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "moe.main", "--dry_run", "--device", "h800", "--models", STEP5,
            "--num_tensor_parallel_workers", "1", "--expert_parallel_sizes", "32",
            "--num_tokens_list", "1", "16", "256",
        ],
    )
    moe_main.main()
    out = capsys.readouterr().out
    plan = json.loads(out[out.index("{\n"):])
    model = plan["models"][STEP5]

    assert model["local_experts_by_expert_parallel_size"] == {"32": 11}
    assert [group["moe_layer_ids"] for group in model["routed_layer_groups"]] == [
        "3-87",
        "88-90",
    ]
    # Three token counts, one EP size and two layer groups.
    assert model["profiled_steps"] == 3 * 1 * 2 * plan["cases_per_parallel_point"]


@pytest.mark.parametrize(
    ("quantization_mode", "block_shape", "weight_dtype_name"),
    [("fp8", (128, 128), "float8_e4m3fn"), ("bf16", None, "bfloat16")],
)
def test_builder_builds_swiglustep_triton_experts_on_cpu(
    monkeypatch, cpu_vllm_platform, quantization_mode, block_shape, weight_dtype_name
):
    torch = pytest.importorskip("torch")
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
    from vllm.model_executor.layers.fused_moe.oracle import fp8 as fp8_oracle
    from vllm.model_executor.layers.fused_moe.oracle import unquantized as unquantized_oracle

    from frontier.profiling.moe.fp8_block_experts import _build_native_routed_experts

    selected = {}

    # Native Step-5 serving on H800 selects Triton experts for both precisions.
    def select_fp8_on_hopper(*, config, weight_key, activation_key, allow_vllm_cutlass):
        selected["config"] = config
        return fp8_oracle.Fp8MoeBackend.TRITON, TritonExperts

    def select_unquantized_on_hopper(moe_config):
        selected["config"] = moe_config
        return unquantized_oracle.UnquantizedMoeBackend.TRITON, TritonExperts

    monkeypatch.setattr(fp8_oracle, "select_fp8_moe_backend", select_fp8_on_hopper)
    monkeypatch.setattr(
        unquantized_oracle, "select_unquantized_moe_backend", select_unquantized_on_hopper
    )
    routed = _build_native_routed_experts(
        quantization_mode=quantization_mode, activation=SWIGLUSTEP_ACTIVATION,
        num_local_experts=11, global_num_experts=352, hidden_dim=256,
        intermediate_dim=128, top_k=8, block_shape=block_shape,
        expert_parallel_size=32, expert_parallel_rank=0, dtype=torch.bfloat16,
        model_type="step5", device=torch.device("cpu"),
    )

    config = selected["config"]
    assert config.activation is MoEActivation.SWIGLUSTEP
    assert (config.num_experts, config.num_local_experts) == (352, 11)
    assert (routed.backend, routed.experts_class) == ("TRITON", "TritonExperts")
    assert type(routed.kernel.fused_experts) is TritonExperts
    assert type(routed.kernel.prepare_finalize).__name__ == (
        "MoEPrepareAndFinalizeNoDPEPModular"
    )
    assert routed.activation is MoEActivation.SWIGLUSTEP
    # No tuned file exists for 11 local experts of width 128.
    assert routed.triton_tuning_config == "default"
    assert routed.w13.dtype == routed.w2.dtype == getattr(torch, weight_dtype_name)
    assert tuple(routed.w13.shape) == (11, 256, 256)
    assert tuple(routed.w2.shape) == (11, 256, 128)


ROUTER_FRESH_PROCESS_SCRIPT = """
import contextlib

import torch

# CPU stand-in for the worker's CUDA device.
torch.cuda.current_device = lambda: "cpu"

from frontier.config import PrecisionType
from frontier.profiling.moe.moe_impl import SigmoidBiasGatingNetwork
from frontier.profiling.moe.routed_layer_groups import RoutedLayerRouter

router = RoutedLayerRouter(
    scoring_func="sigmoid", uses_selection_bias=True, renormalize=True,
    routed_scaling_factor=3.0, gate_precision=PrecisionType.FP32,
)
gating = SigmoidBiasGatingNetwork(
    hidden_dim=64, num_experts=352, router_topk=8, router=router
)
# vLLM's weight loading step; on CPU it binds the CPU GEMM of the gate.
gating.gate.quant_method.process_weights_after_loading(gating.gate)
gating.gating_linear_timer = contextlib.nullcontext()
gating.gating_routing_topk_timer = contextlib.nullcontext()
calls = []


def record_routing(**kwargs):
    calls.append(kwargs)
    num_tokens = kwargs["hidden_states"].shape[0]
    return torch.zeros(num_tokens, 8), torch.zeros(num_tokens, 8, dtype=torch.int32)


gating._fused_topk_bias = record_routing
gating(torch.randn(4, 64, dtype=torch.bfloat16))
call = calls[0]
print("gate", type(gating.gate).__name__, gating.gate.out_dtype)
print("logits", call["gating_output"].dtype, *call["gating_output"].shape)
print("bias", call["e_score_correction_bias"].dtype, *call["e_score_correction_bias"].shape)
print(
    "routing", call["scoring_func"], call["topk"], call["renormalize"],
    call["routed_scaling_factor"],
)
print("weight_policy", gating.routing_weight_policy)
"""


def test_sigmoid_bias_router_runs_fp32_gate_and_biased_topk():
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CPU check; GPU profiling builds the router on CUDA")
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("requires vLLM")
    env = {
        **os.environ,
        "VLLM_TARGET_DEVICE": "cpu",
        "PYTHONPATH": os.pathsep.join(
            filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")])
        ),
    }
    # vLLM's parallel state is process-global, so the router runs in a fresh process.
    result = subprocess.run(
        [sys.executable, "-c", ROUTER_FRESH_PROCESS_SCRIPT],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    lines = result.stdout.splitlines()
    assert "gate GateLinear torch.float32" in lines
    assert "logits torch.float32 4 352" in lines
    assert "bias torch.float32 352" in lines
    assert "routing sigmoid 8 True 3.0" in lines
    assert "weight_policy sigmoid_renorm_scaled" in lines


def test_sigmoid_bias_router_rejects_other_routers():
    pytest.importorskip("torch")
    from dataclasses import replace

    from frontier.profiling.moe.moe_impl import SigmoidBiasGatingNetwork

    for router in (
        replace(STEP5_ROUTER, scoring_func="softmax"),
        replace(STEP5_ROUTER, uses_selection_bias=False),
        replace(STEP5_ROUTER, gate_precision=PrecisionType.BF16),
    ):
        with pytest.raises(ValueError, match="sigmoid routers"):
            SigmoidBiasGatingNetwork(
                hidden_dim=64, num_experts=8, router_topk=2, router=router
            )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"use_vllm_kernel": False}, "use_vllm_kernel=True"),
        ({"routing_runtime_path": "uniform_topk"}, "time their own router"),
    ],
)
def test_wrapper_rejects_layer_groups_on_paths_that_ignore_them(updates, message):
    pytest.importorskip("torch")
    from frontier.profiling.moe.moe_wrapper import MoEWrapper

    model_config = ModelConfig.from_model_name(STEP5)
    kwargs = dict(
        model_config=model_config, num_tensor_parallel_workers=1,
        expert_parallel_size=32, profile_method="cuda_event", rank=0,
        output_dir="unused", routed_layer_group=resolve_routed_layer_groups(model_config)[0],
    )
    kwargs.update(updates)
    with pytest.raises(ValueError, match=message):
        MoEWrapper(**kwargs)


def test_kernel_times_bf16_swiglustep_experts_of_the_last_layers(cuda_runtime):
    torch = cuda_runtime.torch
    cuda_runtime.expert_map.numel.return_value = 352
    cuda_runtime.layout.return_value = (32, 3)
    cuda_runtime.routed_experts.backend = "TRITON"
    cuda_runtime.routed_experts.experts_class = "TritonExperts"
    cuda_runtime.routed_experts.triton_tuning_config = "default"
    cuda_runtime.args.update(
        num_experts=11, global_num_experts=352, hidden_dim=4096,
        expert_hidden_dim=1536, use_fp8=False, block_shape=None,
        dtype=torch.bfloat16, model_type="step5", activation=SWIGLUSTEP_ACTIVATION,
    )

    stats = cuda_runtime.kernel.profile_fused_moe_kernel(**cuda_runtime.args)

    assert stats == {
        "mean": 2.5,
        "native_backend": "TRITON",
        "native_experts_class": "TritonExperts",
        "triton_tuning_config": "default",
    }
    assert cuda_runtime.build.call_args.kwargs == dict(
        quantization_mode="bf16", activation="swiglustep",
        num_local_experts=11, global_num_experts=352, hidden_dim=4096,
        intermediate_dim=1536, top_k=8, block_shape=None,
        expert_parallel_size=32, expert_parallel_rank=3,
        dtype=torch.bfloat16, model_type="step5",
    )


def test_profile_row_records_the_measured_layer_group():
    pytest.importorskip("torch")
    from frontier.profiling.moe.moe_wrapper import MoEWrapper

    model_config = ModelConfig.from_model_name(STEP5)
    group = resolve_routed_layer_groups(model_config)[1]
    wrapper = object.__new__(MoEWrapper)
    for name, value in dict(
        model_config=model_config, routed_layer_group=group,
        num_experts=352, num_experts_per_device=11, expert_parallel_size=32,
        routing_runtime_metadata={"routing_assignment_policy": "biased_score_topk"},
        gating_runtime_context_metadata={}, router_topk=8, hidden_dim=4096,
        expert_hidden_dim=1536, use_gated=True, num_tensor_parallel_workers=1,
    ).items():
        setattr(wrapper, name, value)

    row = wrapper._build_profile_result(
        {"moe_grouped_gemm": {"mean": 2.5}}, num_tokens=16
    )

    assert row["moe_layer_ids"] == "88-90"
    assert row["moe_routed_precision"] == "bf16"
    assert row["moe_activation"] == "swiglustep"
    assert row["routing_assignment_policy"] == "biased_score_topk"
    assert row["expert_hidden_dim"] == 1536
