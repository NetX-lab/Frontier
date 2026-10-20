"""DeepSeek-V4 routed experts: checkpoint MXFP4 weights through vLLM's MARLIN path.

DeepSeek-V4 stores MXFP4 routed experts, so vLLM loads them with ``Mxfp4MoEMethod``
and selects MARLIN on Hopper. These tests pin that selection contract on CPU: the
reported backend label, the guard that pairs a weight source with the encoding,
and the experts' configuration. The MARLIN repack itself needs a CUDA kernel, so
the build test replaces it and checks what the profiler passes to vLLM.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.moe import main as moe_main

DSV4 = "deepseek-ai/DeepSeek-V4-Flash"
GLM = "zai-org/GLM-5.3"


@pytest.fixture(scope="module")
def dsv4_config() -> ModelConfig:
    return ModelConfig.from_model_name(DSV4)


def test_checkpoint_mxfp4_is_labelled_by_the_engine_that_runs_it():
    from frontier.profiling.moe.moe_wrapper import (
        MXFP4_ENGINE,
        resolve_grouped_gemm_backend,
    )

    def resolve(use_vllm_kernel: bool, mxfp4_weight_source) -> str:
        return resolve_grouped_gemm_backend(
            use_vllm_kernel=use_vllm_kernel,
            mxfp4_weight_source=mxfp4_weight_source,
        )

    assert MXFP4_ENGINE == {"online": "aiter", "checkpoint": "marlin"}
    assert resolve(True, "checkpoint") == "vllm_marlin_mxfp4"
    assert resolve(True, "online") == "vllm_aiter_mxfp4"
    assert resolve(True, None) == "vllm_fused"
    assert resolve(False, None) == "frontier_loop"
    with pytest.raises(ValueError, match="requires vLLM kernel profiling"):
        resolve(False, "checkpoint")


def test_weight_source_is_required_exactly_when_the_experts_are_mxfp4():
    from frontier.profiling.moe import moe_vllm_kernel as kernel

    kernel.validate_mxfp4_quantization(use_mxfp4=True, mxfp4_weight_source="checkpoint")
    kernel.validate_mxfp4_quantization(use_mxfp4=False, mxfp4_weight_source=None)
    with pytest.raises(ValueError, match="must be given exactly when use_mxfp4 is set"):
        kernel.validate_mxfp4_quantization(use_mxfp4=True, mxfp4_weight_source=None)
    with pytest.raises(ValueError, match="must be given exactly when use_mxfp4 is set"):
        kernel.validate_mxfp4_quantization(use_mxfp4=False, mxfp4_weight_source="online")
    with pytest.raises(ValueError, match="must be one of"):
        kernel.validate_mxfp4_quantization(
            use_mxfp4=True, mxfp4_weight_source="checkpoint_fp8"
        )


def test_checkpoint_mxfp4_runtime_requires_cuda(monkeypatch):
    from frontier.profiling.moe import moe_vllm_kernel as kernel

    monkeypatch.setattr(
        kernel, "accelerator_platform", lambda _torch: "rocm"
    )
    # The online adapter keeps its ROCm admission; the checkpoint path must
    # reject ROCm before any expert allocation.
    kernel.validate_mxfp4_runtime(
        model_type="qwen3_5_moe_text", mxfp4_weight_source="online"
    )
    with pytest.raises(ValueError, match="Checkpoint-MXFP4/MARLIN profiling requires CUDA"):
        kernel.validate_mxfp4_runtime(
            model_type="deepseek_v4", mxfp4_weight_source="checkpoint"
        )


def test_deepseek_v4_is_mxfp4_in_the_quantization_manager(dsv4_config):
    from frontier.profiling.common.utils import (
        configure_quantization_manager_for_model_name,
        get_operation_precision,
    )

    pytest.importorskip("torch")
    configure_quantization_manager_for_model_name(dsv4_config.name)
    # The routed experts are MXFP4 while the checkpoint's own quant_method is FP8
    # for the base and shared weights, so the source cannot be read from the
    # checkpoint's quantization method.
    assert get_operation_precision("moe_grouped_gemm").name == "FP4"
    assert dsv4_config.quantization_config.quant_method == "fp8"
    assert dsv4_config.expert_dtype == "fp4"
    assert ModelConfig.from_model_name(GLM).expert_dtype is None


def test_dry_run_reports_the_mxfp4_expert_parallel_grid(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "moe.main", "--dry_run", "--device", "h800", "--models", DSV4,
            "--num_tensor_parallel_workers", "1",
            "--expert_parallel_sizes", "8", "16", "32",
            "--max_tokens", "32768", "--num_tokens_list", "1", "256", "32768",
        ],
    )
    moe_main.main()
    import json

    out = capsys.readouterr().out
    plan = json.loads(out[out.index("{\n"):])
    model = plan["models"][DSV4]

    assert model["local_experts_by_expert_parallel_size"] == {"8": 32, "16": 16, "32": 8}
    assert model["num_experts_per_tok"] == 6
    assert model["profiled_steps"] == (
        len(plan["num_tokens"])
        * len(model["local_experts_by_expert_parallel_size"])
        * plan["cases_per_parallel_point"]
    )


def test_builder_configures_vllm_marlin_experts_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as mxfp4_oracle

    from frontier.profiling.moe import mxfp4_marlin_experts as builder

    selected = {}
    repacked = {}

    def select(config):
        selected["config"] = config
        return mxfp4_oracle.Mxfp4MoeBackend.MARLIN, MarlinExperts

    def repack(layer, w13, w2, w13_scale, w2_scale, w13_bias, w2_bias):
        # MARLIN's reorder needs a CUDA kernel; keep the inputs so the test can
        # check the packed layout the profiler asked vLLM to adopt.
        repacked.update(layer=layer, w13_scale=w13_scale, w2_scale=w2_scale)
        return w13, w2, w13_scale, w2_scale, None, None

    monkeypatch.setattr(
        mxfp4_oracle, "select_deepseek_v4_mxfp4_moe_backend", select
    )
    monkeypatch.setattr(
        mxfp4_oracle, "convert_weight_to_mxfp4_moe_kernel_format",
        lambda **kwargs: (repacked.update(layer=kwargs["layer"]) or kwargs) and (
            kwargs["w13_weight"], kwargs["w2_weight"],
            kwargs["w13_weight_scale"], kwargs["w2_weight_scale"], None, None,
        ),
    )
    expert_map = torch.tensor(
        [index if index // 8 == 3 else -1 for index in range(256)],
        dtype=torch.int32,
    )
    routed = builder._build_mxfp4_marlin_routed_experts(
        num_local_experts=8, global_num_experts=256, hidden_dim=256,
        intermediate_dim=128, top_k=2, swiglu_limit=10.0,
        expert_parallel_size=32, expert_parallel_rank=3, expert_map=expert_map,
        dtype=torch.bfloat16, model_type="deepseek_v4", device=torch.device("cpu"),
    )

    config = selected["config"]
    assert routed.backend == "MARLIN"
    assert (config.num_experts, config.num_local_experts) == (256, 8)
    assert config.routing_method.name == "Unspecified"
    assert config.swiglu_limit == 10.0
    assert config.moe_parallel_config.ep_size == 32
    assert config.moe_parallel_config.all2all_backend == "allgather_reducescatter"
    assert routed.experts.quant_config.gemm1_clamp_limit == 10.0
    # Shapes follow the packed MXFP4 layout: one byte per two values, one E8M0
    # byte per group of 32 inputs.
    quant_config = routed.experts.quant_config
    assert routed.w13.shape == (8, 256, 128)
    assert quant_config.w1_scale.shape == (8, 256, 8)
    assert routed.w2.shape == (8, 256, 64)
    assert quant_config.w2_scale.shape == (8, 256, 4)
    assert quant_config.w1_scale.dtype == torch.uint8
    assert int(quant_config.w1_scale[0, 0, 0]) == 127
    assert routed.expert_map is not None


def test_builder_rejects_backends_other_than_marlin(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
    from vllm.model_executor.layers.fused_moe.oracle import mxfp4 as mxfp4_oracle

    from frontier.profiling.moe import mxfp4_marlin_experts as builder

    monkeypatch.setattr(
        mxfp4_oracle, "select_deepseek_v4_mxfp4_moe_backend",
        lambda config: (mxfp4_oracle.Mxfp4MoeBackend.BATCHED_MARLIN, MarlinExperts),
    )
    with pytest.raises(RuntimeError, match="supports only MARLIN"):
        builder._build_mxfp4_marlin_routed_experts(
            num_local_experts=8, global_num_experts=256, hidden_dim=256,
            intermediate_dim=128, top_k=2, swiglu_limit=None,
            expert_parallel_size=32, expert_parallel_rank=3, expert_map=None,
            dtype=torch.bfloat16, model_type="deepseek_v4",
            device=torch.device("cpu"),
        )
