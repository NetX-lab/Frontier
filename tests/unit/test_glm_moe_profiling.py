"""GLM-5.3 routed-expert profiling with vLLM 0.28 block-FP8 experts.

The dry-run and identity tests need only the simulator environment. The kernel
tests fake the CUDA runtime, and the builder test constructs the vLLM experts
on CPU, so neither measures timing.
"""

from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from frontier.profiling.common import runtime_identity
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.moe import main as moe_main

GLM = "zai-org/GLM-5.3"


def _parse_args(monkeypatch, *argv: str):
    monkeypatch.setattr(sys, "argv", ["moe.main", *argv])
    args, _ = moe_main.parse_args()
    return args


def test_dry_run_lists_block_fp8_expert_parallel_grid(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "moe.main", "--dry_run", "--device", "h800", "--models", GLM,
            "--num_tensor_parallel_workers", "1",
            "--expert_parallel_sizes", "8", "16", "32",
            "--max_tokens", "8192", "--num_tokens_list", "1", "256", "8192",
        ],
    )
    moe_main.main()
    out = capsys.readouterr().out
    plan = json.loads(out[out.index("{\n"):])
    model = plan["models"][GLM]

    assert model["use_fp8"] is True
    assert model["block_shape"] == [128, 128]
    assert model["local_experts_by_expert_parallel_size"] == {"8": 32, "16": 16, "32": 8}
    assert model["profiled_steps"] == (
        len(plan["num_tokens"]) * 3 * plan["cases_per_parallel_point"]
    )


def test_revisions_must_be_given_together(monkeypatch):
    with pytest.raises(SystemExit):
        _parse_args(
            monkeypatch, "--dry_run", "--device", "h800", "--model_config_revision", "abc"
        )


def test_runtime_identity_columns_record_the_measured_table(monkeypatch):
    monkeypatch.setattr(runtime_identity, "collect_device_identity", lambda: ("NVIDIA H800", "sm90"))
    monkeypatch.setattr(
        runtime_identity, "collect_runtime_versions", lambda: {"vllm": "0.28.0"}
    )
    args = _parse_args(
        monkeypatch, "--device", "h800", "--models", GLM, "--source_revision", "src",
        "--model_config_revision", "cfg",
    )
    quant_signature = ModelConfig.from_model_name(GLM).get_quant_signature()
    rows = moe_main._attach_runtime_identity(
        pd.DataFrame([{"num_tokens": 1}]), args, GLM, quant_signature
    ).to_dict("records")

    assert rows[0]["schema_version"] == "moe.v1"
    assert rows[0]["source_revision"] == "src"
    assert rows[0]["model_config_revision"] == "cfg"
    assert rows[0]["model_config_sha256"] == runtime_identity.file_sha256(
        runtime_identity.model_config_json_path(GLM)
    )
    assert rows[0]["device_name"] == "NVIDIA H800"
    assert rows[0]["quant_signature"] == quant_signature


def test_expert_parallel_layout_accepts_only_contiguous_aligned_maps():
    torch = pytest.importorskip("torch")
    from frontier.profiling.moe.fp8_block_experts import expert_parallel_layout

    def contiguous(start: int, local: int, total: int = 256):
        expert_map = torch.full((total,), -1, dtype=torch.int32)
        expert_map[start : start + local] = torch.arange(local, dtype=torch.int32)
        return expert_map

    assert expert_parallel_layout(contiguous(64, 32), num_local_experts=32, global_num_experts=256) == (8, 2)
    assert expert_parallel_layout(contiguous(248, 8), num_local_experts=8, global_num_experts=256) == (32, 31)
    assert expert_parallel_layout(None, num_local_experts=256, global_num_experts=256) == (1, 0)

    shuffled = contiguous(0, 8)
    shuffled[:2] = torch.tensor([1, 0], dtype=torch.int32)
    for expert_map, local in ((contiguous(4, 8), 8), (contiguous(0, 7), 8), (shuffled, 8)):
        with pytest.raises(ValueError):
            expert_parallel_layout(expert_map, num_local_experts=local, global_num_experts=256)
    with pytest.raises(ValueError, match="expert_map is required"):
        expert_parallel_layout(None, num_local_experts=8, global_num_experts=256)


@pytest.fixture
def cuda_runtime(monkeypatch):
    torch = pytest.importorskip("torch")
    from frontier.profiling.moe import moe_vllm_kernel as kernel

    runtime = SimpleNamespace(
        version=SimpleNamespace(hip=None, cuda="12.8"),
        cuda=SimpleNamespace(synchronize=Mock()),
        randn=Mock(side_effect=lambda *shape, **kwargs: torch.zeros(*shape, dtype=kwargs["dtype"])),
        empty_like=torch.empty_like,
        bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32,
        int32=torch.int32,
    )
    monkeypatch.setattr(kernel, "torch", runtime)
    monkeypatch.setattr(kernel, "VLLM_AVAILABLE", True)
    monkeypatch.setattr(kernel, "VLLM_API_VERSION", "functional_fused_experts")
    monkeypatch.setattr(
        "frontier.profiling.common.vllm_compat.vllm_config_context", lambda: nullcontext()
    )
    monkeypatch.setattr(
        kernel,
        "_collect_cuda_event_stats",
        lambda step_fn, active_steps: (step_fn(), {"mean": 2.5})[1],
    )
    routed_experts = SimpleNamespace(backend="FLASHINFER_CUTLASS", apply=Mock())
    build = Mock(return_value=routed_experts)
    monkeypatch.setattr(kernel, "get_fp8_block_routed_experts", build)
    layout = Mock(return_value=(32, 0))
    monkeypatch.setattr(kernel, "expert_parallel_layout", layout)
    routing = Mock()
    routing.to.return_value = routing
    routing.contiguous.return_value = routing
    expert_map = Mock()
    expert_map.numel.return_value = 256
    expert_map.to.return_value = expert_map
    args = dict(
        num_tokens=4, num_experts=8, hidden_dim=6144, expert_hidden_dim=2048, top_k=8,
        topk_weights=routing, topk_ids=routing, warmup_steps=0, active_steps=2,
        profile_method="cuda_event", use_fp8=True, block_shape=[128, 128],
        global_num_experts=256, expert_map=expert_map, model_type="glm_moe_dsa",
    )
    return SimpleNamespace(
        kernel=kernel, runtime=runtime, args=args, build=build, layout=layout,
        routed_experts=routed_experts, routing=routing, torch=torch,
    )


@pytest.mark.parametrize(
    "updates, error",
    [
        ({"block_shape": None}, NotImplementedError),
        ({"per_channel_quant": True}, NotImplementedError),
        ({"block_shape": [128]}, ValueError),
        ({"tensor_parallel_size": 2}, ValueError),
    ],
)
def test_block_fp8_kernel_rejects_unsupported_layouts(cuda_runtime, updates, error):
    cuda_runtime.args.update(updates)
    with pytest.raises(error):
        cuda_runtime.kernel.profile_fused_moe_kernel(**cuda_runtime.args)
    cuda_runtime.runtime.randn.assert_not_called()
    cuda_runtime.build.assert_not_called()


def test_block_fp8_kernel_times_the_selected_vllm_experts(cuda_runtime):
    stats = cuda_runtime.kernel.profile_fused_moe_kernel(**cuda_runtime.args)

    assert stats == {"mean": 2.5, "native_backend": "FLASHINFER_CUTLASS"}
    cuda_runtime.layout.assert_called_once_with(
        cuda_runtime.args["expert_map"], num_local_experts=8, global_num_experts=256
    )
    assert cuda_runtime.build.call_args.kwargs == dict(
        num_local_experts=8, global_num_experts=256, hidden_dim=6144,
        intermediate_dim=2048, top_k=8, block_shape=[128, 128],
        expert_parallel_size=32, expert_parallel_rank=0,
        dtype=cuda_runtime.torch.bfloat16, model_type="glm_moe_dsa",
    )
    # Only the activations are allocated; the experts own their FP8 weights.
    cuda_runtime.runtime.randn.assert_called_once()
    output, hidden_states, weights, ids = cuda_runtime.routed_experts.apply.call_args.args
    assert output.shape == hidden_states.shape == (4, 6144)
    assert weights is ids is cuda_runtime.routing


def test_wrapper_reports_the_block_fp8_backend(cuda_runtime):
    from frontier.profiling.moe.moe_wrapper import MoEWrapper

    wrapper = object.__new__(MoEWrapper)
    for name, value in dict(
        model_config=SimpleNamespace(model_type="glm_moe_dsa", swiglu_limit=None),
        use_vllm_kernel=True, use_mxfp4=False, use_fp8=True,
        mxfp4_weight_source=None,
        num_experts=256, num_experts_per_device=8, hidden_dim=6144,
        expert_hidden_dim=2048, router_topk=8, num_tensor_parallel_workers=1,
        _dtype=cuda_runtime.torch.bfloat16, per_channel_quant=False,
        block_shape=[128, 128], profile_method="cuda_event", output_dir=None,
        quantization_mode="fp8",
    ).items():
        setattr(wrapper, name, value)
    row = wrapper.profile_grouped_gemm(4, routing_inputs=dict(
        topk_weights=cuda_runtime.routing, topk_ids=cuda_runtime.routing,
        expert_token_counts=[4] * 8, global_num_experts=256,
        expert_map=cuda_runtime.args["expert_map"],
    ))

    assert row["moe_native_backend"] == "FLASHINFER_CUTLASS"
    assert row["moe_grouped_gemm_backend"] == "vllm_fused"
    assert row["time_stats"] == {"moe_grouped_gemm": {"mean": 2.5}}


def test_builder_configures_flashinfer_block_fp8_experts_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
        FlashInferExperts,
    )
    from vllm.model_executor.layers.fused_moe.oracle import fp8 as fp8_oracle

    from frontier.profiling.moe.fp8_block_experts import _build_fp8_block_routed_experts

    selected = {}

    def select_on_hopper(*, config, weight_key, activation_key, allow_vllm_cutlass):
        selected.update(config=config, allow_vllm_cutlass=allow_vllm_cutlass)
        return fp8_oracle.Fp8MoeBackend.FLASHINFER_CUTLASS, FlashInferExperts

    monkeypatch.setattr(fp8_oracle, "select_fp8_moe_backend", select_on_hopper)
    routed = _build_fp8_block_routed_experts(
        num_local_experts=4, global_num_experts=16, hidden_dim=256,
        intermediate_dim=128, top_k=2, block_shape=(128, 128),
        expert_parallel_size=4, expert_parallel_rank=1, dtype=torch.bfloat16,
        model_type="glm_moe_dsa", device=torch.device("cpu"),
    )

    config = selected["config"]
    assert selected["allow_vllm_cutlass"] is False
    assert (config.num_experts, config.num_local_experts) == (16, 4)
    assert config.moe_parallel_config.all2all_backend == "allgather_reducescatter"
    assert routed.backend == "FLASHINFER_CUTLASS"
    assert (routed.experts.ep_size, routed.experts.ep_rank, routed.experts.tp_size) == (4, 1, 1)
    assert routed.experts.use_deepseek_fp8_block_scale is True
    assert routed.w13.dtype == routed.w2.dtype == torch.float8_e4m3fn
    assert routed.w13.shape[0] == routed.w2.shape[0] == 4


def test_builder_rejects_backends_other_than_flashinfer_cutlass():
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm")
    from frontier.profiling.moe.fp8_block_experts import _build_fp8_block_routed_experts

    # Off Hopper (here: CPU) the oracle does not select FLASHINFER_CUTLASS.
    with pytest.raises(RuntimeError, match="supports only FLASHINFER_CUTLASS"):
        _build_fp8_block_routed_experts(
            num_local_experts=4, global_num_experts=16, hidden_dim=256,
            intermediate_dim=128, top_k=2, block_shape=(128, 128),
            expert_parallel_size=4, expert_parallel_rank=1, dtype=torch.bfloat16,
            model_type="glm_moe_dsa", device=torch.device("cpu"),
        )
