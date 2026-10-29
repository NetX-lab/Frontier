"""GLM-5.3 routed-expert profiling with vLLM 0.28 block-FP8 experts.

The dry-run and identity tests need only the simulator environment. The kernel
tests fake the CUDA runtime, and the builder tests construct the vLLM kernel
on CPU, so none of them measures timing.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

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

    # FlashInfer never runs the Triton kernel, so no tuning config is reported.
    assert stats == {
        "mean": 2.5,
        "native_backend": "FLASHINFER_CUTLASS",
        "native_experts_class": "FlashInferExperts",
    }
    cuda_runtime.layout.assert_called_once_with(
        cuda_runtime.args["expert_map"], num_local_experts=8, global_num_experts=256
    )
    assert cuda_runtime.build.call_args.kwargs == dict(
        quantization_mode="fp8", activation="silu",
        num_local_experts=8, global_num_experts=256, hidden_dim=6144,
        intermediate_dim=2048, top_k=8, block_shape=[128, 128],
        expert_parallel_size=32, expert_parallel_rank=0,
        dtype=cuda_runtime.torch.bfloat16, model_type="glm_moe_dsa",
    )
    # Only the activations are allocated; the experts own their FP8 weights.
    cuda_runtime.runtime.randn.assert_called_once()
    hidden_states, weights, ids, expert_map = cuda_runtime.routed_experts.apply.call_args.args
    assert hidden_states.shape == (4, 6144)
    assert weights is ids is cuda_runtime.routing
    assert expert_map is cuda_runtime.args["expert_map"]


def test_wrapper_reports_the_block_fp8_backend(cuda_runtime):
    from frontier.profiling.moe.moe_wrapper import MoEWrapper

    wrapper = object.__new__(MoEWrapper)
    for name, value in dict(
        model_config=SimpleNamespace(model_type="glm_moe_dsa"),
        use_vllm_kernel=True, use_mxfp4=False, use_fp8=True,
        num_experts=256, num_experts_per_device=8, hidden_dim=6144,
        expert_hidden_dim=2048, router_topk=8, num_tensor_parallel_workers=1,
        _dtype=cuda_runtime.torch.bfloat16, per_channel_quant=False,
        block_shape=[128, 128], profile_method="cuda_event", output_dir=None,
        quantization_mode="fp8", activation="silu",
    ).items():
        setattr(wrapper, name, value)
    row = wrapper.profile_grouped_gemm(4, routing_inputs=dict(
        topk_weights=cuda_runtime.routing, topk_ids=cuda_runtime.routing,
        expert_token_counts=[4] * 8, global_num_experts=256,
        expert_map=cuda_runtime.args["expert_map"],
    ))

    assert row["moe_native_backend"] == "FLASHINFER_CUTLASS"
    assert row["moe_native_experts_class"] == "FlashInferExperts"
    assert "moe_triton_tuning_config" not in row
    assert row["moe_grouped_gemm_backend"] == "vllm_fused"
    assert row["time_stats"] == {"moe_grouped_gemm": {"mean": 2.5}}


def test_builder_configures_flashinfer_block_fp8_experts_on_cpu(
    monkeypatch, cpu_vllm_platform
):
    torch = pytest.importorskip("torch")
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
        FlashInferExperts,
    )
    from vllm.model_executor.layers.fused_moe.oracle import fp8 as fp8_oracle

    from frontier.profiling.moe.fp8_block_experts import _build_native_routed_experts

    selected = {}

    def select_on_hopper(*, config, weight_key, activation_key, allow_vllm_cutlass):
        selected.update(config=config, allow_vllm_cutlass=allow_vllm_cutlass)
        return fp8_oracle.Fp8MoeBackend.FLASHINFER_CUTLASS, FlashInferExperts

    monkeypatch.setattr(fp8_oracle, "select_fp8_moe_backend", select_on_hopper)
    routed = _build_native_routed_experts(
        quantization_mode="fp8", activation="silu",
        num_local_experts=4, global_num_experts=16, hidden_dim=256,
        intermediate_dim=128, top_k=2, block_shape=(128, 128),
        expert_parallel_size=4, expert_parallel_rank=1, dtype=torch.bfloat16,
        model_type="glm_moe_dsa", device=torch.device("cpu"),
    )

    config = selected["config"]
    experts = routed.kernel.fused_experts
    assert selected["allow_vllm_cutlass"] is False
    assert (config.num_experts, config.num_local_experts) == (16, 4)
    assert config.moe_parallel_config.all2all_backend == "allgather_reducescatter"
    assert type(routed.kernel.prepare_finalize).__name__ == (
        "MoEPrepareAndFinalizeNoDPEPModular"
    )
    assert (routed.backend, routed.experts_class) == ("FLASHINFER_CUTLASS", "FlashInferExperts")
    assert routed.triton_tuning_config is None
    assert (experts.ep_size, experts.ep_rank, experts.tp_size) == (4, 1, 1)
    assert experts.use_deepseek_fp8_block_scale is True
    assert routed.w13.dtype == routed.w2.dtype == torch.float8_e4m3fn
    assert routed.w13.shape[0] == routed.w2.shape[0] == 4


def test_builder_rejects_unsupported_fp8_backends(monkeypatch, cpu_vllm_platform):
    torch = pytest.importorskip("torch")
    from vllm.model_executor.layers.fused_moe.oracle import fp8 as fp8_oracle

    from frontier.profiling.moe.fp8_block_experts import _build_native_routed_experts

    monkeypatch.setattr(
        fp8_oracle,
        "select_fp8_moe_backend",
        lambda **_: (fp8_oracle.Fp8MoeBackend.VLLM_CUTLASS, None),
    )
    with pytest.raises(RuntimeError, match="VLLM_CUTLASS fp8 MoE backend"):
        _build_native_routed_experts(
            quantization_mode="fp8", activation="silu",
            num_local_experts=4, global_num_experts=16, hidden_dim=256,
            intermediate_dim=128, top_k=2, block_shape=(128, 128),
            expert_parallel_size=4, expert_parallel_rank=1, dtype=torch.bfloat16,
            model_type="glm_moe_dsa", device=torch.device("cpu"),
        )
