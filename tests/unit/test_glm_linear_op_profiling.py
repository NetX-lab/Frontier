"""CPU checks of the GLM-5.3 linear-op producer: plans, rows, identity, MLA shapes."""

from __future__ import annotations

import json
import os
import sys

import pytest

from frontier.profiling.common import runtime_identity
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op import main as linear_main
from frontier.profiling.linear_op.profiling_plan import (
    build_sequence_parallel_shared_expert_plan,
)

GLM = "zai-org/GLM-5.3"
GLM_CONFIG_SHA256 = "3ac72612095574542f7fff847ada8e59d9199dd8af44bdf625d7e02615572e69"
SHARED_EXPERT_OPS = ["share_expert_up_proj", "share_expert_down_proj", "share_expert_act"]


@pytest.fixture(scope="module")
def glm_config() -> ModelConfig:
    return ModelConfig.from_model_name(GLM)


def _parse_args(monkeypatch, *argv: str):
    monkeypatch.setattr(sys, "argv", ["linear_op.main", *argv])
    return linear_main.parse_args()


def test_sequence_parallel_plan_replicates_full_shared_expert(glm_config):
    plan = build_sequence_parallel_shared_expert_plan(
        glm_config, attention_tp_size=8, attention_dp_size=4, expert_parallel_size=32
    )

    assert plan["tp_size"] == 1
    assert plan["enabled_ops"] == SHARED_EXPERT_OPS
    assert plan["padded_n_expanded_embd"] == 2048
    assert plan["sequence_parallel_moe_topology"] == {
        "attention_tp_size": 8,
        "attention_dp_size": 4,
        "expert_parallel_size": 32,
    }
    for contract in plan["typed_operator_contracts"].values():
        assert contract["operator_family_id"] == "share_expert"
        assert contract["tensor_parallel_mode"] == "replicated"
        assert contract["tensor_parallel_sizes"] == [1]
        assert contract["selected_tensor_parallel_size"] == 1
        assert contract["effective_ffn_width"] == 2048


@pytest.mark.parametrize(
    ("attention_tp", "attention_dp", "expert_parallel"),
    [(8, 1, 32), (1, 4, 32), (8, 4, 1)],
)
def test_sequence_parallel_plan_rejects_non_sequence_parallel_topology(
    glm_config, attention_tp, attention_dp, expert_parallel
):
    with pytest.raises(ValueError, match="resolves to attention_tp"):
        build_sequence_parallel_shared_expert_plan(
            glm_config,
            attention_tp_size=attention_tp,
            attention_dp_size=attention_dp,
            expert_parallel_size=expert_parallel,
        )


def test_sequence_parallel_plan_rejects_profile_without_replication():
    with pytest.raises(ValueError, match="does not replicate the shared expert"):
        build_sequence_parallel_shared_expert_plan(
            ModelConfig.from_model_name("llama3.1-8b"),
            attention_tp_size=8,
            attention_dp_size=4,
            expert_parallel_size=32,
        )


def test_dry_run_lists_tp_sweep_and_sequence_parallel_plans(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "linear_op.main",
            "--dry_run",
            "--models",
            GLM,
            "--num_tensor_parallel_workers",
            "1",
            "8",
            "--max_tokens",
            "8192",
            "--num_tokens_list",
            "1",
            "256",
            "8192",
            "--sequence_parallel_moe_topology",
            "8",
            "4",
            "32",
        ],
    )
    linear_main.main()
    summary = json.loads(capsys.readouterr().out)

    model = summary["models"][GLM]
    assert model["attention_family"] == "dsa_attention"
    assert model["quantization"] == {"quant_method": "fp8", "weight_block_size": [128, 128]}
    assert model["profiled_steps"] == 9
    tp1, tp8, sequence_parallel = model["plans"]
    assert (tp1["tp_size"], tp8["tp_size"], sequence_parallel["tp_size"]) == (1, 8, 1)
    assert tp8["operator_contracts"]["attn_pre_proj"]["tensor_parallel_mode"] == "attention_tp"
    assert tp8["operator_contracts"]["mlp_up_proj"]["effective_ffn_width"] == 12288
    assert tp8["operator_contracts"]["share_expert_up_proj"]["selected_tensor_parallel_size"] == 8
    assert sequence_parallel["enabled_ops"] == SHARED_EXPERT_OPS
    assert {
        contract["tensor_parallel_mode"]
        for contract in sequence_parallel["operator_contracts"].values()
    } == {"replicated"}


def test_revisions_must_be_given_together(monkeypatch):
    with pytest.raises(SystemExit):
        _parse_args(monkeypatch, "--dry_run", "--source_revision", "abc")


def test_rows_keep_only_backends_of_their_operators(glm_config):
    result = {
        "time_stats": {
            "attn_pre_proj": {"median": 1.0},
            "attn_post_proj": {"median": 2.0},
            "input_layernorm": {"median": 3.0},
        },
        "num_tokens": 64,
        "num_tensor_parallel_workers": 8,
        "n_embd": 6144,
        "padded_n_embd": 6144,
        "linear_backends": {
            "attn_pre_proj": ["CutlassFp8BlockScaledMMKernel"],
            "attn_post_proj": ["DeepGemmFp8BlockScaledMMKernel"],
        },
    }

    sharded, replicated = linear_main._materialize_profile_result_rows(
        result,
        measurement_type="CUDA_EVENT",
        should_split=True,
        replicated_op_names={"input_layernorm"},
        model_config=glm_config,
    )

    assert json.loads(sharded["linear_backends"]) == result["linear_backends"]
    assert "linear_backends" not in replicated
    assert replicated["num_tensor_parallel_workers"] == 1


def test_restrict_to_plan_operators_drops_context_modules():
    plan = {"enabled_ops": SHARED_EXPERT_OPS}
    stats = {name: {"median": 1.0} for name in [*SHARED_EXPERT_OPS, "mlp_up_proj"]}

    restricted = linear_main._restrict_to_plan_operators({"time_stats": stats}, plan)
    assert list(restricted["time_stats"]) == SHARED_EXPERT_OPS

    with pytest.raises(RuntimeError, match="lacks planned operators"):
        linear_main._restrict_to_plan_operators(
            {"time_stats": {"share_expert_up_proj": {}}}, plan
        )


def test_runtime_identity_columns_match_model_config(monkeypatch, glm_config):
    monkeypatch.setattr(runtime_identity, "collect_device_identity", lambda: ("NVIDIA H800", "sm90"))
    monkeypatch.setattr(
        runtime_identity, "collect_runtime_versions", lambda: {"vllm": "0.28.0", "torch": "2.10"}
    )
    args = _parse_args(
        monkeypatch,
        "--dry_run",
        "--source_revision",
        "f01a1d7",
        "--model_config_revision",
        "082a2aa294ad5f4056895c42b297b4af24703384",
    )
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(
        [{"num_tokens": 1, "quant_signature": glm_config.get_quant_signature()}]
    )

    attached = linear_main._attach_runtime_identity(frame, args, GLM, glm_config)

    row = attached.iloc[0]
    assert row["model_config_sha256"] == GLM_CONFIG_SHA256
    assert row["schema_version"] == linear_main.LINEAR_OP_SCHEMA_VERSION
    assert row["weight_source"] == "dummy"
    assert row["graph_mode"] == "eager"
    assert json.loads(row["parallel_groups"]) == {"engine_tensor_parallel": 1}
    assert set(runtime_identity.IDENTITY_COLUMNS) <= set(attached.columns)


def test_mla_attention_linear_ops_shapes_on_cpu(glm_config):
    """Build the GLM MLA projections with vLLM's CPU kernels and run one forward."""

    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CPU shape check; GPU runs exercise the CUDA kernels")
    os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")
    pytest.importorskip("vllm")
    import contextlib

    import frontier.profiling.common.parallel_utils.tensor_parallel_layers as layers
    from frontier.profiling.common.cuda_timer import CudaTimer
    from frontier.profiling.linear_op.linear_op_impl import build_linear_op_attention_module
    from frontier.profiling.linear_op.mla_attention import MlaAttentionLinearOps

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.cuda, "current_device", lambda: "cpu")
            patch.setattr(
                layers,
                "_run_unquantized_gemm",
                lambda layer, x, weight, bias: torch.nn.functional.linear(x, weight, bias),
            )
            module = build_linear_op_attention_module(glm_config, 8, None, True)
            for submodule in module.modules():
                for name, value in list(vars(submodule).items()):
                    if isinstance(value, CudaTimer):
                        setattr(submodule, name, contextlib.nullcontext())
            with torch.no_grad():
                output = module(torch.randn(5, 6144), torch.arange(5))
    finally:
        torch.set_default_dtype(previous_dtype)

    assert isinstance(module, MlaAttentionLinearOps)
    shapes = {name: tuple(param.shape) for name, param in module.named_parameters()}
    assert shapes["fused_qkv_a_proj.weight"] == (2048 + 512 + 64, 6144)
    assert shapes["q_b_proj.weight"] == (64 * 256 // 8, 2048)
    assert shapes["o_proj.weight"] == (6144, 64 * 256 // 8)
    assert module.rotary_emb.is_neox_style is False
    assert module.rotary_emb.rotary_dim == 64
    assert tuple(output.shape) == (5, 6144)
