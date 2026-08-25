import json
from pathlib import Path

import pandas as pd
import pytest

from frontier.moe_gating_runtime import (
    DIRECT_MOE_GATING_RUNTIME_CONTEXT,
    PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT,
    get_moe_gating_runtime_context_metadata,
)
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
from tests.performance.profiling.run_h200_frozen_manifest_model import (
    _validate_linear_file,
)
from tests.performance.profiling.validate_h200_six_model_non_dummy_e2e import (
    SUPPORTED_MODELS,
    build_model_contract,
    validate_profile_directory,
    validate_runtime_artifacts,
)


def test_six_model_contract_is_registry_derived() -> None:
    assert SUPPORTED_MODELS == (
        "llama3.1-8b",
        "llama3.3-70b",
        "Qwen3-235B-A22B",
        "qwen3-a3b-30b-moe",
        "step3-moe-noquant",
        "mixtral_8x7b_moe",
    )

    dense = build_model_contract("llama3.1-8b")
    assert dense.is_moe is False
    assert dense.num_pipeline_stages == 1
    assert (
        dense.cluster_num_replicas,
        dense.attn_tensor_parallel_size,
        dense.attn_data_parallel_size,
        dense.moe_tensor_parallel_size,
        dense.moe_expert_parallel_size,
    ) == (1, 1, 1, 1, 1)
    assert dense.profile_filenames == (
        "linear_op.csv",
        "linear_op_kernel_only.csv",
        "attention.csv",
        "attention_kernel_only.csv",
    )

    large_dense = build_model_contract("llama3.3-70b")
    assert large_dense.num_pipeline_stages == 2

    pure_moe = build_model_contract("Qwen3-235B-A22B")
    assert pure_moe.is_moe is True
    assert pure_moe.is_mixed_layer_moe is False
    assert pure_moe.num_pipeline_stages == 2
    assert "time_stats.mlp_up_proj.median" not in pure_moe.linear_target_columns
    assert pure_moe.routing_runtime_path == "standard_fused_topk"
    assert (
        pure_moe.vllm_tensor_parallel_size,
        pure_moe.vllm_data_parallel_size,
        pure_moe.enable_expert_parallel,
        pure_moe.attn_tensor_parallel_size,
        pure_moe.attn_data_parallel_size,
        pure_moe.moe_tensor_parallel_size,
        pure_moe.moe_expert_parallel_size,
    ) == (1, 2, True, 1, 2, 1, 2)

    step3 = build_model_contract("step3-moe-noquant")
    assert step3.is_moe is True
    assert step3.is_mixed_layer_moe is True
    assert step3.num_pipeline_stages == 1
    assert (
        step3.vllm_tensor_parallel_size,
        step3.vllm_data_parallel_size,
        step3.enable_expert_parallel,
        step3.cluster_num_replicas,
        step3.attn_tensor_parallel_size,
        step3.attn_data_parallel_size,
        step3.moe_tensor_parallel_size,
        step3.moe_expert_parallel_size,
    ) == (8, 1, True, 1, 8, 1, 1, 8)
    assert step3.dense_layer_count == 5
    assert step3.moe_layer_count == 56
    assert "time_stats.mlp_up_proj.median" in step3.linear_target_columns
    assert "time_stats.share_expert_up_proj.median" in step3.linear_target_columns
    assert "time_stats.attn_pre_proj_qkv.median" in step3.linear_target_columns
    assert step3.profile_filenames[-2:] == ("moe.csv", "moe_kernel_only.csv")
    assert dict(step3.linear_tp_by_op)["attn_pre_proj"] == 8
    # Dense boundary layers use the dense FFN predictor domain (attention TP8)
    # even though the model also contains routed MoE layers (MoE TP1/EP8).
    assert dict(step3.linear_tp_by_op)["mlp_up_proj"] == 8
    assert dict(step3.linear_tp_by_op)["share_expert_up_proj"] == 1


def test_explicit_uniform_mode_selects_uniform_profile_rows() -> None:
    contract = build_model_contract(
        "mixtral_8x7b_moe",
        moe_routing_mode="uniform_random",
    )
    assert contract.routing_runtime_path == "uniform_topk"


def _write_profile_fixture(profile_dir: Path, model_name: str) -> None:
    contract = build_model_contract(model_name)
    direct_gating_metadata = get_moe_gating_runtime_context_metadata(
        DIRECT_MOE_GATING_RUNTIME_CONTEXT
    )
    common_metadata = {
        "profiling_precision": contract.profiling_precision,
        "quant_signature": contract.quant_signature,
        "model_arch": contract.model_arch,
        "model_architecture_profile": contract.model_architecture_profile,
    }
    linear_metadata = {
        **common_metadata,
        "n_head": contract.num_q_heads,
        "n_kv_head": contract.num_kv_heads,
        "n_embd": contract.embedding_dim,
        "n_expanded_embd": contract.mlp_hidden_dim,
        "vocab_size": contract.vocab_size,
        "use_gated_mlp": contract.use_gated_mlp,
        "use_qk_norm": contract.use_qk_norm,
        "attn_output_gate": contract.attn_output_gate,
        "padded_n_embd": contract.embedding_dim,
        "padded_n_expanded_embd": contract.mlp_hidden_dim,
        "share_expert_dim": contract.share_expert_dim,
        "share_q_dim": contract.share_q_dim,
        "num_tokens": 1,
    }
    attention_metadata = {
        **common_metadata,
        "n_embd": contract.embedding_dim,
        "n_q_head": contract.num_q_heads,
        "n_kv_head": contract.num_kv_heads,
        "block_size": 16,
        "num_tensor_parallel_workers": contract.attn_tensor_parallel_size,
        "max_model_len": 2,
        "batch_size": 1,
        "attention_backend": "FLASHINFER",
        "is_mixed_batch": False,
    }
    moe_metadata = {
        **common_metadata,
        "num_tokens": 1,
        "num_experts": contract.num_experts,
        "num_experts_per_device": (
            contract.num_experts // contract.moe_expert_parallel_size
        ),
        "expert_parallel_size": contract.moe_expert_parallel_size,
        "routing_runtime_path": contract.routing_runtime_path,
        **direct_gating_metadata,
        "router_topk": contract.router_topk,
        "hidden_dim": contract.embedding_dim,
        "expert_hidden_dim": contract.mlp_hidden_dim,
        "use_gated": contract.use_gated_mlp,
        "num_tensor_parallel_workers": contract.moe_tensor_parallel_size,
    }

    profile_dir.mkdir(parents=True)
    for filename in contract.profile_filenames:
        measurement_type = (
            "KERNEL_ONLY" if filename.endswith("_kernel_only.csv") else "CUDA_EVENT"
        )
        if filename.startswith("linear_op"):
            tp_by_op = dict(contract.linear_tp_by_op)
            rows = []
            for tp_size in sorted(set(tp_by_op.values())):
                row = {
                    **linear_metadata,
                    "num_tensor_parallel_workers": tp_size,
                    "measurement_type": measurement_type,
                }
                row.update(
                    {
                        column: (
                            1.0
                            if tp_by_op[
                                column.removeprefix("time_stats.").removesuffix(
                                    ".median"
                                )
                            ]
                            == tp_size
                            else float("nan")
                        )
                        for column in contract.linear_target_columns
                    }
                )
                rows.append(row)
            frame = pd.DataFrame(rows)
        elif filename.startswith("attention"):
            decode_row = {
                **attention_metadata,
                "measurement_type": measurement_type,
                "is_prefill": False,
                "prefill_chunk_size": 0,
                "kv_cache_size": 1,
                "time_stats.attn_decode.median": 1.0,
                "time_stats.attn_prefill.median": 0.0,
            }
            rows = [decode_row]
            if measurement_type == "CUDA_EVENT":
                rows.insert(
                    0,
                    {
                        **attention_metadata,
                        "measurement_type": measurement_type,
                        "is_prefill": True,
                        "prefill_chunk_size": 1,
                        "kv_cache_size": 0,
                        "time_stats.attn_decode.median": 0.0,
                        "time_stats.attn_prefill.median": 1.0,
                    },
                )
            frame = pd.DataFrame(rows)
        else:
            row = {
                **moe_metadata,
                **{column: 1.0 for column in contract.moe_target_columns},
                "measurement_type": measurement_type,
            }
            frame = pd.DataFrame([row])
        frame.to_csv(profile_dir / filename, index=False)


def test_profile_preflight_accepts_matching_step3_dual_family_fixture(
    tmp_path: Path,
) -> None:
    profile_dir = tmp_path / "step3-moe-noquant"
    _write_profile_fixture(profile_dir, "step3-moe-noquant")

    report = validate_profile_directory(
        profile_dir,
        build_model_contract("step3-moe-noquant"),
    )

    assert report["status"] == "PASS"
    assert report["files"]["attention.csv"]["row_count"] == 2
    assert report["files"]["attention_kernel_only.csv"]["row_count"] == 1
    assert report["files"]["moe.csv"]["routing_runtime_path"] == "standard_fused_topk"


def test_profile_preflight_selects_runtime_slice_from_multi_parallel_csvs(
    tmp_path: Path,
) -> None:
    contract = build_model_contract("step3-moe-noquant")
    profile_dir = tmp_path / contract.model_name
    _write_profile_fixture(profile_dir, contract.model_name)

    for filename in contract.profile_filenames:
        path = profile_dir / filename
        frame = pd.read_csv(path)
        non_runtime_rows = frame.copy()
        non_runtime_rows["num_tensor_parallel_workers"] = 2
        if filename.startswith("moe"):
            non_runtime_rows["expert_parallel_size"] = 4
            non_runtime_rows["num_experts_per_device"] = contract.num_experts // 4
            non_runtime_rows["gating_runtime_context"] = "prefill_warmed"
        pd.concat([frame, non_runtime_rows], ignore_index=True).to_csv(
            path,
            index=False,
        )

    report = validate_profile_directory(profile_dir, contract)

    for filename in contract.profile_filenames:
        assert report["files"][filename]["runtime_row_count"] < report["files"][
            filename
        ]["row_count"]


def test_qwen_profile_preflight_requires_prefill_warmed_runtime_slice(
    tmp_path: Path,
) -> None:
    contract = build_model_contract("Qwen3-235B-A22B")
    profile_dir = tmp_path / contract.model_name
    _write_profile_fixture(profile_dir, contract.model_name)

    with pytest.raises(
        ValueError,
        match="gating_runtime_context='prefill_warmed'",
    ):
        validate_profile_directory(profile_dir, contract)


def test_qwen_profile_preflight_selects_both_runtime_contexts(
    tmp_path: Path,
) -> None:
    contract = build_model_contract("Qwen3-235B-A22B")
    profile_dir = tmp_path / contract.model_name
    _write_profile_fixture(profile_dir, contract.model_name)
    warmed_metadata = get_moe_gating_runtime_context_metadata(
        PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT
    )

    for filename in ("moe.csv", "moe_kernel_only.csv"):
        path = profile_dir / filename
        frame = pd.read_csv(path)
        warmed_row = frame.iloc[[0]].copy()
        for column, value in warmed_metadata.items():
            warmed_row[column] = value
        pd.concat([frame, warmed_row], ignore_index=True).to_csv(
            path,
            index=False,
        )

    report = validate_profile_directory(profile_dir, contract)

    assert report["files"]["moe.csv"]["runtime_row_count"] == 2
    assert report["files"]["moe_kernel_only.csv"]["runtime_row_count"] == 2


def test_frozen_runner_accepts_fused_add_norm_and_replicated_tp_split(
    tmp_path: Path,
) -> None:
    contract = build_model_contract("llama3.1-8b")
    model_config = ModelConfig.from_model_name(contract.model_name)
    tp_sizes = (1, 2)
    manifest = {
        "scope": {
            "tp_sizes": list(tp_sizes),
            "linear_op": {"num_tokens": [1]},
        }
    }
    rows = []
    for tp_size in tp_sizes:
        plan = build_profiling_plan(
            model_config=model_config,
            tp_size=tp_size,
            attn_tp=tp_sizes,
            ffn_tp=tp_sizes,
            is_moe=False,
        )
        replicated_ops = set(plan["replicated_ops"])
        row = {
            "measurement_type": "CUDA_EVENT",
            "profiling_precision": contract.profiling_precision,
            "quant_signature": contract.quant_signature,
            "model_arch": contract.model_arch,
            "model_architecture_profile": contract.model_architecture_profile,
            "num_tensor_parallel_workers": tp_size,
            "num_tokens": 1,
        }
        for column in contract.linear_target_columns:
            op_name = column.removeprefix("time_stats.").removesuffix(".median")
            row[column] = (
                1.0 if tp_size == 1 or op_name not in replicated_ops else float("nan")
            )
        rows.append(row)

    accepted_dir = tmp_path / contract.model_name
    accepted_dir.mkdir()
    pd.DataFrame(rows).to_csv(accepted_dir / "linear_op.csv", index=False)

    report = _validate_linear_file(
        accepted_dir=accepted_dir,
        manifest=manifest,
        contract=contract,
        model_config=model_config,
        profile_method="cuda_event",
        measurement_type="CUDA_EVENT",
    )

    assert report["row_count"] == 2
    assert report["validated_target_slices"] == 15


def test_profile_preflight_rejects_wrong_measurement_family(tmp_path: Path) -> None:
    profile_dir = tmp_path / "llama3.1-8b"
    _write_profile_fixture(profile_dir, "llama3.1-8b")
    path = profile_dir / "linear_op.csv"
    frame = pd.read_csv(path)
    frame["measurement_type"] = "KERNEL_ONLY"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="linear_op.csv.*measurement_type"):
        validate_profile_directory(
            profile_dir,
            build_model_contract("llama3.1-8b"),
        )


def _write_runtime_artifacts(
    run_dir: Path,
    *,
    contract,
    profile_dir: Path,
    enable_dummy_mode: bool = False,
) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "system_metrics.json").write_text(
        json.dumps(
            {
                "simulation_metadata": {
                    "total_requests": 1,
                    "completed_requests": 1,
                    "system_architecture": "co-location",
                }
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "request_num_prefill_tokens": 2,
                "request_num_decode_tokens": 2,
                "ttft": 1.25,
                "tpot": 0.75,
                "request_e2e_time": 2.0,
            }
        ]
    ).to_csv(run_dir / "request_metrics.csv", index=False)
    ledger_rows = [
        {"execution_time": {"model_time_ms": 1.25}},
        {"execution_time": {"model_time_ms": 0.75}},
    ]
    (run_dir / "frontier_stage_batch_ledger.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ledger_rows),
        encoding="utf-8",
    )
    trace_rows = [
        {"meta": {"version": "1.0"}},
        {"type": "COMPUTE", "name": "attn_decode", "duration_ms": 0.75},
    ]
    (run_dir / "op_traces.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in trace_rows),
        encoding="utf-8",
    )
    predictor_config = {
        "enable_dummy_mode": enable_dummy_mode,
        "linear_op_input_file": str(profile_dir / "linear_op.csv"),
        "linear_op_kernel_only_input_file": str(
            profile_dir / "linear_op_kernel_only.csv"
        ),
        "atten_input_file": str(profile_dir / "attention.csv"),
        "atten_kernel_only_input_file": str(
            profile_dir / "attention_kernel_only.csv"
        ),
        "moe_input_file": str(profile_dir / "moe.csv"),
        "moe_kernel_only_input_file": str(profile_dir / "moe_kernel_only.csv"),
    }
    replica_config = {
        "model_name": contract.model_name,
        "device": "h200",
        "num_pipeline_stages": contract.num_pipeline_stages,
        "attn_tensor_parallel_size": contract.attn_tensor_parallel_size,
        "attn_data_parallel_size": contract.attn_data_parallel_size,
        "data_parallel_size": contract.attn_data_parallel_size,
        "moe_tensor_parallel_size": contract.moe_tensor_parallel_size,
        "moe_expert_parallel_size": contract.moe_expert_parallel_size,
        "total_expert_num": contract.num_experts if contract.is_moe else 1,
        "router_topk": contract.router_topk if contract.is_moe else 1,
        "moe_routing_mode": "simulation",
    }
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "simulation_mode": "offline",
                "sys_arch": "co-location",
                "decode_cuda_graph_mode": "full_decode_only",
                "request_generator_config": {
                    "name": "synthetic",
                    "num_requests": 1,
                    "length_generator_config": {
                        "name": "fixed",
                        "prefill_tokens": 2,
                        "decode_tokens": 2,
                    },
                },
                "cluster_config": {
                    "num_replicas": contract.cluster_num_replicas,
                    "replica_config": replica_config,
                    "execution_time_predictor_config": predictor_config,
                },
            }
        ),
        encoding="utf-8",
    )


def test_runtime_artifact_validation_closes_numeric_and_config_contract(
    tmp_path: Path,
) -> None:
    contract = build_model_contract("step3-moe-noquant")
    profile_dir = tmp_path / contract.model_name
    _write_profile_fixture(profile_dir, contract.model_name)
    run_dir = tmp_path / "run"
    _write_runtime_artifacts(
        run_dir,
        contract=contract,
        profile_dir=profile_dir,
    )

    report = validate_runtime_artifacts(
        run_dir,
        contract,
        profile_dir=profile_dir,
    )

    assert report["status"] == "PASS"
    assert report["request_metrics"]["ttft_ms"] == pytest.approx(1.25)
    assert report["request_metrics"]["tpot_ms"] == pytest.approx(0.75)
    assert report["request_metrics"]["e2e_ms"] == pytest.approx(2.0)
    assert report["request_metrics"]["prefill_tokens"] == 2
    assert report["ledger"]["row_count"] == 2
    assert report["op_trace"]["event_count"] == 1
    assert report["config"]["replica"]["device"] == "h200"
    assert (
        report["config"]["replica"]["num_pipeline_stages"]
        == contract.num_pipeline_stages
    )
    assert (
        report["config"]["replica"]["attn_tensor_parallel_size"]
        == contract.attn_tensor_parallel_size
    )
    assert (
        report["config"]["replica"]["attn_data_parallel_size"]
        == contract.attn_data_parallel_size
    )
    assert (
        report["config"]["replica"]["moe_expert_parallel_size"]
        == contract.moe_expert_parallel_size
    )


def test_runtime_artifact_validation_rejects_dummy_mode(tmp_path: Path) -> None:
    contract = build_model_contract("llama3.1-8b")
    profile_dir = tmp_path / contract.model_name
    _write_profile_fixture(profile_dir, contract.model_name)
    run_dir = tmp_path / "run"
    _write_runtime_artifacts(
        run_dir,
        contract=contract,
        profile_dir=profile_dir,
        enable_dummy_mode=True,
    )

    with pytest.raises(ValueError, match="enable_dummy_mode=false"):
        validate_runtime_artifacts(
            run_dir,
            contract,
            profile_dir=profile_dir,
        )
