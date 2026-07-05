from pathlib import Path

from frontier.moe_gating_runtime import DEFAULT_MOE_GATING_RUNTIME_CONTEXT
from tests.e2e.operator_parity.profile_prerequisite_audit import (
    REQUIRED_BASE_PROFILE_FILES,
    REQUIRED_MOE_PROFILE_FILES,
    audit_requirements,
    audits_to_dict,
    build_requirements,
)


def _write_config(config_root: Path, name: str, payload: dict) -> None:
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / f"{name}.json").write_text(
        __import__("json").dumps(payload), encoding="utf-8"
    )


def _write_profile(path: Path, mean_value: str = "2.5") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "num_tensor_parallel_workers,model_architecture_profile,time_stats.mean\n"
        f"1,generic,{mean_value}\n",
        encoding="utf-8",
    )


def _write_moe_profile_with_gating_context(
    path: Path,
    gating_runtime_context: str,
    mean_value: str = "2.5",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "num_tensor_parallel_workers,gating_runtime_context,"
        "model_architecture_profile,time_stats.moe_gating_linear.median\n"
        f"1,{gating_runtime_context},generic,{mean_value}\n",
        encoding="utf-8",
    )


def _write_true_mixed_attention_profile(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "num_tensor_parallel_workers,model_architecture_profile,is_true_mixed_batch,"
        "decode_batch_size,decode_avg_kv_cache_size,num_prefill_seqs,"
        "total_prefill_tokens,total_batch_size,batch_composition_ratio,total_tokens,"
        "time_stats.attn_decode.median\n"
        "1,generic,True,1,16,1,16,2,0.5,17,3.25\n",
        encoding="utf-8",
    )


def _write_partial_true_mixed_attention_profile(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "num_tensor_parallel_workers,model_architecture_profile,is_true_mixed_batch,"
        "decode_batch_size,decode_avg_kv_cache_size,num_prefill_seqs,"
        "total_prefill_tokens,total_batch_size,batch_composition_ratio,total_tokens,"
        "time_stats.attn_decode.median\n"
        "1,generic,True,1,16,1,16,2,0.5,17,3.25\n"
        "1,generic,True,1,16,1,16,2,0.5,17,\n",
        encoding="utf-8",
    )


def _write_non_numeric_true_mixed_attention_profile(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "num_tensor_parallel_workers,model_architecture_profile,is_true_mixed_batch,"
        "decode_batch_size,decode_avg_kv_cache_size,num_prefill_seqs,"
        "total_prefill_tokens,total_batch_size,batch_composition_ratio,total_tokens,"
        "time_stats.attn_decode.median\n"
        "1,generic,True,not-a-number,16,1,16,2,0.5,17,3.25\n",
        encoding="utf-8",
    )


def test_build_requirements_uses_model_stem_and_adds_moe_files_for_moe_configs(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    _write_config(config_root, "dense_model", {"model_type": "llama"})
    _write_config(config_root, "moe_model", {"model_type": "qwen3_moe", "num_experts": 16})

    requirements = build_requirements(
        config_root=config_root,
        config_filenames=("dense_model.json", "moe_model.json"),
    )

    dense, moe = requirements
    assert dense.model_name == "dense_model"
    assert dense.required_files == REQUIRED_BASE_PROFILE_FILES
    assert moe.model_name == "moe_model"
    assert moe.required_files == REQUIRED_BASE_PROFILE_FILES + REQUIRED_MOE_PROFILE_FILES


def test_audit_requirements_reports_missing_files_and_numeric_csv_evidence(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    _write_config(config_root, "dense_model", {"model_type": "llama"})
    requirement = build_requirements(
        config_root=config_root,
        config_filenames=("dense_model.json",),
    )[0]
    profile_root = tmp_path / "profiles" / "h800"
    for filename in REQUIRED_BASE_PROFILE_FILES[:-1]:
        _write_profile(profile_root / "dense_model" / filename)

    [result] = audit_requirements(profile_root=profile_root, requirements=(requirement,))

    assert result.model_name == "dense_model"
    assert result.status == "missing"
    assert result.missing_files == ("linear_op_kernel_only.csv",)
    assert result.files["attention.csv"].exists is True
    assert result.files["attention.csv"].row_count == 1
    assert result.files["attention.csv"].time_stats_nan_count == 0
    assert result.files["linear_op_kernel_only.csv"].exists is False


def test_audit_requirements_marks_nan_time_stats_as_invalid(tmp_path: Path) -> None:
    config_root = tmp_path / "models"
    _write_config(config_root, "dense_model", {"model_type": "llama"})
    requirement = build_requirements(
        config_root=config_root,
        config_filenames=("dense_model.json",),
    )[0]
    profile_root = tmp_path / "profiles" / "h800"
    for filename in REQUIRED_BASE_PROFILE_FILES:
        _write_profile(profile_root / "dense_model" / filename)
    _write_profile(profile_root / "dense_model" / "attention.csv", mean_value="NaN")

    [result] = audit_requirements(profile_root=profile_root, requirements=(requirement,))

    assert result.status == "invalid"
    assert result.invalid_files == ("attention.csv",)
    assert result.files["attention.csv"].time_stats_nan_count == 1


def test_audit_requirements_accepts_sparse_wide_time_stats_when_each_row_and_column_has_data(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    _write_config(config_root, "dense_model", {"model_type": "llama"})
    requirement = build_requirements(
        config_root=config_root,
        config_filenames=("dense_model.json",),
    )[0]
    profile_root = tmp_path / "profiles" / "h800"
    for filename in REQUIRED_BASE_PROFILE_FILES:
        _write_profile(profile_root / "dense_model" / filename)

    sparse_profile = profile_root / "dense_model" / "linear_op.csv"
    sparse_profile.write_text(
        "num_tensor_parallel_workers,model_architecture_profile,"
        "time_stats.emb.mean,time_stats.attn_pre_proj.mean\n"
        "1,generic,1.0,\n"
        "2,generic,,2.0\n",
        encoding="utf-8",
    )

    [result] = audit_requirements(profile_root=profile_root, requirements=(requirement,))

    assert result.status == "present"
    assert result.invalid_files == ()
    assert result.files["linear_op.csv"].row_count == 2
    assert result.files["linear_op.csv"].time_stats_nan_count == 2
    assert result.files["linear_op.csv"].time_stats_empty_row_count == 0
    assert result.files["linear_op.csv"].time_stats_empty_column_count == 0


def test_audit_requirements_rejects_time_stats_columns_with_no_measurements(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    _write_config(config_root, "dense_model", {"model_type": "llama"})
    requirement = build_requirements(
        config_root=config_root,
        config_filenames=("dense_model.json",),
    )[0]
    profile_root = tmp_path / "profiles" / "h800"
    for filename in REQUIRED_BASE_PROFILE_FILES:
        _write_profile(profile_root / "dense_model" / filename)

    invalid_profile = profile_root / "dense_model" / "linear_op.csv"
    invalid_profile.write_text(
        "num_tensor_parallel_workers,time_stats.emb.mean,time_stats.attn_pre_proj.mean\n"
        "1,1.0,\n"
        "2,2.0,\n",
        encoding="utf-8",
    )

    [result] = audit_requirements(profile_root=profile_root, requirements=(requirement,))

    assert result.status == "invalid"
    assert result.invalid_files == ("linear_op.csv",)
    assert result.files["linear_op.csv"].time_stats_empty_column_count == 1


def test_audit_requirements_rejects_profiles_missing_architecture_profile_metadata(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    _write_config(config_root, "dense_model", {"model_type": "llama"})
    requirement = build_requirements(
        config_root=config_root,
        config_filenames=("dense_model.json",),
    )[0]
    profile_root = tmp_path / "profiles" / "h800"
    for filename in REQUIRED_BASE_PROFILE_FILES:
        _write_profile(profile_root / "dense_model" / filename)
    legacy_profile = profile_root / "dense_model" / "attention.csv"
    legacy_profile.write_text(
        "num_tensor_parallel_workers,time_stats.mean\n"
        "1,2.5\n",
        encoding="utf-8",
    )

    [result] = audit_requirements(profile_root=profile_root, requirements=(requirement,))

    assert result.status == "invalid"
    assert result.invalid_files == ("attention.csv",)
    assert (
        "model_architecture_profile column is missing"
        in result.files["attention.csv"].semantic_coverage_errors
    )


def test_audit_requirements_can_require_true_mixed_attention_rows(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    _write_config(config_root, "dense_model", {"model_type": "llama"})
    requirement = build_requirements(
        config_root=config_root,
        config_filenames=("dense_model.json",),
    )[0]
    profile_root = tmp_path / "profiles" / "h800"
    for filename in REQUIRED_BASE_PROFILE_FILES:
        _write_profile(profile_root / "dense_model" / filename)

    [result] = audit_requirements(
        profile_root=profile_root,
        requirements=(requirement,),
        require_true_mixed_attention=True,
    )

    assert result.status == "invalid"
    assert result.invalid_files == ("attention.csv", "attention_kernel_only.csv")
    for filename in ("attention.csv", "attention_kernel_only.csv"):
        assert result.files[filename].true_mixed_row_count == 0
        assert (
            "missing true-mixed attention columns: "
            "batch_composition_ratio, decode_avg_kv_cache_size, decode_batch_size, "
            "is_true_mixed_batch, num_prefill_seqs, time_stats.attn_decode.median, "
            "total_batch_size, total_prefill_tokens, total_tokens"
            in result.files[filename].semantic_coverage_errors
        )


def test_audit_requirements_accepts_valid_true_mixed_attention_rows(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    _write_config(config_root, "dense_model", {"model_type": "llama"})
    requirement = build_requirements(
        config_root=config_root,
        config_filenames=("dense_model.json",),
    )[0]
    profile_root = tmp_path / "profiles" / "h800"
    for filename in REQUIRED_BASE_PROFILE_FILES:
        _write_profile(profile_root / "dense_model" / filename)
    _write_true_mixed_attention_profile(profile_root / "dense_model" / "attention.csv")
    _write_true_mixed_attention_profile(
        profile_root / "dense_model" / "attention_kernel_only.csv"
    )

    [result] = audit_requirements(
        profile_root=profile_root,
        requirements=(requirement,),
        require_true_mixed_attention=True,
    )

    assert result.status == "present"
    assert result.invalid_files == ()
    for filename in ("attention.csv", "attention_kernel_only.csv"):
        assert result.files[filename].true_mixed_row_count == 1
        assert result.files[filename].true_mixed_attn_decode_valid_count == 1


def test_audit_requirements_rejects_partial_true_mixed_attention_decode_rows(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    _write_config(config_root, "dense_model", {"model_type": "llama"})
    requirement = build_requirements(
        config_root=config_root,
        config_filenames=("dense_model.json",),
    )[0]
    profile_root = tmp_path / "profiles" / "h800"
    for filename in REQUIRED_BASE_PROFILE_FILES:
        _write_profile(profile_root / "dense_model" / filename)
    _write_partial_true_mixed_attention_profile(
        profile_root / "dense_model" / "attention.csv"
    )
    _write_true_mixed_attention_profile(
        profile_root / "dense_model" / "attention_kernel_only.csv"
    )

    [result] = audit_requirements(
        profile_root=profile_root,
        requirements=(requirement,),
        require_true_mixed_attention=True,
    )

    assert result.status == "invalid"
    assert result.invalid_files == ("attention.csv",)
    assert result.files["attention.csv"].true_mixed_row_count == 2
    assert result.files["attention.csv"].true_mixed_attn_decode_valid_count == 1
    assert (
        "true-mixed attention rows with invalid time_stats.attn_decode.median: 1/2"
        in result.files["attention.csv"].semantic_coverage_errors
    )


def test_audit_requirements_rejects_non_numeric_true_mixed_attention_features(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    _write_config(config_root, "dense_model", {"model_type": "llama"})
    requirement = build_requirements(
        config_root=config_root,
        config_filenames=("dense_model.json",),
    )[0]
    profile_root = tmp_path / "profiles" / "h800"
    for filename in REQUIRED_BASE_PROFILE_FILES:
        _write_profile(profile_root / "dense_model" / filename)
    _write_non_numeric_true_mixed_attention_profile(
        profile_root / "dense_model" / "attention.csv"
    )
    _write_true_mixed_attention_profile(
        profile_root / "dense_model" / "attention_kernel_only.csv"
    )

    [result] = audit_requirements(
        profile_root=profile_root,
        requirements=(requirement,),
        require_true_mixed_attention=True,
    )

    assert result.status == "invalid"
    assert result.invalid_files == ("attention.csv",)
    assert (
        "true-mixed attention rows have invalid numeric columns: decode_batch_size=1"
        in result.files["attention.csv"].semantic_coverage_errors
    )


def test_audit_requirements_rejects_architecture_profile_metadata_mismatch(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    _write_config(
        config_root,
        "step3_model",
        {
            "model_type": "step3_text",
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "hidden_size": 128,
            "head_dim": 16,
            "num_experts": 16,
            "share_expert_dim": 64,
            "use_mfa": True,
            "share_q_dim": 64,
        },
    )
    requirement = build_requirements(
        config_root=config_root,
        config_filenames=("step3_model.json",),
    )[0]
    profile_root = tmp_path / "profiles" / "h800"
    for filename in REQUIRED_BASE_PROFILE_FILES + REQUIRED_MOE_PROFILE_FILES:
        _write_profile(profile_root / "step3_model" / filename)
    mismatched_profile = profile_root / "step3_model" / "linear_op.csv"
    mismatched_profile.write_text(
        "num_tensor_parallel_workers,model_architecture_profile,time_stats.mean\n"
        "1,generic,2.5\n",
        encoding="utf-8",
    )

    [result] = audit_requirements(profile_root=profile_root, requirements=(requirement,))

    assert result.status == "invalid"
    assert "linear_op.csv" in result.invalid_files
    assert (
        "model_architecture_profile mismatch: expected step3_text, observed generic"
        in result.files["linear_op.csv"].semantic_coverage_errors
    )


def test_build_requirements_rejects_structurally_invalid_step3_config(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    _write_config(
        config_root,
        "step3_model",
        {
            "model_type": "step3_text",
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "hidden_size": 128,
            "head_dim": 16,
            "num_experts": 16,
            "share_expert_dim": 64,
            "use_mfa": False,
            "share_q_dim": 64,
        },
    )

    try:
        build_requirements(
            config_root=config_root,
            config_filenames=("step3_model.json",),
        )
    except ValueError as exc:
        assert "Step3Text MFA" in str(exc)
        assert "use_mfa=True" in str(exc)
    else:
        raise AssertionError("structurally invalid Step3 config must fail fast")


def test_audit_requirements_rejects_moe_profiles_without_standalone_legacy_gating_rows(
    tmp_path: Path,
) -> None:
    config_root = tmp_path / "models"
    _write_config(config_root, "moe_model", {"model_type": "qwen3_moe", "num_experts": 16})
    requirement = build_requirements(
        config_root=config_root,
        config_filenames=("moe_model.json",),
    )[0]
    profile_root = tmp_path / "profiles" / "h800"
    for filename in REQUIRED_BASE_PROFILE_FILES:
        _write_profile(profile_root / "moe_model" / filename)
    for filename in REQUIRED_MOE_PROFILE_FILES:
        _write_moe_profile_with_gating_context(
            profile_root / "moe_model" / filename,
            "prefill_hot",
        )

    [result] = audit_requirements(profile_root=profile_root, requirements=(requirement,))
    report = audits_to_dict((result,))

    assert result.status == "invalid"
    assert result.invalid_files == REQUIRED_MOE_PROFILE_FILES
    for filename in REQUIRED_MOE_PROFILE_FILES:
        assert (
            f"gating_runtime_context={DEFAULT_MOE_GATING_RUNTIME_CONTEXT}"
            in result.files[filename].semantic_coverage_errors
        )
    assert "gating_runtime_context=standalone_legacy" in __import__("json").dumps(report)


def test_b0_h800_profiling_script_runs_record_function_one_model_per_process() -> None:
    script = Path("tests/e2e/operator_parity/run_b0_h800_profiling.sh").read_text(
        encoding="utf-8"
    )

    assert 'if [ "$method" = "record_function" ]; then' in script
    assert 'for model in "${ALL_MODELS[@]}"; do' in script
    assert 'attention_${method_slug}_${model_slug}' in script
    assert '--models "$model"' in script


def test_b0_h800_profiling_script_uses_vllm_moe_path_for_uniform_topk() -> None:
    script = Path("tests/e2e/operator_parity/run_b0_h800_profiling.sh").read_text(
        encoding="utf-8"
    )

    assert "--routing_runtime_path uniform_topk" in script
    assert "--disable_load_imbalance" not in script
    assert "--enable_load_imbalance" in script
    assert "--load_distributions uniform" in script
    assert "--num_samples_per_distribution 1" in script


def test_b0_h800_moe_gating_backfill_script_profiles_standalone_legacy_to_stage() -> None:
    script = Path(
        "tests/e2e/operator_parity/run_b0_h800_moe_gating_backfill.sh"
    ).read_text(encoding="utf-8")

    assert 'MOE_GATING_CONTEXT="${MOE_GATING_CONTEXT:-standalone_legacy}"' in script
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-$STAGE_ROOT/profiling}"' in script
    assert "export OUTPUT_ROOT" in script
    assert "--gating_runtime_context" in script
    assert '"$MOE_GATING_CONTEXT"' in script
    assert "--routing_runtime_path uniform_topk" in script
    assert "frontier.profiling.moe.main" in script
    assert "frontier.profiling.attention.main" not in script
    assert "frontier.profiling.linear_op.main" not in script


def test_b0_h800_true_mixed_attention_backfill_script_profiles_both_methods_to_stage() -> None:
    script = Path(
        "tests/e2e/operator_parity/run_b0_h800_true_mixed_attention_backfill.sh"
    ).read_text(encoding="utf-8")

    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-$STAGE_ROOT/profiling}"' in script
    assert 'PROFILE_METHODS="${PROFILE_METHODS:-cuda_event record_function}"' in script
    assert 'FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$STAGE_ROOT/flashinfer_workspace}"' in script
    assert "page_jit_prewarm_ok" in script
    assert "02_flashinfer_attention_jit_prewarm" in script
    assert "attention_jit_prewarm_${method_slug}_${model_slug}" in script
    assert "--profile_only_prefill" in script
    assert 'PREWARM_OUTPUT_ROOT="${PREWARM_OUTPUT_ROOT:-$STAGE_ROOT/prewarm/profiling}"' in script
    assert "attention_combined.csv" in script
    assert "attention_combined_kernel_only.csv" in script
    assert (
        'stat["true_mixed_attn_decode_valid_count"] != stat["true_mixed_row_count"]'
        in script
    )
    assert "true_mixed_required_numeric_valid_row_count" in script
    assert "--enable_true_mixed" in script
    assert "--true_mixed_prefill_batch_sizes" in script
    assert "--true_mixed_prefill_chunk_sizes" in script
    assert "--true_mixed_decode_batch_sizes" in script
    assert "--true_mixed_decode_kv_cache_sizes" in script
    assert "frontier.profiling.attention.main" in script
    assert "frontier.profiling.linear_op.main" not in script
    assert "frontier.profiling.moe.main" not in script
