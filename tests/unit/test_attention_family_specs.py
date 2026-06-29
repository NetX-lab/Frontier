from __future__ import annotations

import pandas as pd
import pytest

from frontier.attention import profiling_mapping
from frontier.attention.families import (
    DENSE_ATTENTION_FAMILY,
    DSA_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
    get_attention_family,
)
from frontier.attention.ops import (
    AttentionFamilySpec,
    AttentionMemoryLayout,
    AttentionOperatorRole,
    AttentionOperatorSpec,
    AttentionPhase,
    ProjectionOwnership,
)
from frontier.attention.profiling_mapping import (
    get_profiling_time_stat_columns,
    get_required_profiling_columns,
    get_required_profiling_feature_columns,
    get_enabled_predictor_feature_columns,
    get_enabled_predictor_median_columns,
    get_enabled_predictor_metric_names,
    get_enabled_predictor_required_feature_columns,
    get_enabled_shared_predictor_feature_columns,
    get_enabled_shared_predictor_required_feature_columns,
    get_e2e_metric_names,
    get_profiling_metric_names,
    validate_attention_catalog_alignment,
)
from frontier.training.attention_trainer import AttentionTrainer
from frontier.metrics.constants import OperationMetrics as E2EOperationMetrics
from frontier.profiling.common.constants import (
    OperationMetrics as ProfilingOperationMetrics,
)
from frontier.types import MeasurementType


DENSE_CORE_OPS = (
    "attn_kv_cache_save",
    "attn_prefill",
    "attn_decode",
)

MLA_PHYSICAL_OPS = (
    "attn_mla_kv_cache_save",
    "attn_mla_prefill_kv_up_proj",
    "attn_mla_prefill",
    "attn_mla_decode_q_latent_proj",
    "attn_mla_decode",
    "attn_mla_v_up_proj",
)


def test_dense_family_lists_exact_gqa_core_ops_in_vllm_order() -> None:
    family = DENSE_ATTENTION_FAMILY

    assert family.family_id == "dense_attention"
    assert family.memory_layout == AttentionMemoryLayout.DENSE_KV
    assert family.dense_compatible is True
    assert family.supported_variants == ("gqa", "mha", "mqa")
    assert tuple(op.name for op in family.operators) == DENSE_CORE_OPS
    assert tuple(op.role for op in family.operators) == (
        AttentionOperatorRole.CACHE_WRITE,
        AttentionOperatorRole.PREFILL_KERNEL,
        AttentionOperatorRole.DECODE_KERNEL,
    )
    assert tuple(op.execution_time_attr for op in family.operators) == (
        "attention_kv_cache_save_execution_time",
        "attention_prefill_execution_time",
        "attention_decode_execution_time",
    )
    assert all(op.profiling_target for op in family.operators)
    assert all(op.predictor_target for op in family.operators)
    assert all(op.e2e_trace_target for op in family.operators)
    assert all(
        op.projection_ownership is ProjectionOwnership.NOT_PROJECTION
        for op in family.operators
    )


def test_mla_family_lists_exact_physical_ops_and_projection_ownership() -> None:
    family = LATENT_MLA_ATTENTION_FAMILY

    assert family.family_id == "latent_mla_attention"
    assert family.memory_layout == AttentionMemoryLayout.LATENT_MLA
    assert family.dense_compatible is False
    assert family.supported_variants == ("mla",)
    assert tuple(op.name for op in family.operators) == MLA_PHYSICAL_OPS
    assert [op.name for op in family.projection_ops()] == [
        "attn_mla_prefill_kv_up_proj",
        "attn_mla_decode_q_latent_proj",
        "attn_mla_v_up_proj",
    ]
    assert tuple(op.execution_time_attr for op in family.operators) == (
        "attn_mla_kv_cache_save_time",
        "attn_mla_prefill_kv_up_proj_time",
        "attn_mla_prefill_time",
        "attn_mla_decode_q_latent_proj_time",
        "attn_mla_decode_time",
        "attn_mla_v_up_proj_time",
    )
    assert family.disjoint_model_projection_attrs == (
        "attention_layer_pre_proj_execution_time",
        "attention_layer_post_proj_execution_time",
    )
    assert all(
        op.projection_ownership
        is ProjectionOwnership.INSIDE_ATTENTION_PHYSICAL_SCOPE
        for op in family.projection_ops()
    )


def test_dsa_family_is_frozen_and_has_no_enabled_operator_targets() -> None:
    family = DSA_ATTENTION_FAMILY

    assert family.family_id == "dsa_attention"
    assert family.dsa_frozen is True
    assert family.operators == ()
    with pytest.raises(NotImplementedError, match="DSA attention is frozen"):
        family.require_enabled_for_execution()


def test_family_lookup_rejects_unknown_family_without_fallback() -> None:
    with pytest.raises(ValueError, match="Unknown attention family"):
        get_attention_family("unknown_attention")


def test_family_specs_have_unique_operator_names() -> None:
    for family_id in (
        "dense_attention",
        "latent_mla_attention",
        "dsa_attention",
    ):
        family = get_attention_family(family_id)
        names = [op.name for op in family.operators]
        assert names == list(dict.fromkeys(names))


def test_family_specs_reject_duplicate_disjoint_projection_attrs() -> None:
    with pytest.raises(ValueError, match="duplicate disjoint model projection attrs"):
        AttentionFamilySpec(
            family_id="duplicate_projection_attrs",
            display_name="Duplicate Projection Attrs",
            supported_variants=("mla",),
            operators=(),
            memory_layout=AttentionMemoryLayout.LATENT_MLA,
            dense_compatible=False,
            requires_runtime_kv_helpers=True,
            disjoint_model_projection_attrs=(
                "attention_layer_pre_proj_execution_time",
                "attention_layer_pre_proj_execution_time",
            ),
        )


def test_e2e_trace_operator_specs_must_declare_execution_time_attr() -> None:
    with pytest.raises(ValueError, match="must declare execution_time_attr"):
        AttentionOperatorSpec(
            name="missing_attr",
            role=AttentionOperatorRole.PREFILL_KERNEL,
            phases=(AttentionPhase.PREFILL,),
        )

    predictor_only = AttentionOperatorSpec(
        name="predictor_only",
        role=AttentionOperatorRole.PREFILL_KERNEL,
        phases=(AttentionPhase.PREFILL,),
        e2e_trace_target=False,
    )

    assert predictor_only.execution_time_attr is None


def test_family_specs_align_with_existing_profiling_and_e2e_enums() -> None:
    validate_attention_catalog_alignment(
        profiling_metric_values={metric.value for metric in ProfilingOperationMetrics},
        e2e_metric_values={metric.value for metric in E2EOperationMetrics},
    )

    assert get_profiling_metric_names(DENSE_ATTENTION_FAMILY) == DENSE_CORE_OPS
    assert get_e2e_metric_names(DENSE_ATTENTION_FAMILY) == DENSE_CORE_OPS
    assert get_profiling_metric_names(LATENT_MLA_ATTENTION_FAMILY) == MLA_PHYSICAL_OPS
    assert get_e2e_metric_names(LATENT_MLA_ATTENTION_FAMILY) == MLA_PHYSICAL_OPS


def test_dense_enabled_predictor_targets_and_columns_come_from_family_spec() -> None:
    assert get_enabled_predictor_metric_names(DENSE_ATTENTION_FAMILY) == DENSE_CORE_OPS
    assert get_enabled_predictor_median_columns(DENSE_ATTENTION_FAMILY) == (
        "time_stats.attn_kv_cache_save.median",
        "time_stats.attn_prefill.median",
        "time_stats.attn_decode.median",
    )


def test_dense_predictor_median_column_lookup_uses_roles_not_operator_names() -> None:
    family = AttentionFamilySpec(
        family_id="renamed_dense_predictor_median_columns",
        display_name="Renamed Dense Predictor Median Columns",
        supported_variants=("gqa",),
        operators=(
            AttentionOperatorSpec(
                name="role_prefill",
                role=AttentionOperatorRole.PREFILL_KERNEL,
                phases=(AttentionPhase.PREFILL,),
                e2e_trace_target=False,
            ),
            AttentionOperatorSpec(
                name="role_decode",
                role=AttentionOperatorRole.DECODE_KERNEL,
                phases=(AttentionPhase.DECODE,),
                e2e_trace_target=False,
            ),
            AttentionOperatorSpec(
                name="role_cache",
                role=AttentionOperatorRole.CACHE_WRITE,
                phases=(AttentionPhase.PREFILL, AttentionPhase.DECODE),
                e2e_trace_target=False,
            ),
        ),
        memory_layout=AttentionMemoryLayout.DENSE_KV,
        dense_compatible=True,
        requires_runtime_kv_helpers=False,
    )

    assert profiling_mapping.get_enabled_predictor_median_column_by_role(
        family,
        AttentionOperatorRole.CACHE_WRITE,
    ) == "time_stats.role_cache.median"
    assert profiling_mapping.get_enabled_predictor_median_column_by_role(
        family,
        AttentionOperatorRole.PREFILL_KERNEL,
    ) == "time_stats.role_prefill.median"
    assert profiling_mapping.get_enabled_predictor_median_column_by_role(
        family,
        AttentionOperatorRole.DECODE_KERNEL,
    ) == "time_stats.role_decode.median"


def test_dense_enabled_predictor_feature_columns_come_from_family_spec() -> None:
    assert get_enabled_predictor_feature_columns(DENSE_ATTENTION_FAMILY) == {
        "attn_kv_cache_save": ("num_tokens",),
        "attn_prefill": ("kv_cache_size", "prefill_chunk_size_squared"),
        "attn_decode": ("batch_size", "kv_cache_size"),
    }


def test_dense_predictor_feature_columns_use_roles_not_operator_names() -> None:
    family = AttentionFamilySpec(
        family_id="renamed_dense_predictor_features",
        display_name="Renamed Dense Predictor Features",
        supported_variants=("gqa",),
        operators=(
            AttentionOperatorSpec(
                name="role_prefill",
                role=AttentionOperatorRole.PREFILL_KERNEL,
                phases=(AttentionPhase.PREFILL,),
                e2e_trace_target=False,
            ),
            AttentionOperatorSpec(
                name="role_decode",
                role=AttentionOperatorRole.DECODE_KERNEL,
                phases=(AttentionPhase.DECODE,),
                e2e_trace_target=False,
            ),
            AttentionOperatorSpec(
                name="role_cache",
                role=AttentionOperatorRole.CACHE_WRITE,
                phases=(AttentionPhase.PREFILL, AttentionPhase.DECODE),
                e2e_trace_target=False,
            ),
        ),
        memory_layout=AttentionMemoryLayout.DENSE_KV,
        dense_compatible=True,
        requires_runtime_kv_helpers=False,
    )

    assert get_enabled_predictor_feature_columns(family) == {
        "role_prefill": ("kv_cache_size", "prefill_chunk_size_squared"),
        "role_decode": ("batch_size", "kv_cache_size"),
        "role_cache": ("num_tokens",),
    }
    assert get_enabled_shared_predictor_feature_columns(family) == {
        "role_prefill": ("kv_cache_size", "prefill_chunk_size_squared"),
        "role_decode": ("batch_size", "kv_cache_size"),
        "role_cache": ("total_tokens", "kv_cache_size", "batch_size"),
    }


def test_dense_enabled_predictor_required_features_are_flattened_from_family_spec() -> None:
    assert get_enabled_predictor_required_feature_columns(DENSE_ATTENTION_FAMILY) == (
        "num_tokens",
        "kv_cache_size",
        "prefill_chunk_size_squared",
        "batch_size",
    )


def test_dense_shared_predictor_feature_columns_keep_runtime_cache_schema() -> None:
    assert get_enabled_shared_predictor_feature_columns(DENSE_ATTENTION_FAMILY) == {
        "attn_kv_cache_save": ("total_tokens", "kv_cache_size", "batch_size"),
        "attn_prefill": ("kv_cache_size", "prefill_chunk_size_squared"),
        "attn_decode": ("batch_size", "kv_cache_size"),
    }


def test_dense_shared_predictor_required_features_are_flattened_from_family_spec() -> None:
    assert get_enabled_shared_predictor_required_feature_columns(
        DENSE_ATTENTION_FAMILY
    ) == (
        "total_tokens",
        "kv_cache_size",
        "batch_size",
        "prefill_chunk_size_squared",
    )


def test_predictor_metric_name_lookup_uses_role_not_family_order() -> None:
    family = AttentionFamilySpec(
        family_id="dense_attention_test",
        display_name="Dense-KV Attention Test",
        supported_variants=("gqa",),
        operators=(
            AttentionOperatorSpec(
                name="role_prefill",
                role=AttentionOperatorRole.PREFILL_KERNEL,
                phases=(AttentionPhase.PREFILL,),
                e2e_trace_target=False,
            ),
            AttentionOperatorSpec(
                name="role_decode",
                role=AttentionOperatorRole.DECODE_KERNEL,
                phases=(AttentionPhase.DECODE,),
                e2e_trace_target=False,
            ),
            AttentionOperatorSpec(
                name="role_cache",
                role=AttentionOperatorRole.CACHE_WRITE,
                phases=(AttentionPhase.PREFILL, AttentionPhase.DECODE),
                e2e_trace_target=False,
            ),
        ),
        memory_layout=AttentionMemoryLayout.DENSE_KV,
        dense_compatible=True,
        requires_runtime_kv_helpers=False,
    )

    assert (
        profiling_mapping.get_enabled_predictor_metric_name_by_role(
            family, AttentionOperatorRole.CACHE_WRITE
        )
        == "role_cache"
    )
    assert (
        profiling_mapping.get_enabled_predictor_metric_name_by_role(
            family, AttentionOperatorRole.PREFILL_KERNEL
        )
        == "role_prefill"
    )
    assert (
        profiling_mapping.get_enabled_predictor_metric_name_by_role(
            family, AttentionOperatorRole.DECODE_KERNEL
        )
        == "role_decode"
    )


def test_profiling_metric_name_lookup_uses_role_not_family_order() -> None:
    family = AttentionFamilySpec(
        family_id="dense_attention_profiling_test",
        display_name="Dense-KV Attention Profiling Test",
        supported_variants=("gqa",),
        operators=(
            AttentionOperatorSpec(
                name="role_prefill_profile",
                role=AttentionOperatorRole.PREFILL_KERNEL,
                phases=(AttentionPhase.PREFILL,),
                e2e_trace_target=False,
            ),
            AttentionOperatorSpec(
                name="role_decode_profile",
                role=AttentionOperatorRole.DECODE_KERNEL,
                phases=(AttentionPhase.DECODE,),
                e2e_trace_target=False,
            ),
            AttentionOperatorSpec(
                name="role_cache_profile",
                role=AttentionOperatorRole.CACHE_WRITE,
                phases=(AttentionPhase.PREFILL, AttentionPhase.DECODE),
                e2e_trace_target=False,
            ),
        ),
        memory_layout=AttentionMemoryLayout.DENSE_KV,
        dense_compatible=True,
        requires_runtime_kv_helpers=False,
    )

    assert (
        profiling_mapping.get_profiling_metric_name_by_role(
            family, AttentionOperatorRole.CACHE_WRITE
        )
        == "role_cache_profile"
    )
    assert (
        profiling_mapping.get_profiling_metric_name_by_role(
            family, AttentionOperatorRole.PREFILL_KERNEL
        )
        == "role_prefill_profile"
    )
    assert (
        profiling_mapping.get_profiling_metric_name_by_role(
            family, AttentionOperatorRole.DECODE_KERNEL
        )
        == "role_decode_profile"
    )


def test_predictor_metric_name_lookup_rejects_missing_or_duplicate_roles() -> None:
    missing_decode_family = AttentionFamilySpec(
        family_id="missing_decode_test",
        display_name="Missing Decode Test",
        supported_variants=("gqa",),
        operators=(
            AttentionOperatorSpec(
                name="role_cache",
                role=AttentionOperatorRole.CACHE_WRITE,
                phases=(AttentionPhase.PREFILL,),
                e2e_trace_target=False,
            ),
        ),
        memory_layout=AttentionMemoryLayout.DENSE_KV,
        dense_compatible=True,
        requires_runtime_kv_helpers=False,
    )
    duplicate_cache_family = AttentionFamilySpec(
        family_id="duplicate_cache_test",
        display_name="Duplicate Cache Test",
        supported_variants=("gqa",),
        operators=(
            AttentionOperatorSpec(
                name="role_cache_a",
                role=AttentionOperatorRole.CACHE_WRITE,
                phases=(AttentionPhase.PREFILL,),
                e2e_trace_target=False,
            ),
            AttentionOperatorSpec(
                name="role_cache_b",
                role=AttentionOperatorRole.CACHE_WRITE,
                phases=(AttentionPhase.DECODE,),
                e2e_trace_target=False,
            ),
        ),
        memory_layout=AttentionMemoryLayout.DENSE_KV,
        dense_compatible=True,
        requires_runtime_kv_helpers=False,
    )

    with pytest.raises(ValueError, match="Expected exactly one predictor operator"):
        profiling_mapping.get_enabled_predictor_metric_name_by_role(
            missing_decode_family, AttentionOperatorRole.DECODE_KERNEL
        )
    with pytest.raises(ValueError, match="Expected exactly one predictor operator"):
        profiling_mapping.get_enabled_predictor_metric_name_by_role(
            duplicate_cache_family, AttentionOperatorRole.CACHE_WRITE
        )


def test_predictor_metric_name_lookup_supports_mla_kernel_roles_and_keeps_dsa_gate() -> None:
    assert (
        profiling_mapping.get_enabled_predictor_metric_name_by_role(
            LATENT_MLA_ATTENTION_FAMILY,
            AttentionOperatorRole.CACHE_WRITE,
        )
        == "attn_mla_kv_cache_save"
    )
    assert (
        profiling_mapping.get_enabled_predictor_metric_name_by_role(
            LATENT_MLA_ATTENTION_FAMILY,
            AttentionOperatorRole.PREFILL_KERNEL,
        )
        == "attn_mla_prefill"
    )
    assert (
        profiling_mapping.get_enabled_predictor_metric_name_by_role(
            LATENT_MLA_ATTENTION_FAMILY,
            AttentionOperatorRole.DECODE_KERNEL,
        )
        == "attn_mla_decode"
    )
    with pytest.raises(ValueError, match="Expected exactly one predictor operator"):
        profiling_mapping.get_enabled_predictor_metric_name_by_role(
            LATENT_MLA_ATTENTION_FAMILY,
            AttentionOperatorRole.PROJECTION,
        )

    with pytest.raises(NotImplementedError, match="DSA attention is frozen"):
        profiling_mapping.get_enabled_predictor_metric_name_by_role(
            DSA_ATTENTION_FAMILY,
            AttentionOperatorRole.DECODE_KERNEL,
        )


def test_mla_enabled_predictor_targets_follow_vllm_imported_scope_order() -> None:
    assert (
        get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY)
        == MLA_PHYSICAL_OPS
    )
    assert get_enabled_predictor_median_columns(LATENT_MLA_ATTENTION_FAMILY) == (
        "time_stats.attn_mla_kv_cache_save.median",
        "time_stats.attn_mla_prefill_kv_up_proj.median",
        "time_stats.attn_mla_prefill.median",
        "time_stats.attn_mla_decode_q_latent_proj.median",
        "time_stats.attn_mla_decode.median",
        "time_stats.attn_mla_v_up_proj.median",
    )


MLA_IMPORTED_PREDICTOR_FEATURE_COLUMNS = (
    "n_q_head",
    "n_kv_head",
    "head_size",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "qk_head_dim",
    "kv_lora_rank",
    "v_head_dim",
    "block_size",
    "num_tensor_parallel_workers",
    "batch_size",
    "batch_num_tokens",
    "batch_num_prefill_tokens",
    "batch_num_decode_tokens",
    "max_seqlen_q",
    "max_seqlen_k",
    "num_actual_tokens",
    "is_prefill",
    "max_seq_len",
)


def test_attention_trainer_dense_layer_targets_follow_family_catalog() -> None:
    trainer = AttentionTrainer.__new__(AttentionTrainer)
    trainer.train_compute_models = False

    model_names = trainer._get_model_names()

    assert tuple(model_names[:3]) == get_enabled_predictor_metric_names(
        DENSE_ATTENTION_FAMILY
    )
    assert tuple(trainer._get_target_col(name) for name in model_names[:3]) == (
        get_enabled_predictor_median_columns(DENSE_ATTENTION_FAMILY)
    )
    assert model_names[3:] == [
        "attn_prefill_mixed",
        "attn_decode_in_mixed",
    ]


def test_attention_trainer_dense_target_columns_use_catalog_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = AttentionTrainer.__new__(AttentionTrainer)

    monkeypatch.setattr(
        AttentionTrainer,
        "DENSE_LAYER_TARGET_COLUMN_BY_MODEL",
        {
            "attn_decode": "time_stats.catalog_attn_decode.median",
        },
        raising=False,
    )

    assert (
        trainer._get_target_col("attn_decode")
        == "time_stats.catalog_attn_decode.median"
    )
    assert (
        trainer._get_target_col("attn_prefill_mixed")
        == "time_stats.attn_prefill.median"
    )
    assert (
        trainer._get_target_col("attn_decode_in_mixed")
        == "time_stats.attn_decode.median"
    )


def test_attention_trainer_loader_fills_role_derived_cache_column(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = AttentionTrainer.__new__(AttentionTrainer)
    trainer.layer_dataset_path = str(tmp_path / "attention.csv")
    trainer.model_config = type(
        "ModelConfig",
        (),
        {
            "embedding_dim": 7168,
            "num_q_heads": 64,
            "num_kv_heads": 8,
        },
    )()
    trainer.block_size = 16
    trainer.tensor_parallel_size = 1
    trainer._set_dataset_metadata = lambda *_args, **_kwargs: None
    trainer._add_derived_features = lambda df: df
    trainer._verify_layer_dataset_columns = lambda df: None

    monkeypatch.setattr(
        AttentionTrainer,
        "DENSE_LAYER_CACHE_WRITE_TARGET_COLUMN",
        "time_stats.role_cache.median",
        raising=False,
    )

    pd.DataFrame(
        {
            "n_embd": [7168],
            "n_q_head": [64],
            "n_kv_head": [8],
            "block_size": [16],
            "num_tensor_parallel_workers": [1],
        }
    ).to_csv(trainer.layer_dataset_path, index=False)

    filtered = trainer._load_layer_dataset()

    assert "time_stats.role_cache.median" in filtered.columns
    assert filtered.iloc[0]["time_stats.role_cache.median"] == 0
    assert "time_stats.attn_kv_cache_save.median" not in filtered.columns


def test_attention_trainer_dense_layer_features_follow_family_catalog() -> None:
    trainer = AttentionTrainer.__new__(AttentionTrainer)

    feature_columns = get_enabled_predictor_feature_columns(DENSE_ATTENTION_FAMILY)

    for model_name, expected_columns in feature_columns.items():
        assert tuple(trainer._get_feature_cols(model_name)) == expected_columns


def test_attention_trainer_dense_layer_features_return_fresh_lists() -> None:
    trainer = AttentionTrainer.__new__(AttentionTrainer)

    feature_cols = trainer._get_feature_cols("attn_decode")
    feature_cols.append("mutated")

    assert trainer._get_feature_cols("attn_decode") == [
        "batch_size",
        "kv_cache_size",
    ]


def test_attention_trainer_layer_column_verification_uses_catalog_required_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = AttentionTrainer.__new__(AttentionTrainer)
    old_required_feature_columns = [
        "num_tokens",
        "prefill_chunk_size_squared",
        "kv_cache_size",
        "batch_size",
    ]
    data_columns = [
        *old_required_feature_columns,
        "is_decode",
        *AttentionTrainer.DENSE_LAYER_TARGET_COLUMNS,
    ]
    df = pd.DataFrame({column: [1] for column in data_columns})

    monkeypatch.setattr(
        AttentionTrainer,
        "DENSE_LAYER_REQUIRED_FEATURE_COLUMNS",
        (*old_required_feature_columns, "catalog_required_feature"),
        raising=False,
    )

    with pytest.raises(ValueError, match="catalog_required_feature"):
        trainer._verify_layer_dataset_columns(df)


def test_mla_enabled_predictor_feature_columns_use_imported_profile_schema() -> None:
    expected_features = MLA_IMPORTED_PREDICTOR_FEATURE_COLUMNS

    feature_columns = get_enabled_predictor_feature_columns(
        LATENT_MLA_ATTENTION_FAMILY
    )

    assert feature_columns == {
        op_name: expected_features for op_name in MLA_PHYSICAL_OPS
    }
    profiling_schema = set(
        get_required_profiling_feature_columns(LATENT_MLA_ATTENTION_FAMILY)
    )
    for columns in feature_columns.values():
        assert set(columns).issubset(profiling_schema)


def test_mla_enabled_predictor_required_feature_columns_are_flattened_from_schema() -> None:
    assert get_enabled_predictor_required_feature_columns(
        LATENT_MLA_ATTENTION_FAMILY
    ) == MLA_IMPORTED_PREDICTOR_FEATURE_COLUMNS


def test_mla_shared_predictor_feature_columns_use_same_imported_schema_contract() -> None:
    expected_features = get_enabled_predictor_required_feature_columns(
        LATENT_MLA_ATTENTION_FAMILY
    )

    assert get_enabled_shared_predictor_feature_columns(
        LATENT_MLA_ATTENTION_FAMILY
    ) == {
        op_name: expected_features for op_name in MLA_PHYSICAL_OPS
    }
    assert get_enabled_shared_predictor_required_feature_columns(
        LATENT_MLA_ATTENTION_FAMILY
    ) == expected_features


def test_dsa_enabled_predictor_feature_columns_remain_frozen() -> None:
    with pytest.raises(NotImplementedError, match="DSA attention is frozen"):
        get_enabled_predictor_feature_columns(DSA_ATTENTION_FAMILY)


def test_dsa_enabled_predictor_required_feature_columns_remain_frozen() -> None:
    with pytest.raises(NotImplementedError, match="DSA attention is frozen"):
        get_enabled_predictor_required_feature_columns(DSA_ATTENTION_FAMILY)


def test_catalog_alignment_fails_fast_when_a_metric_catalog_is_missing_family_op() -> None:
    profiling_values = {metric.value for metric in ProfilingOperationMetrics}
    e2e_values = {metric.value for metric in E2EOperationMetrics}
    e2e_values.remove("attn_mla_decode")

    with pytest.raises(ValueError, match="E2E metrics catalog is missing"):
        validate_attention_catalog_alignment(
            profiling_metric_values=profiling_values,
            e2e_metric_values=e2e_values,
        )


def test_profiling_schema_columns_are_derived_from_family_spec() -> None:
    assert get_required_profiling_feature_columns(DENSE_ATTENTION_FAMILY) == (
        "measurement_type",
        "attention_backend",
        "n_q_head",
        "n_kv_head",
        "block_size",
        "num_tensor_parallel_workers",
        "max_model_len",
        "batch_size",
        "prefill_chunk_size",
        "kv_cache_size",
        "is_prefill",
    )
    assert get_profiling_time_stat_columns(DENSE_ATTENTION_FAMILY) == (
        "time_stats.attn_kv_cache_save.min",
        "time_stats.attn_kv_cache_save.max",
        "time_stats.attn_kv_cache_save.mean",
        "time_stats.attn_kv_cache_save.median",
        "time_stats.attn_kv_cache_save.std",
        "time_stats.attn_kv_cache_save.count",
        "time_stats.attn_prefill.min",
        "time_stats.attn_prefill.max",
        "time_stats.attn_prefill.mean",
        "time_stats.attn_prefill.median",
        "time_stats.attn_prefill.std",
        "time_stats.attn_prefill.count",
        "time_stats.attn_decode.min",
        "time_stats.attn_decode.max",
        "time_stats.attn_decode.mean",
        "time_stats.attn_decode.median",
        "time_stats.attn_decode.std",
        "time_stats.attn_decode.count",
    )
    assert get_required_profiling_columns(LATENT_MLA_ATTENTION_FAMILY) == (
        *get_required_profiling_feature_columns(LATENT_MLA_ATTENTION_FAMILY),
        *get_profiling_time_stat_columns(LATENT_MLA_ATTENTION_FAMILY),
    )


@pytest.mark.parametrize(
    "measurement_type",
    [MeasurementType.CUDA_EVENT, MeasurementType.KERNEL_ONLY],
)
def test_dense_attention_profiling_feature_schema_is_typed_and_measurement_aware(
    measurement_type: MeasurementType,
) -> None:
    assert hasattr(profiling_mapping, "get_attention_profiling_feature_schema")
    schema = profiling_mapping.get_attention_profiling_feature_schema(
        DENSE_ATTENTION_FAMILY,
        measurement_type=measurement_type,
    )

    assert schema.family_id == "dense_attention"
    assert schema.memory_layout is AttentionMemoryLayout.DENSE_KV
    assert schema.measurement_type is measurement_type
    assert schema.required_columns == (
        "measurement_type",
        "attention_backend",
        "n_q_head",
        "n_kv_head",
        "block_size",
        "num_tensor_parallel_workers",
        "max_model_len",
        "batch_size",
        "prefill_chunk_size",
        "kv_cache_size",
        "is_prefill",
    )
    assert schema.predictor_feature_columns is None
    assert get_required_profiling_feature_columns(
        DENSE_ATTENTION_FAMILY,
        measurement_type=measurement_type,
    ) == schema.required_columns


@pytest.mark.parametrize(
    "measurement_type",
    ["cuda_event", "KERNEL_ONLY", MeasurementType.CUDA_EVENT],
)
def test_mla_attention_profiling_feature_schema_drives_imported_predictor_features(
    measurement_type: str | MeasurementType,
) -> None:
    assert hasattr(profiling_mapping, "get_attention_profiling_feature_schema")
    assert hasattr(profiling_mapping, "get_imported_mla_predictor_feature_columns")
    schema = profiling_mapping.get_attention_profiling_feature_schema(
        LATENT_MLA_ATTENTION_FAMILY,
        measurement_type=measurement_type,
    )

    assert schema.family_id == "latent_mla_attention"
    assert schema.memory_layout is AttentionMemoryLayout.LATENT_MLA
    assert schema.measurement_type in {
        MeasurementType.CUDA_EVENT,
        MeasurementType.KERNEL_ONLY,
    }
    assert schema.required_columns == (
        "measurement_type",
        "attention_backend",
        "n_q_head",
        "n_kv_head",
        "head_size",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "qk_head_dim",
        "kv_lora_rank",
        "v_head_dim",
        "block_size",
        "num_tensor_parallel_workers",
        "max_model_len",
        "batch_size",
        "batch_num_tokens",
        "batch_num_prefill_tokens",
        "batch_num_decode_tokens",
        "max_seqlen_q",
        "max_seqlen_k",
        "num_actual_tokens",
        "is_prefill",
        "max_seq_len",
        "is_mla_profile_import",
    )
    assert schema.predictor_feature_columns == MLA_IMPORTED_PREDICTOR_FEATURE_COLUMNS
    assert profiling_mapping.get_imported_mla_predictor_feature_columns() == (
        schema.predictor_feature_columns
    )
    assert "measurement_type" not in schema.predictor_feature_columns
    assert "is_mla_profile_import" not in schema.predictor_feature_columns


def test_attention_profiling_feature_schema_rejects_invalid_measurement_type() -> None:
    with pytest.raises(ValueError, match="Unsupported measurement_type"):
        profiling_mapping.get_attention_profiling_feature_schema(
            DENSE_ATTENTION_FAMILY,
            measurement_type="timer_magic",
        )


def test_attention_profiling_feature_schema_keeps_dsa_frozen() -> None:
    with pytest.raises(NotImplementedError, match="DSA attention is frozen"):
        profiling_mapping.get_attention_profiling_feature_schema(
            DSA_ATTENTION_FAMILY,
            measurement_type=MeasurementType.CUDA_EVENT,
        )


def test_profiling_schema_validation_accepts_both_measurement_families() -> None:
    required_columns = get_required_profiling_columns(DENSE_ATTENTION_FAMILY)
    row = {column: 1 for column in required_columns}
    row.update(
        {
            "measurement_type": "CUDA_EVENT",
            "attention_backend": "FLASHINFER",
            "is_prefill": True,
        }
    )

    profiling_mapping.validate_attention_profiling_dataframe(
        pd.DataFrame([row]),
        DENSE_ATTENTION_FAMILY,
        measurement_type="CUDA_EVENT",
    )
    row["measurement_type"] = "KERNEL_ONLY"
    profiling_mapping.validate_attention_profiling_dataframe(
        pd.DataFrame([row]),
        DENSE_ATTENTION_FAMILY,
        measurement_type="KERNEL_ONLY",
    )
    row["measurement_type"] = "UNKNOWN"
    with pytest.raises(ValueError, match="Unsupported measurement_type"):
        profiling_mapping.validate_attention_profiling_dataframe(
            pd.DataFrame([row]),
            DENSE_ATTENTION_FAMILY,
        )


def test_profiling_schema_validation_rejects_missing_columns_and_frozen_dsa() -> None:
    with pytest.raises(ValueError, match="missing required attention profiling columns"):
        profiling_mapping.validate_attention_profiling_dataframe(
            pd.DataFrame({"measurement_type": ["CUDA_EVENT"]}),
            LATENT_MLA_ATTENTION_FAMILY,
        )

    with pytest.raises(NotImplementedError, match="DSA attention is frozen"):
        get_required_profiling_columns(DSA_ATTENTION_FAMILY)
