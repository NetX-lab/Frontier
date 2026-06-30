"""Cluster C1 — attention feature/layout data is declared on the family spec.

Before C1 the runtime ``kv_factor`` and the profiling feature-column schemas lived
as ``memory_layout``-keyed branches and private module constants inside
``frontier.attention.memory`` / ``frontier.attention.profiling_mapping``. C1 promotes
that data onto :class:`AttentionFamilySpec` so the accessors become branch-free reads.

These tests pin the promoted data directly on the spec objects (exact tuples) so the
de-branching cannot silently drift the dense/MLA profiling or KV-sizing contracts.
"""

from __future__ import annotations

import pytest

from frontier.attention.families import (
    DENSE_ATTENTION_FAMILY,
    DSA_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
)
from frontier.attention.memory import get_attention_runtime_kv_layout
from frontier.attention import profiling_mapping


DENSE_PROFILING_FEATURE_COLUMNS = (
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

MLA_PROFILING_FEATURE_COLUMNS = (
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

MLA_IMPORTED_PREDICTOR_EXCLUDED_FEATURE_COLUMNS = (
    "measurement_type",
    "attention_backend",
    "max_model_len",
    "is_mla_profile_import",
)


def test_dense_family_declares_kv_factor_on_spec() -> None:
    assert DENSE_ATTENTION_FAMILY.kv_factor == 2


def test_latent_mla_family_declares_kv_factor_on_spec() -> None:
    assert LATENT_MLA_ATTENTION_FAMILY.kv_factor == 1


def test_frozen_dsa_family_declares_no_kv_factor() -> None:
    assert DSA_ATTENTION_FAMILY.kv_factor is None


def test_runtime_kv_layout_reads_kv_factor_from_spec() -> None:
    dense_layout = get_attention_runtime_kv_layout(
        DENSE_ATTENTION_FAMILY,
        runtime_num_kv_heads_per_worker=8,
        runtime_head_size=128,
    )
    mla_layout = get_attention_runtime_kv_layout(
        LATENT_MLA_ATTENTION_FAMILY,
        runtime_num_kv_heads_per_worker=1,
        runtime_head_size=576,
    )
    assert dense_layout.kv_factor == DENSE_ATTENTION_FAMILY.kv_factor == 2
    assert mla_layout.kv_factor == LATENT_MLA_ATTENTION_FAMILY.kv_factor == 1


def test_frozen_dsa_runtime_kv_layout_still_raises_before_reading_kv_factor() -> None:
    with pytest.raises(NotImplementedError, match="DSA attention is frozen"):
        get_attention_runtime_kv_layout(
            DSA_ATTENTION_FAMILY,
            runtime_num_kv_heads_per_worker=1,
            runtime_head_size=1,
        )


def test_dense_family_declares_required_profiling_feature_columns_on_spec() -> None:
    assert (
        DENSE_ATTENTION_FAMILY.required_profiling_feature_columns
        == DENSE_PROFILING_FEATURE_COLUMNS
    )


def test_latent_mla_family_declares_required_profiling_feature_columns_on_spec() -> None:
    assert (
        LATENT_MLA_ATTENTION_FAMILY.required_profiling_feature_columns
        == MLA_PROFILING_FEATURE_COLUMNS
    )


def test_frozen_dsa_family_declares_no_profiling_feature_columns() -> None:
    assert DSA_ATTENTION_FAMILY.required_profiling_feature_columns == ()


def test_latent_mla_family_declares_imported_predictor_exclusions_on_spec() -> None:
    assert (
        LATENT_MLA_ATTENTION_FAMILY.imported_predictor_excluded_feature_columns
        == MLA_IMPORTED_PREDICTOR_EXCLUDED_FEATURE_COLUMNS
    )


def test_spec_required_profiling_feature_columns_drive_accessor() -> None:
    assert (
        profiling_mapping.get_required_profiling_feature_columns(DENSE_ATTENTION_FAMILY)
        == DENSE_ATTENTION_FAMILY.required_profiling_feature_columns
    )
    assert (
        profiling_mapping.get_required_profiling_feature_columns(
            LATENT_MLA_ATTENTION_FAMILY
        )
        == LATENT_MLA_ATTENTION_FAMILY.required_profiling_feature_columns
    )


def test_imported_mla_predictor_columns_derive_from_spec_data() -> None:
    excluded = set(
        LATENT_MLA_ATTENTION_FAMILY.imported_predictor_excluded_feature_columns
    )
    expected = tuple(
        column
        for column in LATENT_MLA_ATTENTION_FAMILY.required_profiling_feature_columns
        if column not in excluded
    )
    assert profiling_mapping.get_imported_mla_predictor_feature_columns() == expected


def test_attention_family_runtime_resolvers_match_current_cache_formulas() -> None:
    class DenseRuntimeConfig:
        num_kv_heads = 16

        def get_head_dim(self) -> int:
            return 128

    class MlaRuntimeConfig:
        kv_lora_rank = 512
        qk_rope_head_dim = 64

    assert DENSE_ATTENTION_FAMILY.resolve_runtime_num_kv_heads(
        DenseRuntimeConfig()
    ) == 16
    assert DENSE_ATTENTION_FAMILY.resolve_runtime_head_size(
        DenseRuntimeConfig()
    ) == 128
    assert LATENT_MLA_ATTENTION_FAMILY.resolve_runtime_num_kv_heads(
        MlaRuntimeConfig()
    ) == 1
    assert LATENT_MLA_ATTENTION_FAMILY.resolve_runtime_head_size(
        MlaRuntimeConfig()
    ) == 576


def test_latent_mla_runtime_resolver_fails_fast_on_missing_fields() -> None:
    class MissingRopeDim:
        kv_lora_rank = 512

    with pytest.raises(ValueError, match="qk_rope_head_dim"):
        LATENT_MLA_ATTENTION_FAMILY.resolve_runtime_head_size(MissingRopeDim())


def test_latent_mla_family_declares_runtime_meta_contract() -> None:
    contract = LATENT_MLA_ATTENTION_FAMILY.runtime_meta_contract

    assert contract is not None
    assert contract.expected_runtime_num_kv_heads == 1
    assert contract.runtime_head_size_formula == "kv_lora_rank + qk_rope_head_dim"
    assert contract.supported_block_sizes == (32, 64)
    assert contract.expected_n_q_head is None
