"""C8 — the frozen-DSA execution guard is centralized in one registry accessor.

`iter_execution_enabled_families()` is the single source of truth for "which
attention families participate in execution / profiling / training catalogs".
Catalog consumers iterate it instead of re-implementing a `dsa_frozen` skip.
"""

from __future__ import annotations

from frontier.attention.families import (
    DENSE_ATTENTION_FAMILY,
    DEEPSEEK_V4_ATTENTION_FAMILY,
    DSA_ATTENTION_FAMILY,
    LATENT_MLA_ATTENTION_FAMILY,
    iter_attention_families,
    iter_execution_enabled_families,
)


def test_iter_execution_enabled_families_excludes_frozen_dsa() -> None:
    enabled_ids = {family.family_id for family in iter_execution_enabled_families()}
    assert DSA_ATTENTION_FAMILY.family_id not in enabled_ids
    assert DEEPSEEK_V4_ATTENTION_FAMILY.family_id not in enabled_ids
    assert DENSE_ATTENTION_FAMILY.family_id in enabled_ids
    assert LATENT_MLA_ATTENTION_FAMILY.family_id in enabled_ids


def test_iter_execution_enabled_families_returns_only_runtime_enabled() -> None:
    for family in iter_execution_enabled_families():
        assert family.dsa_frozen is False
        # The guard must not fire for any family this accessor yields.
        family.require_enabled_for_execution()


def test_iter_execution_enabled_families_is_order_preserving_subset() -> None:
    full_ids = [family.family_id for family in iter_attention_families()]
    enabled_ids = [family.family_id for family in iter_execution_enabled_families()]
    unsupported = {
        DSA_ATTENTION_FAMILY.family_id, DEEPSEEK_V4_ATTENTION_FAMILY.family_id,
    }
    assert enabled_ids == [
        family_id for family_id in full_ids if family_id not in unsupported
    ]
    excluded = set(full_ids) - set(enabled_ids)
    assert excluded == unsupported


def test_quantization_registry_attention_ops_match_execution_enabled_helper() -> None:
    """The default quant registry's attention ops are exactly the execution-enabled
    families' profiling ops; structural-only targets do not contribute."""
    from frontier.config.quantization_manager import QuantizationManager

    QuantizationManager.reset()
    try:
        manager = QuantizationManager()
        compute_ops = set(manager.get_supported_operations()["compute_operations"])
        expected_attention_ops = {
            operator.name
            for family in iter_execution_enabled_families()
            for operator in family.profiling_ops()
        }
        assert expected_attention_ops.issubset(compute_ops)
        for op_name in expected_attention_ops:
            assert manager.is_compute_operation(op_name)
        # Structural targets stay outside the measured precision registry.
        frozen_ops = {
            operator.name
            for family in iter_attention_families()
            if family.dsa_frozen
            for operator in family.profiling_ops()
        }
        assert frozen_ops.isdisjoint(compute_ops)
    finally:
        QuantizationManager.reset()
