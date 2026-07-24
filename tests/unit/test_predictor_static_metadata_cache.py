from dataclasses import replace
from types import SimpleNamespace

from frontier.attention.families import DENSE_ATTENTION_FAMILY
from frontier.attention.ops import AttentionOperatorRole
from frontier.attention.profiling_mapping import (
    _get_registered_enabled_predictor_metric_name_by_role,
    get_enabled_predictor_metric_name_by_role,
)
from frontier.entities import time_components
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.model_architectures import ModelArchitectureProfile
from frontier.operators.families import (
    FFN_FAMILY,
    _get_registered_family_profiling_name_set,
    get_family_profiling_name_set,
)


class _ConcreteSklearnExecutionTimePredictor(SklearnExecutionTimePredictor):
    def _get_grid_search_params(self):
        return {}

    def _get_estimator(self):
        return None


def test_predictor_caches_model_architecture_profile_per_instance() -> None:
    profile = ModelArchitectureProfile.generic()
    getter_calls = 0

    def get_model_architecture_profile() -> ModelArchitectureProfile:
        nonlocal getter_calls
        getter_calls += 1
        return profile

    predictor = object.__new__(_ConcreteSklearnExecutionTimePredictor)
    predictor._model_config = SimpleNamespace(
        get_model_architecture_profile=get_model_architecture_profile,
    )

    assert predictor._get_model_architecture_profile() is profile
    assert predictor._get_model_architecture_profile() is profile
    assert getter_calls == 1


def test_attention_role_lookup_reuses_immutable_family_metadata(monkeypatch) -> None:
    family_type = type(DENSE_ATTENTION_FAMILY)
    original_hash = family_type.__hash__
    original_predictor_ops = family_type.predictor_ops
    family_hash_calls = 0
    predictor_ops_calls = 0

    def counted_hash(self):
        nonlocal family_hash_calls
        family_hash_calls += 1
        return original_hash(self)

    def counted_predictor_ops(self):
        nonlocal predictor_ops_calls
        predictor_ops_calls += 1
        return original_predictor_ops(self)

    monkeypatch.setattr(family_type, "__hash__", counted_hash)
    monkeypatch.setattr(family_type, "predictor_ops", counted_predictor_ops)
    _get_registered_enabled_predictor_metric_name_by_role.cache_clear()

    first = get_enabled_predictor_metric_name_by_role(
        DENSE_ATTENTION_FAMILY,
        AttentionOperatorRole.CACHE_WRITE,
    )
    second = get_enabled_predictor_metric_name_by_role(
        DENSE_ATTENTION_FAMILY,
        AttentionOperatorRole.CACHE_WRITE,
    )

    assert first == second == "attn_kv_cache_save"
    assert family_hash_calls == 0
    assert predictor_ops_calls == 1


def test_family_profiling_name_set_reuses_immutable_family_metadata(monkeypatch) -> None:
    family_type = type(FFN_FAMILY)
    original_hash = family_type.__hash__
    original_profiling_ops = family_type.profiling_ops
    family_hash_calls = 0
    profiling_ops_calls = 0

    def counted_hash(self):
        nonlocal family_hash_calls
        family_hash_calls += 1
        return original_hash(self)

    def counted_profiling_ops(self):
        nonlocal profiling_ops_calls
        profiling_ops_calls += 1
        return original_profiling_ops(self)

    monkeypatch.setattr(family_type, "__hash__", counted_hash)
    monkeypatch.setattr(family_type, "profiling_ops", counted_profiling_ops)
    _get_registered_family_profiling_name_set.cache_clear()

    first = get_family_profiling_name_set(FFN_FAMILY)
    second = get_family_profiling_name_set(FFN_FAMILY)

    assert first == second == frozenset(
        {"mlp_up_proj", "mlp_act", "mlp_down_proj"}
    )
    assert family_hash_calls == 0
    assert profiling_ops_calls == 1


def test_attention_role_lookup_preserves_unregistered_same_id_family_semantics() -> None:
    alternate_cache_operator = replace(
        DENSE_ATTENTION_FAMILY.operators[0],
        name="alternate_cache_write",
    )
    unregistered_family = replace(
        DENSE_ATTENTION_FAMILY,
        operators=(
            alternate_cache_operator,
            *DENSE_ATTENTION_FAMILY.operators[1:],
        ),
    )

    assert (
        get_enabled_predictor_metric_name_by_role(
            unregistered_family,
            AttentionOperatorRole.CACHE_WRITE,
        )
        == "alternate_cache_write"
    )


def test_family_profiling_name_set_preserves_unregistered_same_id_family_semantics() -> None:
    alternate_up_projection = replace(
        FFN_FAMILY.operators[0],
        profiling_key="alternate_mlp_up_proj",
    )
    unregistered_family = replace(
        FFN_FAMILY,
        operators=(alternate_up_projection, *FFN_FAMILY.operators[1:]),
    )

    assert get_family_profiling_name_set(unregistered_family) == frozenset(
        {"alternate_mlp_up_proj", "mlp_act", "mlp_down_proj"}
    )


def test_execution_time_operator_mapping_is_cached_but_not_mutably_shared(
    monkeypatch,
) -> None:
    original_iter_families = time_components.iter_execution_enabled_families
    iter_families_calls = 0

    def counted_iter_families():
        nonlocal iter_families_calls
        iter_families_calls += 1
        return original_iter_families()

    monkeypatch.setattr(
        time_components,
        "iter_execution_enabled_families",
        counted_iter_families,
    )
    cache_owner = getattr(
        time_components,
        "_canonical_operator_execution_time_attr_items",
        None,
    )
    if cache_owner is not None:
        cache_owner.cache_clear()

    first = time_components.canonical_operator_execution_time_attrs()
    second = time_components.canonical_operator_execution_time_attrs()

    assert first == second
    assert first is not second
    assert iter_families_calls == 1

    first["test-only-mutation"] = "test_only_attr"
    assert "test-only-mutation" not in (
        time_components.canonical_operator_execution_time_attrs()
    )
