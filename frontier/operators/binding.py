from __future__ import annotations

from dataclasses import dataclass

from frontier.operators.spec import (
    OperatorFamilySpec,
    OperatorSpec,
    TensorParallelMode,
)


@dataclass(frozen=True)
class FamilyBinding:
    """Concrete model-to-operator-family binding."""

    family_id: str
    variant_id: str
    family: OperatorFamilySpec
    frozen: bool = False
    reason: str = ""

    def require_enabled_for_execution(self) -> None:
        if self.frozen:
            raise NotImplementedError(
                f"Operator family binding {self.family_id}/{self.variant_id} "
                f"is frozen: {self.reason}"
            )
        self.family.require_enabled_for_execution()


@dataclass(frozen=True)
class OperatorQueryBinding:
    """Registry facts bound to one runtime operator query."""

    family_id: str
    family: OperatorFamilySpec
    operator: OperatorSpec
    physical_name: str
    profiling_name: str
    tp_mode: TensorParallelMode | None


def bind_operator_query(
    operator_name: str,
    *,
    family_id: str | None = None,
) -> OperatorQueryBinding:
    """Bind a physical or unambiguous profiling operator name to the registry."""

    if not isinstance(operator_name, str) or not operator_name:
        raise ValueError("Operator query name must be a non-empty string")

    from frontier.operators.families import iter_operator_families

    families = tuple(iter_operator_families())
    if family_id is not None:
        family_matches = tuple(family for family in families if family.family_id == family_id)
        if not family_matches:
            raise ValueError(f"Unknown operator family: {family_id}")

    physical_matches = tuple(
        (family, operator)
        for family in families
        for operator in family.operators
        if operator.name == operator_name
    )
    profiling_matches = tuple(
        (family, operator)
        for family in families
        for operator in family.operators
        if operator.profiling_name() == operator_name
    )

    if len(physical_matches) > 1:
        raise ValueError(
            f"Duplicate physical operator registration for {operator_name!r}"
        )
    if len(physical_matches) == 1:
        family, operator = physical_matches[0]
        conflicting_aliases = tuple(
            (candidate_family, candidate_operator)
            for candidate_family, candidate_operator in profiling_matches
            if candidate_operator is not operator
        )
        if conflicting_aliases:
            names = [
                f"{candidate_family.family_id}/{candidate_operator.name}"
                for candidate_family, candidate_operator in conflicting_aliases
            ]
            raise ValueError(
                f"Operator query {operator_name!r} collides with profiling aliases: "
                f"{names}"
            )
    elif not profiling_matches:
        raise ValueError(f"Unknown operator query: {operator_name!r}")
    elif len(profiling_matches) > 1:
        names = [
            f"{candidate_family.family_id}/{candidate_operator.name}"
            for candidate_family, candidate_operator in profiling_matches
        ]
        raise ValueError(
            f"Operator query {operator_name!r} is an ambiguous profiling alias: {names}"
        )
    else:
        family, operator = profiling_matches[0]

    if family_id is not None and family.family_id != family_id:
        raise ValueError(
            f"Operator {operator_name!r} does not belong to operator family "
            f"{family_id!r}; registered family is {family.family_id!r}"
        )

    family.require_enabled_for_execution()
    return OperatorQueryBinding(
        family_id=family.family_id,
        family=family,
        operator=operator,
        physical_name=operator.name,
        profiling_name=operator.profiling_name(),
        tp_mode=operator.tp_mode,
    )


@dataclass(frozen=True)
class OperatorManifest:
    """Ordered operator families selected for one model config."""

    family_bindings: tuple[FamilyBinding, ...]

    def families(self) -> tuple[OperatorFamilySpec, ...]:
        return tuple(binding.family for binding in self.family_bindings)

    def operators(self) -> tuple[OperatorSpec, ...]:
        return tuple(
            operator
            for binding in self.family_bindings
            for operator in binding.family.operators
        )


def _get_model_architecture_profile(config):
    get_profile = getattr(config, "get_model_architecture_profile", None)
    if callable(get_profile):
        return get_profile()

    from frontier.model_architectures import get_model_architecture_profile

    return get_model_architecture_profile(config)


def build_operator_manifest(config) -> OperatorManifest:
    """Build the operator manifest for a model config."""

    from frontier.attention.model_binding import bind_attention_family
    from frontier.operators.families import (
        FFN_FAMILY,
        MEMORY_FAMILY,
        MOE_FAMILY,
        SHARE_EXPERT_FAMILY,
    )

    architecture_profile = _get_model_architecture_profile(config)
    architecture_profile.validate_structural_requirements(config)
    attention_binding = bind_attention_family(config)
    bindings = [
        FamilyBinding(
            family_id=attention_binding.family_id,
            variant_id=attention_binding.variant_id,
            family=attention_binding.family,
            frozen=attention_binding.frozen,
            reason=attention_binding.reason,
        ),
        FamilyBinding(
            family_id=MEMORY_FAMILY.family_id,
            variant_id="replicated",
            family=MEMORY_FAMILY,
        ),
    ]
    if bool(getattr(config, "is_moe", False)):
        bindings.append(
            FamilyBinding(
                family_id=MOE_FAMILY.family_id,
                variant_id="routed",
                family=MOE_FAMILY,
            )
        )
        if not hasattr(config, "supports_share_expert"):
            raise TypeError("MoE operator manifests require config.supports_share_expert()")
        if config.supports_share_expert():
            bindings.append(
                FamilyBinding(
                    family_id=SHARE_EXPERT_FAMILY.family_id,
                    variant_id="shared_dense",
                    family=SHARE_EXPERT_FAMILY,
                )
            )
    else:
        bindings.append(
            FamilyBinding(
                family_id=FFN_FAMILY.family_id,
                variant_id="dense",
                family=FFN_FAMILY,
            )
        )
    return OperatorManifest(
        family_bindings=tuple(bindings)
    )
