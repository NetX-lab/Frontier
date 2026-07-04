from __future__ import annotations

from dataclasses import dataclass

from frontier.operators.spec import OperatorFamilySpec, OperatorSpec


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


def build_operator_manifest(config) -> OperatorManifest:
    """Build the operator manifest for a model config."""

    from frontier.attention.model_binding import bind_attention_family
    from frontier.operators.families import (
        FFN_FAMILY,
        MEMORY_FAMILY,
        MOE_FAMILY,
        SHARE_EXPERT_FAMILY,
    )

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
