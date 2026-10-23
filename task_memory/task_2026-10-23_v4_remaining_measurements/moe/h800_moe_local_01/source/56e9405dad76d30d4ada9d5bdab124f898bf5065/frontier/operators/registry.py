from __future__ import annotations

from collections import OrderedDict

from frontier.operators.spec import OperatorFamilySpec


class OperatorRegistry:
    """Ordered registry for operator families."""

    def __init__(self) -> None:
        self._families_by_id: OrderedDict[str, OperatorFamilySpec] = OrderedDict()

    def register(self, family: OperatorFamilySpec) -> None:
        if family.family_id in self._families_by_id:
            raise ValueError(f"Duplicate operator family: {family.family_id}")
        self._families_by_id[family.family_id] = family

    def get_family(self, family_id: str) -> OperatorFamilySpec:
        try:
            return self._families_by_id[family_id]
        except KeyError as exc:
            raise ValueError(f"Unknown operator family: {family_id}") from exc

    def iter_families(self) -> tuple[OperatorFamilySpec, ...]:
        return tuple(self._families_by_id.values())

    def iter_execution_enabled_families(self) -> tuple[OperatorFamilySpec, ...]:
        return tuple(
            family
            for family in self.iter_families()
            if family.execution_enabled
        )
