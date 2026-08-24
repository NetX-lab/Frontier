"""Typed identity-scope contract for M2N lane validation."""

from __future__ import annotations

import inspect

import pytest

import frontier.scheduler.cluster_scheduler.base_cluster_scheduler as scheduler_module
from frontier.scheduler.cluster_scheduler.base_cluster_scheduler import (
    BaseClusterScheduler,
)


def _identity_scope(name: str):
    scope_type = getattr(scheduler_module, "M2NLaneIdentityScope", None)
    assert scope_type is not None
    return getattr(scope_type, name)


def test_m2n_full_stage_scope_is_independent_of_field_name() -> None:
    normalized = BaseClusterScheduler._normalize_m2n_lane_contract(
        [(0, None)],
        identity_scope=_identity_scope("FULL_STAGE"),
        field_name="renamed transport contract",
        require_nonempty=True,
    )

    assert normalized == [(0, None)]


def test_m2n_replica_local_scope_rejects_absent_local_identity() -> None:
    with pytest.raises(ValueError, match="replica_local_id cannot be None"):
        BaseClusterScheduler._normalize_m2n_lane_contract(
            [(0, None)],
            identity_scope=_identity_scope("REPLICA_LOCAL"),
            field_name="DECODE_ATTN text must not select the scope",
            require_nonempty=True,
        )


def test_m2n_lane_contract_rejects_non_enum_identity_scope() -> None:
    with pytest.raises(ValueError, match="identity_scope must be an exact"):
        BaseClusterScheduler._normalize_m2n_lane_contract(
            [(0, None)],
            identity_scope="full_stage",
            field_name="M2N lane contract",
            require_nonempty=True,
        )


def test_m2n_lane_contract_requires_explicit_identity_scope() -> None:
    signature = inspect.signature(
        BaseClusterScheduler._normalize_m2n_lane_contract
    )

    assert signature.parameters["identity_scope"].default is inspect.Parameter.empty


def test_m2n_lane_contract_does_not_classify_scope_from_field_name() -> None:
    source = inspect.getsource(
        BaseClusterScheduler._normalize_m2n_lane_contract
    )

    assert "field_name.startswith" not in source
