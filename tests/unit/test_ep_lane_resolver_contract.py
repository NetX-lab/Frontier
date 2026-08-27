"""Focused boundary tests for canonical EP-lane descriptor resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from frontier.moe_ep_workload import resolve_ep_lane_workload


class _OptionalLaneProperty:
    @property
    def lane_workload(self):
        return None


def test_missing_lane_attribute_is_optional_only_when_not_required() -> None:
    source = SimpleNamespace()

    assert resolve_ep_lane_workload(source, required=False) is None

    with pytest.raises(ValueError, match="EPLaneWorkload descriptor is required"):
        resolve_ep_lane_workload(source, required=True)


def test_none_lane_property_is_optional_only_when_not_required() -> None:
    source = _OptionalLaneProperty()

    assert resolve_ep_lane_workload(source, required=False) is None

    with pytest.raises(ValueError, match="EPLaneWorkload descriptor is required"):
        resolve_ep_lane_workload(source, required=True)


def test_malformed_lane_descriptor_fails_even_when_optional() -> None:
    source = SimpleNamespace(lane_workload={0: 1})

    with pytest.raises(TypeError, match="EPLaneWorkload"):
        resolve_ep_lane_workload(source, required=False)
