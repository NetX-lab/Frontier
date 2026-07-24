#!/usr/bin/env python3
"""Regression tests for offline PD+AF decode-attn admission defaults."""

from __future__ import annotations

from types import SimpleNamespace

from frontier.config.config import SimulationConfig


def _build_config_stub(*, threshold, num_requests: int = 8) -> SimulationConfig:
    config = object.__new__(SimulationConfig)
    config.simulation_mode = "offline"
    config.sys_arch = "pd-af-disaggregation"
    config.cluster_config = SimpleNamespace(
        decode_attn_request_allocation_threshold=threshold,
    )
    config.request_generator_config = SimpleNamespace(num_requests=num_requests)
    return config


def test_offline_pdaf_default_decode_attn_threshold_stays_disabled() -> None:
    config = _build_config_stub(threshold=None)

    config._maybe_set_default_decode_attn_request_allocation_threshold()

    assert config.cluster_config.decode_attn_request_allocation_threshold is None


def test_explicit_decode_attn_threshold_is_preserved() -> None:
    config = _build_config_stub(threshold=8)

    config._maybe_set_default_decode_attn_request_allocation_threshold()

    assert config.cluster_config.decode_attn_request_allocation_threshold == 8
