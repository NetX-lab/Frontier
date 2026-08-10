"""Tests for the closed-loop (concurrency-capped) request generator and its config."""
from __future__ import annotations

import pytest

from frontier.config import ClosedLoopRequestGeneratorConfig, SyntheticRequestGeneratorConfig
from frontier.config.config import FixedRequestLengthGeneratorConfig, SimulationConfig
from frontier.request_generator.closed_loop_request_generator import (
    ClosedLoopRequestGenerator,
)
from frontier.request_generator.request_generator_registry import (
    RequestGeneratorRegistry,
)
from frontier.types import RequestGeneratorType


def test_get_type_is_closed_loop() -> None:
    assert ClosedLoopRequestGeneratorConfig().get_type() == RequestGeneratorType.CLOSED_LOOP


def test_max_tokens_derived_from_length_generator_config() -> None:
    length_config = FixedRequestLengthGeneratorConfig(prefill_tokens=100, decode_tokens=20)
    config = ClosedLoopRequestGeneratorConfig(closed_loop_length_generator_config=length_config)
    assert config.max_tokens == length_config.max_tokens


def test_max_concurrency_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        ClosedLoopRequestGeneratorConfig(max_concurrency=0)


def test_num_requests_must_be_positive() -> None:
    with pytest.raises(ValueError, match="num_requests"):
        ClosedLoopRequestGeneratorConfig(num_requests=0)


def test_generates_full_population_all_arriving_at_zero() -> None:
    config = ClosedLoopRequestGeneratorConfig(
        num_requests=5,
        max_concurrency=2,
        closed_loop_length_generator_config=FixedRequestLengthGeneratorConfig(
            prefill_tokens=64, decode_tokens=16
        ),
    )
    generator = ClosedLoopRequestGenerator(config)
    requests = generator.generate()

    assert len(requests) == 5
    assert all(request.arrived_at == 0.0 for request in requests)
    assert all(request.num_prefill_tokens == 64 for request in requests)
    assert all(request.num_decode_tokens == 16 for request in requests)
    assert config.num_decode_bound_requests == 5


def test_registered_in_request_generator_registry() -> None:
    config = ClosedLoopRequestGeneratorConfig(num_requests=1)
    generator = RequestGeneratorRegistry.get(config.get_type(), config)
    assert isinstance(generator, ClosedLoopRequestGenerator)


def test_closed_loop_requires_online_simulation_mode() -> None:
    stub_config = object.__new__(SimulationConfig)
    stub_config.request_generator_config = ClosedLoopRequestGeneratorConfig()
    stub_config.simulation_mode = "offline"

    with pytest.raises(ValueError, match="simulation_mode"):
        SimulationConfig._validate_closed_loop_request_generator_config(stub_config)


def test_non_closed_loop_generator_skips_validation_regardless_of_mode() -> None:
    stub_config = object.__new__(SimulationConfig)
    stub_config.request_generator_config = SyntheticRequestGeneratorConfig()
    stub_config.simulation_mode = "offline"

    # Should not raise -- validator only applies when closed_loop is selected.
    SimulationConfig._validate_closed_loop_request_generator_config(stub_config)


def test_closed_loop_allows_online_simulation_mode() -> None:
    stub_config = object.__new__(SimulationConfig)
    stub_config.request_generator_config = ClosedLoopRequestGeneratorConfig()
    stub_config.simulation_mode = "online"

    # Should not raise.
    SimulationConfig._validate_closed_loop_request_generator_config(stub_config)
