#!/usr/bin/env python3
"""Regression tests for request-generator decode-bound workload metadata."""

from __future__ import annotations

import pytest

from frontier.config.config import BaseRequestGeneratorConfig
from frontier.config.request_generator_config import (
    PoissonRequestIntervalGeneratorConfig,
    StaticRequestIntervalGeneratorConfig,
    SyntheticRequestGeneratorConfig,
)
from frontier.request_generator.base_request_generator import BaseRequestGenerator


class _MixedDecodeRequestGenerator(BaseRequestGenerator):
    def generate_requests(self):
        return [
            self._build_request(
                arrived_at=0.0,
                num_prefill_tokens=8,
                num_decode_tokens=0,
            ),
            self._build_request(
                arrived_at=1.0,
                num_prefill_tokens=8,
                num_decode_tokens=3,
            ),
        ]


def test_generate_records_decode_bound_request_count() -> None:
    config = BaseRequestGeneratorConfig()
    generator = _MixedDecodeRequestGenerator(config)

    requests = generator.generate()

    assert len(requests) == 2
    assert config.num_decode_bound_requests == 1


def test_static_arrivals_require_request_count_instead_of_positive_duration() -> None:
    with pytest.raises(ValueError, match="num_requests without duration"):
        SyntheticRequestGeneratorConfig(
            interval_generator_config=StaticRequestIntervalGeneratorConfig(),
            num_requests=8,
            duration=1.0,
        )


@pytest.mark.parametrize("qps", [0.0, -1.0, float("nan"), float("inf")])
def test_poisson_rejects_invalid_offered_load(qps: float) -> None:
    with pytest.raises(ValueError, match="Poisson qps"):
        PoissonRequestIntervalGeneratorConfig(qps=qps)
