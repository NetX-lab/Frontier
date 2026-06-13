#!/usr/bin/env python3
"""Regression tests for request-generator decode-bound workload metadata."""

from __future__ import annotations

from frontier.config.config import BaseRequestGeneratorConfig
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
