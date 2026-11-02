"""Binding and lookup of the diagnostic V4 step-cost table (H4a)."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.config import RandomForrestExecutionTimePredictorConfig, ReplicaConfig
from frontier.execution_time_predictor.dsv4_step_cost_input import (
    FILE_NAME,
    QUALIFICATION,
    SCHEMA,
    TOPOLOGY_FIELDS,
    load_step_cost_table,
)


@pytest.fixture
def replica_config(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    return ReplicaConfig(
        model_name="deepseek-ai/DeepSeek-V4-Flash", device="h800",
        network_device="h800_dgx", attn_tensor_parallel_size=8, attn_dp=2,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=16,
    )


def _decode(prefill_requests, prefill_tokens, decode_min, decode_max, time_ms):
    return {"num_prefill_requests": prefill_requests, "num_prefill_tokens": prefill_tokens,
            "num_decode_requests": {"min": decode_min, "max": decode_max}, "time_ms": time_ms}


@pytest.fixture
def diagnostic(replica_config, tmp_path):
    """A predictor config that selects a table in a diagnostic directory through the attention input."""
    table = {
        "schema_version": SCHEMA, "qualification": QUALIFICATION, "formal_gate_admission": False,
        "model_name": replica_config.model_config.get_name(), "device": replica_config.device,
        "topology": {name: getattr(replica_config, name) for name in TOPOLOGY_FIELDS},
        "roles": {
            "PREFILL": [_decode(1, 6144, 0, 0, 287.0)],
            "DECODE": [_decode(0, 0, 1, 128, 19.0), _decode(1, 1, 0, 0, 21.0), _decode(1, 1, 1, 128, 30.0)],
        },
    }
    (tmp_path / FILE_NAME).write_text(json.dumps(table))
    return RandomForrestExecutionTimePredictorConfig(atten_input_file=str(tmp_path / "attention.csv"))


def _lane(*requests):
    """One lane batch from (prompt tokens, scheduled tokens, computed tokens) rows."""
    return SimpleNamespace(
        requests=[SimpleNamespace(num_prefill_tokens=prompt) for prompt, _, _ in requests],
        num_tokens=[tokens for _, tokens, _ in requests],
        num_context_tokens=[done for _, _, done in requests],
    )


def test_table_binds_only_when_a_diagnostic_driver_selects_it(replica_config, diagnostic):
    # ADR 0002: a measured whole-step table is never in the default profiling path.
    assert load_step_cost_table(RandomForrestExecutionTimePredictorConfig(), replica_config) is None
    assert load_step_cost_table(replace(diagnostic, enable_dummy_mode=True), replica_config) is None
    assert load_step_cost_table(diagnostic, replica_config).path == str(Path(diagnostic.atten_input_file).parent / FILE_NAME)
    with pytest.raises(DeepSeekV4CoverageError, match="topology"):
        load_step_cost_table(diagnostic, replace(replica_config, attn_dp=4, moe_expert_parallel_size=32))


def test_lane_step_is_keyed_by_prefill_and_decode_requests(replica_config, diagnostic):
    table = load_step_cost_table(diagnostic, replica_config)
    assert table.step_time_ms("PREFILL", _lane((6144, 6144, 0))) == 287.0
    # The first decode-lane step after the KV transfer still consumes one prompt token.
    first = (6144, 1, 6143)
    decoding = (6144, 1, 6200)
    assert table.step_time_ms("DECODE", _lane(first)) == 21.0
    assert table.step_time_ms("DECODE", _lane(decoding, decoding)) == 19.0
    assert table.step_time_ms("DECODE", _lane(first, decoding)) == 30.0
    with pytest.raises(DeepSeekV4CoverageError, match="no DECODE step with 2 prefill requests of 2 tokens"):
        table.step_time_ms("DECODE", _lane(first, first))
    with pytest.raises(DeepSeekV4CoverageError, match="no MONOLITHIC step"):
        table.step_time_ms("MONOLITHIC", _lane(decoding))
