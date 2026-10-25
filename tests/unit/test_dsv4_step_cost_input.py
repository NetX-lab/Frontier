"""Binding and lookup of the diagnostic V4 step-cost table (H4a)."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.config import RandomForrestExecutionTimePredictorConfig, ReplicaConfig
from frontier.execution_time_predictor.dsv4_step_cost_input import load_step_cost_table


@pytest.fixture
def replica_config(monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    return ReplicaConfig(
        model_name="deepseek-ai/DeepSeek-V4-Flash", device="h800",
        network_device="h800_dgx", attn_tensor_parallel_size=8, attn_dp=2,
        moe_tensor_parallel_size=1, moe_expert_parallel_size=16,
    )


def _lane(*requests):
    """One lane batch from (prompt tokens, scheduled tokens, computed tokens) rows."""
    return SimpleNamespace(
        requests=[SimpleNamespace(num_prefill_tokens=prompt) for prompt, _, _ in requests],
        num_tokens=[tokens for _, tokens, _ in requests],
        num_context_tokens=[done for _, _, done in requests],
    )


def test_table_binds_only_measured_v4_steps_of_its_topology(replica_config, tmp_path):
    measured = RandomForrestExecutionTimePredictorConfig()
    table = load_step_cost_table(measured, replica_config)
    assert table.path.endswith("h800/deepseek-ai/DeepSeek-V4-Flash/dsv4_step_cost.json")
    dummy = RandomForrestExecutionTimePredictorConfig(enable_dummy_mode=True)
    assert load_step_cost_table(dummy, replica_config) is None
    elsewhere = RandomForrestExecutionTimePredictorConfig(atten_input_file=str(tmp_path / "attention.csv"))
    assert load_step_cost_table(elsewhere, replica_config) is None
    with pytest.raises(DeepSeekV4CoverageError, match="topology"):
        load_step_cost_table(measured, replace(replica_config, attn_dp=4, moe_expert_parallel_size=32))


def test_lane_step_is_keyed_by_prefill_and_decode_requests(replica_config):
    table = load_step_cost_table(RandomForrestExecutionTimePredictorConfig(), replica_config)
    assert table.step_time_ms("PREFILL", _lane((6144, 6144, 0))) == pytest.approx(287.35362365841866)
    # The first decode-lane step after the KV transfer still consumes one prompt token.
    first = (6144, 1, 6143)
    decoding = (6144, 1, 6200)
    assert table.step_time_ms("DECODE", _lane(first)) == pytest.approx(21.100847534179685)
    assert table.step_time_ms("DECODE", _lane(decoding, decoding)) == pytest.approx(19.2992880859375)
    assert table.step_time_ms("DECODE", _lane(first, decoding)) == pytest.approx(30.146481201171873)
    with pytest.raises(DeepSeekV4CoverageError, match="no DECODE step with 2 prefill requests of 2 tokens"):
        table.step_time_ms("DECODE", _lane(first, first))
    with pytest.raises(DeepSeekV4CoverageError, match="no MONOLITHIC step"):
        table.step_time_ms("MONOLITHIC", _lane(decoding))
