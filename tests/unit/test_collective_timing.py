from types import SimpleNamespace

import pytest

from frontier.scheduler.utils.collective_timing import (
    attention_delay_seconds,
    prepare_decode_final_timing,
    prepare_prefill_final_timing,
    select_active_batch,
    validate_decode_layer_advance,
)


def test_select_active_batch_prefers_first_non_idle_batch():
    idle = SimpleNamespace(is_idle=True)
    active = SimpleNamespace(is_idle=False)
    assert select_active_batch({0: idle, 1: active}) is active
    assert select_active_batch({0: idle}) is None


def test_attention_delay_converts_milliseconds():
    execution_time = SimpleNamespace(get_single_layer_attention_scope_time=lambda: 2.5)
    assert attention_delay_seconds(execution_time) == pytest.approx(0.0025)


def test_prefill_final_timing_preserves_component_and_pipeline_units():
    execution_time = SimpleNamespace(pipeline_time=4.0, total_time=10.0, model_time=8.0)
    timing = prepare_prefill_final_timing(execution_time, [3.0, 5.0], 1.0, 0.0)
    assert timing.pipeline_time == pytest.approx(0.004)
    assert timing.cpu_overhead == pytest.approx(2.0)
    assert timing.explicit_model_time == pytest.approx(0.008)
    assert timing.total_time == pytest.approx(2.004)
    assert timing.completion_time == pytest.approx(3.004)
    assert timing.actual_execution_time == pytest.approx(3.004)


def test_prefill_final_timing_rejects_negative_cpu_overhead():
    execution_time = SimpleNamespace(pipeline_time=4.0, total_time=7.0, model_time=8.0)
    with pytest.raises(ValueError, match="CPU overhead"):
        prepare_prefill_final_timing(execution_time, [1.0], 1.0, 0.0)


def test_decode_final_timing_matches_existing_decomposition():
    execution_time = SimpleNamespace(
        pipeline_time=4.0,
        total_time=10.0,
        model_time=8.0,
        decode_draft_proposer_time=3.0,
        mtp_terminal_overshoot_time=2.0,
    )
    timing = prepare_decode_final_timing(execution_time)
    assert timing.pipeline_time == pytest.approx(0.004)
    assert timing.cpu_overhead == pytest.approx(2.0)
    assert timing.draft_proposer_time == pytest.approx(0.003)
    assert timing.mtp_terminal_overshoot_time == pytest.approx(0.002)
    assert timing.total_time == pytest.approx(2.007)


def test_decode_layer_advance_rejects_completed_request():
    request = SimpleNamespace(
        id=4,
        completed=True,
        completed_layer_count=8,
        current_decode_token_index=2,
    )
    with pytest.raises(ValueError, match="layer counter"):
        validate_decode_layer_advance([request], 8)
