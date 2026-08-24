from pathlib import Path


EVENT_SOURCE = Path(
    "frontier/events/replica_stage_schedule_event.py"
).read_text()


def test_shared_layer_attention_probe_disables_ffn_prediction() -> None:
    """The first attention-only probe must not materialize an FFN workload."""

    assert "include_ffn=False" in EVENT_SOURCE
