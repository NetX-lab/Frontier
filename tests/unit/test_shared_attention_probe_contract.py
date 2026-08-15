from pathlib import Path


EVENT_SOURCE = Path(
    "frontier/events/replica_stage_schedule_event.py"
).read_text()


def test_shared_layer_attention_probe_disables_moe_prediction() -> None:
    """The first attention-only probe must not materialize a MoE workload."""

    assert "include_moe=False" in EVENT_SOURCE

