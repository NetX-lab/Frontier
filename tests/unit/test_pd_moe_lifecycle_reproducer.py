from copy import deepcopy
import re

import pytest

from tests.performance.sim_walltime_scaling.pd_moe_lifecycle_reproducer import (
    validate_lifecycle_state,
)


def _healthy_state() -> dict:
    return {
        "run_error": None,
        "expected_requests": 1,
        "completed_requests": 1,
        "global_scheduler_is_empty": True,
        "requests": [
            {
                "id": 7,
                "completed": True,
                "completed_layer_count": 0,
                "active_memberships": [],
            }
        ],
        "event_queues": {
            "sequential_event_queue_length": 0,
            "parallel_cluster_runtime": {},
        },
        "inter_cluster_stats": {},
        "cluster_schedulers": {
            "PREFILL": {
                "is_empty": True,
                "debug_state": {
                    "request_queue": {"count": 0},
                    "af_queue": {"status": "not_applicable"},
                    "m2n_waiting_groups": {"status": "not_applicable"},
                    "m2n_ready_groups": {"status": "not_applicable"},
                    "raw_batch_waiting_map": {"count": 0},
                },
                "replicas": {
                    "(0, 0)": {
                        "active_batch_request_counts": {},
                        "num_running_batches": 0,
                        "allocation_map": {"count": 0},
                        "request_queue": {"count": 0},
                        "waiting_requests": {"count": 0},
                        "running_requests": {"count": 0},
                        "stage_schedulers": {
                            "0": {
                                "is_busy": False,
                                "is_empty": True,
                                "batch_queue": {"count": 0},
                            }
                        },
                    }
                },
            }
        },
    }


def test_validate_lifecycle_state_accepts_fully_clean_terminal_state() -> None:
    assert validate_lifecycle_state(_healthy_state()) == []


@pytest.mark.parametrize(
    ("mutate", "error_match"),
    [
        (
            lambda state: state.update(
                completed_requests=0,
                global_scheduler_is_empty=False,
            ),
            "completed_requests|global scheduler",
        ),
        (
            lambda state: state["requests"][0].update(
                completed=False,
                completed_layer_count=94,
                active_memberships=["DECODE/(1, 0)"],
            ),
            "request 7|active",
        ),
        (
            lambda state: state["cluster_schedulers"]["PREFILL"]["replicas"][
                "(0, 0)"
            ].update(
                active_batch_request_counts={"7": 1},
                num_running_batches=1,
                allocation_map={"count": 1},
            ),
            "active batch|running batch|allocation",
        ),
        (
            lambda state: state["cluster_schedulers"]["PREFILL"]["replicas"][
                "(0, 0)"
            ]["stage_schedulers"]["0"].update(
                is_busy=True,
                is_empty=False,
                batch_queue={"count": 1},
            ),
            "stage|busy|queue",
        ),
        (
            lambda state: state["event_queues"].update(
                sequential_event_queue_length=1,
                parallel_cluster_runtime={
                    "DECODE": {
                        "queue_size": 1,
                        "is_processing_event": True,
                        "is_running": True,
                    }
                },
            ),
            "event queue|DECODE|processing|running",
        ),
        (
            lambda state: state.update(
                inter_cluster_stats={
                    "events_sent": 3,
                    "events_delivered": 2,
                    "queue_size": 1,
                    "queue_full_count": 1,
                    "buffer_sizes": {"DECODE": 1},
                    "total_buffered_events": 1,
                }
            ),
            "inter-cluster|sent|delivered|buffer|queue_full",
        ),
    ],
)
def test_validate_lifecycle_state_reports_retained_state(
    mutate,
    error_match: str,
) -> None:
    state = deepcopy(_healthy_state())
    mutate(state)

    errors = validate_lifecycle_state(state)

    assert errors
    assert any(
        re.search(error_match, error, flags=re.IGNORECASE)
        for error in errors
    )
