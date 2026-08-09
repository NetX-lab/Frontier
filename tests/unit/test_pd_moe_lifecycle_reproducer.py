from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re

import pytest

from tests.performance.sim_walltime_scaling import pd_moe_lifecycle_reproducer
from tests.performance.sim_walltime_scaling.pd_moe_lifecycle_reproducer import (
    summarize_claim_order,
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


def test_summarize_claim_order_reports_first_priority_inversion() -> None:
    summary = summarize_claim_order(
        [
            (1.0, 1, 1),
            (1.0, 3, 1),
            (1.0, 2, 1),
        ]
    )

    assert summary["claim_count"] == 3
    assert summary["priority_inversion_count"] == 1
    assert summary["monotonic"] is False
    assert summary["first_priority"] == [1.0, 1, 1]
    assert summary["last_priority"] == [1.0, 2, 1]
    assert summary["first_inversion"] == {
        "claim_index": 2,
        "previous": [1.0, 3, 1],
        "current": [1.0, 2, 1],
    }
    assert len(summary["priority_sequence_sha256"]) == 64


def test_summarize_claim_order_reports_duplicate_priority_as_violation() -> None:
    summary = summarize_claim_order(
        [
            (1.0, 1, 1),
            (1.0, 1, 1),
        ]
    )

    assert summary["claim_count"] == 2
    assert summary["priority_inversion_count"] == 1
    assert summary["monotonic"] is False
    assert summary["first_inversion"] == {
        "claim_index": 1,
        "previous": [1.0, 1, 1],
        "current": [1.0, 1, 1],
    }


def test_internal_parallel_pdd_factory_is_explicitly_available() -> None:
    factory = getattr(
        pd_moe_lifecycle_reproducer,
        "create_parallel_pdd_config_with_release_policy_suppressed",
        None,
    )

    assert callable(factory)


def test_lifecycle_reproducer_parser_requires_explicit_internal_policy_flag() -> None:
    parser = pd_moe_lifecycle_reproducer._build_parser()
    args = parser.parse_args(
        [
            "--case-json",
            "case.json",
            "--state-json",
            "state.json",
            "--suppress-pdd-release-policy-for-internal-test",
        ]
    )

    assert args.suppress_pdd_release_policy_for_internal_test is True


def _write_case(path: Path, *, mode: str) -> None:
    path.write_text(
        json.dumps(
            {
                "model": "moe",
                "total_gpus": 32,
                "mode": mode,
                "attempt_index": 0,
                "shape": {
                    "attn_tp": 4,
                    "attn_dp": 2,
                    "moe_tp": 1,
                    "moe_ep": 8,
                    "pp": 2,
                },
                "num_requests": 1,
                "qps": 8.0,
                "prefill_tokens": 16,
                "decode_tokens": 2,
            }
        ),
        encoding="utf-8",
    )


def test_lifecycle_reproducer_rejects_parallel_case_without_internal_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_path = tmp_path / "parallel.case.json"
    _write_case(case_path, mode="parallel")
    monkeypatch.setattr(
        pd_moe_lifecycle_reproducer,
        "run_reproducer",
        lambda *args, **kwargs: pytest.fail("run_reproducer must not be called"),
    )

    with pytest.raises(SystemExit) as exc_info:
        pd_moe_lifecycle_reproducer.main(
            [
                "--case-json",
                str(case_path),
                "--state-json",
                str(tmp_path / "state.json"),
            ]
        )

    assert exc_info.value.code == 2
    assert (
        "parallel CaseSpec requires "
        "--suppress-pdd-release-policy-for-internal-test"
        in capsys.readouterr().err
    )


def test_lifecycle_reproducer_rejects_internal_flag_for_sequential_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    case_path = tmp_path / "sequential.case.json"
    _write_case(case_path, mode="sequential")
    monkeypatch.setattr(
        pd_moe_lifecycle_reproducer,
        "run_reproducer",
        lambda *args, **kwargs: pytest.fail("run_reproducer must not be called"),
    )

    with pytest.raises(SystemExit) as exc_info:
        pd_moe_lifecycle_reproducer.main(
            [
                "--case-json",
                str(case_path),
                "--state-json",
                str(tmp_path / "state.json"),
                "--suppress-pdd-release-policy-for-internal-test",
            ]
        )

    assert exc_info.value.code == 2
    assert (
        "--suppress-pdd-release-policy-for-internal-test may only be used "
        "with a parallel CaseSpec"
        in capsys.readouterr().err
    )


def test_internal_parallel_pdd_factory_rejects_non_main_thread() -> None:
    factory = getattr(
        pd_moe_lifecycle_reproducer,
        "create_parallel_pdd_config_with_release_policy_suppressed",
        None,
    )
    assert callable(factory)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(factory, [])
        with pytest.raises(RuntimeError, match="main thread"):
            future.result()


def test_validate_lifecycle_state_reports_priority_inversion() -> None:
    state = _healthy_state()
    state["claim_order"] = {
        "claim_count": 3,
        "priority_inversion_count": 1,
        "monotonic": False,
        "first_inversion": {
            "claim_index": 2,
            "previous": [1.0, 3, 1],
            "current": [1.0, 2, 1],
        },
    }

    assert validate_lifecycle_state(state) == [
        "parallel claim order contains 1 priority inversion(s): "
        "first={'claim_index': 2, 'previous': [1.0, 3, 1], "
        "'current': [1.0, 2, 1]}"
    ]


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
