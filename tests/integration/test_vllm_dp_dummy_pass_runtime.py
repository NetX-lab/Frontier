"""vLLM's attention-DP dummy pass in the engine loop of a MoE replica.

vLLM's `DPEngineCoreProc.run_busy_loop` runs `execute_dummy_batch` on an engine
whose step executed no batch while its DP wave runs, so the MoE collectives of
the other engines always find a partner. These cases run the simulator through
the DP placement harness and check the engine-loop properties that only a run
can show: when a dummy pass is issued, that its engine is held until every
stage part ends, that dummy forwards pair with each other, and that the wave
ends once no lane has an unfinished request.
"""

from __future__ import annotations

from collections import defaultdict

import pytest

from tests.integration.test_vllm_dp_placement_runtime import (
    _assert_run_conserves_work,
    _run_child,
)


def _records(run: dict, kind: str) -> list[dict]:
    return [record for record in run["engine_loop"] if record["kind"] == kind]


def _output_effects(rows: dict) -> dict:
    return {
        request_id: (row["completed"], row["completion_recorded"], row["tool_wait"])
        for request_id, row in rows.items()
    }


@pytest.mark.parametrize("case", ["moe_dp2_pp2_online", "moe_dp2_pp4"])
def test_a_lane_runs_a_dummy_pass_after_a_pass_that_left_the_pipeline_unfilled(
    tmp_path, case
):
    evidence = _run_child(tmp_path, case)

    for run in evidence.values():
        passes = _records(run, "dummy_pass")
        # A pass that queued its batch without filling the pipeline blocks on
        # the oldest batch, returns model_executed=False, and runs a dummy.
        assert any(
            record["lane_has_requests"] and record["after_empty_pass"]
            for record in passes
        ), passes
        _assert_run_conserves_work(run)


@pytest.mark.parametrize("case", ["moe_dp2_pp2_online", "moe_dp2_pp4_release"])
def test_a_lane_without_requests_runs_dummy_passes_while_a_sibling_has_work(
    tmp_path, case
):
    evidence = _run_child(tmp_path, case)

    for run in evidence.values():
        passes = _records(run, "dummy_pass")
        assert any(not record["lane_has_requests"] for record in passes), passes
        _assert_run_conserves_work(run)


@pytest.mark.parametrize("case", ["moe_dp2_pp2_online", "moe_dp2_pp4"])
def test_a_lane_is_held_until_every_stage_part_of_its_dummy_pass_ends(
    tmp_path, case
):
    num_stages = {"moe_dp2_pp2_online": 2, "moe_dp2_pp4": 4}[case]
    evidence = _run_child(tmp_path, case)

    for run in evidence.values():
        held = _records(run, "held")
        assert held, "no engine iteration was attempted during a dummy pass"
        assert all(record["returned"] == 0 for record in held), held
        ends = _records(run, "dummy_end")
        num_passes = len(_records(run, "dummy_pass"))
        assert len(ends) == num_stages * num_passes
        assert sum(record["pass_ended"] for record in ends) == num_passes


@pytest.mark.parametrize("case", ["moe_dp2_pp2_online", "moe_dp2_pp4", "moe_dp2_pp4_release"])
def test_dummy_forwards_pair_with_each_other_and_the_wave_ends_when_drained(
    tmp_path, case
):
    evidence = _run_child(tmp_path, case)

    for run in evidence.values():
        assert run["all_dummy_waves"], "no MoE wave paired only dummy forwards"
        assert set(run["all_dummy_waves"]) == {run["num_lanes"]}
        # Every forward of one lane met one forward of each sibling, and no
        # dummy forward outlived the last request.
        assert len(set(run["lane_forward_counts"])) == 1, run["lane_forward_counts"]
        assert run["dummy_forwards_in_flight"] == [0] * run["num_lanes"]
        _assert_run_conserves_work(run)


@pytest.mark.parametrize(
    ("case", "a_batch_ends_during_the_sync"),
    [("moe_dp2_pp2_online", False), ("moe_dp2_pp4_kv_pressure", True)],
)
def test_every_lane_leaves_the_dp_finish_sync_together_after_32_iterations(
    tmp_path, case, a_batch_ends_during_the_sync
):
    """`DPEngineCoreProc._has_global_unfinished_reqs` all-reduces over the DP
    group after every 32nd busy-loop iteration of a wave. Each iteration runs
    one forward, so every lane has run as many forwards when the last lane
    joins and the sync releases them all. A batch that ends while its engine
    waits keeps its output until an iteration after the release pops it."""

    evidence = _run_child(tmp_path, case)

    for run in evidence.values():
        releases = run["dp_sync_releases"]
        assert releases, "no wave ran 32 iterations"
        for release in releases:
            assert sorted(release["lanes"]) == list(range(run["num_lanes"]))
            assert len(set(release["forward_counts"])) == 1, release
        # Once drained, the last sync found no unfinished request and ended the
        # wave on every lane.
        assert run["dp_wave_states"] == [
            {"running": False, "steps": 0, "waits": False, "unpopped": 0}
        ] * run["num_lanes"]
        _assert_run_conserves_work(run)
    if a_batch_ends_during_the_sync:
        assert any(run["held_outputs"] for run in evidence.values())


@pytest.mark.parametrize(
    "case", ["moe_dp2_pp4_kv_pressure", "moe_dp2_pp4_kv_pressure_thinking"]
)
def test_an_output_held_at_the_dp_sync_is_applied_when_its_engine_pops_it(
    tmp_path, case
):
    """vLLM applies a batch's output (`Scheduler.update_from_output`) only when
    an iteration pops it from the batch queue, and that iteration schedules
    before it pops. An output that ends while its host waits in the DP finish
    sync therefore leaves its requests, their KV blocks, the completion
    metrics and any tool wait untouched until the first iteration after the
    release, whose schedule call still sees that occupancy."""

    evidence = _run_child(tmp_path, case)

    held_transitions = []
    for run in evidence.values():
        applied = defaultdict(list)
        for output in run["applied_outputs"]:
            applied[output["batch"]].append(output)
        assert all(len(outputs) == 1 for outputs in applied.values()), applied
        for held in run["held_outputs"]:
            assert held["rows_after_hold"] == held["rows"], held
            (output,) = applied[held["batch"]]
            assert output["applied_at"] > held["ended_at"], (held, output)
            # The schedule call before the pop may preempt or readmit a row,
            # but no row's output effect happens before the pop.
            assert _output_effects(output["before"]) == _output_effects(
                held["rows"]
            ), (held, output)
            # The lane's first schedule call after the hold still sees the
            # held rows' KV blocks.
            first_schedule = next(
                schedule
                for schedule in run["iteration_schedules"][held["schedules_before"]:]
                if schedule["lane"] == held["lane"]
            )
            kv_holders = {
                int(request_id)
                for request_id, row in held["rows"].items()
                if row["allocated"] is not None
            }
            assert first_schedule["held"] >= 1, first_schedule
            assert kv_holders <= set(first_schedule["kv_holders"]), (
                first_schedule,
                held,
            )
            # The iteration that pops the output schedules first.
            pop_schedule = run["iteration_schedules"][output["schedules_before"] - 1]
            assert pop_schedule["lane"] == held["lane"], (pop_schedule, held)
            assert pop_schedule["time"] == output["applied_at"], (pop_schedule, output)
            assert pop_schedule["held"] >= 1, pop_schedule
            held_transitions.append(output)
        _assert_run_conserves_work(run)

    assert held_transitions, "no output ended while its engine waited at the DP sync"
    if case.endswith("_thinking"):
        # A round that stops in a held output starts its tool wait at the pop.
        assert any(
            output["thinking_requeues"]
            and any(row["tool_wait"] for row in output["after"].values())
            for output in held_transitions
        ), held_transitions
    else:
        assert any(
            row["completion_recorded"]
            for output in held_transitions
            for row in output["after"].values()
        ), held_transitions


def test_an_attention_dp_one_replica_runs_no_dummy_pass(tmp_path):
    evidence = _run_child(tmp_path, "moe_dp1_pp3")

    for run in evidence.values():
        assert _records(run, "dummy_pass") == []
        assert run["all_dummy_waves"] == []
        _assert_run_conserves_work(run)
