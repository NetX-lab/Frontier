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

import pytest

from tests.integration.test_vllm_dp_placement_runtime import (
    _assert_run_conserves_work,
    _run_child,
)


def _records(run: dict, kind: str) -> list[dict]:
    return [record for record in run["engine_loop"] if record["kind"] == kind]


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


def test_an_attention_dp_one_replica_runs_no_dummy_pass(tmp_path):
    evidence = _run_child(tmp_path, "moe_dp1_pp3")

    for run in evidence.values():
        assert _records(run, "dummy_pass") == []
        assert run["all_dummy_waves"] == []
        _assert_run_conserves_work(run)
