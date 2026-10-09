"""Synthetic checks of the stage-admission vLLM comparison tools.

``compare_lanes`` is driven end to end on hand-built vLLM traces and Frontier
sets: ideal lane pairing on both sides, and the base rule's two negative
controls (a MoE admission deadlock, dense lanes one forward apart).
``vllm_burst_driver`` is checked for overlay acceptance and patch parsing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.comparison.stage_admission_pp import compare_lanes
from tests.comparison.stage_admission_pp.vllm_burst_driver import apply_patch, build_overlay
from tests.e2e.stage_admission_matrix import ADMISSION_DEADLOCK, MATRIX_DIR_NAME, SUCCESS


FORWARD = 0.12
ROUNDS = 3


def forwards(num_requests: int, late_lane: bool = False) -> list[tuple[int, int, float, float, int]]:
    """Two-stage forwards: pair ``k`` runs stage 0 in ``[k*F, (k+1)*F)``.

    With ``late_lane`` lane 1 starts one forward after lane 0, as under the
    base rule.
    """
    rows = []
    for index in range(num_requests):
        lane, slot = index % 2, index // 2
        start = (slot + (1 if late_lane and lane == 1 else 0)) * FORWARD
        rows.append((lane, 0, start, start + FORWARD, index))
        rows.append((lane, 1, start + FORWARD, start + 2 * FORWARD, index))
    return rows


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def write_vllm_scenario(run_dir: Path, model: str, late_round: int | None = None) -> None:
    scenario = run_dir / "runs" / model
    requests, boundaries, rounds = [], [], []
    placement = {0: [], 1: []}
    wall_offset, origin = 1.7e9, 100.0
    labels = [("warmup", 0, 4)] + [
        (f"b{burst}-r{r}", r, burst) for burst in compare_lanes.BURSTS for r in range(ROUNDS)
    ]
    for label, round_index, num_requests in labels:
        late = label != "warmup" and round_index == late_round
        for lane, stage, start, end, index in forwards(num_requests, late):
            request_id = f"{label}-q{index}"
            boundaries.append({
                "request_ids": [request_id], "pp_rank": stage, "is_last_rank": stage == 1,
                "forward_start_ts": origin + start,
                "send_start_ts": None if stage else origin + end,
                "timestamp": wall_offset + origin + end,
            })
            if stage == 0:
                placement[lane].append({"kind": "engine_iteration", "engine": lane,
                                        "scheduled_new_req_ids": [request_id]})
        requests.extend(
            {"request_id": f"{label}-q{index}", "burst": label, "round": round_index,
             "index": index, "rank": index % 2, "num_output_tokens": 1}
            for index in range(num_requests)
        )
        rounds.append({"label": label, "wall_minus_monotonic_before": wall_offset,
                       "wall_minus_monotonic_after": wall_offset})
        origin += 10.0
    write_jsonl(scenario / "requests.jsonl", requests)
    write_jsonl(scenario / "pp_boundary.jsonl", boundaries)
    (scenario / "summary.json").write_text(json.dumps({"rounds": rounds}))
    for engine, records in placement.items():
        write_jsonl(scenario / "dp_placement" / f"dp_placement_{engine}.jsonl", records)


def write_frontier_case(set_dir: Path, model: str, burst: int, outcome: str,
                        late_lane: bool = False) -> None:
    case_dir = set_dir / f"G7-{model}-dp2-pp2-n{burst}"
    case_dir.mkdir(parents=True)
    (case_dir / "run.json").write_text(json.dumps({"outcome": outcome}))
    (case_dir / "case.json").write_text(json.dumps({"num_requests": burst}))
    if outcome == SUCCESS:
        write_jsonl(
            case_dir / "metrics" / "run" / "frontier_stage_batch_ledger.jsonl",
            ({"execution_scope": "ATTN_DP_LANE", "replica_local_id": lane, "stage_id": stage,
              "stage_start_ts": start, "stage_end_ts": end, "request_ids": [str(index)]}
             for lane, stage, start, end, index in forwards(burst, late_lane)),
        )


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("FRONTIER_TMP_ROOT", str(tmp_path / "scratch"))
    return tmp_path


def run_compare(workspace: Path, *, fixed_base: bool = False, late_round: int | None = None,
                drop_placement_engine: int | None = None) -> tuple[dict, list[dict]]:
    vllm_run = workspace / "vllm"
    for model in compare_lanes.MODELS:
        write_vllm_scenario(vllm_run, model, late_round if model == "dense" else None)
    if drop_placement_engine is not None:
        (vllm_run / "runs" / "moe" / "dp_placement"
         / f"dp_placement_{drop_placement_engine}.jsonl").unlink()
    frontier = workspace / "scratch" / MATRIX_DIR_NAME
    for burst in compare_lanes.BURSTS:
        for model in compare_lanes.MODELS:
            write_frontier_case(frontier / "after", model, burst, SUCCESS)
            if fixed_base:
                write_frontier_case(frontier / "base", model, burst, SUCCESS)
            elif model == "moe":
                write_frontier_case(frontier / "base", model, burst, ADMISSION_DEADLOCK)
            else:
                write_frontier_case(frontier / "base", model, burst, SUCCESS, late_lane=True)
    output = workspace / "analysis"
    compare_lanes.main(["--vllm-run", str(vllm_run), "--output", str(output)])
    status = json.loads((output / "workflow_gap_status.json").read_text())
    with (output / "workflow_gap_table.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return status, rows


def statuses(rows: list[dict], checks: tuple[str, ...]) -> set[str]:
    return {row["status"] for row in rows if row["check"] in checks}


def test_ideal_after_revision_matches_and_base_controls_hold(workspace) -> None:
    status, rows = run_compare(workspace)

    assert status["status"] == "PASS"
    assert status["mismatches"] == 0
    assert status["negative_control_holds"] is True
    assert statuses(rows, ("V1", "V2", "V3", "V4")) == {"MATCH"}
    assert {(row["model"], row["status"]) for row in rows if row["check"] == "V5"} == {
        ("moe", "MATCH"), ("dense", compare_lanes.INFORMATIONAL)
    }
    assert sorted((row["check"], row["model"], row["status"]) for row in rows
                  if row["check"] in ("N1", "N4")) == [
        ("N1", "moe", "HOLDS"), ("N1", "moe", "HOLDS"),
        ("N4", "dense", "HOLDS"), ("N4", "dense", "HOLDS"),
    ]


def test_fixed_base_loses_the_controls_without_a_mismatch(workspace) -> None:
    status, rows = run_compare(workspace, fixed_base=True)

    assert status["status"] == "PASS"
    assert status["mismatches"] == 0
    assert status["negative_control_holds"] is False
    assert statuses(rows, ("V1", "V2", "V3", "V4")) == {"MATCH"}
    assert statuses(rows, ("N1", "N4")) == {"LOST"}


def test_vllm_round_with_a_late_lane_is_reported(workspace) -> None:
    status, rows = run_compare(workspace, late_round=1)

    mismatched = sorted((row["check"], row["model"], row["burst"], row["round"])
                        for row in rows if row["status"] == "MISMATCH")
    assert mismatched == [
        ("V3", "dense", "16", "1"), ("V3", "dense", "8", "1"),
        ("V4", "dense", "16", "1"), ("V4", "dense", "8", "1"),
    ]
    assert status["status"] == "FAIL"


def test_missing_placement_log_fails_the_comparison(workspace) -> None:
    status, _ = run_compare(workspace, drop_placement_engine=1)

    assert status["vllm_placement_ok"] is False
    assert status["vllm_placement_unseen_requests"] > 0
    assert status["status"] == "FAIL"


def write_tree(root: Path, files: dict[str, str]) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


@pytest.mark.parametrize("extra_change, accepted", [(False, True), (True, False)])
def test_overlay_acceptance_compares_file_sets(tmp_path, extra_change, accepted) -> None:
    # Path order puts vllm/a/c.py first; string order puts vllm/a-b.py first.
    changed = ["vllm/a-b.py", "vllm/a/c.py"]
    write_tree(tmp_path / "site", {"vllm/a-b.py": "old\n", "vllm/a/c.py": "old\n", "vllm/d.py": "same\n"})
    checkout_files = {"vllm/a-b.py": "new\n", "vllm/a/c.py": "new\n", "vllm/d.py": "same\n"}
    if extra_change:
        checkout_files["vllm/d.py"] = "changed\n"
    write_tree(tmp_path / "checkout", checkout_files)
    expected = tmp_path / "expected_changes.txt"
    expected.write_text("".join(f"{name}\n" for name in sorted(changed)))

    report = build_overlay(tmp_path / "site" / "vllm", tmp_path / "checkout",
                           tmp_path / "overlay", expected)

    assert report["accepted"] is accepted
    assert report["unexpected"] == ([] if accepted else ["vllm/d.py"])


def test_apply_patch_keeps_every_section_of_a_repeated_file(tmp_path) -> None:
    write_tree(tmp_path, {"pkg/mod.py": "a = 1\nb = 2\nc = 3\n"})
    patch = tmp_path / "change.patch"
    patch.write_text(
        "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 10\n"
        "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -3,1 +3,1 @@\n-c = 3\n+c = 30\n"
    )

    assert apply_patch(patch, tmp_path) == ["pkg/mod.py"]
    assert (tmp_path / "pkg" / "mod.py").read_text() == "a = 10\nb = 2\nc = 30\n"


def test_apply_patch_reads_a_trimmed_context_line(tmp_path) -> None:
    write_tree(tmp_path, {"pkg/mod.py": "a = 1\n\nb = 2\n"})
    patch = tmp_path / "change.patch"
    patch.write_text("--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,3 +1,3 @@\n a = 1\n\n-b = 2\n+b = 20\n")

    apply_patch(patch, tmp_path)

    assert (tmp_path / "pkg" / "mod.py").read_text() == "a = 1\n\nb = 20\n"


def test_apply_patch_rejects_an_unknown_hunk_line(tmp_path) -> None:
    write_tree(tmp_path, {"pkg/mod.py": "a = 1\n"})
    patch = tmp_path / "change.patch"
    patch.write_text(
        "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -1,1 +1,1 @@\n-a = 1\n"
        "\\ No newline at end of file\n+a = 2\n"
    )

    with pytest.raises(ValueError, match="unexpected hunk line"):
        apply_patch(patch, tmp_path)
