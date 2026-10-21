"""Prepare case-aligned R2 capture inputs without claiming an independent reference.

The describe command is runnable offline. It deliberately does not seal or submit
a G3 package while the independent functional reference decision is unresolved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from tests.integration.step5_pd import canonical_replay, canonical_stage  # noqa: E402


def definition(source_root: Path) -> dict:
    """Reuse the frozen clean engine and R2 collector, not the obsolete short run."""
    settings = json.loads(Path(__file__).with_name("r2_canonical.json").read_text())
    run = canonical_stage.case_definition(json.loads(
        Path(__file__).with_name("canonical_run_h200.json").read_text()), settings["case_id"])
    common = run.pop("canonical_replay")
    run["description"] = (
        "D21 P0 canonical R2 capture, not clean timing or G3 acceptance. "
        "6144/1024 content and native no-fence requests; adjacent6145/1023 is "
        "a boundary control only, retaining the frozen7168 context. "
        "Independent functional reference is not yet selected or implemented.")
    run["canonical_g3"] = {
        "workload": common["workload"],
        "manifest_sha256": canonical_replay.MANIFEST_SHA256,
        "sums_sha256": canonical_replay.SUMS_SHA256,
        "independent_reference": "BLOCKED_ON_REFERENCE_DECISION",
        "pending_read_abort": "REQUIRE_OBSERVED_NATIVE_IN_FLIGHT_READ_ABORT_NOT_DEADLINE_ALONE",
        "pressure": "UNVERIFIED_WITHOUT_OBSERVED_NATIVE_ALLOCATION_PRESSURE",
    }
    run["waves"] = [wave | {"stream": True} for wave in settings["waves"]]
    run["gating_wave"] = settings["gating_wave"]
    run["pd_state_observation"] = {
        "region_size": settings["region_size"], "expected_tp_ranks": run["engine"]["tensor_parallel_size"],
        "source_root": str(source_root), **settings["capture_limits"]}
    run["scrape_timeout_s"] = 10
    run["release_margin_s"] = 5
    validate_definition(run)
    return run


def validate_definition(run: dict) -> None:
    """Reject short-run substitutions and accidental changes to the native case."""
    frozen = canonical_stage.case_definition(json.loads(
        Path(__file__).with_name("canonical_run_h200.json").read_text()), "h800_32")
    if run["engine"] != frozen["engine"] or run["resources"] != frozen["resources"]:
        raise ValueError("Canonical G3 capture must preserve the frozen P0 engine and resources")
    if "role_num_gpu_blocks_override" in run or "canonical_replay" in run:
        raise ValueError("R2 diagnostic waves must not reuse short-run block limits or clean replay")
    waves = {wave["name"]: wave for wave in run["waves"]}
    if len(waves) != len(run["waves"]):
        raise ValueError("Duplicate canonical R2 wave")
    main = waves[run["gating_wave"]]
    if main["prompt_lengths"] != [6144] or main["num_decode_tokens"] != 1024:
        raise ValueError("Gating content request must be6144/1024")
    if main.get("capture_content", True) is not True:
        raise ValueError("Canonical content request cannot be a no-content probe")
    if (waves["concurrent8"]["prompt_lengths"] != [6144] * 8
            or waves["concurrent8"]["num_decode_tokens"] != 1024
            or waves["concurrent8"].get("capture_content") is not False):
        raise ValueError("Canonical concurrent wave requires eight6144/1024 no-fence requests")
    if not waves["cancel6144"].get("cancel_after_first_content"):
        raise ValueError("Missing declared cancellation control")
    abort = waves["abort_read6144"]
    if (type(abort.get("cancel_after_decode_submit_s")) not in (int, float)
            or not 0 < abort["cancel_after_decode_submit_s"] < float("inf")
            or abort.get("capture_content") is not False):
        raise ValueError("Pending READ abort requires a positive diagnostic deadline without content fences")
    lengths = {length for wave in waves.values() for length in wave["prompt_lengths"]}
    for size in (run["engine"]["block_size"], run["pd_state_observation"]["region_size"]):
        if not any(n % size == 0 and {n - 1, n, n + 1} <= lengths for n in lengths):
            raise ValueError("Missing adjacent native region/page boundary controls")
    if any(length + wave["num_decode_tokens"] > run["engine"]["max_model_len"]
           or wave.get("stream") is not True
           for wave in waves.values() for length in wave["prompt_lengths"]):
        raise ValueError("Canonical R2 trace exceeds frozen context or disables streaming")


def prepare_traces(run: dict, stage: Path) -> None:
    """Use original canonical prompt templates and the existing token-ID client."""
    validate_definition(run)
    workload = REPO / run["canonical_g3"]["workload"]
    _, formal, warmup = canonical_replay.canonical_workload(workload)
    shutil.copytree(workload, stage / "canonical_workload")
    for wave in run["waves"]:
        templates = warmup if wave.get("templates") == "warmup" else formal
        path = stage / "trace" / wave["name"] / "request_ids.json"
        rows = json.loads(path.read_text())["rows"]
        if len(rows) > len(templates):
            raise ValueError("Canonical diagnostic wave exceeds its frozen prompt template population")
        for row, template in zip(rows, templates):
            row.update(prompt_id=template["request_id"], template_request_id=template["request_id"],
                       return_token_ids=True)
        path.write_text(json.dumps({"rows": rows}, indent=1))


def describe(source_root: Path) -> dict:
    run = definition(source_root)
    return {"task_id": "D21-CANONICAL-G3", "definition": run,
            "status": "CAPTURE_CALLER_IMPLEMENTED_REFERENCE_BLOCKED", "sealed": False,
            "ready_for_formal": False, "submitted": False,
            "required_before_sealing": [
                "Approve and implement the independent native functional reference and complete parameter/buffer/scales identity comparison",
                "Complete reset/reuse/pressure component and alias conservation, including independent-reference state checks"],
            "parent_budget": {"replica": 4, "gpu_per_replica": 8, "gpu_type": "H800",
                              "charged_group": "codesign", "preemptible": False,
                              "timeout_seconds": run["resources"]["timeout_seconds"]}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(describe(args.source_root), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
