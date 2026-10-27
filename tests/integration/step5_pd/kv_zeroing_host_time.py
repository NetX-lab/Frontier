"""Write the Step-5 KV side-storage reset host-time profile from a native host_probe run.

A native Step-5 engine whose step resets the side storage of new blocks first waits, in the
first reset hook's `.tolist()`, for the previous step's device work. From that moment the
device idles until the host has run the remaining resets, the input preparation and the DP
padding all-reduce, and launched the forward. Frontier composes the DP forward start as
`max_i(a_i + h_i)` over the DP engines, so `h_i` must exclude the time engine `i` waited in
the all-reduce for the other engines.

The host-step probe records, per worker process, one line per model-runner step. The k-th
DP all-reduce call of every engine at TP rank `r` is the same collective, so the call
ordinal joins the engines. For collective k at rank r, the engine that arrived last waited
least; its duration is the collective's own cost, and every other engine's duration above
it is its wait for the others (`excess`). For a resetting step of engine i:

    h_i = max over TP ranks r of (launch_r - excess_r - first_sync_r)

because the forward starts when the engine's slowest TP rank launches. The profile holds
the mean of `h_i` over the resetting steps of the formal measurement windows, per role.

The report adds checks that settle whether `first_sync` is the device idle point: the share
of steps whose previous step was still running at the first hook, the device idle gap from
the CUDA events (`gap_ms - excess`), the hook count per step, the TP-rank spread, values by
request count, and the client TTFT/TPOT shift against a clean run (the probe perturbs native
timing; a shift beyond the clean repeat range marks the profile as perturbed).

Usage: kv_zeroing_host_time.py --case h200_32 --stage-dir <stage> --clean-summary <json>
       --job-id <rjob> --min-samples 100 --output <profile.json> --report <report.json>
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from frontier.config import ReplicaConfig  # noqa: E402
from frontier.execution_time_predictor.kv_block_zeroing_host_time import SCHEMA  # noqa: E402
from tests.integration.step5_pd.frontier_workload import CASES, DEVICES, MODEL  # noqa: E402

ROLES = ("prefill", "decode")
PROBE = "host_probe_bootstrap/host_step_probe.py"


def read_processes(nodes: Path) -> dict[tuple[str, int, int], list[dict]]:
    """Map (role, dp_rank, tp_rank) to the probe files of its worker processes."""
    processes = defaultdict(list)
    for path in sorted(nodes.glob("*/host_probe/*.jsonl.gz")):
        with gzip.open(path, "rt") as stream:
            header, *records = (json.loads(line) for line in stream)
        if header["kind"] != "header":
            raise ValueError(f"Probe file does not start with its header: {path}")
        processes[header["role"], header["dp_rank"], header["tp_rank"]].append(
            {"path": str(path), "header": header, "records": records})
    return processes


def formal_windows(summary: dict) -> list[tuple[float, float]]:
    """Wall-clock windows from each formal phase's first submission to its last output."""
    windows = []
    for repetition in summary["repetitions"]:
        formal, metrics = repetition["phases"]["formal"], repetition["metrics"]
        offset = formal["origin_wall"] - formal["origin_monotonic"]
        windows.append((metrics["first_submission_monotonic"] + offset, metrics["last_output_monotonic"] + offset))
    return windows


def role_processes(processes: dict, role: str, dp_size: int, tp_size: int) -> dict[tuple[int, int], dict]:
    """Exactly one complete, error-free process per engine and TP rank, or the role is unusable."""
    expected = {(dp, tp) for dp in range(dp_size) for tp in range(tp_size)}
    found = {(dp, tp): files for (name, dp, tp), files in processes.items() if name == role}
    if set(found) != expected:
        raise ValueError(f"{role}: probe processes {sorted(found)} differ from DP{dp_size} x TP{tp_size}")
    restarted = sorted(key for key, files in found.items() if len(files) != 1)
    if restarted:
        raise ValueError(f"{role}: more than one process generation at (dp, tp) {restarted}")
    failed = [files[0]["path"] for files in found.values()
              if any(record["kind"] == "error" for record in files[0]["records"])]
    if failed:
        raise ValueError(f"{role}: the probe disabled itself in {failed}")
    for (dp, tp), [process] in found.items():
        header = process["header"]
        if (header["dp_size"], header["tp_size"]) != (dp_size, tp_size):
            raise ValueError(f"{role}: {process['path']} reports DP{header['dp_size']} x TP{header['tp_size']}")
    return {key: files[0] for key, files in found.items()}


def engine_steps(processes: dict[tuple[int, int], dict], dp_size: int, tp_size: int,
                 windows: list[tuple[float, float]]) -> tuple[list[dict], dict]:
    """Join the engines by DP all-reduce ordinal and derive each resetting step's host time."""
    by_ordinal = {key: {record["sync"]["ordinal"]: record for record in process["records"]
                        if record["kind"] == "step" and "sync" in record}
                  for key, process in processes.items()}
    by_sequence = {key: {record["seq"]: record for record in process["records"] if record["kind"] == "step"}
                   for key, process in processes.items()}
    ordinals = set.intersection(*(set(records) for records in by_ordinal.values()))
    every = set.union(*(set(records) for records in by_ordinal.values()))
    steps = []
    for ordinal in sorted(ordinals):
        records = {key: by_ordinal[key][ordinal] for key in by_ordinal}
        excess = {}
        for tp in range(tp_size):
            durations = {dp: records[dp, tp]["sync"]["exit"] - records[dp, tp]["sync"]["enter"]
                         for dp in range(dp_size)}
            own = min(durations.values())
            excess.update({(dp, tp): duration - own for dp, duration in durations.items()})
        for dp in range(dp_size):
            ranks = [records[dp, tp] for tp in range(tp_size)]
            batch = {(record["dummy"], record["reqs"], record["tokens"]) for record in ranks}
            if len(batch) != 1:
                raise ValueError(f"DP engine {dp} TP ranks disagree on the batch at all-reduce {ordinal}: "
                                 f"{sorted(batch)}; the ordinal join is broken")
            dummy, reqs, tokens = batch.pop()
            wall = ranks[0]["wall_ns"] / 1e9
            if dummy or not any(start <= wall <= end for start, end in windows):
                continue
            resetting = all(record.get("zero", {}).get("hooks", 0) > 0 and "first_sync" in record["zero"]
                            and "launch" in record for record in ranks)
            step = {"dp": dp, "ordinal": ordinal, "reqs": reqs, "tokens": tokens, "resetting": resetting,
                    "wait_ms": max(excess[dp, tp] for tp in range(tp_size)) / 1e6}
            gaps = []
            for tp, record in enumerate(ranks):
                previous = by_sequence[dp, tp].get(record["seq"] - 1)
                if "gap_ms" in record and previous is not None and "span_ms" in previous:
                    gaps.append(record["gap_ms"] - excess[dp, tp] / 1e6)
            if len(gaps) == tp_size:
                step["device_gap_ms"] = max(gaps)
            if resetting:
                local = [(record["launch"] - excess[dp, tp] - record["zero"]["first_sync"]) / 1e6
                         for tp, record in enumerate(ranks)]
                step.update(host_ms=max(local), rank0_host_ms=local[0], tp_spread_ms=max(local) - min(local),
                            hooks=sorted({record["zero"]["hooks"] for record in ranks}),
                            prior_pending=all(record["zero"]["prior_pending"] for record in ranks))
            steps.append(step)
    join = {"joined_all_reduces": len(ordinals), "unjoined_all_reduces": len(every - ordinals)}
    return steps, join


def distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    quartiles = statistics.quantiles(values, n=4) if len(values) > 1 else values * 3
    return {"n": len(values), "mean": statistics.fmean(values), "p25": quartiles[0],
            "median": quartiles[1], "p75": quartiles[2], "min": min(values), "max": max(values)}


def role_report(steps: list[dict]) -> dict:
    resetting = [step for step in steps if step["resetting"]]
    plain = [step for step in steps if not step["resetting"]]
    by_requests = defaultdict(list)
    for step in resetting:
        by_requests[step["reqs"]].append(step["host_ms"])
    paired = [step for step in resetting if "device_gap_ms" in step]
    return {
        "host_ms": distribution([step["host_ms"] for step in resetting]),
        "rank0_host_ms": distribution([step["rank0_host_ms"] for step in resetting]),
        "tp_spread_ms": distribution([step["tp_spread_ms"] for step in resetting]),
        "waited_share": (sum(step["prior_pending"] for step in resetting) / len(resetting)) if resetting else None,
        "host_ms_when_waited": distribution([step["host_ms"] for step in resetting if step["prior_pending"]]),
        "host_ms_when_idle": distribution([step["host_ms"] for step in resetting if not step["prior_pending"]]),
        "device_gap_check": {
            "host_ms": distribution([step["host_ms"] for step in paired]),
            "device_gap_ms": distribution([step["device_gap_ms"] for step in paired])},
        "non_resetting_device_gap_ms": distribution([step["device_gap_ms"] for step in plain
                                                     if "device_gap_ms" in step]),
        "dp_wait_ms": distribution([step["wait_ms"] for step in resetting]),
        "hooks_per_step": {str(hooks): count for hooks, count in sorted(Counter(
            tuple(step["hooks"]) for step in resetting).items())},
        "host_ms_by_num_requests": {str(reqs): distribution(values) for reqs, values in sorted(by_requests.items())},
        "steps": {"resetting": len(resetting), "non_resetting": len(plain)},
    }


def perturbation(probe: dict, clean: dict) -> dict:
    """The probe run passes when each client mean stays within the clean run's repeat range."""
    checks = {}
    for metric in ("ttft_ms", "tpot_ms"):
        shift = probe["repeat_metric_means"][metric] - clean["repeat_metric_means"][metric]
        spread = clean["repeat_variation"][metric]["max"] - clean["repeat_variation"][metric]["min"]
        checks[metric] = {"probe_mean": probe["repeat_metric_means"][metric],
                          "clean_mean": clean["repeat_metric_means"][metric],
                          "shift": shift, "clean_repeat_range": spread, "within": abs(shift) <= spread}
    return {"status": "PASS" if all(check["within"] for check in checks.values()) else "PERTURBED",
            "metrics": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--stage-dir", type=Path, required=True,
                        help="host_probe stage with config.json and the copied evidence/")
    parser.add_argument("--clean-summary", type=Path, required=True,
                        help="canonical_summary.json of a clean workload run of the same case")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--min-samples", type=int, required=True,
                        help="Fewest resetting engine steps per role the profile needs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads((args.stage_dir / "config.json").read_text())
    if config.get("canonical_replay", {}).get("kind") != "host_probe" or config.get("case_id") != args.case:
        raise ValueError("The stage is not a host_probe run of this case")
    topology = CASES[args.case]
    engine = config["engine"]
    tp, dp = engine["tensor_parallel_size"], engine["data_parallel_size"]
    if dp != topology["data_parallel_size"] or engine["pipeline_parallel_size"] != 1:
        raise ValueError("The staged engine differs from the case topology")
    device, network_device = DEVICES[topology["gpu_type"]]
    replica = ReplicaConfig(model_name=MODEL, device=device, network_device=network_device,
                            attn_tensor_parallel_size=tp, attn_dp=dp, moe_tensor_parallel_size=1,
                            moe_expert_parallel_size=tp * dp, attention_backend="STEP5_DSA",
                            kv_cache_dtype="auto")

    evidence = args.stage_dir / "evidence"
    summary = json.loads((evidence / "replay/canonical_summary.json").read_text())
    windows = formal_windows(summary)
    processes = read_processes(evidence / "nodes")
    report = {"case": args.case, "stage": str(args.stage_dir), "job_id": args.job_id,
              "formal_windows_wall_s": windows, "roles": {}}
    for role in ROLES:
        steps, join = engine_steps(role_processes(processes, role, dp, tp), dp, tp, windows)
        report["roles"][role] = {"join": join, **role_report(steps)}
    report["perturbation"] = perturbation(summary, json.loads(args.clean_summary.read_text()))
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    short = {role: report["roles"][role]["host_ms"]["n"] for role in ROLES
             if report["roles"][role]["host_ms"]["n"] < args.min_samples}
    if short:
        raise ValueError(f"Resetting steps per role {short} are fewer than {args.min_samples}; no profile written")
    roles = {role: {"time_ms": report["roles"][role]["host_ms"]["mean"]} for role in ROLES}

    profile = {
        "schema": SCHEMA,
        "provenance": {"job_id": args.job_id, "stage": str(args.stage_dir),
                       "source_commit": config["source"]["commit"],
                       "probe_sha256": config["tool_hashes"][PROBE],
                       "statistic": "mean over the resetting engine steps of the formal windows",
                       "resetting_steps": {role: report["roles"][role]["steps"]["resetting"] for role in ROLES},
                       "perturbation": report["perturbation"]["status"],
                       "report": str(args.report)},
        "model_architecture_profile": replica.model_config.get_model_architecture_profile().profile_id,
        "device": device,
        "decode_cuda_graph_mode": engine["compilation_config"]["cudagraph_mode"].lower(),
        "topology": {"attention_tp": tp, "attention_dp": dp, "moe_ep": tp * dp, "pipeline_stages": 1},
        "roles": roles,
    }
    args.output.write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps({"roles": roles, "perturbation": report["perturbation"]["status"],
                      "join": {role: report["roles"][role]["join"] for role in ROLES}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
