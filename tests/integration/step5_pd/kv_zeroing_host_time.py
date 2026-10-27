"""Write the Step-5 KV side-storage reset host-time profile from a native host_probe run.

A native Step-5 engine whose step resets the side storage of new blocks first waits, in the
first reset hook's `.tolist()`, for the previous step's device work. From that moment the
device idles until the host has run the remaining resets, the input preparation and the DP
padding all-reduce, and launched the forward. Frontier composes the DP forward start as
`max_i(a_i + h_i)` over the DP engines, so `h_i` must exclude the time engine `i` waited in
the all-reduce for the other engines.

The host-step probe records, per worker process, one line per model-runner step. The k-th
DP sync call of every engine at TP rank `r` is the same gloo all_reduce, so the call ordinal
joins the engines. For all_reduce k at rank r, the engine that arrived last waited least; its
all_reduce duration is the collective's own cost, and every other engine's duration above it
is its wait for the others (`excess`). The local work before and after the all_reduce stays
in the host time. The TP ranks of one engine share a node and its monotonic clock, so for a
resetting step of engine i:

    h_i = max over TP ranks r of (launch_r - excess_r) - min over r of first_sync_r

The forward starts when the slowest TP rank launches, and the earliest first `.tolist()`
return marks the end of the previous step's device work (on a waited step that rank waited
for the device). The profile holds the mean of `h_i` over the resetting steps of the formal
measurement windows, per role; when the least-squares trend of `h_i` over the observed
request counts exceeds `TREND_LIMIT_MS`, the role's entry is the mean per request count.

The probe files are complete for the formal windows: on every TP rank of every engine, the
non-dummy steps inside a window hold exactly the tokens of the window's client-completed formal
requests, no request was preempted (recomputed work could hide a lost step), and an all-reduce
lacks a participant only after the role's last step with requests.

The report adds checks that settle whether `first_sync` is the device idle point: the share
of steps on which a rank's previous step was still running at the first hook, the
main-stream gap from the CUDA events (`gap_ms - excess`), the hook count per step, the
per-rank variant and the TP skew of `first_sync`, values by request count, the clearance of
the window edges from the nearest step, and the client TTFT/TPOT shift against a clean run
(the probe perturbs native timing; a shift beyond the clean repeat range marks the profile
as perturbed).

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
from tests.integration.step5_pd.canonical_replay import (  # noqa: E402
    ROLES, boundary_reached, completed_work, phase_completions)
from tests.integration.step5_pd.frontier_workload import CASES, DEVICES, MODEL  # noqa: E402
# canonical_replay put the calibration helpers on the path.
from serving_metrics import engine_values, parse_prometheus  # noqa: E402

PROBE = "host_probe_bootstrap/host_step_probe.py"
PREEMPTIONS = "vllm:num_preemptions_total"
# Design 3.3: a request-count dependence above this over the observed range needs a table.
TREND_LIMIT_MS = 1.0


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
    """Wall-clock windows between the idle boundaries around each formal phase.

    Every engine was idle at both boundaries, so only formal requests run between them, and
    the edges lie in idle gaps away from the steps of either side. The servers' wall clocks
    may differ from the client's by less than the reported edge clearance.

    Each boundary is checked from its last recorded observation with the replay's rule,
    counting from the length sweep boundary. Replays before that rule expected completions
    also for the requests that failed at the client, so their recorded verdicts may be wrong.
    The idle gauges are snapshots, so a failed request may still have run after a boundary;
    `window_tokens` rejects a window that holds any work besides its formal requests.
    """
    boundaries = {boundary["boundary"]: boundary for boundary in summary["idle_boundaries"]}
    sweep = boundaries["after_length_sweep"]
    if not sweep["verified"]:
        raise ValueError("The idle boundary after_length_sweep is unverified")
    completed = {role: sweep["observations"][-1]["roles"][role]["completed_requests"] for role in ROLES}
    windows = []
    for repetition in summary["repetitions"]:
        formal = repetition["phases"]["formal"]
        offset = formal["origin_wall"] - formal["origin_monotonic"]
        edges = []
        for phase in ("warmup", "formal"):
            completed = {role: count + phase_completions(repetition["phases"][phase])
                         for role, count in completed.items()}
            name = f"after_repeat{repetition['repeat']:02d}_{phase}"
            last = boundaries[name]["observations"][-1]
            if not boundary_reached(last, completed):
                raise ValueError(f"The idle boundary {name} is not idle or misses completions {completed}")
            edges.append(last["monotonic"] + offset)
        windows.append((edges[0], edges[1]))
    return windows


def window_preemptions(summary: dict) -> None:
    """Reject a formal window in which a role preempted a request: its recomputed tokens could
    hide a lost step from the exact token count."""
    pages = {boundary["boundary"]: boundary["observations"][-1]["pages"] for boundary in summary["idle_boundaries"]}
    for repetition in summary["repetitions"]:
        start, end = (pages[f"after_repeat{repetition['repeat']:02d}_{phase}"] for phase in ("warmup", "formal"))
        for role in ROLES:
            before, after = (engine_values(parse_prometheus(page[role]), PREEMPTIONS) for page in (start, end))
            if not after or sum(after.values()) != sum(before.values()):
                raise ValueError(f"{role}: {PREEMPTIONS} changed or is missing in formal window "
                                 f"{repetition['repeat']}: {before} -> {after}")


def expected_window_tokens(records: list[dict], summary: dict, block_size: int) -> dict[str, list[int]]:
    """Tokens each role schedules in each formal window: the work of its client-completed formal
    requests (`canonical_replay.completed_work`)."""
    expected = {role: [] for role in ROLES}
    for repetition in summary["repetitions"]:
        work = completed_work([record for record in records if record["segment"] == "formal"
                               and record["repeat"] == repetition["repeat"]], block_size)
        expected["prefill"].append(work["prefill_prompt_sources"]["local_compute"])
        expected["decode"].append(work["decode_prompt_sources"]["local_compute"]
                                  + work["decode_inter_token_latencies"])
    return expected


def edge_clearance_s(processes: dict, windows: list[tuple[float, float]]) -> float:
    """Smallest distance from a window edge to a non-dummy step of any process."""
    starts = [record["wall_ns"] / 1e9 for files in processes.values() for process in files
              for record in process["records"] if record["kind"] == "step" and not record["dummy"]]
    return min(abs(start - edge) for start in starts for window in windows for edge in window)


def role_processes(processes: dict, role: str, dp_size: int, tp_size: int) -> dict[tuple[int, int], dict]:
    """Exactly one error-free process per engine and TP rank, or the role is unusable.

    A file holds steps 1 to N in order. The native P/D launcher stops the engines with an abort
    shutdown that force-kills the workers, so `GPUModelRunner.shutdown` never runs: the probe's
    `end` line is absent and the last step or two, still waiting for their device events, stay
    unwritten. The file is line-buffered, so every earlier step is on disk; `window_tokens`
    and `engine_steps` check that no step with requests is missing.
    """
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
        header, records = process["header"], process["records"]
        if (header["dp_size"], header["tp_size"]) != (dp_size, tp_size):
            raise ValueError(f"{role}: {process['path']} reports DP{header['dp_size']} x TP{header['tp_size']}")
        steps = [record["seq"] for record in records if record["kind"] == "step"]
        if not steps or steps != list(range(1, len(steps) + 1)):
            raise ValueError(f"{role}: {process['path']} skips steps")
    for dp in range(dp_size):
        hosts = {found[dp, tp][0]["header"]["host"] for tp in range(tp_size)}
        if len(hosts) != 1:
            raise ValueError(f"{role}: the TP ranks of DP engine {dp} run on {sorted(hosts)}; "
                             "their monotonic clocks are not comparable")
    return {key: files[0] for key, files in found.items()}


def window_tokens(processes: dict[tuple[int, int], dict], role: str, windows: list[tuple[float, float]],
                  expected: list[int]) -> list[list[int]]:
    """Tokens each DP engine scheduled in each formal window, checked on every TP rank.

    The TP ranks of an engine run the same batches, and the engines together run exactly the
    work of the window's client-completed formal requests (`expected`). A probe file that lost
    a step with requests, or work of another request inside the window, breaks one of the two.
    """
    dp_size = 1 + max(dp for dp, _ in processes)
    per_engine = []
    for index, (start, end) in enumerate(windows):
        ranks = defaultdict(set)
        for (dp, tp), process in processes.items():
            ranks[dp].add(sum(record["tokens"] for record in process["records"] if record["kind"] == "step"
                              and not record["dummy"] and start <= record["wall_ns"] / 1e9 <= end))
        uneven = {dp: sorted(values) for dp, values in ranks.items() if len(values) != 1}
        if uneven:
            raise ValueError(f"{role}: the TP ranks of DP engines {uneven} scheduled different tokens in "
                             f"formal window {index + 1}; a probe file lost steps")
        engines = [ranks[dp].pop() for dp in range(dp_size)]
        if sum(engines) != expected[index]:
            raise ValueError(f"{role}: formal window {index + 1} scheduled {engines} tokens per engine, not the "
                             f"{expected[index]} of its client-completed formal requests")
        per_engine.append(engines)
    return per_engine


def engine_steps(processes: dict[tuple[int, int], dict], dp_size: int, tp_size: int,
                 windows: list[tuple[float, float]]) -> tuple[list[dict], dict]:
    """Join the engines by DP all-reduce ordinal and derive each resetting step's host time."""
    by_ordinal = {key: {record["sync"]["ordinal"]: record for record in process["records"]
                        if record["kind"] == "step" and "sync" in record}
                  for key, process in processes.items()}
    by_sequence = {key: {record["seq"]: record for record in process["records"] if record["kind"] == "step"}
                   for key, process in processes.items()}
    ordinals = set.intersection(*(set(records) for records in by_ordinal.values()))
    partial = set.union(*(set(records) for records in by_ordinal.values())) - ordinals
    # A force-killed worker leaves only its last steps unwritten. An all-reduce whose missing
    # participant ran with requests, or was a dummy peer of one, would bias or drop samples.
    useful = max(ordinal for records in by_ordinal.values() for ordinal, record in records.items()
                 if not record["dummy"])
    lost = sorted(ordinal for ordinal in partial if ordinal <= useful)
    if lost:
        raise ValueError(f"All-reduces {lost[:5]} lack a participant at or before the last step with requests "
                         f"(all-reduce {useful}); a probe file lost steps")
    steps = []
    for ordinal in sorted(ordinals):
        records = {key: by_ordinal[key][ordinal] for key in by_ordinal}
        excess = {}
        for tp in range(tp_size):
            durations = {dp: records[dp, tp]["sync"]["reduce_exit"] - records[dp, tp]["sync"]["reduce_enter"]
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
                ready = [record["launch"] - excess[dp, tp] for tp, record in enumerate(ranks)]
                first = [record["zero"]["first_sync"] for record in ranks]
                step.update(host_ms=(max(ready) - min(first)) / 1e6,
                            rank_host_ms=max(r - f for r, f in zip(ready, first)) / 1e6,
                            rank0_host_ms=(ready[0] - first[0]) / 1e6,
                            first_sync_skew_ms=(max(first) - min(first)) / 1e6,
                            hooks=sorted({record["zero"]["hooks"] for record in ranks}),
                            waited=any(record["zero"]["prior_pending"] for record in ranks))
            steps.append(step)
    join = {"joined_all_reduces": len(ordinals), "trailing_partial_all_reduces": len(partial)}
    return steps, join


def distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    quartiles = statistics.quantiles(values, n=4) if len(values) > 1 else values * 3
    return {"n": len(values), "mean": statistics.fmean(values), "p25": quartiles[0],
            "median": quartiles[1], "p75": quartiles[2], "min": min(values), "max": max(values)}


def request_trend_ms(steps: list[dict]) -> float:
    """Change of the least-squares line of host time over the observed request-count range."""
    counts = [step["reqs"] for step in steps]
    if len(set(counts)) < 2:
        return 0.0
    line = statistics.linear_regression(counts, [step["host_ms"] for step in steps])
    return line.slope * (max(counts) - min(counts))


def role_entry(steps: list[dict]) -> dict:
    """The profile entry: the mean, or the mean per request count when the trend needs it."""
    resetting = [step for step in steps if step["resetting"]]
    if abs(request_trend_ms(resetting)) <= TREND_LIMIT_MS:
        return {"time_ms": statistics.fmean(step["host_ms"] for step in resetting)}
    by_requests = defaultdict(list)
    for step in resetting:
        by_requests[step["reqs"]].append(step["host_ms"])
    return {"by_num_requests": [[reqs, statistics.fmean(values)] for reqs, values in sorted(by_requests.items())]}


def role_report(steps: list[dict]) -> dict:
    resetting = [step for step in steps if step["resetting"]]
    plain = [step for step in steps if not step["resetting"]]
    by_requests = defaultdict(list)
    for step in resetting:
        by_requests[step["reqs"]].append(step["host_ms"])
    paired = [step for step in resetting if "device_gap_ms" in step]
    return {
        "host_ms": distribution([step["host_ms"] for step in resetting]),
        "rank_host_ms": distribution([step["rank_host_ms"] for step in resetting]),
        "rank0_host_ms": distribution([step["rank0_host_ms"] for step in resetting]),
        "first_sync_skew_ms": distribution([step["first_sync_skew_ms"] for step in resetting]),
        "waited_share": (sum(step["waited"] for step in resetting) / len(resetting)) if resetting else None,
        "host_ms_when_waited": distribution([step["host_ms"] for step in resetting if step["waited"]]),
        "host_ms_when_idle": distribution([step["host_ms"] for step in resetting if not step["waited"]]),
        "device_gap_check": {
            "host_ms": distribution([step["host_ms"] for step in paired]),
            "device_gap_ms": distribution([step["device_gap_ms"] for step in paired])},
        "non_resetting_device_gap_ms": distribution([step["device_gap_ms"] for step in plain
                                                     if "device_gap_ms" in step]),
        "dp_wait_ms": distribution([step["wait_ms"] for step in resetting]),
        "hooks_per_step": {str(hooks): count for hooks, count in sorted(Counter(
            tuple(step["hooks"]) for step in resetting).items())},
        "host_ms_by_num_requests": {str(reqs): distribution(values) for reqs, values in sorted(by_requests.items())},
        "request_trend_ms": request_trend_ms(resetting),
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
    window_preemptions(summary)
    with (evidence / "replay/client_requests.jsonl").open() as stream:
        expected = expected_window_tokens([json.loads(line) for line in stream], summary, engine["block_size"])
    processes = read_processes(evidence / "nodes")
    report = {"case": args.case, "stage": str(args.stage_dir), "job_id": args.job_id,
              "formal_windows_wall_s": windows, "window_edge_clearance_s": edge_clearance_s(processes, windows),
              "roles": {}}
    steps = {}
    for role in ROLES:
        engines = role_processes(processes, role, dp, tp)
        tokens = window_tokens(engines, role, windows, expected[role])
        steps[role], join = engine_steps(engines, dp, tp, windows)
        report["roles"][role] = {"join": join, "window_tokens": tokens, **role_report(steps[role])}
    report["perturbation"] = perturbation(summary, json.loads(args.clean_summary.read_text()))
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    short = {role: report["roles"][role]["host_ms"]["n"] for role in ROLES
             if report["roles"][role]["host_ms"]["n"] < args.min_samples}
    if short:
        raise ValueError(f"Resetting steps per role {short} are fewer than {args.min_samples}; no profile written")
    roles = {role: role_entry(steps[role]) for role in ROLES}

    profile = {
        "schema": SCHEMA,
        "provenance": {"job_id": args.job_id, "stage": str(args.stage_dir),
                       "source_commit": config["source"]["commit"],
                       "probe_sha256": config["tool_hashes"][PROBE],
                       "statistic": ("mean over the resetting engine steps of the formal windows; per request "
                                     f"count when the trend exceeds {TREND_LIMIT_MS} ms"),
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
