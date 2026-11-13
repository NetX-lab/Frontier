"""Write the Step-5 PREFILL DP dummy-step group cadence profile from a native host_probe run.

Every forward of a DP engine, real or dummy, first passes `dispatch_cg_and_sync_dp`, a CPU
all-reduce over the DP engines that releases dispatch n of every engine when the last engine
arrives. While every engine runs a dummy step the dummy steps run as CUDA graphs and nothing in
them waits on the device, so the hosts dispatch at one group cadence ahead of the GPU, which runs
the queued dummy steps in order. Frontier paces the hosts of such all-dummy forwards by this
cadence (`dp_dummy_step_host_time.v1`).

The host-step probe records, per worker process, one line per model-runner step, with the DP
all-reduce ordinal that joins the engines. A cadence sample is the interval between the
launches of all-reduce ordinals k-1 and k when both are dummy steps on every engine and TP rank,
k falls inside a formal measurement window, and the last ordinal with a real step on any engine
lies more than `--exclude-after-real` ordinals before k. Within the excluded ordinals the engine
that ran the real step pops and processes its output and reaches the all-reduce late; that host
time belongs to the real step, not to the dummy cadence, and the report gives its excess per real
cohort. A new wave starts with a real step on the engine that received the request, so an idle
gap between waves never lies between two all-dummy ordinals.

The engines run on separate nodes whose monotonic clocks are not comparable, so each engine's
interval uses the latest launch over its own TP ranks (one node), and the sample is the mean of
the engines' intervals. The barrier makes the intervals of all engines equal up to their local
launch work. The profile holds the mean of the samples, because the queued dummy steps on the
GPU add up the intervals.

The probe files are checked for completeness as in `kv_zeroing_host_time.py`. The probe perturbs
native timing; a client TTFT/TPOT shift beyond the clean run's repeat range marks the profile
`exploratory`, otherwise it is `admitted`.

Usage: dp_dummy_step_host_time.py --case h200_64 --stage-dir <stage> --clean-summary <json>
       --job-id <rjob> --exclude-after-real 3 --min-samples 1000 --output <profile.json>
       --report <report.json>
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from frontier.config import ReplicaConfig  # noqa: E402
from frontier.execution_time_predictor.dp_dummy_step_host_time import SCHEMA  # noqa: E402
from tests.integration.step5_pd.frontier_workload import CASES, DEVICES, MODEL  # noqa: E402
from tests.integration.step5_pd.kv_zeroing_host_time import (  # noqa: E402
    PROBE, expected_window_tokens, formal_windows, join_by_ordinal, perturbation, read_processes,
    role_processes, window_preemptions, window_tokens)

ROLE = "prefill"


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * (len(ordered) - 1) + 0.5))]


def cadence_samples(processes: dict[tuple[int, int], dict], dp_size: int, windows: list[tuple[float, float]],
                    exclude_after_real: int) -> tuple[list[dict], dict]:
    """The cadence samples, those near a real step marked with its ordinal (`after_real`), and the
    exclusion counts."""
    by_ordinal, ordinals, _ = join_by_ordinal(processes)
    real = sorted(ordinal for ordinal in ordinals
                  if any(not records[ordinal]["dummy"] for records in by_ordinal.values()))
    counts = defaultdict(int)
    samples = []
    last_real = None
    real_index = 0
    for ordinal in sorted(ordinals):
        while real_index < len(real) and real[real_index] < ordinal:
            last_real = real[real_index]
            real_index += 1
        if ordinal - 1 not in ordinals:
            counts["previous_ordinal_missing"] += 1
            continue
        current = {key: records[ordinal] for key, records in by_ordinal.items()}
        previous = {key: records[ordinal - 1] for key, records in by_ordinal.items()}
        if not all(record["dummy"] for record in [*current.values(), *previous.values()]):
            counts["real_step"] += 1
            continue
        if not all("launch" in record for record in [*current.values(), *previous.values()]):
            counts["launch_missing"] += 1
            continue
        wall = min(record["wall_ns"] for record in current.values()) / 1e9
        window = next((index for index, (start, end) in enumerate(windows) if start <= wall <= end), None)
        if window is None:
            counts["outside_formal_windows"] += 1
            continue
        intervals = []
        for dp in range(dp_size):
            launch = [record["launch"] for (engine, _), record in current.items() if engine == dp]
            launched = [record["launch"] for (engine, _), record in previous.items() if engine == dp]
            intervals.append((max(launch) - max(launched)) / 1e6)
        since_real = None if last_real is None else ordinal - last_real
        sample = {"ordinal": ordinal, "window": window, "since_real": since_real,
                  "cadence_ms": statistics.fmean(intervals), "engine_intervals_ms": intervals}
        if since_real is not None and since_real <= exclude_after_real:
            counts["near_real_step"] += 1
            sample["after_real"] = last_real
        samples.append(sample)
    return samples, dict(counts)


def role_report(samples: list[dict], counts: dict, windows: list[tuple[float, float]]) -> dict:
    kept = [sample for sample in samples if "after_real" not in sample]
    values = [sample["cadence_ms"] for sample in kept]
    by_window = defaultdict(list)
    for sample in kept:
        by_window[sample["window"]].append(sample["cadence_ms"])
    window_means = [statistics.fmean(by_window[index]) for index in range(len(windows)) if by_window[index]]
    by_since_real = defaultdict(list)
    for sample in samples:
        by_since_real[sample["since_real"]].append(sample["cadence_ms"])
    mean = statistics.fmean(values) if values else None
    near = [sample for sample in samples if "after_real" in sample]
    # Spread of the engines' intervals within one sample: the barrier keeps it at local launch noise.
    spread = [max(sample["engine_intervals_ms"]) - min(sample["engine_intervals_ms"]) for sample in kept]
    return {
        "cadence_ms": {"n": len(values), "mean": mean,
                       **({f"p{int(q * 100)}": quantile(values, q) for q in (0.1, 0.5, 0.9, 0.99)} if values else {}),
                       "max": max(values, default=None)},
        "window_means_ms": window_means,
        "window_samples": [len(by_window[index]) for index in range(len(windows))],
        "window_mean_cv": (statistics.pstdev(window_means) / statistics.fmean(window_means)
                           if len(window_means) > 1 else None),
        "engine_interval_spread_ms": {"p50": quantile(spread, 0.5), "p99": quantile(spread, 0.99)} if spread else {},
        "by_ordinals_after_real": {str(since): {"n": len(group), "mean": statistics.fmean(group),
                                                "p50": quantile(group, 0.5)}
                                   for since, group in sorted(by_since_real.items(), key=lambda item: item[0] or 0)},
        # Host time above the cadence that the excluded samples carry, per real cohort they follow.
        "near_real_excess_ms_per_real_cohort": (sum(sample["cadence_ms"] - mean for sample in near)
                                                / len({sample["after_real"] for sample in near})
                                                if mean is not None and near else None),
        "counts": counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--stage-dir", type=Path, required=True,
                        help="host_probe stage with config.json and the copied evidence/")
    parser.add_argument("--clean-summary", type=Path, required=True,
                        help="canonical_summary.json of a clean workload run of the same case")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--exclude-after-real", type=int, required=True,
                        help="Drop samples at ordinals up to this many after the last real step on any engine")
    parser.add_argument("--min-samples", type=int, required=True, help="Fewest cadence samples the profile needs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads((args.stage_dir / "config.json").read_text())
    if config.get("canonical_replay", {}).get("kind") != "host_probe" or config.get("case_id") != args.case:
        raise ValueError("The stage is not a host_probe run of this case")
    topology = CASES[args.case]
    engine = config["engine"]
    tp, dp = engine["tensor_parallel_size"], engine["data_parallel_size"]
    if dp != topology["data_parallel_size"] or engine["pipeline_parallel_size"] != 1 or dp < 2:
        raise ValueError("The staged engine differs from the case topology or has no DP group")
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
    engines = role_processes(read_processes(evidence / "nodes"), ROLE, dp, tp)
    tokens = window_tokens(engines, ROLE, windows, expected[ROLE])
    samples, counts = cadence_samples(engines, dp, windows, args.exclude_after_real)
    report = {"case": args.case, "stage": str(args.stage_dir), "job_id": args.job_id,
              "exclude_after_real": args.exclude_after_real, "formal_windows_wall_s": windows,
              "window_tokens": tokens, ROLE: role_report(samples, counts, windows),
              "perturbation": perturbation(summary, json.loads(args.clean_summary.read_text()))}
    args.report.write_text(json.dumps(report, indent=1) + "\n")
    result = report[ROLE]
    if result["cadence_ms"]["n"] < args.min_samples:
        raise ValueError(f"{result['cadence_ms']['n']} cadence samples are fewer than {args.min_samples}; "
                         "no profile written")

    estimator = Path(__file__).resolve()
    profile = {
        "schema": SCHEMA,
        "admission": "admitted" if report["perturbation"]["status"] == "PASS" else "exploratory",
        "provenance": {"job_id": args.job_id, "stage": str(args.stage_dir),
                       "source_commit": config["source"]["commit"],
                       "probe_sha256": config["tool_hashes"][PROBE],
                       "estimator": str(estimator.relative_to(REPO)),
                       "estimator_sha256": hashlib.sha256(estimator.read_bytes()).hexdigest(),
                       "statistic": ("mean of the all-dummy group cadence samples of the formal windows, "
                                     f"excluding ordinals up to {args.exclude_after_real} after a real step"),
                       "perturbation": report["perturbation"]["status"],
                       "report": str(args.report)},
        "model_architecture_profile": replica.model_config.get_model_architecture_profile().profile_id,
        "device": device,
        "decode_cuda_graph_mode": engine["compilation_config"]["cudagraph_mode"].lower(),
        "topology": {"attention_tp": tp, "attention_dp": dp, "moe_ep": tp * dp, "pipeline_stages": 1},
        "roles": {ROLE: {"group_cadence_ms": result["cadence_ms"]["mean"], "samples": result["cadence_ms"]["n"],
                         "p50_ms": result["cadence_ms"]["p50"], "p90_ms": result["cadence_ms"]["p90"],
                         "window_means_ms": result["window_means_ms"],
                         "window_mean_cv": result["window_mean_cv"]}},
    }
    args.output.write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps({"admission": profile["admission"], "roles": profile["roles"],
                      "near_real_excess_ms_per_real_cohort": result["near_real_excess_ms_per_real_cohort"],
                      "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
