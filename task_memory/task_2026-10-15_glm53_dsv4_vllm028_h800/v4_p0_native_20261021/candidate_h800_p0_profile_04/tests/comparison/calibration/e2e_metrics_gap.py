"""E2E gate of one calibration case: clean vLLM request rows against Frontier runs.

Joins the formal requests of the case's request-id map once on each side and reports
the calibration contract's five metrics independently: mean TTFT, mean TPOT (requests
with more than one decode token), mean request E2E time, request throughput and token
throughput over the formal window (first formal arrival to last formal completion,
each side on its own clock). A metric passes when its relative error
abs(frontier - vllm) / abs(vllm) is at most 0.10.

The vLLM rows follow the case manifest's topology.frontier.sys_arch: request_metrics.jsonl
of the one instance for co-location, the joined 1P1D rows of pd_metrics.py
(pd_request_metrics.jsonl) for pd-disaggregation.

Several Frontier runs of the same case may be given. The first is the gated checkout;
the others are reference checkouts reported next to it and never change the verdict.

  python tests/comparison/calibration/e2e_metrics_gap.py \
      --manifest <case>/manifest.yaml \
      --request-ids <case>/inputs/trace/request_ids.json \
      --vllm-run-dir <case>/runs/groundtruth_clean/<run> \
      --frontier-run stack_head=<simulator output> --frontier-run main=<simulator output> \
      --routing-status MATCH --output-dir <fresh dir>
"""

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

RELATIVE_ERROR_LIMIT = 0.10
NORMALIZED_COLUMNS = [
    "request_id", "arrival_s", "completion_s", "ttft_ms", "tpot_ms",
    "request_e2e_time_ms", "request_num_prefill_tokens", "request_num_decode_tokens",
]
TOKEN_COLUMNS = ["request_num_prefill_tokens", "request_num_decode_tokens"]
# The vLLM OpenAI completions server names the engine request cmpl-<X-Request-Id>-0.
VLLM_COMPLETION_REQUEST_ID = re.compile(r"cmpl-(?P<request_id>.+)-0")
# MISMATCH and UNSET block the numeric gate (calibration contract, MoE routing gate).
ROUTING_GATE = {"MATCH": None, "NOT_APPLICABLE": None,
                "MISMATCH": "FAIL", "UNSET": "INSUFFICIENT_EVIDENCE"}


def file_record(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "modified_at_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
    }


def require_written_after(artifact: Path, reference: Path) -> None:
    """An artifact older than the file its run writes first belongs to an earlier run."""
    if artifact.stat().st_mtime < reference.stat().st_mtime:
        raise SystemExit(f"{artifact} is older than {reference}: it was not written by this run")


def vllm_rows(run_dir: Path) -> tuple:
    manifest = run_dir / "run_manifest.json"
    metrics = run_dir / "run" / "request_metrics.jsonl"
    require_written_after(metrics, manifest)
    rows = pd.read_json(metrics, lines=True, convert_dates=False)
    request_ids = rows["request_id"].str.fullmatch(VLLM_COMPLETION_REQUEST_ID.pattern)
    if not request_ids.all():
        raise SystemExit(f"{metrics}: request ids not in the cmpl-<id>-0 form: "
                         f"{rows.loc[~request_ids, 'request_id'].tolist()[:5]}")
    rows["request_id"] = rows["request_id"].str.extract(VLLM_COMPLETION_REQUEST_ID.pattern,
                                                        expand=False)
    rows = rows.rename(columns={
        "arrival_time": "arrival_s", "completion_time": "completion_s", "ttft": "ttft_ms",
        "tpot": "tpot_ms", "request_e2e_time": "request_e2e_time_ms",
    })
    return rows[NORMALIZED_COLUMNS], [file_record(manifest), file_record(metrics)]


def vllm_pd_rows(run_dir: Path) -> tuple:
    manifest = run_dir / "run_manifest.json"
    metrics = run_dir / "run" / "pd_request_metrics.jsonl"
    require_written_after(metrics, manifest)
    rows = pd.read_json(metrics, lines=True, convert_dates=False, dtype={"request_id": str})
    return rows[NORMALIZED_COLUMNS], [file_record(manifest), file_record(metrics),
                                      file_record(run_dir / "run" / "pd_join_report.json")]


VLLM_ROWS = {"co-location": vllm_rows, "pd-disaggregation": vllm_pd_rows}


def frontier_rows(run_dir: Path, trace_rows: pd.DataFrame) -> tuple:
    metrics_paths = sorted(run_dir.rglob("request_metrics.csv"))
    if len(metrics_paths) != 1:
        raise SystemExit(f"{run_dir}: expected one request_metrics.csv, found {metrics_paths}")
    (metrics_path,) = metrics_paths
    # Frontier writes config.json when it builds the run's configuration.
    config_path = metrics_path.with_name("config.json")
    require_written_after(metrics_path, config_path)
    metrics = pd.read_csv(metrics_path).rename(columns={
        "Request Id": "frontier_request_id", "ttft": "ttft_ms", "tpot": "tpot_ms",
        "request_e2e_time": "request_e2e_time_ms",
    })
    rows = metrics.merge(trace_rows[["frontier_request_id", "request_id", "arrived_at"]],
                         on="frontier_request_id", how="left", validate="one_to_one")
    rows["arrival_s"] = rows["arrived_at"]
    rows["completion_s"] = rows["arrived_at"] + rows["request_e2e_time_ms"] / 1000.0
    return rows[NORMALIZED_COLUMNS], [file_record(config_path), file_record(metrics_path)]


def formal_sample(rows: pd.DataFrame, formal_ids: list, classified_ids: set) -> dict:
    counts = rows["request_id"].value_counts()
    sample = rows[rows["request_id"].isin(formal_ids)].drop_duplicates("request_id")
    return {
        "sample": sample.set_index("request_id"),
        "missing": sorted(set(formal_ids) - set(rows["request_id"])),
        "duplicates": sorted(set(counts[counts > 1].index) & set(formal_ids)),
        "unclassified": sorted(map(str, set(rows["request_id"]) - classified_ids)),
    }


def metric_values(sample: pd.DataFrame) -> dict:
    decoding = sample[sample["request_num_decode_tokens"] > 1]
    window_s = sample["completion_s"].max() - sample["arrival_s"].min()
    tokens = int(sample[TOKEN_COLUMNS].to_numpy().sum())
    return {
        "ttft_ms": (sample["ttft_ms"].sum(), len(sample), len(sample)),
        "tpot_ms": (decoding["tpot_ms"].sum(), len(decoding), len(decoding)),
        "request_e2e_time_ms": (sample["request_e2e_time_ms"].sum(), len(sample), len(sample)),
        "request_throughput_rps": (len(sample), window_s, len(sample)),
        "token_throughput_tps": (tokens, window_s, len(sample)),
    }


def metric_row(metric: str, label: str, vllm: tuple, frontier: tuple, formal_count: int,
               blocked_status: str) -> dict:
    (vllm_numerator, vllm_denominator, _), (numerator, denominator, eligible) = vllm, frontier
    denominators_valid = all(math.isfinite(value) and value > 0
                             for value in (vllm_denominator, denominator))
    vllm_value = vllm_numerator / vllm_denominator if denominators_valid else None
    frontier_value = numerator / denominator if denominators_valid else None
    # Relative error is undefined when the vLLM value is zero.
    relative_error = (abs(frontier_value - vllm_value) / abs(vllm_value)
                      if denominators_valid and vllm_value != 0 else None)
    if blocked_status:
        status = blocked_status
    elif relative_error is None:
        status = "INSUFFICIENT_EVIDENCE"
    else:
        status = "PASS" if relative_error <= RELATIVE_ERROR_LIMIT else "FAIL"
    return {
        "frontier_run": label, "metric": metric,
        "metric_statistic": "mean" if metric.endswith("_ms") else "window_rate",
        "vllm": vllm_value, "frontier": frontier_value,
        "absolute_error": None if frontier_value is None else frontier_value - vllm_value,
        "relative_error": relative_error, "status": status,
        "sample_count": formal_count, "eligible_count": eligible,
        "excluded_count": formal_count - eligible,
        "vllm_numerator": vllm_numerator, "vllm_denominator": vllm_denominator,
        "frontier_numerator": numerator, "frontier_denominator": denominator,
    }


def token_count_mismatches(vllm_sample: pd.DataFrame, frontier_sample: pd.DataFrame) -> list:
    shared = vllm_sample.index.intersection(frontier_sample.index)
    differs = (vllm_sample.loc[shared, TOKEN_COLUMNS]
               != frontier_sample.loc[shared, TOKEN_COLUMNS]).any(axis=1)
    return sorted(shared[differs.to_numpy()])


def number(value) -> str:
    return "-" if pd.isna(value) else f"{value:.4f}"


def summary_markdown(table: pd.DataFrame, labels: list, gate: str) -> str:
    header = ["Metric", "vLLM"] + [f"{label} ({kind})" for label in labels
                                   for kind in ("value", "rel. err", "status")]
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    for metric, rows in table.groupby("metric", sort=False):
        rows = rows.set_index("frontier_run")
        cells = [metric, number(rows["vllm"].iloc[0])]
        for label in labels:
            row = rows.loc[label]
            cells += [number(row["frontier"]),
                      "undefined" if pd.isna(row["relative_error"]) else number(row["relative_error"]),
                      row["status"]]
        lines.append("| " + " | ".join(cells) + " |")
    return (f"Gate on `{labels[0]}`: **{gate}**. Other runs are references and do not "
            f"change the verdict.\n\n" + "\n".join(lines) + "\n")


def frontier_run_argument(value: str) -> tuple:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(f"expected LABEL=DIR, got {value!r}")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--request-ids", type=Path, required=True)
    parser.add_argument("--vllm-run-dir", type=Path, required=True)
    parser.add_argument("--frontier-run", type=frontier_run_argument, action="append",
                        required=True, help="LABEL=DIR; the first run is the gated checkout")
    parser.add_argument("--routing-status", choices=sorted(ROUTING_GATE), required=True,
                        help="MoE routing-alignment status; NOT_APPLICABLE for dense models")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    labels = [label for label, _ in args.frontier_run]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate --frontier-run labels: {labels}")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    manifest = yaml.safe_load(args.manifest.read_text())
    request_ids = json.loads(args.request_ids.read_text())
    formal_ids = request_ids["formal_request_ids"]
    setup_ids = set(request_ids["warmup_request_ids"]) | set(request_ids["sizing_request_ids"])
    if setup_ids & set(formal_ids):
        raise SystemExit(f"{args.request_ids}: warmup or sizing ids listed as formal: "
                         f"{sorted(setup_ids & set(formal_ids))}")
    classified_ids = setup_ids | set(formal_ids)
    trace_rows = pd.DataFrame(request_ids["rows"])

    rows, vllm_files = VLLM_ROWS[manifest["topology"]["frontier"]["sys_arch"]](args.vllm_run_dir)
    sides = {"vllm": formal_sample(rows, formal_ids, classified_ids)}
    inputs = {"manifest": file_record(args.manifest), "request_ids": file_record(args.request_ids),
              "vllm": vllm_files, "frontier": {}}
    for label, run_dir in args.frontier_run:
        rows, inputs["frontier"][label] = frontier_rows(run_dir, trace_rows)
        sides[label] = formal_sample(rows, formal_ids, classified_ids)
    for side, value in sides.items():
        value["sample"].to_csv(args.output_dir / f"normalized_{side}.csv")
        value["metrics"] = metric_values(value["sample"])

    evidence_gaps = {
        side: {"missing_request_ids": value["missing"], "duplicate_request_ids": value["duplicates"],
               "unclassified_request_ids": value["unclassified"]}
        for side, value in sides.items()
        if value["missing"] or value["duplicates"] or value["unclassified"]
    }
    mismatches = {label: token_count_mismatches(sides["vllm"]["sample"], sides[label]["sample"])
                  for label in labels}
    table = []
    for label in labels:
        blocked_status = ROUTING_GATE[args.routing_status] or (
            "INSUFFICIENT_EVIDENCE"
            if {"vllm", label} & set(evidence_gaps) or mismatches[label] else None)
        table += [metric_row(metric, label, sides["vllm"]["metrics"][metric],
                             sides[label]["metrics"][metric], len(formal_ids), blocked_status)
                  for metric in sides["vllm"]["metrics"]]
    table = pd.DataFrame(table)
    table.to_csv(args.output_dir / "e2e_metrics_table.csv", index=False)

    gated = table[table["frontier_run"] == labels[0]]["status"]
    gate = ("PASS" if (gated == "PASS").all()
            else "FAIL" if (gated == "FAIL").any() else "INSUFFICIENT_EVIDENCE")
    status = {
        "entry": "e2e-metrics-gap",
        "case_id": manifest["case_id"],
        "run_generation": manifest["run_generation"],
        "gated_frontier_run": labels[0],
        "reference_frontier_runs": labels[1:],
        "ttft_semantic": "request_arrival_to_prefill_completion",
        "formal_request_count": len(formal_ids),
        "warmup_request_ids": request_ids["warmup_request_ids"],
        "request_id_gaps": evidence_gaps,
        "token_count_mismatches": mismatches,
        "routing_status": args.routing_status,
        "formal_window": {
            side: {"first_formal_arrival_s": value["sample"]["arrival_s"].min(),
                   "last_formal_completion_s": value["sample"]["completion_s"].max()}
            for side, value in sides.items()
        },
        "inputs": inputs,
        "tool": file_record(Path(__file__).resolve()),
        "fresh_file": "PASS: every metrics file is newer than the file its run writes first",
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate": gate,
    }
    (args.output_dir / "e2e_metrics_status.json").write_text(
        json.dumps(status, indent=1, default=str) + "\n")
    summary = summary_markdown(table, labels, gate)
    (args.output_dir / "e2e_metrics_summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
