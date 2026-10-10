"""Normalized request rows of one scenario-matrix cell from its raw vLLM and Frontier runs.

The rows reuse e2e_metrics_gap.py's readers: ``vllm_rows`` (co-location request_metrics.jsonl),
``vllm_pd_rows`` (the 1P1D rows that pd_metrics.py joined) and ``frontier_rows`` (Frontier's
request_metrics.csv mapped through the cell's request-id map), and its ``formal_sample``. Each side
keeps its formal requests; the formal ids it lacks, holds twice or the rows of no id list are
recorded in ``sample.json``.

Gate-specific columns:

- ``pd_s33``: endpoint_offset_proxy_api_ms of each vLLM request, the proxy hop to the prefill
  instance's arrival plus the prefill API-server offset (prefill ttft less its model execution
  time), from proxy_requests.jsonl and prefill/request_metrics.jsonl (pd_metrics.py names).
- ``dp_pp_pairs``: ttft_mode and tpot_mode of each request (dp_pp_modes.py). A vLLM run's modes
  are on its engines' monotonic clock, from each request's client dispatch; a Frontier member's on
  its own trace (trace.csv beside metrics/, the arrivals with the member's route delays). Each
  side's window counts per mode go to sample.json.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from tests.comparison.calibration.e2e_metrics_gap import VLLM_ROWS, formal_sample, frontier_rows
from tests.comparison.calibration.pd_metrics import by_request_id, engine_request_id, read_jsonl
from tests.comparison.e2e_gate import dp_pp_modes


def endpoint_offsets_ms(run_dir: Path) -> dict[str, float]:
    run = run_dir / "run"
    pd_rows = {row["request_id"]: row for row in read_jsonl(run / "pd_request_metrics.jsonl")}
    prefill = by_request_id(read_jsonl(run / "prefill" / "request_metrics.jsonl"))
    offsets = {}
    for request_id, [record] in by_request_id(read_jsonl(run / "proxy_requests.jsonl")).items():
        if request_id not in pd_rows:
            continue
        [row] = prefill[engine_request_id(record["proxied_request_id"])]
        offsets[request_id] = ((pd_rows[request_id]["prefill_arrival_s"] - pd_rows[request_id]["arrival_s"]) * 1e3
                               + row["ttft"] - row["request_model_execution_time"])
    return offsets


def label_modes(sample: pd.DataFrame, arrival: dict[str, float], lanes: dp_pp_modes.Lanes,
                window_s: float) -> tuple[pd.DataFrame, dict[str, int]]:
    start = min(arrival[request_id] for request_id in sample.index)
    modes = dp_pp_modes.window_modes(lanes, start, window_s)
    labels = {}
    for request_id, row in sample.iterrows():
        begin = arrival[request_id]
        ttft_mode, tpot_mode = dp_pp_modes.request_modes(
            begin, begin + row["ttft_ms"] * 1e-3, begin + row["request_e2e_time_ms"] * 1e-3, start, window_s, modes)
        labels[request_id] = (ttft_mode, tpot_mode if row["request_num_decode_tokens"] > 1 else None)
    labeled = sample.assign(ttft_mode=[labels[r][0] for r in sample.index],
                            tpot_mode=[labels[r][1] for r in sample.index])
    return labeled, dict(sorted(Counter(modes.values()).items()))


def normalize_cell(cell: dict, case: dict, request_ids: dict, vllm_runs: dict[str, Path],
                   frontier_runs: dict[str, Path], output_dir: Path) -> dict:
    """Write <output_dir>/sample.json, vllm/<run>.csv and frontier/<label>.csv; return the sample."""
    formal_ids = request_ids["formal_request_ids"]
    classified = set(formal_ids) | set(request_ids["warmup_request_ids"]) | set(request_ids["sizing_request_ids"])
    trace_rows = pd.DataFrame(request_ids["rows"])
    formal_frontier_ids = {str(row["frontier_request_id"]) for row in request_ids["rows"] if row["role"] != "warmup"}
    window_s = cell.get("dp_pp_gate", {}).get("mode_window_s")
    output_dir.mkdir(parents=True, exist_ok=False)
    sides, written = {}, {}
    for run, run_dir in vllm_runs.items():
        rows, _ = VLLM_ROWS[case["architecture"]](run_dir)
        sides[f"vllm/{run}"] = formal_sample(rows, formal_ids, classified)
        frame = sides[f"vllm/{run}"]["sample"]
        if cell["gate"] == "pd_s33":
            offsets = endpoint_offsets_ms(run_dir)
            frame = frame.assign(endpoint_offset_proxy_api_ms=[offsets[r] for r in frame.index])
        if window_s:
            frame, sides[f"vllm/{run}"]["windows"] = label_modes(
                frame, dp_pp_modes.vllm_formal_dispatch(run_dir), dp_pp_modes.vllm_lanes(run_dir), window_s)
        written[f"vllm/{run}"] = frame
    for label, run_dir in frontier_runs.items():
        [metrics_csv] = (run_dir / "metrics").rglob("request_metrics.csv")
        rows, _ = frontier_rows(metrics_csv.parent, trace_rows)
        sides[f"frontier/{label}"] = formal_sample(rows, formal_ids, classified)
        frame = sides[f"frontier/{label}"]["sample"]
        if window_s:
            trace = pd.read_csv(run_dir / "trace.csv")
            arrival = {row["request_id"]: float(trace.at[row["frontier_request_id"], "arrived_at"])
                       for row in request_ids["rows"]}
            frame, sides[f"frontier/{label}"]["windows"] = label_modes(
                frame, arrival, dp_pp_modes.frontier_lanes(metrics_csv.parent, formal_frontier_ids), window_s)
        written[f"frontier/{label}"] = frame
    for side, frame in written.items():
        (output_dir / side).parent.mkdir(parents=True, exist_ok=True)
        # 17 significant digits keep the wall-clock seconds exact through the CSV.
        frame.to_csv(output_dir / f"{side}.csv", index_label="request_id", float_format="%.17g")
    sample = {
        "cell_id": cell["cell_id"],
        "formal_request_ids": formal_ids,
        "sides": {side: {key: value[key] for key in ("missing", "duplicates", "unclassified", "windows")
                         if key in value} for side, value in sides.items()},
    }
    (output_dir / "sample.json").write_text(json.dumps(sample, indent=1) + "\n")
    return sample

