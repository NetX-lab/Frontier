"""Gate readings of the E2E scenario-matrix cells from normalized request rows.

A cell's rows directory holds ``sample.json`` and one CSV per side: ``vllm/<run>.csv`` for each
ground-truth vLLM run and ``frontier/<label>.csv`` for each Frontier run, with the normalized
columns of e2e_metrics_gap.py indexed by the trace's request id. ``normalize.py`` writes it from
raw runs; the fixtures ship it.

The metric contract is e2e_metrics_gap.py's: mean TTFT, mean TPOT over requests with more than
one output token, mean request E2E time, request throughput and token throughput over the formal
window (first formal arrival to last formal completion, each side on its own clock), each passing
when abs(frontier - vllm) / abs(vllm) <= RELATIVE_ERROR_LIMIT. The five metrics are decided
independently; the cell passes when all of its gated metrics pass.

Evidence admission, before any number is gated:

- request identity: every side holds each formal request exactly once and no other request;
- token counts: each Frontier row has the vLLM row's prompt and output token counts;
- clocks and units: seconds for arrival_s and completion_s, milliseconds for the latencies, with
  request_e2e_time_ms = 1000 (completion_s - arrival_s) and 0 <= ttft_ms <= request_e2e_time_ms;
- routing: the case's MoE routing-alignment status, from the scenario matrix (ROUTING_GATE:
  MISMATCH fails the cell, UNSET leaves it without evidence).

A failed identity, token or clock check makes every metric INSUFFICIENT_EVIDENCE.

Gate kinds (scenario matrix ``gate``):

- ``single``: one vLLM run against one Frontier run.
- ``pd_s33``: as ``single``, but the vLLM TTFT is reduced by the minimum endpoint offset of the
  formal requests (proxy hop plus prefill API-server offset, column endpoint_offset_proxy_api_ms of
  the vLLM rows), decision S33-TTFT (ii).
- ``dp_pp_pairs``: decision T43-PAIRGATE. Every ground-truth run is paired with every Frontier
  ensemble member; a request's TTFT counts for mode M when its ttft_mode is M on both sides, its
  TPOT when its tpot_mode is. A (mode, metric) pool is gated when MIN_MODE_REQUESTS distinct
  requests reached it. An empty gated set is INSUFFICIENT_EVIDENCE: no single numerical
  comparison exists. The modes holding REQUIRED_MODE_SHARE of the vLLM runs' windows are the
  reference modes; more than one is a multi-mode reference. The whole-cell metrics (mean of the
  vLLM runs, mean of the members) are reported, not gated.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd

from tests.comparison.calibration.e2e_metrics_gap import (
    NORMALIZED_COLUMNS,
    RELATIVE_ERROR_LIMIT,
    ROUTING_GATE,
    metric_row,
    metric_values,
    token_count_mismatches,
)

MIN_MODE_REQUESTS = 20
REQUIRED_MODE_SHARE = 0.2
CLOCK_TOLERANCE_MS = 1e-3
MODE_METRICS = ("ttft_ms", "tpot_ms")


def read_rows(rows_dir: Path) -> tuple[dict, dict[str, pd.DataFrame]]:
    sample = json.loads((rows_dir / "sample.json").read_text())
    sides = {side: pd.read_csv(rows_dir / f"{side}.csv", dtype={"request_id": str},
                               float_precision="round_trip").set_index("request_id")
             for side in sample["sides"]}
    return sample, sides


def evidence_gaps(rows: pd.DataFrame, formal_ids: list[str], recorded: dict) -> dict:
    """Identity and clock violations of one side's rows, with the gaps normalize.py recorded (it
    writes one row per request, so duplicates are known only from its record)."""
    index = rows.index
    elapsed_ms = (rows["completion_s"] - rows["arrival_s"]) * 1000.0
    clock = ((elapsed_ms - rows["request_e2e_time_ms"]).abs() > CLOCK_TOLERANCE_MS) | (rows["ttft_ms"] < 0) | (
        rows["ttft_ms"] > rows["request_e2e_time_ms"])
    gaps = {
        "missing_request_ids": sorted(set(recorded.get("missing", [])) | (set(formal_ids) - set(index))),
        "duplicate_request_ids": recorded.get("duplicates", []),
        "unclassified_request_ids": sorted(set(recorded.get("unclassified", [])) | (set(index) - set(formal_ids))),
        "clock_unit_violations": sorted(index[clock.to_numpy()]),
    }
    return {kind: ids for kind, ids in gaps.items() if ids}


def verdict(statuses: list[str]) -> str:
    if statuses and all(status == "PASS" for status in statuses):
        return "PASS"
    return "FAIL" if "FAIL" in statuses else "INSUFFICIENT_EVIDENCE"


def admission(sample: dict, sides: dict[str, pd.DataFrame], routing_status: str) -> dict:
    formal_ids = sample["formal_request_ids"]
    gaps = {side: found for side, rows in sides.items()
            if (found := evidence_gaps(rows, formal_ids, sample["sides"][side]))}
    vllm_sides = [side for side in sides if side.startswith("vllm/")]
    mismatches = {}
    for side in sides:
        if side.startswith("frontier/"):
            found = sorted({request_id for vllm in vllm_sides
                            for request_id in token_count_mismatches(sides[vllm], sides[side])})
            if found:
                mismatches[side] = found
    blocked = ROUTING_GATE[routing_status] or ("INSUFFICIENT_EVIDENCE" if gaps or mismatches else None)
    return {"routing_status": routing_status, "request_id_gaps": gaps, "token_count_mismatches": mismatches,
            "blocked_status": blocked}


def s33_vllm_ttft(rows: pd.DataFrame) -> tuple[tuple, dict]:
    bound = float(rows["endpoint_offset_proxy_api_ms"].min())
    count = len(rows)
    numerator = float(rows["ttft_ms"].sum()) - bound * count
    return (numerator, count, count), {"decision": "S33-TTFT (ii)", "bound": "endpoint_offset_proxy_api_ms.min",
                                       "bound_ms": bound, "vllm_ttft_mean_ms": float(rows["ttft_ms"].mean()),
                                       "vllm_ttft_adjusted_ms": numerator / count}


def single_reading(cell: dict, sample: dict, sides: dict[str, pd.DataFrame], checks: dict) -> dict:
    [vllm_side] = [side for side in sides if side.startswith("vllm/")]
    [frontier_side] = [side for side in sides if side.startswith("frontier/")]
    vllm, frontier = sides[vllm_side], sides[frontier_side]
    vllm_metrics, frontier_metrics = metric_values(vllm), metric_values(frontier)
    reading = {}
    if cell["gate"] == "pd_s33":
        vllm_metrics["ttft_ms"], reading["ttft_s33"] = s33_vllm_ttft(vllm)
    count = len(sample["formal_request_ids"])
    metrics = [metric_row(metric, frontier_side, vllm_metrics[metric], frontier_metrics[metric], count,
                          checks["blocked_status"]) for metric in vllm_metrics]
    return {**reading, "metrics": metrics, "verdict": verdict([row["status"] for row in metrics])}


def mode_pairs(sides: dict[str, pd.DataFrame]) -> dict[tuple[str, str], dict]:
    """Per (mode, metric), the values of the requests that reached that mode on both sides."""
    vllm_sides = [side for side in sides if side.startswith("vllm/")]
    members = [side for side in sides if side.startswith("frontier/")]
    pairs: dict[tuple[str, str], dict] = {}
    for run in vllm_sides:
        for member in members:
            joined = sides[run].join(sides[member], rsuffix="_member", how="inner")
            for metric in MODE_METRICS:
                mode_column = metric.replace("_ms", "_mode")
                agree = joined[joined[mode_column].notna() & (joined[mode_column] == joined[f"{mode_column}_member"])]
                if metric == "tpot_ms":
                    agree = agree[agree["request_num_decode_tokens"] > 1]
                for request_id, row in agree.iterrows():
                    entry = pairs.setdefault((row[mode_column], metric), {"vllm": [], "frontier": [], "requests": set()})
                    entry["vllm"].append(row[metric])
                    entry["frontier"].append(row[f"{metric}_member"])
                    entry["requests"].add(request_id)
    return pairs


def dp_pp_reading(cell: dict, sample: dict, sides: dict[str, pd.DataFrame], checks: dict) -> dict:
    vllm_sides = [side for side in sides if side.startswith("vllm/")]
    members = [side for side in sides if side.startswith("frontier/")]
    vllm_windows = Counter()
    for side in vllm_sides:
        vllm_windows.update(sample["sides"][side]["windows"])
    share = {mode: count / sum(vllm_windows.values()) for mode, count in sorted(vllm_windows.items())}
    reference_modes = sorted(mode for mode, value in share.items() if value >= REQUIRED_MODE_SHARE)
    per_mode, gated = {}, []
    for (mode, metric), entry in sorted(mode_pairs(sides).items()):
        vllm, frontier = statistics.mean(entry["vllm"]), statistics.mean(entry["frontier"])
        relative_error = abs(frontier - vllm) / abs(vllm) if vllm != 0 else None
        reading = {"vllm": vllm, "frontier": frontier, "relative_error": relative_error,
                   "pairs_n": len(entry["vllm"]), "requests_n": len(entry["requests"]),
                   "gated": len(entry["requests"]) >= MIN_MODE_REQUESTS}
        if reading["gated"]:
            reading["status"] = checks["blocked_status"] or (
                "INSUFFICIENT_EVIDENCE" if relative_error is None
                else "PASS" if relative_error <= RELATIVE_ERROR_LIMIT else "FAIL")
            gated.append(reading["status"])
        per_mode.setdefault(mode, {})[metric] = reading
    cell_metrics = {}
    vllm_values = [metric_values(sides[side]) for side in vllm_sides]
    member_values = [metric_values(sides[side]) for side in members]
    for metric in vllm_values[0]:
        vllm = statistics.mean(numerator / denominator for numerator, denominator, _ in
                               (values[metric] for values in vllm_values))
        frontier = statistics.mean(numerator / denominator for numerator, denominator, _ in
                                   (values[metric] for values in member_values))
        cell_metrics[metric] = {"vllm": vllm, "frontier": frontier,
                                "relative_error": abs(frontier - vllm) / abs(vllm) if vllm else None}
    return {
        "dp_pp_modes": {
            "decision": "T43-PAIRGATE", "mode_window_s": cell["dp_pp_gate"]["mode_window_s"],
            "windows": {side: sample["sides"][side]["windows"] for side in sides},
            "vllm_window_share": share, "reference_modes": reference_modes,
            "multi_mode_reference": len(reference_modes) > 1,
            "reference_modes_without_gated_pairs": [mode for mode in reference_modes
                                                    if not any(r["gated"] for r in per_mode.get(mode, {}).values())],
            "mode_outcome": "paired" if gated else "empty_pair_set",
            "modes": per_mode,
        },
        "cell_metrics_reported": cell_metrics,
        "verdict": verdict(gated) if gated else "INSUFFICIENT_EVIDENCE",
    }


def gate_reading(cell: dict, case: dict, rows_dir: Path) -> dict:
    sample, sides = read_rows(rows_dir)
    if sample["cell_id"] != cell["cell_id"]:
        raise ValueError(f"{rows_dir} holds the rows of {sample['cell_id']}, not {cell['cell_id']}")
    for side, rows in sides.items():
        missing = set(NORMALIZED_COLUMNS[1:]) - set(rows.columns)
        if missing:
            raise ValueError(f"{rows_dir / side}.csv lacks the normalized columns {sorted(missing)}")
    checks = admission(sample, sides, case["routing"]["status"])
    reading = (dp_pp_reading if cell["gate"] == "dp_pp_pairs" else single_reading)(cell, sample, sides, checks)
    return {"cell_id": cell["cell_id"], "gate": cell["gate"], "relative_error_limit": RELATIVE_ERROR_LIMIT,
            "ttft_semantic": "request_arrival_to_prefill_completion", "sides": sorted(sides),
            "formal_request_count": len(sample["formal_request_ids"]), **checks, **reading}

