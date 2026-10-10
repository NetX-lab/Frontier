"""Admission of a CPU-probe vLLM run into a published CPU-overhead table.

A CPU-probe run stamps every engine step, so its host must have run as fast as a clean run's for
its terms to stand for the clean run's host work. Each rule compares a probe run with runs of the
same job (one node), within ACCEPTED:

- ``client_e2e`` (issues.md I-6): mean client E2E (response - dispatch) of the formal requests,
  against the clean run of the same cell.
- ``send_span`` (decision T43-C4PROBE, PD prefill instances; with ``client_e2e``): the median over
  the isolated single-request prefills of the summed per-layer KV-send gaps, against the mean of
  the job's clean runs of the cell. The sends are stamped when each layer's attention saves its KV,
  so their span follows the host's launch progress through the forward.
- ``isolated_prefill`` (decision T43-ISOPREFILL, DP x PP cells whose mode is not reproducible): the
  first formal request's isolated prefill step (its scheduling step's end to the engine's next
  step end), against the median of the job's CPU-probe cohort. The probes' own stamping cost is
  the same for all of them, so the clean runs are not the reference.

Run directories are a ground-truth run's ``runs/groundtruth_<mode>/<tag>/``.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

ACCEPTED = (0.97, 1.03)
# Two prefills are isolated when their send intervals are this far apart (s).
ISOLATION_GAP_S = 0.002


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean_client_e2e_ms(run_dir: Path) -> float:
    formal = [r for r in read_jsonl(run_dir / "run" / "client_requests.jsonl") if r["segment"] != "warmup"]
    return 1000.0 * statistics.fmean(r["response_monotonic"] - r["dispatch_monotonic"] for r in formal)


def send_span_ms(run_dir: Path) -> float:
    sends: dict[str, list[float]] = defaultdict(list)
    for record in read_jsonl(run_dir / "run" / "prefill" / "kv_transfer.jsonl"):
        if record["event"] == "producer_layer_send_start":
            sends[record["request_id"]].append(record["timestamp"])
    layers = max(len(stamps) for stamps in sends.values())
    intervals = sorted((min(stamps), max(stamps), np.diff(sorted(stamps)) * 1e3)
                       for stamps in sends.values() if len(stamps) == layers)
    isolated = [gaps for i, (first, last, gaps) in enumerate(intervals)
                if (i == 0 or first > intervals[i - 1][1] + ISOLATION_GAP_S)
                and (i + 1 == len(intervals) or intervals[i + 1][0] > last + ISOLATION_GAP_S)]
    return float(np.median([gaps.sum() for gaps in isolated]))


def isolated_prefill_ms(run_dir: Path) -> float:
    client = read_jsonl(run_dir / "run" / "client_requests.jsonl")
    first = min((r for r in client if r["role"] == "formal"), key=lambda r: r["dispatch_monotonic"])
    request_id = first["engine_request_id"]
    records = sorted((record for log in (run_dir / "run" / "dp_placement").glob("*.jsonl")
                      for record in read_jsonl(log)), key=lambda r: r["monotonic"])
    scheduled = next(r for r in records if r["kind"] == "engine_iteration"
                     and request_id in (r.get("scheduled_new_req_ids") or []))
    following = next(r for r in records if r["kind"] == "engine_iteration"
                     and r["engine"] == scheduled["engine"] and r["monotonic"] > scheduled["monotonic"])
    return 1e3 * (following["monotonic"] - scheduled["monotonic"])


def decision(rule: str, probe_value: float, reference_value: float, reference: dict) -> dict:
    ratio = probe_value / reference_value
    return {"rule": rule, "probe": probe_value, "reference": reference_value, "reference_values": reference,
            "ratio": ratio, "accepted": list(ACCEPTED), "passed": ACCEPTED[0] <= ratio <= ACCEPTED[1]}


def client_e2e(probe: Path, references: list[Path]) -> dict:
    [clean] = references
    return decision("client_e2e", mean_client_e2e_ms(probe), mean_client_e2e_ms(clean),
                    {clean.name: mean_client_e2e_ms(clean)})


def send_span(probe: Path, clean_runs: list[Path]) -> dict:
    spans = {run.name: send_span_ms(run) for run in clean_runs}
    return decision("send_span", send_span_ms(probe), statistics.fmean(spans.values()), spans)


def isolated_prefill(probe: Path, cohort: list[Path]) -> dict:
    prefills = {run.name: isolated_prefill_ms(run) for run in cohort}
    if probe.name not in prefills:
        raise ValueError(f"{probe.name} is not in its probe cohort {sorted(prefills)}")
    return decision("isolated_prefill", prefills[probe.name], statistics.median(prefills.values()), prefills)


RULES = {"client_e2e": client_e2e, "send_span": send_span, "isolated_prefill": isolated_prefill}
