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

import hashlib
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


def observer_pin(pin: dict, document: Path, reasons: list[str]) -> bytes | None:
    if (type(pin) is not dict or set(pin) != {"path", "sha256"}
            or type(pin["path"]) is not str or not pin["path"]
            or type(pin["sha256"]) is not str or len(pin["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in pin["sha256"])):
        raise ValueError(f"{document}: expected a path/SHA-256 pin")
    path = document.parent / pin["path"]
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        reasons.append(f"INCOMPLETE: missing {path}")
        return None
    if hashlib.sha256(raw).hexdigest() != pin["sha256"]:
        reasons.append(f"FAIL: SHA-256 mismatch {path}")
        return None
    return raw


def observer_envelope(raw: bytes, observer: dict | None, reasons: list[str]) -> None:
    lines, rows = raw.splitlines(keepends=True), []
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            prefix = "INCOMPLETE" if not line.endswith(b"\n") else "FAIL"
            reasons.append(f"{prefix}: JSON line {number}")
            break
        if type(row) is not dict:
            reasons.append(f"FAIL: object required on line {number}")
            break
        rows.append(row)
    if not rows or rows[0].get("kind") != "manifest":
        reasons.append("INCOMPLETE: missing first manifest")
        return
    manifest = rows[0]
    common = "run_id source_sha instrumentation_sha process_instance".split()
    common += ["clock_domain", "clock_unit"]
    invalid = any(type(manifest.get(k)) is not str for k in common +
                  "node config_id replica_id role mode coverage".split())
    invalid |= manifest.get("clock_unit") != "ns" or manifest.get(
        "clock_domain") != f"{manifest.get('process_instance')}:monotonic"
    invalid |= type(manifest.get("schema_version")) is not int or manifest.get(
        "schema_version") != 1
    if observer is not None:
        invalid |= any(manifest.get(k) != observer[k] for k in common[:3] + [
            "config_id", "replica_id", "role", "expected_ranks"])
    if manifest.get("mode") != "light" or manifest.get("coverage") not in (
            "engine_transport_only", "worker_host_transport_only"):
        reasons.append("INCOMPLETE: unsupported recorder mode or coverage")
    if rows[-1].get("kind") != "footer":
        reasons.append("INCOMPLETE: missing final footer")
    footer = rows[-1] if rows[-1].get("kind") == "footer" else None
    records = rows[1:-1] if footer else rows[1:]
    invalid |= any(any(r.get(k) != manifest.get(k) for k in common) for r in rows[1:])
    sequence = [r.get("sequence") for r in records]
    invalid |= any(type(s) is not int or s <= 0 for s in sequence)
    invalid |= any(type(r.get("time_ns")) is not int or type(r.get("kind")) is not str
                   or r.get("kind") in ("manifest", "footer") for r in records)
    invalid |= any(type(manifest.get(k)) is not int or manifest.get(k, -1) < minimum
                   for k, minimum in (("capacity_bytes", 1), ("pid", 1),
                                      ("engine_request_owner", 0)))
    invalid |= type(manifest.get("expected_ranks")) is not list
    if type(manifest.get("expected_ranks")) is list:
        invalid |= any(type(r) is not int for r in manifest["expected_ranks"])
    if not invalid:
        invalid |= any(a >= b for a, b in zip(sequence, sequence[1:]))
        invalid |= sum(map(len, lines[1:1 + len(records)])) > manifest["capacity_bytes"]
    if footer is not None:
        counts = ("final_sequence step_count record_count "
                  "drop_count pending_steps").split()
        invalid |= any(type(footer.get(k)) is not int or footer.get(k, -1) < 0
                       for k in counts)
        invalid |= footer.get("record_count") != len(records)
        invalid |= type(footer.get("final_sequence")) is int and any(
            type(s) is int and s > footer["final_sequence"] for s in sequence)
        if not footer.get("drop_count") and footer.get("observer_error") is None:
            invalid |= sequence != list(range(1, len(records) + 1))
            invalid |= footer.get("final_sequence") != len(records)
    if invalid:
        reasons.append("FAIL: process identity, sequence, capacity or footer counts")


def observer_host(probe: Path, references: list[Path]) -> dict:
    if len(references) != 1:
        raise ValueError("observer_host requires exactly one reference")
    reference = references[0]
    documents, inputs = [], {}
    for name, path in (("probe", probe), ("reference", reference)):
        raw = path.read_bytes()
        documents.append(json.loads(raw))
        inputs[name] = {"path": str(path.resolve()),
                        "sha256": hashlib.sha256(raw).hexdigest()}
    index, expected = documents
    if (type(index) is not dict or set(index) != {"files"} or type(expected) is not dict
            or set(expected) != {"observer", "artifacts", "engine_owners",
                                 "worker_ranks"}
            or type(index["files"]) is not list
            or type(expected["artifacts"]) is not list):
        raise ValueError("observer_host: invalid trace index or expected document")
    reasons = ["INCOMPLETE: inventory, causal and lifecycle checks deferred to A2"]
    inputs.update(observer=expected["observer"], artifacts=expected["artifacts"])
    raw = observer_pin(expected["observer"], reference, reasons)
    try:
        observer = json.loads(raw) if raw is not None else None
    except (ValueError, UnicodeError):
        observer = None
        reasons.append("FAIL: invalid pinned observer JSON")
    if raw is not None and (type(observer) is not dict or set(observer) != {
            "run_id", "source_sha", "instrumentation_sha", "config_id", "replica_id",
            "role", "expected_ranks", "output_dir"}):
        reasons.append("FAIL: invalid observer fields")
        observer = None
    for pin in expected["artifacts"]:
        observer_pin(pin, reference, reasons)
    files = []
    for pin in index["files"]:
        local = []
        raw = observer_pin(pin, probe, local)
        if raw is not None:
            observer_envelope(raw, observer, local)
        status = "FAIL" if any(r.startswith("FAIL:") for r in local) else (
            "INSUFFICIENT_EVIDENCE")
        files.append({**pin, "status": status, "reasons": local})
    reasons.extend(r for f in files for r in f["reasons"])
    status = "FAIL" if any(r.startswith("FAIL:") for r in reasons) else (
        "INSUFFICIENT_EVIDENCE")
    return dict(rule="observer_host", status=status, structural_status=status,
                coverage_status="INSUFFICIENT_EVIDENCE", files=files,
                reasons=reasons, inputs=inputs, qualification_verified=False)


RULES = {"client_e2e": client_e2e, "send_span": send_span,
         "isolated_prefill": isolated_prefill, "observer_host": observer_host}
