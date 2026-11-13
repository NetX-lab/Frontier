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

from tests.integration.step5_pd.formal_release import DIAGNOSTIC_KINDS

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


def client_reference(cell: dict, case: dict, run_dir: Path, run_check: Path | None) -> dict:
    """Bind the existing clean-run checker to the actual client input and runtime evidence."""
    evidence = dict(status="UNVERIFIED", reasons=[], inputs={})
    reasons = evidence["reasons"]
    if run_check is None:
        reasons.append("Missing pinned groundtruth-run-check receipt")
        return evidence
    receipt = json.loads(run_check.read_text())
    origin = run_check.resolve().parent.parent
    check = receipt.get("runs", {}).get(run_dir.name, {})
    if (run_check.resolve() != origin / "analysis/run_check_status.groundtruth_clean.json"
            or receipt.get("entry") != "groundtruth-run-check"
            or receipt.get("mode") != "groundtruth_clean" or receipt.get("case_id") != cell["case_id"]
            or check.get("status") != "PASS" or not check.get("checks")
            or not all(value.startswith("PASS") for value in check["checks"].values())):
        reasons.append("Missing passing case-bound clean-run check")
        return evidence
    if case.get("native_client_source", "vllm028") == "vllm028":
        expected = f"runs/groundtruth_clean/{run_dir.name}/run_manifest.json"
        if (check.get("run_manifest") == expected
                and (origin / expected).resolve() == (run_dir / "run_manifest.json").resolve()):
            evidence["status"] = "PASS"
        else:
            reasons.append("Native run-check origin does not identify the input manifest")
        return evidence

    # Image-native runs have no clean native Git checkout. The original stage declaration,
    # actual worker receipts and the post-run check replace that unsupported claim.
    expected_runtime = case.get("native_runtime")
    pins = check.get("artifacts", [])
    runtime_evidence = check.get("runtime_evidence", {})
    required_checks = {"clean_runtime", "runtime_identity", "native_qualification", "artifact_integrity"}
    if (not expected_runtime or not required_checks <= check["checks"].keys()
            or set(runtime_evidence) != {"platform_image", "installed_sources", "native_qualification"}
            or check.get("run_directory") is None
            or (origin / check["run_directory"]).resolve() != run_dir.resolve() or not pins):
        reasons.append("Missing Step runtime declaration, origin or required native checks")
        return evidence
    artifacts = {}
    for pin in pins:
        raw = observer_pin(pin | {"path": str(origin / pin["path"])}, run_check, reasons)
        path = (origin / pin["path"]).resolve()
        if path in artifacts or not path.is_relative_to(origin):
            reasons.append("FAIL: duplicate or external native evidence pin")
        artifacts[path] = raw
    if any(pin not in pins for pin in runtime_evidence.values()):
        reasons.append("Missing immutable actual platform/source/qualification receipts")
    required = [run_dir / "config.json", run_dir / "replay/client_requests.jsonl",
                run_dir / "replay/canonical_summary.json"]
    nodes = sorted((run_dir / "nodes").glob("*/result.json"))
    if any(artifacts.get(path.resolve()) is None for path in required + nodes) or not nodes:
        reasons.append("Missing pinned stage, client population or native node receipts")
    if reasons:
        return evidence
    config = json.loads(artifacts[required[0].resolve()])
    kinds = {config.get("canonical_replay", {}).get("kind"),
             json.loads(artifacts[required[2].resolve()]).get("replay_kind")}
    if kinds & DIAGNOSTIC_KINDS or "host_step_probe" in config:
        reasons.append("A diagnostic run perturbs native timing and is never a clean reference")
    identity = expected_runtime | {"source_hashes": config.get("expected_source_hashes"),
                                    "harness_commit": config.get("source", {}).get("commit")}
    topology = case.get("native_topology")
    if topology is not None:
        identity["topology"] = topology
        engine = config.get("engine", {})
        observed = {key: config.get(key) for key in ("gpu_type", "replica", "gpu", "data_parallel_size_local")}
        observed.update({key: engine.get(key) for key in (
            "tensor_parallel_size", "data_parallel_size", "pipeline_parallel_size")})
        observed["expert_parallel_size"] = (engine.get("tensor_parallel_size", 0)
                                            * engine.get("data_parallel_size", 0)
                                            if engine.get("enable_expert_parallel") else 1)
        if observed != topology or config.get("case_id", cell["case_id"]) != cell["case_id"]:
            reasons.append("Step hardware/topology declaration disagrees with case")
    if (config.get("mode") != "clean" or config.get("image") != expected_runtime.get("image")
            or config.get("engine", {}).get("native_reference", {}).get("image") != config.get("image")
            or config.get("engine", {}).get("vllm_version") != expected_runtime.get("adapter")
            or not identity["source_hashes"] or not identity["harness_commit"]
            or check.get("runtime_identity") != identity
            or any(key in config for key in ("pd_state_observation", "serve_observers"))
            or len(nodes) != config.get("replica")):
        reasons.append("Step stage/source/image declaration disagrees with native run check")
    lanes = []
    for path in nodes:
        node = json.loads(artifacts[path.resolve()])
        plan = node.get("plan", {})
        lanes.append((plan.get("role"), plan.get("lane")))
        if topology is not None:
            devices = [line for line in node.get("nvidia_smi", "").splitlines() if line.startswith("GPU ")]
            node_engine = plan.get("engine", {})
            if (len(devices) != topology["gpu"] or any(topology["gpu_type"] not in line for line in devices)
                    or any(node_engine.get(key) != topology[key] for key in (
                        "tensor_parallel_size", "data_parallel_size", "pipeline_parallel_size"))
                    or node_engine.get("enable_expert_parallel") is not True):
                reasons.append(f"Step actual GPU/role topology mismatch: {path}")
        if (node.get("stage") != "finished" or node.get("workload_returncode") != 0
                or node.get("observation_errors") or node.get("exception")
                or "STEP5_HOST_STEP_PROBE" in node.get("server_env_set", {})
                or node.get("versions", {}).get("vllm") != expected_runtime.get("vllm_version")
                or plan.get("engine", {}).get("native_reference") != config["engine"].get("native_reference")):
            reasons.append(f"Unverified actual native runtime: {path}")
    expected_lanes = [(role, lane) for role in ("prefill", "decode")
                      for lane in range(0, config["engine"]["data_parallel_size"], config["data_parallel_size_local"])]
    if sorted(lanes) != sorted(expected_lanes):
        reasons.append("Native role/lane runtime population mismatch")
    evidence["inputs"] = dict(run_check=str(run_check.resolve()), runtime_identity=identity, artifacts=pins)
    if not reasons:
        evidence["status"] = "PASS"
    return evidence


def step_frontier_runtime(case: dict, run_dir: Path) -> dict:
    """Bind a Step case to its original resolved Frontier P/D shape, not a name switch."""
    [truth] = (run_dir / "metrics").rglob("metrics_ground_truth.jsonl")
    path = truth.with_name("config.json")
    config = json.loads(path.read_text())
    topology = case["native_topology"]
    cluster = config.get("cluster_config", {})
    if config.get("sys_arch") != case["architecture"]:
        raise ValueError("Step Frontier architecture does not match the case")
    expected = dict(device=topology["gpu_type"].lower(),
                    model_name=case["frontier_model_name"],
                    attn_tensor_parallel_size=topology["tensor_parallel_size"],
                    attn_dp=topology["data_parallel_size"],
                    moe_expert_parallel_size=topology["expert_parallel_size"],
                    moe_tensor_parallel_size=1, num_pipeline_stages=topology["pipeline_parallel_size"])
    for role in ("prefill", "decode"):
        replica = cluster.get(f"{role}_replica_config", {})
        if (cluster.get(f"{role}_cluster_num_replicas") != 1
                or any(replica.get(key) != value for key, value in expected.items())):
            raise ValueError(f"Step Frontier {role} hardware/model/topology does not match the case")
    return dict(config_path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                topology=topology, model_name=case["frontier_model_name"])


RULES = {"client_e2e": client_e2e, "send_span": send_span,
         "isolated_prefill": isolated_prefill, "observer_host": observer_host}
