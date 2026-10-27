"""Release formal native cases from existing case-bound checks and client-gate artifacts.

The 30% dependency reads the existing normalizer and comparator, not user-authored
scalar errors or a readiness boolean. Missing evidence blocks formal submission only.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile

CASES = json.loads(Path(__file__).with_name("priority_cases.json").read_text())
MATRIX = Path(__file__).resolve().parents[2] / "comparison/e2e_gate/cases/step5_priority.json"
METRICS = {"ttft_ms", "tpot_ms", "request_e2e_time_ms", "token_throughput_tps"}
DEPENDENCY_ERROR_LIMIT = .30
COLLECTION_KINDS = {
    "qualification": "Basic qualification has no accuracy dependency",
    "workload": "Full native workload collection; calibration readiness and accuracy remain unverified",
    "device_timeline": "Diagnostic decode device timeline; profiler windows perturb native timing, never a reference",
    "host_probe": "Diagnostic host-step probe; probe hooks perturb native timing, never a reference",
}


def pin(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def pinned(record: dict, root=Path("/")) -> Path:
    path = (root / record["path"]).resolve()
    if pin(path)["sha256"] != record["sha256"]:
        raise ValueError(f"Formal prerequisite artifact hash mismatch: {path}")
    return path


def declaration(case_id: str) -> tuple[dict, dict]:
    matrix = json.loads(MATRIX.read_text())
    return next(cell for cell in matrix["cells"] if cell["case_id"] == case_id), matrix["cases"][case_id]


def native_evidence(config: dict, case_id: str, inputs: dict) -> tuple[dict, Path]:
    from tests.comparison.e2e_gate.admission import client_reference

    run = Path(inputs["run_directory"]).resolve()
    receipt = pinned(inputs["run_check"])
    cell, case = declaration(case_id)
    evidence = client_reference(cell, case, run, receipt)
    if evidence["status"] != "PASS":
        raise ValueError(f"Formal {case_id} native readiness unverified: {evidence['reasons']}")
    observed = json.loads((run / "config.json").read_text())
    if any(observed.get(key) != config.get(key) for key in ("image", "model_hashes", "expected_source_hashes")):
        raise ValueError("Formal prerequisite native image/model/source identity mismatch")
    # Actual qualification disposition belongs to the existing clean-run checker.
    # A successful client summary alone is never readiness evidence.
    return evidence, run


def accuracy_dependency(config: dict, case_id: str, inputs: dict, output: Path) -> dict:
    from tests.comparison.e2e_gate.gate import gate_reading
    from tests.comparison.e2e_gate.normalize import normalize_cell
    from tests.integration.step5_pd.canonical_replay import canonical_workload, MANIFEST_SHA256

    matrix_path = pinned(inputs["matrix"])
    matrix = json.loads(matrix_path.read_text())
    expected_cell, expected_case = declaration(case_id)
    cells = [cell for cell in matrix["cells"] if cell["case_id"] == case_id]
    if len(cells) != 1:
        raise ValueError("Formal dependency requires the same-hardware 32-GPU case")
    cell, case = cells[0], matrix["cases"][case_id]
    if (cell["cell_id"] != expected_cell["cell_id"] or cell.get("gate") != "client"
            or cell.get("status") != "claimed" or any(case.get(key) != expected_case[key] for key in (
                "architecture", "native_client_source", "native_runtime", "native_topology", "frontier_model_name"))):
        raise ValueError("Formal dependency case/runtime/topology identity mismatch")
    root = Path(inputs["artifacts_root"]).resolve()
    workload_path = pinned(cell["trace"]["workload"], root)
    if cell["trace"]["workload"]["sha256"] != MANIFEST_SHA256:
        raise ValueError("Formal dependency workload identity mismatch")
    workload = canonical_workload(workload_path.parent)
    ids_path = pinned(cell["trace"]["request_ids"], root)
    if ids_path.read_bytes() != (workload_path.parent / "request_ids.json").read_bytes():
        raise ValueError("Formal dependency request population differs from frozen workload")
    run_check = pinned(cell["run_check"], root)
    evidence, run = native_evidence(config, case_id, {
        "run_directory": inputs["run_directory"], "run_check": pin(run_check)})
    frontier = Path(inputs["frontier_run_directory"]).resolve()
    [truth] = (frontier / "metrics").rglob("metrics_ground_truth.jsonl")
    expected = {truth.resolve(), truth.with_name("config.json").resolve()}
    if {pinned(record) for record in inputs["frontier_artifacts"]} != expected:
        raise ValueError("Formal dependency must pin original Frontier configuration and request truth")
    normalize_cell(cell, case, json.loads(ids_path.read_text()), {run.name: run},
                   {"measured": frontier}, output,
                   workload, run_check)
    reading = gate_reading(cell, case, output)
    if (reading["blocked_status"] is not None or reading.get("client_metric_gaps")
            or reading["formal_request_count"] != 128 or len(reading["repetitions"]) != 3):
        raise ValueError("Formal dependency lacks three admitted complete clean repetitions")
    for repeat in reading["repetitions"].values():
        rows = repeat["metrics"]
        if (len(rows) != 4 or {row["metric"] for row in rows} != METRICS
                or any(row["status"] not in ("PASS", "FAIL")
                       or not isinstance(row["relative_error"], (int, float))
                       or not math.isfinite(row["relative_error"])
                       or not 0 <= row["relative_error"] <= DEPENDENCY_ERROR_LIMIT for row in rows)):
            raise ValueError("Formal dependency requires all four errors <=30% in every clean repeat")
    return {"case_id": case_id, "error_limit": DEPENDENCY_ERROR_LIMIT,
            "reading": reading, "native_evidence": evidence, "matrix": pin(matrix_path)}


def validate(config: dict, *, output: Path | None = None) -> dict:
    """Check before staging and again before SDK construction; never a training hard gate."""
    kind = config.get("canonical_replay", {}).get("kind")
    if kind is None:
        return {"status": "NOT_REQUIRED", "reason": "Non-canonical diagnostic run"}
    if kind in COLLECTION_KINDS:
        return {"status": "NOT_REQUIRED", "reason": COLLECTION_KINDS[kind]}
    if kind != "canonical":
        raise ValueError(f"Unsupported native replay kind: {kind}")
    inputs = config.get("formal_inputs")
    if not inputs or "readiness" not in inputs:
        raise ValueError("Formal submission requires explicit case-bound native readiness artifacts")
    case_id = config["case_id"]
    readiness, _ = native_evidence(config, case_id, inputs["readiness"])
    result = {"status": "PASS", "case_id": case_id, "readiness": readiness}
    dependency = CASES[case_id]["formal_dependency"]
    if dependency:
        if "accuracy" not in inputs:
            raise ValueError("Formal64 requires same-hardware32 three-repeat four-metric accuracy artifacts")
        if output is None:
            with tempfile.TemporaryDirectory(prefix="step5_formal_", dir=config["workspace"]) as scratch:
                result["accuracy"] = accuracy_dependency(config, dependency, inputs["accuracy"], Path(scratch) / "rows")
        else:
            result["accuracy"] = accuracy_dependency(config, dependency, inputs["accuracy"], output)
    return result
