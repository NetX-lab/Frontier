"""Step raw-schema fixtures exercise the real CLI; no fixture qualifies native ground truth."""

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.comparison.calibration.client_metrics import native_client_rows
from tests.comparison.e2e_gate import run_gate
from tests.comparison.e2e_gate.gate import gate_reading
from tests.comparison.e2e_gate.normalize import normalize_cell
from tests.unit.test_canonical_client_metrics import IDS, TRACE, WORKLOAD, native, simulator, write_jsonl

MATRIX = Path(run_gate.__file__).parent / "cases/step5_canonical.json"
DECLARATION = json.loads(MATRIX.read_text())
CELL = DECLARATION["cells"][0]
CASE = DECLARATION["cases"][CELL["case_id"]]


def pin(path, root):
    return dict(path=str(path.relative_to(root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def step_native(native):
    root, records, glm_summary = native
    run = root / "replay"
    run.mkdir()
    summary = copy.deepcopy(glm_summary["canonical_repetitions"])
    summary.update(status="CLIENT_WORKLOAD_VERIFIED", reference_admission="UNVERIFIED",
                   runtime_qualification="UNVERIFIED",
                   clock="prefill-head Linux monotonic; proxy and client on the same host")
    for repetition in summary["repetitions"]:
        repeat = repetition["repeat"]
        repetition.update(client_workload_verified=True, phases={phase: dict(
            origin_monotonic=repeat * 1000 + (100 if phase == "formal" else 0),
            planned_requests=128 if phase == "formal" else 16,
            recorded_requests=128 if phase == "formal" else 16, failed_request_ids=[])
            for phase in ("warmup", "formal")})
    for row in records:
        row["stream_completed"] = row.pop("stream_done")
        row["generated_token_count"] = row.pop("output_token_count")
        row.update(accepted_prompt_token_count=6144, prompt_sha256="a" * 64,
                   accepted_prompt_sha256="a" * 64, response_monotonic=row["last_token_monotonic"] + .01)
    write_jsonl(run / "client_requests.jsonl", records)
    (run / "canonical_summary.json").write_text(json.dumps(summary))
    return root, records, summary


def synthetic_step_check(root):
    """An explicit synthetic oracle of the receipt contract, never an actual native receipt."""
    origin = root.parents[2]
    runtime = CASE["native_runtime"]
    engine = dict(vllm_version="step5", native_reference=dict(image=runtime["image"]), data_parallel_size=4)
    config = dict(mode="clean", image=runtime["image"], engine=engine, replica=8,
                  data_parallel_size_local=1, source=dict(commit="fixture-not-native"),
                  expected_source_hashes={"vllm/models/step5/nvidia/model.py": "b" * 64})
    (root / "config.json").write_text(json.dumps(config))
    for role in ("prefill", "decode"):
        for lane in range(4):
            path = root / f"nodes/{role}{lane}/result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(stage="finished", workload_returncode=0, observation_errors=[],
                versions=dict(vllm=runtime["vllm_version"]), plan=dict(role=role, lane=lane, engine=engine))))
    receipt = origin / "analysis/run_check_status.groundtruth_clean.json"
    receipt.parent.mkdir(exist_ok=True)
    files = [root / "config.json", root / "replay/client_requests.jsonl", root / "replay/canonical_summary.json",
             *sorted((root / "nodes").glob("*/result.json"))]
    actual_receipts = {}
    for name in ("platform_image", "installed_sources", "native_qualification"):
        path = root / f"{name}.json"
        path.write_text(json.dumps(dict(evidence="SYNTHETIC FIXTURE ONLY, NOT NATIVE", kind=name)))
        actual_receipts[name] = pin(path, origin)
        files.append(path)
    check = dict(status="PASS", run_directory=str(root.relative_to(origin)),
                 checks=dict.fromkeys(("clean_runtime", "runtime_identity", "native_qualification", "artifact_integrity"),
                                      "PASS: synthetic fixture only"),
                 runtime_identity=runtime | dict(source_hashes=config["expected_source_hashes"], harness_commit="fixture-not-native"),
                 runtime_evidence=actual_receipts,
                 artifacts=[pin(path, origin) for path in files])
    receipt.write_text(json.dumps(dict(entry="groundtruth-run-check", mode="groundtruth_clean",
                                      case_id=CELL["case_id"], runs={root.name: check})))
    return receipt


def test_step_cli_computes_four_metrics_without_masquerading_as_vllm028(tmp_path, step_native, simulator):
    root, _, _ = step_native
    matrix = copy.deepcopy(DECLARATION)
    # NOT_APPLICABLE is an arithmetic-oracle setting, not the real Step routing claim.
    matrix["cases"][CELL["case_id"]]["routing"] = dict(status="NOT_APPLICABLE")
    receipt = synthetic_step_check(root)
    matrix["cells"][0]["run_check"] = pin(receipt, tmp_path)
    for record in matrix["cells"][0]["trace"].values():
        record["path"] = str(TRACE / Path(record["path"]).name)
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix))
    rows = tmp_path / "rows"
    common = [sys.executable, str(Path(run_gate.__file__))]
    result = subprocess.run(common + ["normalize", "--matrix", str(matrix_path), "--cell", CELL["cell_id"],
        "--artifacts-root", str(tmp_path), "--vllm-run-dir", str(root), "--frontier-run", f"sim={simulator}",
        "--output-dir", str(rows)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(common + ["gate", "--matrix", str(matrix_path), "--cell", CELL["cell_id"],
        "--rows", str(rows), "--output", str(tmp_path / "reading.json")], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    reading = json.loads((tmp_path / "reading.json").read_text())
    assert reading["verdict"] == "PASS"  # Synthetic arithmetic/admission oracle only.
    assert len(reading["repetitions"]) == 3
    expected = dict(ttft_ms=2000., tpot_ms=2., request_e2e_time_ms=4046., token_throughput_tps=917504 / 131.046)
    for repetition in reading["repetitions"].values():
        for row in repetition["metrics"]:
            assert row["vllm"] == pytest.approx(expected[row["metric"]])
            assert row["frontier"] == pytest.approx(expected[row["metric"]])
    assert "pooled_diagnostic" in reading
    assert all(w["total_tokens"] == 917504 for w in reading["formal_windows"].values())


@pytest.mark.parametrize("fault", ["missing_counts", "count", "prompt", "clock", "origin", "repeat", "duplicate", "missing", "unstable"])
def test_step_raw_schema_rejects_population_count_clock_failures(step_native, fault):
    root, records, summary = step_native
    row = records[-1]
    if fault == "missing_counts": row.pop("generated_token_count")
    if fault == "count": row["generated_token_count"] = 1023
    if fault == "prompt": row["accepted_prompt_sha256"] = "wrong"
    if fault == "clock": summary["clock"] = "remote clock"
    if fault == "origin": summary["repetitions"][-1]["phases"]["formal"]["origin_monotonic"] += 1
    if fault == "repeat": summary["repetitions"].pop()
    if fault == "duplicate": records.append(dict(row))
    if fault == "missing": records.pop()
    if fault == "unstable": summary["status"] = "REFERENCE_UNSTABLE"
    write_jsonl(root / "replay/client_requests.jsonl", records)
    (root / "replay/canonical_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        native_client_rows(root, WORKLOAD, IDS, "step5")


@pytest.mark.parametrize("fault", ["missing_check", "other_origin", "runtime", "image", "source", "artifact", "node", "declaration", "runtime_receipt", "node_version", "node_role", "stage_image"])
def test_step_unverified_real_provenance_cannot_be_admitted(tmp_path, step_native, simulator, fault):
    root, _, _ = step_native
    receipt = synthetic_step_check(root)
    document = json.loads(receipt.read_text())
    check = document["runs"][root.name]
    if fault == "missing_check": receipt = None
    if fault == "runtime": check["runtime_identity"]["vllm_version"] = "0.28.0"
    if fault == "image": check["runtime_identity"]["image"] = "wrong-image"
    if fault == "source": check["runtime_identity"]["source_hashes"] = {}
    if fault == "artifact": check["artifacts"][1]["sha256"] = "f" * 64
    if fault == "node": check["artifacts"] = [p for p in check["artifacts"] if "decode3" not in p["path"]]
    if fault == "declaration": check["checks"].pop("runtime_identity")
    if fault == "runtime_receipt": check["runtime_evidence"].pop("platform_image")
    if fault in ("node_version", "node_role", "stage_image"):
        path = root / ("config.json" if fault == "stage_image" else "nodes/decode3/result.json")
        value = json.loads(path.read_text())
        if fault == "stage_image": value["engine"]["native_reference"]["image"] = "wrong-image"
        if fault == "node_version": value["versions"]["vllm"] = "0.28.0"
        if fault == "node_role": value["plan"]["role"] = "prefill"
        path.write_text(json.dumps(value))
        check["artifacts"] = [pin(path, root.parents[2]) if p["path"].endswith(str(path.relative_to(root))) else p
                              for p in check["artifacts"]]
    if receipt is not None: receipt.write_text(json.dumps(document))
    if fault == "other_origin":
        root = shutil.copytree(root, tmp_path / "other/native")
    rows = tmp_path / "rows"
    normalize_cell(CELL, CASE, IDS, {root.name: root}, {"sim": simulator}, rows, WORKLOAD, receipt)
    reading = gate_reading(CELL, CASE | dict(routing=dict(status="NOT_APPLICABLE")), rows)
    assert reading["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_shipped_step_matrix_never_qualifies_fixture_without_real_check_and_routing(tmp_path, step_native, simulator):
    root, _, _ = step_native
    rows = tmp_path / "rows"
    normalize_cell(CELL, CASE, IDS, {root.name: root}, {"sim": simulator}, rows, WORKLOAD)
    assert gate_reading(CELL, CASE, rows)["verdict"] == "INSUFFICIENT_EVIDENCE"
