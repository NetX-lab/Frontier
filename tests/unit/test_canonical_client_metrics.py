"""Canonical client metrics: real Request exports and native HTTP records, not accuracy evidence."""

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import pytest

from tests.comparison.calibration.client_metrics import native_client_rows
from tests.comparison.calibration.pd_replay import canonical_workload
from tests.comparison.e2e_gate import run_gate
from tests.comparison.e2e_gate.gate import gate_reading
from tests.comparison.e2e_gate.normalize import normalize_cell

TRACE = Path(__file__).resolve().parents[2] / "data/processed_traces/canonical_fixed_6144_1024_req128_qps1"
WORKLOAD = canonical_workload(TRACE)
IDS = json.loads((TRACE / "request_ids.json").read_text())
CELL = {"cell_id": "canonical/fixed", "case_id": "canonical", "status": "claimed", "gate": "client"}
CASE = {"routing": {"status": "NOT_APPLICABLE"}, "architecture": "pd-disaggregation"}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def synthetic_run_check(root):
    receipt = root.parents[2] / "analysis/run_check_status.groundtruth_clean.json"
    receipt.parent.mkdir(exist_ok=True)
    receipt.write_text(json.dumps(dict(entry="groundtruth-run-check", mode="groundtruth_clean", case_id="canonical", runs={
        root.name: dict(status="PASS", checks={"synthetic_oracle": "PASS: fixture only, not native admission"},
                        run_manifest=f"runs/groundtruth_clean/{root.name}/run_manifest.json")})))
    return receipt


def export_simulator(root):
    import logging
    from types import SimpleNamespace
    from frontier.config import global_vars
    from frontier.config.metrics_config import MetricsConfig
    from frontier.entities import Request
    from frontier.metrics.metrics_store import MetricsStore
    from frontier.types import ClusterType

    logging.disable(logging.CRITICAL)
    global_vars.KV_CONNECTOR = "nixl"
    global_vars.SIMULATION_MODE = "offline"
    output = Path(root) / "metrics"
    store = MetricsStore(SimpleNamespace(sys_arch="pd-disaggregation", metrics_config=MetricsConfig(
        output_dir=str(output), store_plots=False, enable_metrics_ground_truth_trace=True)), {})
    for row in IDS["rows"]:
        arrival = row["arrived_at"]
        request = Request(arrival, 6144, 1024)
        request._id = str(row["frontier_request_id"])
        request.on_batch_schedule(arrival + .1, ClusterType.PREFILL)
        request.on_batch_end(arrival + .5, 6144, ClusterType.PREFILL)
        request.on_kv_cache_transfer_start(arrival + .5)
        request.on_kv_cache_transfer_complete(arrival + 1, .5)
        request.on_batch_schedule(arrival + 1.5, ClusterType.DECODE)
        for token in range(1024):
            request.on_batch_end(arrival + 2 + token * .002, 1, ClusterType.DECODE)
        assert request.completed and request.num_emitted_decode_tokens == 1024
        assert request.first_decode_token_completed_at == arrival + 2
        assert request.decode_first_token_completed_at == pytest.approx(arrival + 2.002)
        store._on_request_end(request.completed_at, request)
    (output / "config.json").write_text("{}")
    os.utime(output / "config.json", (1, 1))


@pytest.fixture(scope="module")
def simulator(tmp_path_factory):
    root = tmp_path_factory.mktemp("simulator")
    code = "from tests.unit.test_canonical_client_metrics import export_simulator; export_simulator(%r)" % str(root)
    result = subprocess.run([os.environ.get("FRONTIER_SIMULATOR_PYTHON", sys.executable), "-c", code],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, result.stderr
    return root


@pytest.fixture
def native(tmp_path):
    root = tmp_path / "canonical/runs/groundtruth_clean/native"
    root.mkdir(parents=True)
    (root / "run_manifest.json").write_text(json.dumps(
        dict(mode="clean", groundtruth_tree_dirty=False, groundtruth_commit="synthetic-oracle-not-native")))
    records = []
    for repeat in (1, 2, 3):
        for phase, templates in (("warmup", WORKLOAD[2]),
                                 ("formal", IDS["rows"])):
            for template in templates:
                jitter = .013 if 0 < template["arrived_at"] < 15 else 0
                arrival = repeat * 1000 + (100 if phase == "formal" else 0) + template["arrived_at"] + jitter
                records.append(dict(template_request_id=template["request_id"],
                    request_id=f"{template['request_id']}-repeat{repeat:02d}", repeat=repeat,
                    arrived_at=template["arrived_at"],
                    role=phase, segment=phase, status="completed", http_status=200, validation_errors=[],
                    stream_done=True, finish_reason="length", num_prefill_tokens=6144, num_decode_tokens=1024,
                    output_token_count=1024, usage=dict(prompt_tokens=6144, completion_tokens=1024, total_tokens=7168),
                    dispatch_monotonic=arrival, first_token_monotonic=arrival + 2,
                    last_token_monotonic=arrival + 4.046, ttft_ms=2000., tpot_ms=2., e2e_ms=4046.))
    summary = dict(mode="clean", vllm_version="0.28.0", status=0, canonical_repetitions=dict(
        client_workload_verified=True, observation_errors=[], reference_admission="PASS",
        repetitions=[dict(repeat=i) for i in (1, 2, 3)], idle_boundaries=[
            dict(boundary=name, verified=True) for name in ["before_repeat01_warmup"] + [
                f"after_repeat{i:02d}_{phase}" for i in (1, 2, 3) for phase in ("warmup", "formal")]]))
    write_jsonl(root / "run/client_requests.jsonl", records)
    (root / "run/replay_summary.json").write_text(json.dumps(summary))
    return root, records, summary


@pytest.mark.parametrize("run_location", ["bound", "same_name_other_case"])
def test_canonical_run_gate_cli_preserves_three_windows_and_first_d(tmp_path, native, simulator, run_location):
    root, _, _ = native
    cell = copy.deepcopy(CELL)
    cell["trace"] = {key: dict(path=str(TRACE / name), sha256=hashlib.sha256((TRACE / name).read_bytes()).hexdigest())
                     for key, name in (("request_ids", "request_ids.json"), ("workload", "manifest.json"))}
    matrix = tmp_path / "matrix.json"
    receipt = synthetic_run_check(root)
    cell["run_check"] = dict(path=str(receipt), sha256=hashlib.sha256(receipt.read_bytes()).hexdigest())
    matrix.write_text(json.dumps(dict(cases={"canonical": CASE}, cells=[cell])))
    selected = root
    if run_location == "same_name_other_case":
        selected = tmp_path / "other/runs/groundtruth_clean/native"
        shutil.copytree(root, selected)
    common = [sys.executable, str(Path(run_gate.__file__)), "normalize", "--matrix", str(matrix),
              "--cell", CELL["cell_id"], "--artifacts-root", str(tmp_path), "--vllm-run-dir", str(selected),
              "--frontier-run", f"sim={simulator}", "--output-dir", str(tmp_path / "rows")]
    subprocess.run(common, check=True, capture_output=True)
    subprocess.run([sys.executable, str(Path(run_gate.__file__)), "gate", "--matrix", str(matrix),
                    "--cell", CELL["cell_id"], "--rows", str(tmp_path / "rows"),
                    "--output", str(tmp_path / "reading.json")], check=True, capture_output=True)
    reading = json.loads((tmp_path / "reading.json").read_text())
    if run_location != "bound":
        assert reading["verdict"] == "INSUFFICIENT_EVIDENCE"
        return
    assert reading["verdict"] == "PASS"  # Synthetic oracle only, not real native acceptance.
    assert len(reading["repetitions"]) == 3 and len(reading["metrics"]) == 4
    values = {r["metric"]: r for r in reading["metrics"]}
    assert values["ttft_ms"]["frontier"] == values["ttft_ms"]["vllm"] == 2000.
    assert values["tpot_ms"]["frontier"] == pytest.approx(2.)
    assert values["request_e2e_time_ms"]["frontier"] == pytest.approx(4046.)
    assert values["token_throughput_tps"]["frontier"] == pytest.approx(917504 / 131.046)
    rows = pd.read_csv(tmp_path / "rows/frontier/sim.csv")
    assert rows.internal_ttft_ms.mean() == 500. and rows.ttft_ms.mean() == 2000.
    assert [w["first_submission_s"] for side, w in reading["formal_windows"].items()
            if side.startswith("vllm/")] == [1100., 2100., 3100.]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "failed", "unfinished", "tokens", "endpoint",
                                   "provenance", "idle", "overlap", "wrong_metric"])
def test_raw_native_admission_never_falls_back_to_p_ttft(native, fault):
    root, records, summary = native
    r = records[-1]
    if fault == "missing": records.pop()
    if fault == "duplicate": records.append(dict(r))
    if fault in ("failed", "unfinished"): r["status"] = fault
    if fault == "tokens": r["output_token_count"] = 1023
    if fault == "endpoint": r["first_token_monotonic"] = None
    if fault == "wrong_metric": r["ttft_ms"] = 500.
    if fault == "overlap": records[144]["dispatch_monotonic"] = 0.
    if fault == "idle": summary["canonical_repetitions"]["idle_boundaries"].pop()
    if fault == "provenance": (root / "run_manifest.json").write_text('{"mode":"clean","groundtruth_tree_dirty":true}')
    write_jsonl(root / "run/client_requests.jsonl", records)
    (root / "run/replay_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        native_client_rows(root, WORKLOAD, IDS)


def test_missing_simulator_first_d_rejects_even_when_later_decode_boundary_exists(tmp_path, simulator):
    from tests.comparison.calibration.client_metrics import frontier_client_rows
    rows = [json.loads(line) for line in (simulator / "metrics/metrics_ground_truth.jsonl").read_text().splitlines()]
    rows[0].pop("first_decode_token_completed_at")
    assert rows[0]["decode_first_token_completed_at"] > rows[0]["arrived_at"]
    write_jsonl(tmp_path / "metrics/metrics_ground_truth.jsonl", rows)
    (tmp_path / "metrics/config.json").write_text("{}")
    os.utime(tmp_path / "metrics/config.json", (1, 1))
    with pytest.raises(ValueError, match="client output-token"):
        frontier_client_rows(tmp_path, pd.DataFrame(IDS["rows"]))


@pytest.mark.parametrize("fault", ["semantic", "tokens", "duplicate", "missing", "reference", "one_bad_round"])
def test_gate_rejects_normalized_gaps_and_never_averages_away_a_bad_round(tmp_path, native, simulator, fault):
    root, _, _ = native
    output = tmp_path / "normalized"
    normalize_cell(CELL, CASE, IDS, {root.name: root}, {"sim": simulator}, output, WORKLOAD, synthetic_run_check(root))
    path = output / "vllm/native-repeat01.csv"
    rows = pd.read_csv(path)
    if fault == "semantic": rows["ttft_semantic"] = "request_arrival_to_prefill_completion"
    if fault == "tokens": rows.loc[0, "request_num_decode_tokens"] = 1023
    if fault == "duplicate": rows = pd.concat([rows, rows.iloc[:1]])
    if fault == "missing": rows = rows.iloc[1:]
    if fault == "one_bad_round": rows["ttft_ms"] *= 1.2; rows["tpot_ms"] = (rows.request_e2e_time_ms - rows.ttft_ms) / 1023
    rows.to_csv(path, index=False, float_format="%.17g")
    if fault == "reference":
        sample = json.loads((output / "sample.json").read_text())
        sample["reference_admission"]["native"] = "UNVERIFIED"
        (output / "sample.json").write_text(json.dumps(sample))
    reading = gate_reading(CELL, CASE, output)
    assert reading["verdict"] == ("FAIL" if fault == "one_bad_round" else "INSUFFICIENT_EVIDENCE")
    if fault == "one_bad_round":
        assert reading["metrics"][0]["status"] == "PASS"
        assert reading["repetitions"]["vllm/native-repeat01"]["metrics"][0]["status"] == "FAIL"


@pytest.mark.parametrize("side", ["native-formal", "native-warmup", "simulator"])
def test_swapped_intermediate_arrivals_preserving_latencies_and_windows_are_rejected(tmp_path, native, simulator, side):
    root, records, _ = native
    if side == "simulator":
        simulator = shutil.copytree(simulator, tmp_path / "swapped-simulator")
        path = simulator / "metrics/metrics_ground_truth.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        pair = records[1:3]
        keys = [key for key in pair[0] if key.endswith("_at")]
    else:
        path = root / "run/client_requests.jsonl"
        pair = records[17:19] if side == "native-formal" else records[1:3]
        keys = ["arrived_at", "dispatch_monotonic", "first_token_monotonic", "last_token_monotonic"]
    before = [(r.get("ttft_ms"), r.get("tpot_ms"), r.get("e2e_ms")) for r in pair]
    for key in keys:
        pair[0][key], pair[1][key] = pair[1][key], pair[0][key]
    assert before == [(r.get("ttft_ms"), r.get("tpot_ms"), r.get("e2e_ms")) for r in pair]
    write_jsonl(path, records)
    with pytest.raises(ValueError, match="arrival"):
        normalize_cell(CELL, CASE, IDS, {root.name: root}, {"sim": simulator}, tmp_path / "rows", WORKLOAD)


@pytest.mark.skipif(sys.version_info < (3, 11), reason="Native replay uses asyncio.timeout_at")
def test_compressed_streaming_http_arrivals_do_not_admit_the_frozen_qps1_trace(tmp_path, monkeypatch, simulator):
    from tests.comparison.calibration import pd_replay
    from tests.unit.test_native_streaming_replay import test_three_repetitions_use_existing_replay_and_observed_idle

    saved = []
    replay = pd_replay.replay_repetitions
    async def retain(*args, **kwargs):
        result, records = await replay(*args, **kwargs)
        saved.extend([result, records])
        return result, records
    monkeypatch.setattr(pd_replay, "replay_repetitions", retain)
    test_three_repetitions_use_existing_replay_and_observed_idle(tmp_path, monkeypatch, None)
    repetitions, records = saved
    root = tmp_path / "http-native"
    root.mkdir()
    (root / "run_manifest.json").write_text(json.dumps(dict(mode="clean", groundtruth_tree_dirty=False,
        groundtruth_commit="local-http-fixture-not-native")))
    write_jsonl(root / "run/client_requests.jsonl", records)
    (root / "run/replay_summary.json").write_text(json.dumps(dict(mode="clean", vllm_version="0.28.0",
        status=0, canonical_repetitions=repetitions)))
    output = tmp_path / "http-rows"
    with pytest.raises(ValueError, match="arrival"):
        normalize_cell(CELL, CASE, IDS, {root.name: root}, {"sim": simulator}, output, WORKLOAD)
    assert all(r["output_token_count"] == 1024 for r in records)


def test_glm_p0_missing_native_receipt_and_routing_stay_insufficient(tmp_path, native, simulator):
    """Exercise the existing CLI, not a P0 native or accuracy acceptance fixture."""
    root, _, _ = native
    cell = dict(CELL, cell_id="glm53_p0_h800_32/qps1", case_id="glm53_p0_h800_32")
    cell["trace"] = {key: dict(path=str(TRACE / name), sha256=hashlib.sha256((TRACE / name).read_bytes()).hexdigest())
                     for key, name in (("request_ids", "request_ids.json"), ("workload", "manifest.json"))}
    case = dict(CASE, routing={"status": "UNSET"})
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(dict(cases={cell["case_id"]: case}, cells=[cell])))
    subprocess.run([sys.executable, str(Path(run_gate.__file__)), "normalize", "--matrix", str(matrix),
                    "--cell", cell["cell_id"], "--artifacts-root", str(tmp_path),
                    "--vllm-run-dir", str(root), "--frontier-run", f"sim={simulator}",
                    "--output-dir", str(tmp_path / "rows")], check=True, capture_output=True)
    subprocess.run([sys.executable, str(Path(run_gate.__file__)), "gate", "--matrix", str(matrix),
                    "--cell", cell["cell_id"], "--rows", str(tmp_path / "rows"),
                    "--output", str(tmp_path / "reading.json")], check=True, capture_output=True)
    reading = json.loads((tmp_path / "reading.json").read_text())
    assert reading["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert reading["blocked_status"] == "INSUFFICIENT_EVIDENCE"
    assert {r["status"] for r in reading["metrics"]} == {"INSUFFICIENT_EVIDENCE"}
    assert len(reading["repetitions"]) == 3
