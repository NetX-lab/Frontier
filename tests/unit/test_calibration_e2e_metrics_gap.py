import json
import os
import sys

import pandas as pd
import pytest

import tests.comparison.calibration.e2e_metrics_gap as gap

FORMAL = {"a": 1.0, "b": 2.0}
VLLM_ROWS = {
    # request: (ttft ms, tpot ms, e2e ms, prefill tokens, decode tokens)
    "w": (10.0, 5.0, 30.0, 8, 4),
    "a": (20.0, 10.0, 50.0, 16, 4),
    "b": (40.0, 10.0, 70.0, 32, 4),
}


def write_after(path, content, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    os.utime(path, (mtime, mtime))


def write_case(root, frontier_runs, vllm_rows=VLLM_ROWS):
    """frontier_runs: label -> (rows like VLLM_ROWS, metrics mtime minus config mtime)."""
    write_after(root / "manifest.yaml", "case_id: synthetic\nrun_generation: 1\n", 0)
    trace = [{"frontier_request_id": 0, "request_id": "w", "arrived_at": 0.0}]
    trace += [{"frontier_request_id": index + 1, "request_id": request_id, "arrived_at": arrival}
              for index, (request_id, arrival) in enumerate(FORMAL.items())]
    (root / "request_ids.json").write_text(json.dumps({
        "formal_request_ids": list(FORMAL), "warmup_request_ids": ["w"],
        "sizing_request_ids": [], "rows": trace}))
    arrivals = {"w": 0.0, **FORMAL, "x": 3.0}  # x: a row in no id list
    vllm_run = root / "vllm_run"
    write_after(vllm_run / "run_manifest.json", "{}", 100)
    write_after(vllm_run / "run" / "request_metrics.jsonl", "".join(
        json.dumps({"request_id": f"cmpl-{request_id}-0", "arrival_time": 1000 + arrivals[request_id],
                    "completion_time": 1000 + arrivals[request_id] + e2e / 1000, "ttft": ttft,
                    "tpot": tpot, "request_e2e_time": e2e, "request_num_prefill_tokens": prefill,
                    "request_num_decode_tokens": decode}) + "\n"
        for request_id, (ttft, tpot, e2e, prefill, decode) in vllm_rows.items()), 200)
    arguments = []
    for label, (rows, metrics_age) in frontier_runs.items():
        metrics = root / label / "metrics" / "model" / "online_serving" / "run"
        ids = {row["request_id"]: row["frontier_request_id"] for row in trace}
        write_after(metrics / "config.json", "{}", 300)
        write_after(metrics / "request_metrics.csv", pd.DataFrame(
            [{"Request Id": ids[request_id], "ttft": ttft, "tpot": tpot, "request_e2e_time": e2e,
              "request_num_prefill_tokens": prefill, "request_num_decode_tokens": decode}
             for request_id, (ttft, tpot, e2e, prefill, decode) in rows.items()]).to_csv(index=False),
            300 + metrics_age)
        arguments += ["--frontier-run", f"{label}={root / label}"]
    return ["--manifest", str(root / "manifest.yaml"), "--request-ids", str(root / "request_ids.json"),
            "--vllm-run-dir", str(vllm_run), *arguments]


def run_gap(monkeypatch, root, arguments, routing_status="NOT_APPLICABLE"):
    output = root / "gate"
    monkeypatch.setattr(sys, "argv", ["e2e_metrics_gap.py", *arguments,
                                      "--routing-status", routing_status, "--output-dir", str(output)])
    gap.main()
    return (pd.read_csv(output / "e2e_metrics_table.csv"),
            json.loads((output / "e2e_metrics_status.json").read_text()))


def scaled(rows, factor):
    return {request_id: (ttft * factor, tpot * factor, e2e * factor, prefill, decode)
            for request_id, (ttft, tpot, e2e, prefill, decode) in rows.items()}


def test_the_first_run_is_gated_and_the_others_are_references(tmp_path, monkeypatch):
    arguments = write_case(tmp_path, {"head": (VLLM_ROWS, 1), "main": (scaled(VLLM_ROWS, 1.2), 1)})

    table, status = run_gap(monkeypatch, tmp_path, arguments)

    head = table[table["frontier_run"] == "head"].set_index("metric")
    main = table[table["frontier_run"] == "main"].set_index("metric")
    assert (head["status"] == "PASS").all()
    assert head["relative_error"].tolist() == pytest.approx([0.0] * 5, abs=1e-12)
    assert main.loc["ttft_ms", "relative_error"] == pytest.approx(0.2)
    assert main.loc["ttft_ms", "status"] == "FAIL"
    assert head.loc["ttft_ms", "vllm"] == pytest.approx(30.0)  # warmup w excluded
    assert head.loc["token_throughput_tps", "frontier_numerator"] == 56
    assert status["gate"] == "PASS" and status["reference_frontier_runs"] == ["main"]


def test_a_zero_vllm_value_leaves_the_relative_error_undefined(tmp_path, monkeypatch):
    zero_ttft = {request_id: (0.0, *row[1:]) for request_id, row in VLLM_ROWS.items()}
    arguments = write_case(tmp_path, {"head": (VLLM_ROWS, 1)}, vllm_rows=zero_ttft)

    table, status = run_gap(monkeypatch, tmp_path, arguments)

    ttft = table.set_index("metric").loc["ttft_ms"]
    assert pd.isna(ttft["relative_error"]) and ttft["status"] == "INSUFFICIENT_EVIDENCE"
    assert status["gate"] == "INSUFFICIENT_EVIDENCE"


@pytest.mark.parametrize("frontier_rows, vllm_rows, gap_side, gap_field", [
    ({"w": VLLM_ROWS["w"], "a": VLLM_ROWS["a"]}, VLLM_ROWS, "head", "missing_request_ids"),
    (VLLM_ROWS, {**VLLM_ROWS, "x": VLLM_ROWS["a"]}, "vllm", "unclassified_request_ids"),
], ids=["missing_formal_request", "unclassified_vllm_row"])
def test_a_request_id_gap_blocks_the_gate(tmp_path, monkeypatch, frontier_rows, vllm_rows,
                                          gap_side, gap_field):
    arguments = write_case(tmp_path, {"head": (frontier_rows, 1)}, vllm_rows=vllm_rows)

    table, status = run_gap(monkeypatch, tmp_path, arguments)

    assert (table["status"] == "INSUFFICIENT_EVIDENCE").all()
    assert status["request_id_gaps"][gap_side][gap_field] in (["b"], ["x"])
    assert status["gate"] == "INSUFFICIENT_EVIDENCE"


def test_a_token_count_mismatch_blocks_the_gate(tmp_path, monkeypatch):
    longer_b = {**VLLM_ROWS, "b": (*VLLM_ROWS["b"][:4], 5)}
    arguments = write_case(tmp_path, {"head": (longer_b, 1)})

    table, status = run_gap(monkeypatch, tmp_path, arguments)

    assert status["token_count_mismatches"] == {"head": ["b"]}
    assert (table["status"] == "INSUFFICIENT_EVIDENCE").all()


def test_a_metrics_file_older_than_its_run_config_is_rejected(tmp_path, monkeypatch):
    arguments = write_case(tmp_path, {"head": (VLLM_ROWS, -1)})

    with pytest.raises(SystemExit, match="not written by this run"):
        run_gap(monkeypatch, tmp_path, arguments)


def test_a_routing_mismatch_fails_every_metric(tmp_path, monkeypatch):
    arguments = write_case(tmp_path, {"head": (VLLM_ROWS, 1)})

    table, status = run_gap(monkeypatch, tmp_path, arguments, routing_status="MISMATCH")

    assert (table["status"] == "FAIL").all() and status["gate"] == "FAIL"
