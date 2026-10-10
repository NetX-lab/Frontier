import json
from pathlib import Path

import pandas as pd
import pytest

from frontier.profiling.cpu_overhead import vllm_cpu_probe
from tests.comparison.e2e_gate import admission, dp_pp_modes, run_gate
from tests.comparison.e2e_gate.gate import MIN_MODE_REQUESTS, gate_reading

FIXTURES = Path(run_gate.HERE) / "fixtures"
SINGLE = {"cell_id": "synthetic/qps1", "gate": "single"}
PD = {"cell_id": "synthetic/qps1", "gate": "pd_s33"}
DP_PP = {"cell_id": "synthetic/qps1", "gate": "dp_pp_pairs", "dp_pp_gate": {"mode_window_s": 1.0}}
# request: (arrival s, ttft ms, tpot ms, e2e ms, prompt tokens, output tokens); c's TPOT is not eligible.
ROWS = {
    "a": (0.0, 20.0, 10.0, 50.0, 16, 4),
    "b": (1.0, 40.0, 10.0, 70.0, 32, 4),
    "c": (2.0, 20.0, 99.0, 30.0, 8, 1),
}


def case(routing="NOT_APPLICABLE"):
    return {"routing": {"status": routing}}


def frame(rows, **columns):
    table = pd.DataFrame([{"request_id": request_id, "arrival_s": 1000 + arrival,
                           "completion_s": 1000 + arrival + e2e / 1000, "ttft_ms": ttft, "tpot_ms": tpot,
                           "request_e2e_time_ms": e2e, "request_num_prefill_tokens": prefill,
                           "request_num_decode_tokens": decode}
                          for request_id, (arrival, ttft, tpot, e2e, prefill, decode) in rows.items()])
    return table.assign(**columns)


def write_rows(root, sides, formal_ids=tuple(ROWS), windows=None, cell=SINGLE, recorded=None):
    """sides: side name ("vllm/<run>", "frontier/<label>") -> DataFrame of normalized rows; recorded:
    side -> the identity gaps normalize.py found in the raw rows."""
    for side, table in sides.items():
        (root / side).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(root / f"{side}.csv", index=False, float_format="%.17g")
    sample = {"cell_id": cell["cell_id"], "formal_request_ids": list(formal_ids),
              "sides": {side: {**(recorded or {}).get(side, {}), **({"windows": windows[side]} if windows else {})}
                        for side in sides}}
    (root / "sample.json").write_text(json.dumps(sample))
    return root


def statuses(reading):
    return {row["metric"]: row["status"] for row in reading["metrics"]}


def scaled(rows, metric_index, factor):
    return {request_id: tuple(value * factor if index == metric_index else value
                              for index, value in enumerate(row)) for request_id, row in rows.items()}


@pytest.mark.parametrize("name", sorted(path.parent.parent.name for path in FIXTURES.glob("*/rows/sample.json")))
def test_published_cell_replays_from_shipped_rows(name):
    rows = FIXTURES / name / "rows"
    cell_id = json.loads((rows / "sample.json").read_text())["cell_id"]
    published = {entry["cell_id"]: entry for entry in
                 json.loads((Path(run_gate.HERE) / "cases" / "published_readings.json").read_text())["cells"]}
    expected = published[cell_id]
    cell, matrix_case = run_gate.find_cell(run_gate.load_matrix(), cell_id)
    reading = gate_reading(cell, matrix_case, rows)
    if cell["gate"] == "dp_pp_pairs":
        modes = reading["dp_pp_modes"]
        errors = {f"{mode}.{metric}": entry["relative_error"] for mode, metrics in modes["modes"].items()
                  for metric, entry in metrics.items() if entry["gated"]}
        assert modes["mode_outcome"] == expected["mode_outcome"]
        assert modes["reference_modes"] == expected["reference_modes"]
        assert modes["windows"] == expected["windows_per_run"]
        for metric, error in expected["cell_metrics_reported"].items():
            assert reading["cell_metrics_reported"][metric]["relative_error"] == pytest.approx(error, abs=1e-12)
    else:
        errors = {row["metric"]: row["relative_error"] for row in reading["metrics"]}
    assert errors == pytest.approx({name: entry["relative_error"] for name, entry in expected["gated"].items()},
                                   abs=1e-12)
    assert reading["verdict"] == expected["verdict"]


def test_matrix_resolves_every_cell():
    matrix = run_gate.load_matrix()
    cells = {cell["cell_id"]: cell for cell in matrix["cells"]}
    assert sum(cell["status"] == "claimed" for cell in cells.values()) == 22
    for cell in cells.values():
        if cell["status"] != "claimed":
            assert cell["reason"]
            continue
        assert cell["gate"] in ("single", "pd_s33", "dp_pp_pairs")
        assert cell["coverage"]["class"] in ("in_profile", "outside_profile")
        assert cell["frontier_runs"] and not cell["frontier_runs_missing"]
        assert matrix["cases"][cell["case_id"]]["routing"]["status"] in ("MATCH", "UNSET", "NOT_APPLICABLE")
        for name in ("workload", "trace_csv", "request_ids"):
            assert len(cell["trace"][name]["sha256"]) == 64


def test_five_metrics_are_decided_independently(tmp_path):
    rows = write_rows(tmp_path, {"vllm/run": frame(ROWS), "frontier/x": frame(scaled(ROWS, 1, 1.2))})
    reading = gate_reading(SINGLE, case(), rows)
    assert statuses(reading) == {"ttft_ms": "FAIL", "tpot_ms": "PASS", "request_e2e_time_ms": "PASS",
                                 "request_throughput_rps": "PASS", "token_throughput_tps": "PASS"}
    assert reading["verdict"] == "FAIL"


def test_tpot_counts_only_requests_with_more_than_one_output_token(tmp_path):
    frontier = {**ROWS, "c": (2.0, 20.0, 500.0, 30.0, 8, 1)}
    rows = write_rows(tmp_path, {"vllm/run": frame(ROWS), "frontier/x": frame(frontier)})
    reading = gate_reading(SINGLE, case(), rows)
    [tpot] = [row for row in reading["metrics"] if row["metric"] == "tpot_ms"]
    assert tpot["relative_error"] == 0.0
    assert reading["verdict"] == "PASS"


@pytest.mark.parametrize("frontier, recorded, kind", [
    (frame({k: v for k, v in ROWS.items() if k != "b"}), {}, "missing_request_ids"),
    (frame(ROWS), {"duplicates": ["a"]}, "duplicate_request_ids"),
    (frame(ROWS), {"unclassified": ["x"]}, "unclassified_request_ids"),
    (frame({**ROWS, "x": (3.0, 20.0, 10.0, 50.0, 16, 4)}), {}, "unclassified_request_ids"),
    (frame(ROWS).assign(request_e2e_time_ms=lambda t: t["request_e2e_time_ms"] * 1000), {}, "clock_unit_violations"),
    (frame(ROWS).assign(ttft_ms=1e6), {}, "clock_unit_violations"),
])
def test_identity_and_clock_gaps_leave_every_metric_without_evidence(tmp_path, frontier, recorded, kind):
    rows = write_rows(tmp_path, {"vllm/run": frame(ROWS), "frontier/x": frontier}, recorded={"frontier/x": recorded})
    reading = gate_reading(SINGLE, case(), rows)
    assert kind in reading["request_id_gaps"]["frontier/x"]
    assert set(statuses(reading).values()) == {"INSUFFICIENT_EVIDENCE"}
    assert reading["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_token_count_mismatch_leaves_every_metric_without_evidence(tmp_path):
    frontier = {**ROWS, "b": (1.0, 40.0, 10.0, 70.0, 33, 4)}
    rows = write_rows(tmp_path, {"vllm/run": frame(ROWS), "frontier/x": frame(frontier)})
    reading = gate_reading(SINGLE, case(), rows)
    assert reading["token_count_mismatches"] == {"frontier/x": ["b"]}
    assert set(statuses(reading).values()) == {"INSUFFICIENT_EVIDENCE"}


@pytest.mark.parametrize("routing, verdict", [
    ("MATCH", "PASS"), ("NOT_APPLICABLE", "PASS"), ("UNSET", "INSUFFICIENT_EVIDENCE"), ("MISMATCH", "FAIL"),
])
def test_routing_status_gates_matching_numbers(tmp_path, routing, verdict):
    rows = write_rows(tmp_path, {"vllm/run": frame(ROWS), "frontier/x": frame(ROWS)})
    reading = gate_reading(SINGLE, case(routing), rows)
    assert reading["verdict"] == verdict
    assert set(statuses(reading).values()) == {verdict}


def test_pd_s33_reduces_the_vllm_ttft_by_the_minimum_endpoint_offset(tmp_path):
    vllm = frame(ROWS, endpoint_offset_proxy_api_ms=[5.0, 3.0, 4.0])
    rows = write_rows(tmp_path, {"vllm/run": vllm, "frontier/x": frame(ROWS).assign(ttft_ms=lambda t: t["ttft_ms"] - 3.0)})
    reading = gate_reading(PD, case(), rows)
    assert reading["ttft_s33"]["bound_ms"] == 3.0
    assert reading["ttft_s33"]["vllm_ttft_adjusted_ms"] == pytest.approx(frame(ROWS)["ttft_ms"].mean() - 3.0)
    assert reading["verdict"] == "PASS"


def mode_rows(count, ttft_mode, tpot_mode="in_phase", ttft=20.0):
    rows = {f"r{index:02d}": (float(index), ttft, 10.0, 200.0, 16, 4) for index in range(count)}
    return frame(rows, ttft_mode=ttft_mode, tpot_mode=tpot_mode)


def dp_sides(vllm, members, windows):
    sides = {"vllm/run": vllm, **{f"frontier/s{seed}": member for seed, member in enumerate(members, 1)}}
    return sides, {side: windows.get(side, {"in_phase": 10}) for side in sides}


def test_dp_pairs_count_a_request_only_where_both_sides_share_its_mode(tmp_path):
    count = MIN_MODE_REQUESTS
    sides, windows = dp_sides(mode_rows(count, "in_phase"),
                              [mode_rows(count, "in_phase", ttft=21.0), mode_rows(count, "split", ttft=99.0)], {})
    rows = write_rows(tmp_path, sides, mode_rows(count, "x").request_id, windows, DP_PP)
    reading = gate_reading(DP_PP, case(), rows)
    ttft = reading["dp_pp_modes"]["modes"]["in_phase"]["ttft_ms"]
    assert (ttft["pairs_n"], ttft["requests_n"], ttft["gated"]) == (count, count, True)
    assert ttft["relative_error"] == pytest.approx(0.05)
    assert "split" not in reading["dp_pp_modes"]["modes"]
    assert reading["dp_pp_modes"]["mode_outcome"] == "paired"
    assert reading["verdict"] == "PASS"


def test_dp_cell_without_a_gated_pair_set_has_no_numerical_comparison(tmp_path):
    count = MIN_MODE_REQUESTS - 1
    sides, windows = dp_sides(mode_rows(count, "in_phase"), [mode_rows(count, "in_phase")],
                              {"vllm/run": {"in_phase": 5, "anti_phase": 5}})
    rows = write_rows(tmp_path, sides, mode_rows(count, "x").request_id, windows, DP_PP)
    reading = gate_reading(DP_PP, case(), rows)
    modes = reading["dp_pp_modes"]
    assert modes["modes"]["in_phase"]["ttft_ms"]["gated"] is False
    assert modes["mode_outcome"] == "empty_pair_set"
    assert modes["multi_mode_reference"] is True
    assert modes["reference_modes_without_gated_pairs"] == ["anti_phase", "in_phase"]
    assert reading["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert reading["cell_metrics_reported"]["ttft_ms"]["relative_error"] == 0.0


def test_window_modes_and_request_modes():
    period = 0.1
    lane0 = [(index * period, False) for index in range(30)]
    # Window 0: lane 1 starts with lane 0; window 1: half a period later; window 2: lane 1 holds two batches.
    lane1 = ([(index * period, False) for index in range(10)]
             + [((index + 0.5) * period, False) for index in range(10, 20)]
             + [(index * period, True) for index in range(20, 30)])
    modes = dp_pp_modes.window_modes({0: lane0, 1: lane1}, 0.0, 1.0)
    assert modes == {0: "in_phase", 1: "anti_phase", 2: "split"}
    assert dp_pp_modes.request_modes(0.5, 1.1, 1.9, 0.0, 1.0, modes) == ("in_phase", "anti_phase")
    # Decode over windows 1 and 2 has no mode covering DECODE_MODE_SHARE of them.
    assert dp_pp_modes.request_modes(0.5, 1.5, 2.5, 0.0, 1.0, modes) == ("in_phase", None)


def test_cpu_probe_producer_replays_the_shipped_probe_slice(tmp_path):
    fixture = FIXTURES / "cpu_probe_c1"
    source = json.loads((fixture / "source.json").read_text())
    outputs = {"eager": tmp_path / "cpu_overheads.csv", "kernel_only": tmp_path / "cpu_overheads_kernel_only.csv"}
    vllm_cpu_probe.main(["--cpu_probe_logs", str(fixture / "cpu_probe.jsonl"), *source["producer_arguments"],
                         "--eager_output_file", str(outputs["eager"]),
                         "--kernel_only_output_file", str(outputs["kernel_only"])])
    for path in outputs.values():
        assert path.read_bytes() == (fixture / "expected" / path.name).read_bytes()


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def client_run(root, name, e2e_ms):
    run = root / name
    write_jsonl(run / "run" / "client_requests.jsonl",
                [{"segment": "warmup", "role": "warmup", "dispatch_monotonic": 0.0, "response_monotonic": 9.0}]
                + [{"segment": "formal", "role": "formal", "dispatch_monotonic": float(index),
                    "response_monotonic": index + value / 1000} for index, value in enumerate(e2e_ms, 1)])
    return run


def test_client_e2e_admission(tmp_path):
    clean = client_run(tmp_path, "clean", [100.0, 100.0])
    assert admission.client_e2e(client_run(tmp_path, "fast", [101.0, 103.0]), [clean])["passed"]
    rejected = admission.client_e2e(client_run(tmp_path, "slow", [104.0, 104.0]), [clean])
    assert rejected["ratio"] == pytest.approx(1.04) and not rejected["passed"]


def send_run(root, name, starts, gap_ms):
    run = root / name
    write_jsonl(run / "run" / "prefill" / "kv_transfer.jsonl",
                [{"event": "producer_layer_send_start", "request_id": f"q{index}", "timestamp": start + layer * gap_ms / 1000}
                 for index, start in enumerate(starts) for layer in range(4)])
    return run


def test_send_span_uses_isolated_prefills_against_the_mean_clean_span(tmp_path):
    # q1 and q2 overlap and are not isolated; q0 and q3 are.
    probe = send_run(tmp_path, "probe", [0.0, 1.0, 1.001, 2.0], 2.0)
    assert admission.send_span_ms(probe) == pytest.approx(6.0)
    clean = [send_run(tmp_path, "c1", [0.0, 1.0], 1.9), send_run(tmp_path, "c2", [0.0, 1.0], 2.1)]
    reading = admission.send_span(probe, clean)
    assert reading["reference"] == pytest.approx(6.0) and reading["passed"]


def prefill_run(root, name, prefill_s):
    run = client_run(root, name, [100.0])
    client = json.loads((run / "run" / "client_requests.jsonl").read_text().splitlines()[1])
    write_jsonl(run / "run" / "client_requests.jsonl", [{**client, "engine_request_id": "e1"}])
    write_jsonl(run / "run" / "dp_placement" / "dp_placement_0.jsonl", [
        {"kind": "engine_iteration", "engine": 0, "monotonic": 1.0, "scheduled_new_req_ids": ["e1"]},
        {"kind": "engine_iteration", "engine": 1, "monotonic": 1.0 + prefill_s / 2},
        {"kind": "engine_iteration", "engine": 0, "monotonic": 1.0 + prefill_s},
    ])
    return run


def test_isolated_prefill_admission_against_the_probe_cohort_median(tmp_path):
    cohort = [prefill_run(tmp_path, name, value) for name, value in
              [("f", 0.0287), ("g", 0.0232), ("h", 0.0292)]]
    assert admission.isolated_prefill(cohort[0], cohort)["passed"]
    assert not admission.isolated_prefill(cohort[1], cohort)["passed"]
    with pytest.raises(ValueError):
        admission.isolated_prefill(prefill_run(tmp_path, "outside", 0.028), cohort)
