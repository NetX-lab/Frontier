import csv
import hashlib
import json
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parents[2] / "data/processed_traces/canonical_sharegpt_req256_qps4"


def test_canonical_sharegpt_files_match_their_checksums():
    for line in (CASE_DIR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((CASE_DIR / name).read_bytes()).hexdigest() == digest, name


def test_canonical_sharegpt_trace_and_native_rows_describe_the_same_requests():
    manifest = json.loads((CASE_DIR / "manifest.json").read_text())
    rows = json.loads((CASE_DIR / "request_ids.json").read_text())["rows"]
    with (CASE_DIR / "trace.csv").open() as handle:
        trace = list(csv.DictReader(handle))
    with (CASE_DIR / "warmup_pool.csv").open() as handle:
        warmup = list(csv.DictReader(handle))

    assert len(rows) == len(trace) == manifest["selection"]["num_formal_requests"] == 256
    for row, trace_row in zip(rows, trace):
        assert float(trace_row["arrived_at"]) == row["arrived_at"]
        assert int(trace_row["num_prefill_tokens"]) == row["num_prefill_tokens"]
        assert int(trace_row["num_decode_tokens"]) == row["num_decode_tokens"]
    arrivals = [row["arrived_at"] for row in rows]
    assert arrivals == sorted(arrivals) and arrivals[0] == 0.0
    assert (len(rows) - 1) / arrivals[-1] == manifest["arrivals"]["target_qps"] == 4.0
    formal_indices = {row["source_index"] for row in rows}
    assert formal_indices.isdisjoint(int(row["source_index"]) for row in warmup)
    assert len(formal_indices) + len(warmup) == 300
