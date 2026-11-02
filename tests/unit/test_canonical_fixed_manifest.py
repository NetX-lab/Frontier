import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from frontier.config import TraceRequestGeneratorConfig
from frontier.request_generator.trace_replay_request_generator import TraceReplayRequestGenerator

REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_DIR = REPO_ROOT / "data/processed_traces/canonical_fixed_6144_1024_req128_qps1"
BUILDER = REPO_ROOT / "tests/integration/canonical_case/build_fixed_manifest.py"


def test_canonical_fixed_files_match_pinned_checksums():
    assert hashlib.sha256((CASE_DIR / "SHA256SUMS").read_bytes()).hexdigest() == (
        "594c3cb671f13dff4804dc269f957820c51fd6fe5af5f841194091c623e013b9"
    )
    for line in (CASE_DIR / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((CASE_DIR / name).read_bytes()).hexdigest() == digest, name
    manifest = json.loads((CASE_DIR / "manifest.json").read_text())
    assert manifest["reproducibility"]["builder_sha256"] == hashlib.sha256(BUILDER.read_bytes()).hexdigest()


def test_canonical_fixed_rows_have_exact_counts_arrivals_and_separate_phase_ids():
    manifest = json.loads((CASE_DIR / "manifest.json").read_text())
    request_ids = json.loads((CASE_DIR / "request_ids.json").read_text())
    rows = request_ids["rows"]
    with (CASE_DIR / "trace.csv").open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ["arrived_at", "num_prefill_tokens", "num_decode_tokens"]
        trace = list(reader)
    with (CASE_DIR / "warmup_pool.csv").open() as handle:
        warmup = list(csv.DictReader(handle))

    assert len(rows) == len(trace) == manifest["selection"]["num_formal_requests"] == 128
    assert [float(row["arrived_at"]) for row in trace] == list(range(128))
    assert [row["arrived_at"] for row in rows] == list(range(128))
    assert [row["frontier_request_id"] for row in rows] == list(range(128))
    assert all(row["role"] == "formal" for row in rows)
    assert len(warmup) == manifest["warmup"]["num_requests_per_repetition"] == 16
    assert [float(row["arrived_at"]) for row in warmup] == list(range(16))
    for row in trace + rows + warmup:
        assert int(row["num_prefill_tokens"]) == 6144
        assert int(row["num_decode_tokens"]) == 1024
    assert sum(int(row["num_prefill_tokens"]) for row in trace) == 786432
    assert sum(int(row["num_decode_tokens"]) for row in trace) == 131072
    assert manifest["lengths"] == {
        "prompt_tokens_per_request": 6144,
        "output_tokens_per_request": 1024,
        "total_tokens_per_request": 7168,
        "formal_prompt_tokens": 786432,
        "formal_output_tokens": 131072,
        "formal_total_tokens": 917504,
    }
    assert manifest["arrivals"]["target_qps"] == 1.0
    assert manifest["arrivals"]["first_arrival_s"] == 0.0
    assert manifest["arrivals"]["last_arrival_s"] == manifest["arrivals"]["realized_span_s"] == 127.0
    assert manifest["warmup"]["included_in_metrics"] is False
    assert manifest["frontier_replay"]["minimum_max_tokens"] == 7168
    assert manifest["measurement"]["arrival_end_s"] == 127.0

    formal_ids = [row["request_id"] for row in rows]
    warmup_ids = [row["request_id"] for row in warmup]
    assert request_ids["formal_request_ids"] == formal_ids
    assert request_ids["warmup_request_ids"] == warmup_ids
    assert len(set(formal_ids)) == 128
    assert len(set(warmup_ids)) == 16
    assert set(formal_ids).isdisjoint(warmup_ids)
    assert all("-formal-" in request_id for request_id in formal_ids)
    assert all("-warmup-" in request_id for request_id in warmup_ids)
    repetitions = manifest["repetitions"]
    assert repetitions["clean_repetitions"] == 3
    assert repetitions["repeat_indices"] == [1, 2, 3]
    runtime_ids = [repetitions["runtime_request_id_rule"].format(request_id=request_id, repeat=repeat)
                   for repeat in [1, 2, 3] for request_id in formal_ids + warmup_ids]
    assert len(set(runtime_ids)) == 432


def test_canonical_fixed_trace_loads_all_128_requests_without_clipping():
    config = TraceRequestGeneratorConfig(
        trace_file=str(CASE_DIR / "trace.csv"),
        prefill_scale_factor=1.0,
        decode_scale_factor=1.0,
        time_scale_factor=1.0,
        max_tokens=7168,
    )
    requests = TraceReplayRequestGenerator(config).generate()
    assert len(requests) == 128
    assert [request.arrived_at for request in requests] == list(range(128))
    assert all(request.num_prefill_tokens == 6144 for request in requests)
    assert all(request.num_decode_tokens == 1024 for request in requests)
    assert sum(request.num_prefill_tokens + request.num_decode_tokens for request in requests) == 917504


def test_canonical_fixed_builder_reproduces_every_frozen_byte():
    scratch_root = Path(os.environ.get("FRONTIER_TMP_ROOT", "/data/ycfeng/tmp"))
    with tempfile.TemporaryDirectory(prefix="canonical_fixed_", dir=scratch_root) as scratch:
        subprocess.run([sys.executable, str(BUILDER), "--output-dir", scratch],
                       cwd=REPO_ROOT, check=True, capture_output=True, text=True)
        rebuilt = Path(scratch)
        assert sorted(path.name for path in rebuilt.iterdir()) == sorted(path.name for path in CASE_DIR.iterdir())
        for frozen in CASE_DIR.iterdir():
            assert (rebuilt / frozen.name).read_bytes() == frozen.read_bytes(), frozen.name
