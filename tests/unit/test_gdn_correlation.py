import gzip
import json
from pathlib import Path

import pytest

from frontier.validation.batch_record import BatchRecord
from frontier.validation.gdn_correlation import correlate
from tests.unit.test_gdn_sglang_trace import _write_trace


MODEL = "Qwen3.8-2.4T-A95B-Quark-MXFP4"


def capture_fixture(tmp_path: Path):
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "manifest.json").write_text(json.dumps({
        "status": "complete", "model_path": MODEL, "versions": {"sglang": "test"},
        "topology": {"tp": 8, "ep": 1, "pp": 1, "nodes": 1},
    }))
    records = []
    for size in (16, 24, 32):
        events, offset = [], 0.0
        for step, phase in enumerate(("prefill", "decode", "decode")):
            template = _write_trace(tmp_path, phase, repeats=69)
            with gzip.open(template, "rt") as stream:
                kernels = json.load(stream)["traceEvents"]
            batch_id = f"b{size}-s{step}"
            # A linear, analytically checkable latency curve in each phase.
            scale = size / 16
            for kernel in kernels:
                kernel["ts"] = offset + float(kernel["ts"]) * scale + 1
                kernel["dur"] *= scale
                kernel["args"] = {"device": 0}
            duration = kernels[-1]["ts"] + kernels[-1]["dur"] - offset + 1
            events.append({"name": f"frontier.batch:{batch_id}", "ph": "X",
                           "cat": "user_annotation", "ts": offset, "dur": duration})
            events.extend(kernels)
            offset += duration + 10
            record = BatchRecord(batch_id=batch_id, rank=0, phase=phase,
                request_ids=tuple(str(i) for i in range(size)),
                query_lens=(1024 if phase == "prefill" else 1,) * size,
                context_lens=(0 if phase == "prefill" else 1024 + step - 1,) * size,
                prefill_mask=(phase == "prefill",) * size,
                graph_mode="NONE" if phase == "prefill" else "FULL",
                capture_size=0 if phase == "prefill" else size, profiled=True, step=step)
            records.append(record)
        with gzip.open(capture / f"b{size}-rank0.trace.json.gz", "wt") as stream:
            json.dump({"traceEvents": events}, stream)
    (capture / "batches.jsonl").write_text("".join(json.dumps(r.to_dict()) + "\n" for r in records))
    return capture


def test_trace_calibration_pools_repeated_decode_rows_and_excludes_validation(tmp_path):
    frame, summaries, rows, report = correlate(capture_fixture(tmp_path), model=MODEL,
        device="mi355x", calibration_sizes={16, 32}, validation_sizes={24})
    assert set(frame["batch_size"]) == {16, 32}
    assert len(frame) == 4  # repeated decode passes pool samples before fitting
    assert len(rows) == 3
    assert len(summaries) == 9
    assert max(row["absolute_error_pct"] for row in rows) < 1e-10
    assert report["by_phase"]["decode"]["batches"] == 2


def test_calibration_rejects_overlap_and_missing_validation_shapes(tmp_path):
    path = capture_fixture(tmp_path)
    with pytest.raises(ValueError, match="disjoint"):
        correlate(path, model=MODEL, device="mi355x", calibration_sizes={16, 24}, validation_sizes={24})
    with pytest.raises(ValueError, match="Missing complete"):
        correlate(path, model=MODEL, device="mi355x", calibration_sizes={16, 32}, validation_sizes={8})
