"""Check one worker's personal-volume append behavior using retained serialized rows."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_trace(path, buffer):
    with path.open("a", encoding="utf-8") as stream:
        stream.writelines(buffer)
    buffer.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    settings = plan["append"]
    personal = Path(plan["cloud_personal_root"])
    root = Path(plan["cloud_test_root"])
    require(personal == Path("/mnt/codesign-exp/ycfeng") and root.is_relative_to(personal),
            "Cloud path is outside the personal volume")
    require(personal.is_dir(), "Personal cloud volume is unavailable")
    root.mkdir(parents=True, exist_ok=False)
    ledger_path, trace_path = root / "ledger.jsonl", root / "operators.jsonl"
    receipt = {"task_id": plan["task_id"], "status": "IN_PROGRESS",
        "started_utc": datetime.now(timezone.utc).isoformat(), "cloud_root": str(root),
        "reference_only": True, "production_pricing_calls": 0, "production_emission_calls": 0,
        "trace_buffer_events": settings["trace_buffer_events"], "files": []}
    try:
        ledger_spec = settings["reference_ledger"]
        require(file_hash(Path(ledger_spec["path"])) == ledger_spec["sha256"], "Ledger reference changed")
        ledger_rows = json.loads(Path(ledger_spec["path"]).read_bytes())
        ledger_lines = [json.dumps(row, sort_keys=True) + "\n" for row in ledger_rows]
        op_spec = settings["reference_operators"]
        with Path(op_spec["path"]).open("rb") as stream:
            for _ in range(op_spec["skip_header_lines"]):
                stream.readline()
            op_lines = [stream.readline() for _ in range(op_spec["record_count"])]
        require(all(op_lines) and hashlib.sha256(b"".join(op_lines)).hexdigest() == op_spec["sha256"],
                "Operator reference changed")
        op_lines = [line.decode("utf-8") for line in op_lines]
        require(len(ledger_lines) == 16 and len(op_lines) == 208, "Incomplete reference wave")
        wave_bytes = sum(len(line.encode()) for line in ledger_lines + op_lines)
        trace_path.open("x").close()
        expected_hashes = {ledger_path: hashlib.sha256(), trace_path: hashlib.sha256()}
        buffer = []
        written = repetitions = 0
        started = time.monotonic()
        with ledger_path.open("x", encoding="utf-8") as ledger_stream:
            while time.monotonic() - started < settings["deadline_s"] and written + wave_bytes <= settings["max_bytes"]:
                for line in ledger_lines:
                    ledger_stream.write(line)
                    ledger_stream.flush()
                    expected_hashes[ledger_path].update(line.encode())
                for line in op_lines:
                    buffer.append(line)
                    expected_hashes[trace_path].update(line.encode())
                    if len(buffer) == settings["trace_buffer_events"]:
                        append_trace(trace_path, buffer)
                repetitions += 1
                written += wave_bytes
            if buffer:
                append_trace(trace_path, buffer)
        appended = time.monotonic()
        require(repetitions > 0, "No complete serialized wave was appended")
        for path in (ledger_path, trace_path):
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        synced = time.monotonic()
        for path, expected in expected_hashes.items():
            actual = file_hash(path)
            require(actual == expected.hexdigest(), f"Read-back hash mismatch: {path.name}")
            receipt["files"].append({"path": str(path), "bytes": path.stat().st_size,
                "sha256": actual, "read_back": "PASS"})
        verified = time.monotonic()
        require(sum(item["bytes"] for item in receipt["files"]) == written, "Output byte count differs")
        receipt.update(status="PASS_SERIALIZED_REFERENCE_APPEND_ONLY", repetitions=repetitions,
            bytes=written, ledger_rows=16 * repetitions, operator_rows=208 * repetitions,
            append_and_close_s=appended-started, fsync_s=synced-appended,
            read_back_s=verified-synced, append_bytes_per_s=written/(appended-started),
            append_with_fsync_bytes_per_s=written/(synced-started),
            whole_cli_storage_floor_from_projection_bytes_per_s=
                plan["sample_projection"]["bytes_per_mode"] / plan["sample_projection"]["main_subprocess_cap_s"],
            full_workload_memory_verified=False, cleanup="PENDING")
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        for path in (ledger_path, trace_path):
            require(path.parent == root and path.is_relative_to(personal), "Invalid cleanup path")
            path.unlink()
        root.rmdir()
        receipt["cleanup"] = "PASS_TWO_COMPLETED_TEST_FILES_REMOVED"
    except BaseException as error:
        receipt["status"] = "FAIL"
        receipt["error"] = f"{type(error).__name__}: {error}"
        receipt["cleanup"] = "PRESERVE_UNVERIFIED_FILES_FOR_SCOPED_DIAGNOSIS"
        args.receipt.with_suffix(".failure.log").write_text(traceback.format_exc())
        raise
    finally:
        receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
        receipt["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "bytes", "append_bytes_per_s", "cleanup")}))


if __name__ == "__main__":
    main()
