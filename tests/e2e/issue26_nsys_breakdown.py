"""Break down a clipped Nsight Systems CUDA activity window.

The ``cuda_gpu_trace`` report is the primary input because it retains event
timestamps, durations, and a device label.  Nsight summary reports are
aggregated by operation and cannot provide the interval accounting needed for
this report.  This script intentionally treats the control marker timestamps
as an external window and clips every selected activity to that window before
performing interval unions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REQUIRED_COLUMNS = ("Start (ns)", "Duration (ns)", "Device", "Name")
MEMORY_ACTIVITY_RE = re.compile(r"\[cuda\s+(?:memcpy|memset)\b", re.IGNORECASE)
MEMORY_KERNEL_RE = re.compile(
    r"(?:memcpy|memset|copy_|copykernel|\bcast\b|\bconvert\b|"
    r"\btranspose\b|index_select)",
    re.IGNORECASE,
)
COMMUNICATION_RE = re.compile(
    r"(?:nccl|all.?reduce|all.?to.?all|alltoall|broadcast|reduce.?scatter|"
    r"allgather|cross.?device.?reduce|\bsend\b|\brecv\b|\bscatter\b|"
    r"\bgather\b|collective)",
    re.IGNORECASE,
)
DEVICE_ID_RE = re.compile(r"\((\d+)\)\s*$")


def read_marker(path: Path) -> int:
    """Read a wall-clock ``time_ns`` value from a control marker."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = int(payload["time_ns"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid control marker {path}: expected integer time_ns") from exc
    return value


def parse_number(value: str, *, field: str, row_number: int) -> float:
    """Parse a finite Nsight numeric field, accepting blank optional fields."""
    text = value.strip()
    if not text:
        raise ValueError(f"row {row_number}: {field} is empty")
    try:
        number = float(text.replace(",", ""))
    except ValueError as exc:
        raise ValueError(f"row {row_number}: invalid {field} value {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"row {row_number}: non-finite {field} value {value!r}")
    return number


def as_ns(value: float, *, field: str, row_number: int) -> int:
    """Convert an Nsight timestamp/duration to integer nanoseconds."""
    if value < 0:
        raise ValueError(f"row {row_number}: negative {field} value {value}")
    return int(round(value))


def classify(name: str) -> tuple[str, str, bool]:
    """Return category, rule name, and whether the rule is heuristic."""
    lowered = name.casefold()
    if not lowered:
        return "unknown", "missing_name", False
    if MEMORY_ACTIVITY_RE.search(name):
        return "memory", "cuda_memcpy_or_memset_activity", False
    # CUPTI can report copy/transform work as a kernel rather than a memcpy
    # activity.  Keep this explicit heuristic visible in the output.
    if MEMORY_KERNEL_RE.search(name):
        return "memory", "memory_kernel_name_heuristic", True
    if COMMUNICATION_RE.search(name):
        return "communication", "collective_or_p2p_name", True
    return "compute", "remaining_named_gpu_activity", True


def union_duration(intervals: Iterable[tuple[int, int]]) -> tuple[int, int | None, int | None]:
    """Return union duration and its first/last boundaries."""
    ordered = sorted(intervals)
    if not ordered:
        return 0, None, None
    total = 0
    current_start, current_end = ordered[0]
    first = current_start
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    total += current_end - current_start
    return total, first, current_end


def device_identity(label: str) -> tuple[str, int | None]:
    """Preserve Nsight's device label and extract a trailing numeric ID."""
    match = DEVICE_ID_RE.search(label)
    return label, int(match.group(1)) if match else None


def analyze(
    trace_csv: Path,
    start_marker: Path,
    stop_marker: Path,
    output: Path,
    timestamp_offset_ns: int = 0,
) -> dict[str, Any]:
    """Parse and report interval accounting for one control window."""
    window_start = read_marker(start_marker)
    window_stop = read_marker(stop_marker)
    if window_start >= window_stop:
        raise ValueError(
            f"control window must be positive: start={window_start}, stop={window_stop}"
        )

    by_device: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "intervals": defaultdict(list),
            "operation_durations": defaultdict(lambda: defaultdict(int)),
            "event_counts": defaultdict(int),
            "heuristic_counts": defaultdict(int),
            "raw_rows": 0,
        }
    )
    input_rows = 0
    skipped_rows = 0
    selected_rows = 0
    unknown_rows = 0
    unknown_duration_ns = 0

    with trace_csv.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"trace CSV has no header: {trace_csv}")
        missing = [field for field in REQUIRED_COLUMNS if field not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"trace CSV is not cuda_gpu_trace: missing columns {missing}; "
                f"found {reader.fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            input_rows += 1
            raw_start = as_ns(
                parse_number(row["Start (ns)"], field="Start (ns)", row_number=row_number),
                field="Start (ns)",
                row_number=row_number,
            )
            duration = as_ns(
                parse_number(row["Duration (ns)"], field="Duration (ns)", row_number=row_number),
                field="Duration (ns)",
                row_number=row_number,
            )
            if duration <= 0:
                skipped_rows += 1
                continue
            start = raw_start + timestamp_offset_ns
            end = start + duration
            # Filter-time selects overlapping rows, but does not clip their
            # durations.  Apply the clip here before any union accounting.
            clipped_start = max(start, window_start)
            clipped_end = min(end, window_stop)
            if clipped_start >= clipped_end:
                skipped_rows += 1
                continue
            device = (row.get("Device") or "").strip() or "<unknown-device>"
            name = (row.get("Name") or "").strip()
            category, rule, heuristic = classify(name)
            state = by_device[device]
            state["raw_rows"] += 1
            state["intervals"][category].append((clipped_start, clipped_end))
            state["operation_durations"][category][name or "<unknown>"] += (
                clipped_end - clipped_start
            )
            state["event_counts"][category] += 1
            if heuristic:
                state["heuristic_counts"][rule] += 1
            selected_rows += 1
            if category == "unknown":
                unknown_rows += 1
                unknown_duration_ns += clipped_end - clipped_start

    devices: list[dict[str, Any]] = []
    for device in sorted(by_device):
        state = by_device[device]
        intervals: dict[str, list[tuple[int, int]]] = state["intervals"]
        category_union: dict[str, int] = {}
        category_boundaries: dict[str, dict[str, int | None]] = {}
        for category in ("compute", "communication", "memory", "unknown"):
            duration, first, last = union_duration(intervals.get(category, []))
            category_union[category] = duration
            category_boundaries[category] = {"start_ns": first, "end_ns": last}
        all_intervals = [interval for values in intervals.values() for interval in values]
        all_union, envelope_start, envelope_end = union_duration(all_intervals)
        envelope_ns = (
            envelope_end - envelope_start
            if envelope_start is not None and envelope_end is not None
            else 0
        )
        category_total = sum(category_union.values())
        devices.append(
            {
                "device": device,
                "device_id": device_identity(device)[1],
                "event_count": sum(state["event_counts"].values()),
                "event_count_by_category": {
                    category: state["event_counts"].get(category, 0)
                    for category in ("compute", "communication", "memory", "unknown")
                },
                "unknown_event_count": state["event_counts"].get("unknown", 0),
                "heuristic_classification_count": sum(state["heuristic_counts"].values()),
                "category_union_ns": category_union,
                "category_boundaries": category_boundaries,
                "all_activity_union_ns": all_union,
                "all_kernel_or_activity_union_ns": all_union,
                "unknown_duration_ns": category_union["unknown"],
                "activity_envelope_start_ns": envelope_start,
                "activity_envelope_end_ns": envelope_end,
                "activity_envelope_ns": envelope_ns,
                "idle_or_non_activity_ns": max(0, envelope_ns - all_union),
                "category_overlap_ns": max(0, category_total - all_union),
                "top_operations": {
                    category: [
                        {"name": name, "clipped_duration_ns": duration}
                        for name, duration in sorted(
                            operations.items(), key=lambda item: (-item[1], item[0])
                        )[:20]
                    ]
                    for category, operations in state["operation_durations"].items()
                },
            }
        )

    report: dict[str, Any] = {
        "status": "PASS",
        "trace_csv": str(trace_csv),
        "control_markers": {"start": str(start_marker), "stop": str(stop_marker)},
        "window": {
            "start_ns": window_start,
            "stop_ns": window_stop,
            "control_window_ns": window_stop - window_start,
        },
        "timestamp_alignment": {
            "timestamp_offset_ns": timestamp_offset_ns,
            "meaning": "adjusted_trace_timestamp = csv_timestamp + timestamp_offset_ns",
            "marker_domain": "wall_clock_time_ns",
        },
        "rows": {
            "input_rows": input_rows,
            "selected_rows_after_clip": selected_rows,
            "skipped_rows": skipped_rows,
            "unknown_event_count": unknown_rows,
            "unknown_duration_ns": unknown_duration_ns,
        },
        "devices": devices,
        "schema": {
            "required_columns": list(REQUIRED_COLUMNS),
            "primary_report": "cuda_gpu_trace",
            "filter_time_semantics": "overlap selection; event durations are clipped by this parser",
        },
        "limits": [
            "Device labels do not establish DP/TP/rank identity; device_id is retained only when Nsight provides it.",
            "Communication and copy/transform classification uses operation-name heuristics and is reported as such.",
            "cuda_api_sum and aggregate summary reports are not included in GPU interval accounting.",
            "The control marker window is a host wall-clock boundary and may not coincide exactly with GPU event boundaries.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--start-marker", type=Path, required=True)
    parser.add_argument("--stop-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--timestamp-offset-ns",
        type=int,
        default=0,
        help="add this offset to CSV timestamps before clipping to markers",
    )
    args = parser.parse_args()
    analyze(
        args.trace_csv,
        args.start_marker,
        args.stop_marker,
        args.output,
        args.timestamp_offset_ns,
    )


if __name__ == "__main__":
    main()
