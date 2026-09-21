## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-13 | Added validation evidence for the independent Nsight CSV breakdown parser. |

# Nsight Breakdown Parser Validation

## Execution

- Environment: system Python `3.12.3` (`GCC 13.3.0`), repository worktree `issue26-ttft-h200-20260907`.
- Syntax check:

  ```bash
  python -m py_compile tests/e2e/issue26_nsys_breakdown.py
  ```

- Behavior check: generated a temporary CSV with the actual `cuda_gpu_trace` header and five activities, two JSON control markers (`start=1000`, `stop=5000`), and two device labels. Ran:

  ```bash
  python tests/e2e/issue26_nsys_breakdown.py \
    --trace-csv /data/ycfeng/tmp/issue26-nsys-breakdown-NY255G/trace2.csv \
    --start-marker /data/ycfeng/tmp/issue26-nsys-breakdown-NY255G/start \
    --stop-marker /data/ycfeng/tmp/issue26-nsys-breakdown-NY255G/stop \
    --output /data/ycfeng/tmp/issue26-nsys-breakdown-NY255G/report2.json
  ```

## Criteria

- Require the parser to accept the observed Nsight CSV column names.
- Require event intervals to be clipped to `[start, stop]` before union accounting.
- Require separate compute, communication, memory, and unknown categories.
- Require per-device union duration, activity envelope, idle remainder, and cross-category overlap.
- Require unknown names and device IDs to remain explicit in the report.

## Evidence

`PASS`. The report contained two devices, selected all five overlapping rows, and retained one blank-name event as `unknown`. For `H200 (0)`, the report measured:

| Quantity | Value |
| --- | ---: |
| Compute union | 3,000 ns |
| Communication union | 800 ns |
| Memory union | 1,400 ns |
| Unknown union | 0 ns |
| All activity union | 4,000 ns |
| Activity envelope | `[1,000, 5,000)` ns |
| Idle/non-activity | 0 ns |
| Cross-category overlap | 1,200 ns |

The values include clipping of the activity that begins before `start` and the activity that ends after `stop`. The parser also records `device_id=0` from `H200 (0)`, marks name-based compute/communication rules as heuristic, and reports the marker/time-domain limitations in the JSON output.

This synthetic check establishes parser behavior only. It does not establish that a complete H200 Nsight capture exists or that host wall-clock markers are aligned with the target's raw Nsight clock domain.
