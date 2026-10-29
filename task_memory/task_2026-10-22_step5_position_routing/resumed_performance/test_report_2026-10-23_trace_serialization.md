# Parent trace serialization verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Verify reviewed source, exact trace bytes and complete actual proofs; accept and integrate the bounded component. |

## Source and environment

Code `390bb5a2`, reviewed tree `9ae60ebc`, integration `64881373`.
Candidate worktree `/data/ycfeng/Frontier/.worktrees/step5-d29-trace-serialization`.
Python `/data/ycfeng/envs/frontier-py310/bin/python`, CPython3.10.6. Heavy checks
use60s outer limits and MemoryMax30G, with the Python processes running as brainpp.
Source SHA256 is `07d7cbbb081829f558b6a4c101201baf112e5387e2045ee244436853fb463a0d`.
Merged source/test/config bytes match the reviewed tree.

## Exact commands

Run from the candidate worktree, except the archive command which runs from
`/data/ycfeng/Frontier/.worktrees/step5-preview`:

```bash
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-d29-trace-serialization PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_measured_attention_root_metrics.py tests/unit/test_dsv4_structural_execution.py tests/unit/test_dsv4_cache_zero_execution.py tests/unit/test_mla_core_native_op_tracing.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_metrics_stage_execution_time.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_trace_serialization_parent01/pytest --junitxml=/data/ycfeng/tmp/d29_trace_serialization_parent01/pytest.xml > /data/ycfeng/tmp/d29_trace_serialization_parent01/pytest.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_trace_serialization_parent01/verify_trace.py > /data/ycfeng/tmp/d29_trace_serialization_parent01/verification.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_trace_serialization_parent01/verify_wave.py > /data/ycfeng/tmp/d29_trace_serialization_parent01/wave02.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_trace_serialization_parent01/compare.py > /data/ycfeng/tmp/d29_trace_serialization_parent01/comparison.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_trace_serialization_parent01/archive.py
```

The trace wrapper changes only the owner's whitespace-check command to inspect
the source path: committed raw logs retain their original whitespace. The wave
wrapper changes only the pre-edit HEAD assertion and result revision to reviewed
`9ae60ebc`; the exact source hash and all business assertions remain. Both original
and derived script identities are retained in `trace_parent_acceptance.json`.

Before that explicit rebind, this command failed with the old HEAD assertion at
wave-candidate.py:23, before configuration parsing:

```bash
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_trace_serialization_repair/wave-candidate.py --output /data/ycfeng/tmp/d29_trace_serialization_parent01/wave --iterations 2 > /data/ycfeng/tmp/d29_trace_serialization_parent01/wave.log 2>&1
```

The failure log and empty output directory remain preserved; they are not a pass.

## Expected and observed

| Check | Passing condition | Observed | Result |
| --- | --- | --- | --- |
| Existing test statuses | No change from retained baseline |98PASS,1same mapper failure;99/99 statuses equal | PASS comparison; raw suite FAIL |
| Real-disk events | Same complete event bytes after timestamped header |4 files,8 events identical | PASS |
| Captured serialization | Same598 events and required bytes/hash |8,353,308 bytes; SHA256 `3ea6239cea9295e5d532722200c78238c9e01ecb58b32383fe8c8aaa20b7e004` | PASS |
| Fields and failures | Same declared order/None filtering/error identity/restoration |All checks pass; dynamic attributes excluded | PASS |
| Complete actual wave | Same layer60/position6147/16-EP durations and all proof fields |208/208 pairs in each of2 waves;0 mismatches | PASS |
| Whole event output | Same baseline event bytes |416 events,5,780,831 bytes; SHA256 `a9c36f5ad944f6d3bf27afa4db9b874e4c8c7fb6a184c7707bd1bff113bb2c5d` | PASS |
| Source scope |Only allocated AST delta and<=20 gross |7 added/7 deleted;14gross; source whitespace clean | PASS |

Known raw failure is
`test_mla_core_native_op_tracing.py::test_frontier_stage_batch_component_ledger_dense_attention_uses_shared_mapper`:
the old test-local fake rejects `allow_structural`. No fixture or mapper was changed.

The598-event alternating-order medians are0.370868s baseline and0.066694s candidate.
The parent actual wave's warm emit/flush times are0.059494s/0.023546s. These are
local measurements; pricing2.703395s is unchanged code and is not credited to this
repair. Full pricing/proof/plan payload SHA256 remains
`3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83`.

## Acceptance and limits

Both independent review axes report No findings. Parent accepts the trace
component and merges it at64881373. Git-derived accounting from the accepted79
audit plus the sole14-line source delta gives native1350/CORE3984/PRICING3920,
source9254 and global9268. The prepared-point allowance is not measured consumption.

Raw logs, XML, comparison JSON and source bindings are indexed and hashed by
`trace_parent_acceptance.json`. Retained-main08 comparisons are simulator replay
checks, not GPU ground truth. Prepared implementation, residual collective/trace
diagnosis, final-union regressions, complete CLI and native/G3/E2E gates remain open.
