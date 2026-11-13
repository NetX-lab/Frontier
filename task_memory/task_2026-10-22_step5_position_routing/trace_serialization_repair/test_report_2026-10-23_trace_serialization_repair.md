# D29 trace serialization repair focused verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record exact source binding, byte/proof differentials, focused tests, measurements, and limits. |

TaskID: `D29-TRACE-SERIALIZATION-REPAIR`.
Code commit: `390bb5a2`; parent: `767da5437609301a6183dbfdcfd87b2402a32007`.
Worktree: `/data/ycfeng/Frontier/.worktrees/step5-d29-trace-serialization`.
Scratch root: `/data/ycfeng/tmp/d29_trace_serialization_repair`.

## Source and environment

Python: `/data/ycfeng/envs/frontier-py310/bin/python`, CPython 3.10.6,
existing `frontier-py310` environment; no conda environment or dependencies changed.
Baseline `trace_store.py` SHA256:
`f66f9df95422ad734707e723a38a516143e90fe80274c80dd169babf9a375939`.
Candidate SHA256:
`07d7cbbb081829f558b6a4c101201baf112e5387e2045ee244436853fb463a0d`.
The successful checks imported the candidate by its exact current-worktree path
and verified this hash. At execution time Git HEAD was parent767da543 with the
single owned source modification; the code commit subsequently contains exactly
those verified bytes. Baseline serialized from `git show 767da543:...`; the full
wave baseline imports frozen79, whose relevant `frontier/` source matches767da543.

All memory-heavy commands use `timeout 60s` and
`sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp --`.
Probe scripts also set `RLIMIT_AS=30 GiB`. They run as brainpp, not root.
Unprivileged system scope initially failed with `Interactive authentication
required`; the user scope failed with `Job failed`. Noninteractive sudo scope
worked. No daemon, permission, or runtime configuration was changed.

The new worktree initially had an empty collective-sim submodule directory.
The candidate wave failed before pricing with
`ModuleNotFoundError: No module named 'collective_sim_core'`. It was populated
locally, without network or changes to the peer, at the already-pinned gitlink:

```bash
git clone --shared --no-checkout /data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06/frontier/cc_backend/backends/collective-sim /data/ycfeng/Frontier/.worktrees/step5-d29-trace-serialization/frontier/cc_backend/backends/collective-sim
git -C frontier/cc_backend/backends/collective-sim switch --detach 3dac53c544e546574cb6e8078cd14053d0916113
```

## Exact successful commands

Run from the candidate worktree. All scripts are new scratch files; existing
source scripts and fixtures were read-only. Original script identities and
explicit source/sample transformations are in `evidence/wave-binding.json` and
`evidence/trace-verification.json`. The wave adapters remove the old diagnostic
flag and JSON/read interception, leaving real pricing, producer, proofs/plans,
and `TraceStore.flush` in use. No runtime diagnostic flag is passed.

```bash
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-d29-trace-serialization PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_measured_attention_root_metrics.py tests/unit/test_dsv4_structural_execution.py tests/unit/test_dsv4_cache_zero_execution.py tests/unit/test_mla_core_native_op_tracing.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_metrics_stage_execution_time.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_trace_serialization_repair/pytest-baseline --junitxml=/data/ycfeng/tmp/d29_trace_serialization_repair/pytest-baseline.xml > /data/ycfeng/tmp/d29_trace_serialization_repair/pytest-baseline.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-d29-trace-serialization PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_measured_attention_root_metrics.py tests/unit/test_dsv4_structural_execution.py tests/unit/test_dsv4_cache_zero_execution.py tests/unit/test_mla_core_native_op_tracing.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_metrics_stage_execution_time.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d29_trace_serialization_repair/pytest-candidate --junitxml=/data/ycfeng/tmp/d29_trace_serialization_repair/pytest-candidate.xml > /data/ycfeng/tmp/d29_trace_serialization_repair/pytest-candidate.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_trace_serialization_repair/verify_trace.py /data/ycfeng/tmp/d29_trace_serialization_repair/verified02 > /data/ycfeng/tmp/d29_trace_serialization_repair/trace-verification02.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_trace_serialization_repair/prepare_wave.py
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_trace_serialization_repair/wave-baseline.py --output /data/ycfeng/tmp/d29_trace_serialization_repair/wave-baseline01 --iterations 5 > /data/ycfeng/tmp/d29_trace_serialization_repair/wave-baseline01.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_trace_serialization_repair/wave-candidate.py --output /data/ycfeng/tmp/d29_trace_serialization_repair/wave-candidate02 --iterations 5 > /data/ycfeng/tmp/d29_trace_serialization_repair/wave-candidate02.log 2>&1
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_trace_serialization_repair/compare_outputs.py > /data/ycfeng/tmp/d29_trace_serialization_repair/comparison.log 2>&1
timeout 60s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/d29_trace_serialization_repair/budget.py
git diff --check 767da543
git diff --numstat 767da543 -- frontier/metrics/trace_store.py
```

The baseline pytest command ran before the source edit. The first scratch
verification attempt finished byte/producer/failure checks but failed in the
auxiliary AST assertion: `ast.dump` does not accept lists. The scratch assertion
was corrected to dump each node, then all checks reran in fresh `verified02`.
The first artifact comparison counted pytest's `current` symlink twice;
excluding that symlink fixed the census without changing any product/test input.
Failed logs remain in scratch and are not reported as passes.

## Expectations and observations

| Check | Passing condition | Actual | Result |
| --- | --- | --- | --- |
| Focused pytest baseline/candidate | Exact same case/status and known failure set | Both 98 passed / 1 failed; 99/99 statuses identical; 9.19s / 9.39s | PASS comparison; raw suites FAIL |
| Real disk root trace | All event bytes and complete metadata identical | 4 files, 8 events, byte-identical after header | PASS |
| Captured DECODE replay | 598 events, 8,353,308 bytes, required SHA256 | Exact bytes in all seven alternating-order baseline/candidate iterations | PASS |
| Real `record_ep_wave` producer | Full payload bytes and JSON-compatible metadata | 7 events, 21,051 bytes; no unsupported leaf | PASS |
| Field contract | Declared key order, top-level None omission only | Nested null/tuple preserved; dynamic attribute excluded; no `to_dict` | PASS |
| Failure handling | Same exception identity and buffer restoration order | Injected OSError rethrown unchanged; pending events precede concurrent additions; unsupported-object TypeError restores buffer | PASS |
| Complete actual layer60 wave | 208/208 matching duration and full proofs | Exact pair set; 0 mismatches; 5 iterations, 1,040 events, baseline/candidate bytes identical | PASS |
| Scope/AST/whitespace | Only allowed method/import/flush-body delta; <=20 gross | +7/-7=14; exact AST delta; unchanged handler and all other methods/BaseEvent; whitespace clean | PASS |
| Donor allowance | Untouched fixed-probe file below45 | +40/-1=41 from D29 baseline; unchanged by repair | PASS |

Known failure:
`tests/unit/test_mla_core_native_op_tracing.py::test_frontier_stage_batch_component_ledger_dense_attention_uses_shared_mapper`.
The test-local fake rejects `allow_structural`, called at
`frontier/metrics/metrics_store.py:4482`. It failed identically before and after;
no fixture or mapper source was changed. The real-disk root file has 36 passing
tests in the selected suite. Pytest metadata headers have different wall-clock
timestamps; all event bytes after the headers match. The captured replay fixes
the header clock only in its scratch harness, so whole JSONL bytes also match.

Raw retained pytest logs/XML and the exact `.diff` preserve original trailing
spaces. The record-wide staged whitespace check therefore warns on those raw
artifacts; the source-only and maintained Markdown/JSON checks are clean. Raw
evidence was not trimmed or rewritten after its hashes were recorded.

Captured replay payload SHA256:
`3ea6239cea9295e5d532722200c78238c9e01ecb58b32383fe8c8aaa20b7e004`.
Producer payload SHA256:
`3eff3f2657e05c40e271477e86eda3f3357867f6c9d5ba87de494f39c4597844`.
Full wave pricing/proof/plan payload SHA256 in every iteration:
`3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83`.
Five-wave trace body SHA256, 14,464,252 bytes:
`b4d8f598fb428125be3e813032d33e482f1ecb8fc7afc3bd671858a6f35c6155`.

## Source-based measurements

No profiling or runtime serialization patch was enabled. Captured-event
serialization uses seven alternating-order iterations in the same Python
process; complete-wave baseline/candidate run sequentially, five iterations
each, with warm medians excluding the first iteration.

| Actual operation | Baseline median (s) | Candidate median (s) | Interpretation |
| --- | --- | --- | --- |
| 598-event log/flush | 0.373688 | 0.066070 | 0.307618s less serialization work |
| 208-event wave flush | 0.131004 | 0.023990 | 0.107013s less; 81.69% reduction |
| Wave pricing | 2.773583 | 2.677960 | No pricing change; observed noise, not attributed to this repair |
| Whole bounded wave | 2.965517 | 2.762631 | Observed 6.84% reduction includes pricing noise; not causal CLI speedup evidence |

The supported change is removal of redundant recursive copying, not omission of
proofs. All 208 operator checks compare `collective_plan`, `collective_cost`,
`operator_cost`, `native_context`, `routing_source`, and duration against actual
captured layer60 events. Full proof-bearing traces are retained under
`wave-baseline01/op_traces.jsonl` and `wave-candidate02/op_traces.jsonl` in scratch;
selected comparison evidence contains all 208 per-operator proof hashes.

## Limits and next owner

Captured reads were limited to the retained 8 MiB tail. No full CSV or full
1.5GB trace scan, full simulator CLI, GPU/platform/network action, peer
cherry-pick, additional agent, schema, config, cache, or persistent test was used.
The original contract seam establishes unsupported nested dataclass metadata as
outside the producer domain; this repair deliberately adds no generic fallback.
All supported producer metadata remains unchanged.

The parent owns independent review, final union accounting and source freeze,
strict77/full-unit regressions, complete CLI, and workflow/native qualification.
Focused PASS results do not qualify those pending gates.
