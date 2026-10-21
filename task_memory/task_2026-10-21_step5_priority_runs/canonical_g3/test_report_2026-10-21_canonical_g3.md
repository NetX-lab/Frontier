# D21 canonical G3 capture CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded current-source CPU/source/caller results and nonqualification limits. |

## Environment and exact command

Worktree `/data/ycfeng/Frontier/.worktrees/step5-canonical-g3`, initialb6755f7; Linux; existing `/data/ycfeng/envs/asyncopd-py312/bin/python` Python3.12.13; no conda activation or dependency changes. The source-bound corrected R2 replay test and fixtures are read from `/data/ycfeng/Frontier/.worktrees/step5-r2-correction`, never edited. PYTHONPATH explicitly points to this new worktree. All tests execute on CPU; HTTP cases use localhost aiohttp mock servers.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-canonical-g3 \
/data/ycfeng/envs/asyncopd-py312/bin/python -m pytest \
  tests/unit/test_step5_canonical_g3.py \
  tests/unit/test_step5_native_pd_r2_stage.py \
  tests/unit/test_step5_native_pd_r2_observation.py \
  tests/unit/test_step5_native_pd_r2_verdict.py \
  tests/unit/test_step5_native_pd.py tests/unit/test_native_pd_worker.py \
  tests/unit/test_calibration_pd_harness.py \
  /data/ycfeng/Frontier/.worktrees/step5-r2-correction/tests/unit/test_step5_native_pd_r2_trace_replay.py \
  --import-mode=importlib -q --tb=short -p no:cacheprovider \
  --basetemp=/data/ycfeng/tmp/d21_g3_cpu_native_final
```

Measured by a one-off Python parent invoking the exact argv in `cpu_native_verification.json` with `subprocess.run`, `time.monotonic()` and `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`. This is maximum childRSS, not an aggregate cgroup/process-tree measurement. `/usr/bin/time` was not available (exit127); no tests ran in that failed timing invocation.

## Expected vs observed

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Focused native/caller tests | >=50; all selected pass |612 passed in26.87s, exit0 | PASS |
| New canonical G3 module | Shape/caller/negative-control regressions pass |53 cases pass | PASS |
| CPU harness wall | <120s, preregistered before final measurement |28.345140215s | PASS |
| CPU harness maximum childRSS | <2097152KiB |779000KiB | PASS |
| Synthetic actual builder/standalone R2 verifier |32 requests;5 content/27 fence-free; canonical config/IDs retained | Exactly these values; G3false, diagnostic-only | PASS, unit fixture only |
| Original prompt template classes | Original16 warmup and8 formal IDs; disjoint | Exact equality against shared canonical loader | PASS |
| Native client body path | P max1, unchanged actual P transfer metadata to D, D6144/1024, token-ID accounting |8 concurrent mock HTTP P/D/proxy completions all satisfy existing canonical validator | PASS, not native compute |
| Pending-READ cancellation caller | Proxy closes native D HTTP client without first token/READ fence | Mock D received one request; canceled handler/closed transport; client0 output IDs and explicit cancellation | PASS, not actual READ |
| Abort conservation consumer | Native pending state+same-host overlapping READ;8 completions/readiness; notifications/free/pool counts | Positive synthetic case passes;9 negative cases reject | PASS, not native lifetime evidence |
| P6128/D6128+16 attribution | Use observed truncated P count once, not subtract another16 |6143/6144/6145 cases pass correct prefix and reject +/-16;6144 gives6128 external and16 local | PASS, regression values only |
| P1 policy preservation | Exact c0a0e63 token/retirement/free/lifetime logic | Six functions AST-identical; retained native AST/trace controls included | PASS |
| Temporary debug cleanup | No DEBUG-g3abort text | grep returned no matches | PASS |
| Actual new native capture/reference/G3 | Requires platform execution and complete reference/controls | Not run; no new native witnesses or independent reference | UNVERIFIED / incomplete |

The CPU fixtures fabricate only test inputs under the pytest temporary directory and clearly retain sourcecommit=unit. They are not retained as historical/native proof and are not a real-image offline-sealed package.

## Additional recorded iteration outcomes

- Initial capture slice469 PASS; expanded native/builder suite527 PASS before the abort extension. Neither supersedes the final612 pass result.
- One new test initially requested nonexistent `local_tail_tokens`; existing returned prompt/external counts settle the invariant. The common canonical loader returns `(manifest, formal, warmup)`; implementation and assertions use that order and compare disjoint original classes.
- Pending abort tests initially failed because the positive synthetic abort timestamp omitted the fixture's clock base and because the actual mock D handler was canceled on disconnect before the test set its event. Falsifiable probe showed received=1/transport=None/task.cancelling=1. Fixes were to test event timing correctly and recognize a genuine canceled native-client connection, not weaken conservation or timeout semantics. Production READ and allocator code are unchanged.
- Optional broad run also selected `test_canonical_client_metrics.py` and `test_step5_client_metrics.py`:631 PASS,27 setup ERROR,35.69s. All errors share unchanged `frontier/metrics/metrics_store.py:8 -> import plotly.express` and `ModuleNotFoundError: No module named 'plotly'`. Exact argv/resource values remain in `cpu_verification.json`. No environment repair, simulator inspection/edit or claim that these27 passed. A scoped affected native/caller rerun produced the final612 PASS above.

## Limits / disposition

No native kernels, FP8 policy, thresholds, sealed01/02 package, simulator or profiling source changed. Existing auxiliary callback guards and run-worker failure propagation remain; cancellation is explicitly diagnostic and never a default/native-training exception fallback. New output/accounting checks are auxiliary verdicts only, not platform stop gates.

No fresh actual stage was sealed because independent functional reference and remaining reset/reuse/pressure component/alias controls are incomplete. No platform submit command is valid yet. Original P0/H200 outcomes are not edited; parent owns qualification03 and any later GPU actions. Reference choice is isolated in `reference_decision.md`.
