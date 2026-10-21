# D18-NATIVE offline test report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded HTTP/consumer, legacy regressions and native qualification limits. |

## Result and environment

**PASS, source/CPU harness only:** 574 affected tests and 9 narrow simulator client-metric tests (distinct commands). No native latency, H200 capacity, installed-image, per-rank graph or R2 verdict is admitted. No GPU, authentication, platform, cloud, SSH or Docker operation was executed. The tests use local aiohttp HTTP endpoints, not fake request-send counters.

- Workspace: `/data/ycfeng/Frontier/.worktrees/step5-native-pd`.
- Harness Python: `/data/ycfeng/envs/asyncopd-py312/bin/python`, CPython **3.12.13**, Clang22.1.3.
- Existing simulator environment: `/data/ycfeng/envs/frontier-py310/bin/python`, CPython **3.10.6**, GCC11.3.0. These are existing environments; no conda activation, dependency installation or SDK construction was needed.
- `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/data/ycfeng/tmp`, pytest cache disabled. The new native consumer targets the locked image's Python3.12 (`asyncio.timeout`), not the simulator environment.

## Commands and observations

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest \
  tests/unit/test_step5_canonical_native.py tests/unit/test_step5_native_pd.py \
  tests/unit/test_native_pd_worker.py tests/unit/test_calibration_pd_harness.py \
  tests/unit/test_dp_placement_harness.py tests/unit/test_step5_native_pd_r2_stage.py \
  tests/unit/test_step5_native_pd_r2_observation.py tests/unit/test_step5_native_pd_r2_verdict.py \
  tests/unit/test_canonical_fixed_manifest.py -q -p no:cacheprovider \
  --basetemp=/data/ycfeng/tmp/d18_native_tests05
/data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_pd_client_ttft.py \
  -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d18_native_client_metric_tests01
git diff --check
```

Observed: **574 passed in52.47s**, raw stdout `/data/ycfeng/tmp/d18_native_tests05.log`; **9 passed in5.69s**. `git diff --check` passed. Counts include the existing R2 suites solely as regression checks; none of their frozen evidence or verdict rules changed. No simulator code changed and no simulation sweep ran.

After narrowing prompt-ID parsing to the source-confirmed choice-level schema, the three affected suites (`test_step5_canonical_native.py`, `test_calibration_pd_harness.py`, `test_dp_placement_harness.py`) passed **160 tests in39.17s**, using the same Python3.12 command options and `--basetemp=/data/ycfeng/tmp/d18_native_tests06`. This is an overlapping subset, not160 additional unique tests.

| Criterion | Expected | Observed / verdict |
| --- | --- | --- |
| Common package pins | Exact D18 manifest/SHA256SUMS and four checksum members | PASS; every individually modified member rejected |
| Canonical phase population | Three ×(16 warmup +128 formal) actual P/D HTTP requests | PASS;432 P sends and432 D sends in each healthy/negative-control scenario |
| Prompt/repeat identity |144 stable prompt templates,432 unique transport IDs | PASS; exact6144-ID body equality across all three repeats |
| Native output identity/count |1024 real output IDs, final native usage corroborates | PASS; each healthy fixture has two512-ID deltas, not1024 SSE chunks; native text is empty |
| Timing and denominator |First/last generated-token delta on client clock; final usage excluded | PASS; last output precedes HTTP response completion; every healthy formal population totals917504 tokens |
| Four metrics |Mean TTFT, per-request TPOT=(last-first)/1023, mean request E2E, prompt+output THPT | PASS; formal-only metric windows equal min(actual dispatch)..max(last output), with per-repeat means and variation |
| Open-loop dispatch |Later requests do not await an earlier slow completion | PASS; healthy third send occurs before first slow request completes |
| Failure isolation |Broken JSON SSE / HTTP failure / bad usage / text-only output must not cancel healthy requests | PASS; invalid population gets no admitted metric, later repeats still run |
| Soft observations |Missing idle metrics or checkpoint write failure must leave unverified diagnostics without stopping requests | PASS; all432 requests still sent and retained; reference staysUNVERIFIED |
| Fresh idle |All four lanes require finite running/waiting/KV gauges and fresh completion counters | PASS; missing lane, busy lane, NaN and stale completion counters rejected |
| Full H200 stage config |P32/D32, TP8/DP4/EP32/PP1;7168/8192/128; chunk/prefix/MTPoff | PASS; all eight node argv/environment plans verified; no R2 bootstrap or block override |

The large canonical HTTP tests put their origin128 seconds in the past to exercise the exact432 bodies without waiting for the real426-second combined injection spans. They separately assert unchanged0..15 and0..127 offsets. A separate real-clock open-loop test proves actual scheduling independently. These fixture timings are not performance measurements.

## Preserved failures and corrections

1. Initial new suite:14PASS/1FAIL. The failure-injection fixture guessed a template suffix that the frozen package does not use; it therefore injected no failure. It now selects the actual first formal template ID. Production IDs/bytes were not changed.
2. A broader Python3.12 test collection failed because that existing harness environment lacks`plotly`, imported by unchanged simulator metric code. Harness tests remained in Python3.12; the9 simulator metric tests passed in the existing frontier Python3.10 environment. No dependency fallback or installation was added.

## Native prerequisites and limits

Source derivation is in`source_contract.md`; staging receipts and parent-only commands are in`stage_handoff.md`. Native success requires source identity, real H200/cache capacity, exact accepted/generated lengths, per-lane actual graph evidence and the separate parent-owned R2 disposition. The stage's `reference_admission` remains`UNVERIFIED`; a process exit0 cannot close those gates. Auxiliary verdicts remain nonblocking for the64-GPU workload.

## Final stage observations

Both stages were built from clean source`4a708ff` and independently verified from their own workspace with no repository`PYTHONPATH`: **27 staged tool/input hashes,38 native-source bindings and3 unchanged metadata pins each PASS**. Build and verify receipts match; no `launcher/` directory exists. The isolated staged canonical CLI imports and accepts`--help`. Exact commands, config SHA-256 values, cloud output paths, resource estimates and parent-only submissions are in`stage_handoff.md`. These are local offline checks only; no cloud paths were accessed.
