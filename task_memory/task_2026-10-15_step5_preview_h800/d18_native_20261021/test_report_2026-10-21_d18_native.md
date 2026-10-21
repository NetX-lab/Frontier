# D18-NATIVE offline test report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded HTTP/consumer, legacy regressions and native qualification limits. |
| 2026-10-21 | Recorded MRv2 source-contract and throughput red/green, real qualification HTTP coverage and585-test follow-up PASS. |

## Result and environment

**Current follow-up PASS, source/CPU harness only:585 affected tests in59.09s.** Earlier verification was574 affected tests plus9 narrow simulator client-metric tests; those are historical, not additional unique current tests. No native latency, H200 capacity, installed-image, per-rank graph or R2 verdict is admitted. No GPU, authentication, platform, cloud, SSH or Docker operation was executed. The tests use local aiohttp HTTP endpoints, not fake request-send counters.

- Workspace: `/data/ycfeng/Frontier/.worktrees/step5-native-pd`.
- Harness Python: `/data/ycfeng/envs/asyncopd-py312/bin/python`, CPython **3.12.13**, Clang22.1.3.
- Existing simulator environment: `/data/ycfeng/envs/frontier-py310/bin/python`, CPython **3.10.6**, GCC11.3.0. These are existing environments; no conda activation, dependency installation or SDK construction was needed.
- `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/data/ycfeng/tmp`, pytest cache disabled. The new native consumer targets the locked image's Python3.12 (`asyncio.timeout`), not the simulator environment.

## Commands and observations

Current follow-up command, executed from the worktree:

```bash
STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm \
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest \
  tests/unit/test_step5_canonical_native.py tests/unit/test_step5_native_source_contract.py \
  tests/unit/test_step5_native_pd.py tests/unit/test_native_pd_worker.py \
  tests/unit/test_calibration_pd_harness.py tests/unit/test_dp_placement_harness.py \
  tests/unit/test_step5_native_pd_r2_stage.py tests/unit/test_step5_native_pd_r2_observation.py \
  tests/unit/test_step5_native_pd_r2_verdict.py tests/unit/test_canonical_fixed_manifest.py \
  -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d18_followup_affected01
```

Observed **585 PASS,0 skipped**; stdout`/data/ycfeng/tmp/d18_followup_affected01.log`. The exported source is a prerequisite for the9 source-contract checks; without`STEP5_NATIVE_SOURCE_ROOT`, those checks explicitly skip rather than claim native evidence. No source/gate failure is hidden by this prerequisite.

### Review follow-up red/green

Each command below used the same Python/env/pytest options above, with the named`--basetemp` under`/data/ycfeng/tmp/`:

| Slice | Exact test selection | Red | Green |
| --- | --- | --- | --- |
| Non-gating throughput |`tests/unit/test_step5_canonical_native.py -k formal_diagnostic_throughputs` |1FAIL, missing`output_tokens`;`d18_followup_metric_red01` |1PASS;`d18_followup_metric_green01` |
| MRv2 source binding |`tests/unit/test_step5_native_source_contract.py -k bindings_cover` |1FAIL, six required MRv2/DP/logging bindings absent;`d18_followup_source_red01` |Full source-contract module9PASS;`d18_followup_source_green02` |
| Qualification HTTP |`tests/unit/test_step5_canonical_native.py -k 'qualification or formal_diagnostic_throughputs'` |Coverage addition; existing qualification behavior needed no production fix |2PASS, including1 new qualification case;`d18_followup_qualification_green01` |

The first expanded source-test run had6PASS/1FAIL because its AST fixture selected an earlier unrelated`if not dummy_run` in the same native method. Restricting the fixture to the branch containing the actual stat call fixed the test; native code was not edited. The final9 source checks execute native positive/negative runner selection, check post-DP MRv2 wiring and handoff, exercise the native FULL route with a CPU replay sentinel, and render the actual native Runtime Mode/Count table. None executes CUDA.

New observed acceptance evidence:

- Diagnostic worked example:128 formal requests,131072 output tokens,100s last-token window → output throughput1310.72 tokens/s; request throughput1.28 requests/s; total throughput9175.04 tokens/s. Response time900s does not change the100s window. A failed request removes the full-population metrics. The four gate-summary fields are unchanged.
- Qualification:24 actual P HTTP sends and24 D HTTP sends,24 unique diagnostic transport IDs and24 exact6144/1024-token completions. Warmup offsets0..15, then both roles report a deliberately busy lane before fresh idle; only then do8 concurrent diagnostic requests arrive. The8 D requests wait on an all-arrived barrier, so a serialized client would fail this test. No formal/repeat metric is fabricated for qualification.

### Earlier implementation verification

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

## Stage observations

Historical stage01 packages from`4a708ff` passed their hash checks but their extra source contract cited MRv1; **they are superseded and must not be submitted**. Stage02 will bind the selected MRv2 path and its actual metrics chain. New committed-source build/isolated verification receipts will be recorded in`stage_handoff.md`. These are local offline checks only; no cloud paths were accessed.
