# D22 native timing verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Parent submitted the unchanged full probe, verified platform identity and judged the retained native outputs. |

## Run and provenance

- Job `exp-1021-201658-886733`, source `74df010`, one **preemptible H200**, `step_main`, creator `i-fengyicheng`.
- Sealed stage `/data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01`; manifest SHA256 `a1dc28ceca71235b000e7f7448464a5ff76d27e76d2d51ee63fe0811908b5312`. Parent preflight verified 3407 payloads. The approved 18 event + 4 GPU-active attention cells and both eight-row linear methods were retained.
- Platform image digest, creator and same-path local NFS source were checked. Platform Succeeded at `2026-10-21T12:26:14Z`; container exit0, restart0, `12:24:03–12:26:11Z` (128 seconds). This is container execution time, not full billed allocation time.
- Worker logs report `NVIDIA H200`, Torch `2.13.0+cu130`, and compatibility `libcuda.so.580.82.07`. Native scope remains physical TP1 with TP8-local dimensions; this is not full serving.
- Cloud root: `/mnt/codesign-exp/ycfeng/frontier/step5_profiling/fixed_h200_d22_probe_20261021_01`. Full CSVs/traces remain there. Selected local copies: `/data/ycfeng/tmp/d22_timing_20261021/native_results01`.
- Retained receipt and selected logs: `native_judgment/`. Download receipts bind each original CSV/log with SHA256 and exact cloud path. No old result was overwritten.

## Exact execution

```bash
setsid bash /data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01/submit_with_env.sh \
  > /data/ycfeng/tmp/d22_timing_20261021/parent_launcher01.log 2>&1

env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH=/data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01 \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python -B \
  /data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01/tests/integration/step5_profiling/judge_attention_smoke.py \
  /data/ycfeng/tmp/d22_timing_20261021/native_results01 \
  --expected-event-rows 18 --expected-gpu-active-rows 4 --max-time-cv 0.05 \
  --report /data/ycfeng/tmp/d22_timing_20261021/native_results01/d22_strict_verdict01.json
```

Parent uses the existing Python3.12 `oneshot-opd-py312` environment for the CPU judge. The launcher uses the existing personal StepMind Python3.10 environment. Download operations used read-only `ssh shai-ycfeng`, checking `id -un = i-fengyicheng`, and reading only the corresponding `/mnt/ws-jfs/ycfeng/` paths.

## Expected versus observed

| Check | Required | Observed | Result |
| --- | --- | --- | --- |
| Rows | 18 event + 4 GPU-active | 18 + 4; all expected cells | PASS |
| Independent candidate quality | All 18 candidates meet CV and reference agreement <=5% | 18/18; no failed cells/errors | PASS |
| Output/cache state | All 22 rows | 22/22 | PASS |
| Prefix continuation | All 18 native cells | 18/18 | PASS |
| Protocol/counts/windows/graph/scopes | Strict judge checks all required fields | Every check true | PASS |
| CUPTI | All required GPU-active rows | 4/4 | PASS |
| Linear retained plan | Eight rows/method, tokens1/2/16/6144 | Eight event and eight kernel-only rows | PASS for collection only |

Event layer-reference errors range from 0.01331% to 3.60715%; candidate layer CVs across the 18 event rows range from 0.11727% to 2.79961%. The strict judge reports 14 FULL-event candidates plus 4 GPU-active candidates, all passing; the other four eager event rows remain diagnostics. `admitted=false` is preserved: a successful small probe is not full-case admission.

The linear methods were not changed by D22. Their row/shape completion does not establish independent linear timing quality or full profile coverage. No E2E predicted/actual pair was collected here, so no four-metric error is reported.

## Disposition

The approved D22 native verification step passes. Historical D18 H200 and H800 Smoke03 failures remain unchanged. Ticket10 still needs full hardware/scale-compatible coverage and admission; native graph/G3/G4, case-aligned transfer and the four final10% metrics remain open. Next use this verified protocol for the required track-specific profiles rather than continuing the closed boundary-event investigation.
