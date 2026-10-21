# D22-TIMING completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Parent native verification PASS:18/18 strict candidates,22/22 state and18/18 prefix checks; scope remains isolated probe. |
| 2026-10-21 | Archive verified source/CPU correction and sealed parent-only native rerun. |

## Outcome

Implemented the approved independent outer-window reference cohort through existing Step attention timing owners. Candidate/reference samples are separately executed using distinct primed outer event pairs and matching replay counts/minimum-window treatment. Boundary singleton observations remain explicitly labelled instrumentation diagnostics. The5% agreement/CV bounds, native geometry/capture, scoped timing and linear protocol are unchanged. Existing CLI quality-dependent retry was removed.

Source commit: `74df0100ca0f39d6ff134d999909e863c3db6962`. Final affected suite:873 PASS with14 existing deprecation warnings; all851 prior IDs retain PASS,22 added. Meaningful real-seam RED->GREEN includes3.33us/event injection on34.7us/replay and strict independent-execution assertions.

## Deliverables

- Production: `frontier/profiling/step5/{timing,rows,vllm_wrapper,main,job}.py`, existing common stage resource wiring, existing launcher/verifier/judge and focused tests.
- Persistent task records: `task_memory/task_2026-10-21_step5_d22_timing/requirements.md`, `plan.md`, `progress.md`, `test_report_2026-10-21_d22_timing.md`, `handoff.md`, `summary.md`, and `evidence/`.
- Fresh full original D18 stage: `/data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01`, manifest SHA256 `a1dc28ceca71235b000e7f7448464a5ff76d27e76d2d51ee63fe0811908b5312`.
- Offline preflight:3407 payload hashes/exact inventory PASS;525 staged tracked source payloads match source commit;18 event+4 GPU-active attention rows and16 original linear rows; both linear dry runs PASS. All13 linear protocol files are byte-identical to old stage02, scoped timing AST unchanged. Original stage02's3407 payloads/inventory and manifest are unchanged.

## Parent native completion

Parent submitted `exp-1021-201658-886733` on one preemptible H200 after quota checks. The unchanged stage completed with platformSucceeded, container exit0/restart0 and128 container seconds. Frozen strict judge passes all18 candidates, all22 output/cache rows and18 prefix comparisons, with no failed cells. Both original linear methods produced8 rows. Exact commands, provenance, bounds and original hashes are in `test_report_2026-10-21_d22_native.md` and `native_judgment/`.

The approved D22 native probe step is complete. Historical H200 D18 and H800 Smoke03 FAIL receipts were not recalculated or relabelled. No full-grid, canonical serving/PD, independent linear-quality or predictive acceptance is claimed; the judge retains `admitted=false`. Parent integrated the D22 source separately from the ongoing compute branch. The original implementation handoff performed no platform operations; the parent run and read-only retrieval are recorded above.
