# D18-NATIVE completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archived completed implementation and offline staging, with native dependencies unverified. |

## Outcome

Implemented Step native consumption of the byte-pinned common fixed workload using the **existing** HTTP replay client and P/D proxy. Three repeats each have16 disjoint warmups and128 formal requests. Stable template IDs determine prompt bytes; transport IDs are repeat-qualified. The locked Step API's real generated token-ID deltas supply client TTFT and last-output endpoints; final native usage and returned prompt identity corroborate exact6144/1024 counts. Complete-population metrics report mean client TTFT, TPOT and request E2E plus prompt-and-output throughput over first actual formal submission to last formal output.

Per-request HTTP/SSE/count failures remain in the population. Native idle/fresh-completion checks and auxiliary persistence/metric errors are soft diagnostics: healthy requests and subsequent repetitions continue, while reference admission remainsUNVERIFIED. No custom R2 bootstrap or content capture is staged.

## Deliverables

- `tests/comparison/dp_placement_pp/vllm_replay.py`: explicitly authorized opt-in token-ID stream reader, fixed prompt identity/count metadata and per-request failure retention; existing defaults preserved. Native server argv supports the existing graph-statistics flag.
- `tests/integration/step5_pd/canonical_replay.py`: pinned consumer, warmup/formal boundaries, complete-population metrics and soft native idle diagnostics.
- `tests/integration/step5_pd/canonical_run_h200.json`:64-H200 P32/D32 TP8/DP4/EP32/PP1,7168 context/8192 budget/128 sequences, native graph ceiling256; chunk/prefix/thinking/MTPoff and unchanged native FP8 metadata.
- `tests/integration/step5_pd/build_stage.py` and`canonical_stage.py`: existing worker/stage integration, immutable common bytes, source bindings,8-node preflight, qualification/canonical policies, offline verifier and credential-aware submit wrapper.
- `tests/unit/test_step5_canonical_native.py`: real local HTTP tests,432 actual P/D sends, prompt repeat identity, token timestamps/counts, metric windows, open-loop behavior and fault controls.
- Task-local`requirements.md`, `plan.md`, `progress.md`, `source_contract.md`, `test_report_2026-10-21_d18_native.md`, and`stage_handoff.md` preserve decisions, commands/evidence and exact parent submissions.
- Offline stages: `/data/ycfeng/tmp/step5_d18_native_20261021/qualification01/stage` and`/data/ycfeng/tmp/step5_d18_native_20261021/canonical01/stage`, frozen from code commit`4a708ff6681f7b1031f7a2a1ed9fbf655aed6d45`.

## Observed verification

574 affected tests PASS under Python3.12.13;9 narrow existing simulator metric tests PASS under Python3.10.6. A final160-test overlapping subset passed after parsing was narrowed to the source-confirmed schema. Both stages independently verify27 staged hashes,38 native-source pins and3 unchanged original metadata pins. No simulator sweep ran; simulator code and all frozen workload packages/environment tooling/R2 verdict and observations remain unchanged. No push, main merge or GPU/auth/platform/cloud/SSH/Docker operation occurred.

## Open work

Parent-owned R2 disposition; H200 installed identity, legal startup/cache capacity, per-lane/rank actual decode-graph evidence and transfer behavior; then canonical collection and formal metric admission/stability. Each run uses64 H200 GPUs. Qualification estimate20–35 active minutes /21.3–37.3 GPU-hours, hard worker allocation budget64 GPU-hours. Canonical estimate35–60 minutes /37.3–64.0 GPU-hours, hard budget96 GPU-hours. These are estimates, not native performance observations. No native correctness/performance or formal ticket acceptance is claimed by this archive.
