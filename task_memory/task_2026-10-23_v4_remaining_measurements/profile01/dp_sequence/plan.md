# DP sequence source/CPU plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Defined bounded source/CPU acceptance and dependencies. |

`source/Worker inspection + existing-suite baseline -> local fixture/regression RED -> bounded source change -> focused GREEN + inherited matrix/frozen byte checks -> records/commit -> root review/integration`

1. Read parent allocation and actual pinned Worker, V2 runner and DP source. Record reset scope and baseline failures without changing environment or dependencies.
2. Extend existing local fixtures/tests within140 gross lines. Require an actual explicit group, post-reduction counter reads, nonzero startup, zero/repeated vectors, soft getter isolation and unchanged frozen-schema controls.
3. Add only inventory thread-local counter recording/reset, optional final-step enrichment and the existing context/annotation envelope within40 gross lines. Active dispatch uses its raw side state; post-execution sample_tokens uses the detached step, never a later direct dummy execution's counter. Native collective and CPU-vector observation remain independent of the getter read.
4. Run four focused suites in separate CPU processes with `FRONTIER_VLLM_BS_ROOT` set; run directly imported adjacent regression suites only if justified. Preserve original errors, test counts and skips. Check final detached-row capacity and per-execution stale-state controls.
5. Hash unchanged frozen R2/R3 sources and native source; explicitly compare inherited matrix domains and verify retained77-case comparison hash. No fresh matrix capture or worker pairing claim.
6. Record exact source/base/record hashes, source.diff, changed-file budgets and limitations; re-read requirements, run check-progress self-check, commit only owned files, verify clean branch and stop.

Acceptance: optional raw integer on profile steps and existing context/annotation records; base row keys, exact three dispatch keys, hook set, native outcomes and bounded observation errors unchanged. Missing/raising getter leaves sequence absent. Every execution resets before native work; early Worker failure and V2 no-forward cannot reuse a previous execution's value. Direct execute_dummy_batch is source-backed and emits annotation counters with step=None, not fabricated step rows. A later dummy counter never replaces sample_tokens' recorded step counter. Final admitted rows are detached and their complete encoded size is charged.
