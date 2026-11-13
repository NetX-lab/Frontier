# D23-MRV2-INPUTS progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Start the isolated producer-to-consumer implementation. |

- completed: created clean worktree/branch from `7fa73f9`; session `ses_ea08efc2cffe6Z1Rc7srkVdebh`. Parent untracked `.pi/.cortexkit` left untouched. The request's `primary7fa73f9` notation was not a Git ref; confirmed and used exact existing `7fa73f9`.
- completed: read parent D20/D21/D22 and latest native/producer integration report, measured-compute plan, relevant source and existing test seams. Existing loader rejects the dummy scope and enforces one scalar scope; exact feature keys lack kind. Existing no-metadata measured fixture mutates a request-owned row, leaving its producer kind/scope mislabeled.
- completed: inherited producer baseline465 PASS/24.84s and initial affected measured baseline141 PASS/55.42s; no inherited failure in those commands. Original logs `/data/ycfeng/tmp/d23_inputs_baseline_{producer,measured}.{log,xml}`.
- completed: real build_row/rows_to_dataframe → shared fit regression RED:3 failures/2.12s at mixed scalar physical_scope, `/data/ycfeng/tmp/d23_inputs_red.{log,xml}`. Minimal reproduced NONE/FULL/PIECEWISE request/dummy numeric collisions.
- completed: extracted the existing descriptor into neutral `frontier/attention/step5_inputs.py`; producer/loader/runtime reuse one kind label/code/scope table. Added kind feature, kind-bound scope identity and observed-only domain through union/cache/cost. The initial regression plus measured training/replay GREEN:43 PASS/12.96s, `/data/ycfeng/tmp/d23_inputs_green01.{log,xml}`. Fixed the existing no-metadata unit fixture by constructing its truthful producer cell, not by relabeling its output.
- completed: expanded invalid-field, missing-kind, union/bundle and observed FULL/null regressions. Parent's decisive local report confirms5/13 actual groups rejected by the inherited FULL/max_query_len check; legal FULL/null now retains null with uniform_token_count1 and source-derived single-token split. Role graph-family mapping remains unchanged.
- completed: actual13 local dummy descriptor groups pass;26 SWA/DSA producer-shaped CPU rows retain actual splits/raw null and reach existing cold/warm root/count caches. Selected evidence `observed_descriptors.json`; no native timing admission.
- completed: final producer467 PASS/16.70s, final measured263 PASS/191.88s, final focused49 PASS/17.56s. All final XML failures/errors/skips0. First producer attempt465 PASS/2 FAIL preserved: two old isolated-scope rejection assertions were corrected to assert truthful isolated identity and reject native relabeling; no producer source/timing workaround. Exact logs/commands and verification limits in `test_report_2026-10-22_mrv2_inputs.md`.
- completed: final requirements reread, mainline self-check and scoped diff inspection. Source changes remain only in exclusive approved files/new neutral module; no giant manager, scheduler, graph-policy or actual-main runner/fixture changes.
- completed: verified source/tests committed as `ba484b0d9e1534f5f8e742127addee5cafe62560`. Full source step is scoped clean; no unrelated edits, scheduler/graph-policy edits, external action or delegation occurred. Final archive/test records are retained in this task directory and committed separately for parent cherry-pick. All assigned CPU work complete; no failed or unfinished in-scope subtask remains. Native admission and parent scheduler/native binding remain outside scope.

## Parent handoff

1. Integrate source `ba484b0` and the following task-record commit from `feat/step5-mrv2-inputs`; primary should re-run acceptance on the integrated tree.
2. Future MRv2 runtime binding must supply `native_workload=mrv2_dp_dummy` and the observed descriptor/provenance to the existing row-feature path. A metadata-bearing workload alone remains request-owned. Existing DummyForwardBatch stays isolated diagnostic; no native binder was introduced here.
3. Role graph-family mapping, first-prediction group agreement/rendezvous, actual-main producer grid/native admission and original native runtime/reference/pressure failures remain parent/other-owner dependencies, not this CPU acceptance.
