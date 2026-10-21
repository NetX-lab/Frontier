# D21-RUNS completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archived verified priority-native campaign deliverables. |

## Outcome

Completed the delegated CPU-only native staging/release work on `feat/step5-priority-runs`, based on b65609b. P0 H80032 qualification was sealed and handed off first at source2033376. The other three cases use source90724ba. All historical native/staged packages, the frozen common workload, D20 simulator work, R2 and GLM are preserved.

## Deliverables

- `tests/integration/step5_pd/priority_cases.json`: four hardware/scale/scheduling-policy entries through the existing canonical builder.32 total=P16/D16 TP8/DP2/EP16/PP1;64=P32/D32 TP8/DP4/EP32/PP1.
- `tests/integration/step5_pd/{canonical_stage,build_stage,submit_native_pd}.py`: complete immutable native packages, ResourceConfig preemptible propagation, standalone dry-run and formal checks before auth/SDK.
- `tests/integration/step5_pd/formal_release.py`: existing clean-run readiness plus same-hardware32 raw three-repeat/four-metric dependency, using the existing normalizer/comparator. Formal64 requires readiness and every independent error<=30%; formal32 requires own readiness. Final four10% gates and metric formulas are unchanged. Qualification has no30% prerequisite.
- `tests/comparison/e2e_gate/{admission,normalize}.py` and `cases/step5_priority.json`: Step-only original native/Frontier hardware, model, TP/DP/EP/PP and actual role/device accounting. Original image/model/source identity and receipt artifacts are pinned.
- `tests/integration/step5_pd/canonical_replay.py`, Step-only `tests/comparison/calibration/client_metrics.py` schema and `tests/fixtures/step5_pd/startup_metrics_qualification02.json`: truthful one-snapshot first-start diagnostic, with strict unchanged post-warmup/formal/inter-repeat idle admission. Original qualification02 verdict stays unchanged.
- Focused persistent tests: `tests/unit/test_step5_{canonical_native,client_metrics,priority_runs}.py`.
- Full handoff, commands, exact stage/config/resource payloads and evidence: `stage_handoff.md`, `sealed_stages.json`, `test_report_2026-10-21_priority_runs.md` in this task directory.

## Observed validation

-204 CPU tests passed, including both-topology real loopback HTTP qualification, both-hardware formal release positives/negatives, original GLM/client/E2E and R2-stage regressions.37 priority tests passed again after the final scratch-placement change.
- All four sealed packages pass standalone hash/config/source-role verification, exact frozen ResourceConfig dry-run, shell syntax and Python source compilation. P0 has28 verified staged files; later packages have36.
- No GPU, authentication, platform, cloud, SSH, Docker or external network operation was performed. No synthetic test value is claimed as native evidence.

## Remaining parent-owned work and limitations

Parent owns live personal reservation/capacity reconciliation, native runtime qualification, admitted run checks and calibration. Global H200 non-preemptible nonterminal reservations must remain<=32 across all sessions; H20064 is preemptible with no80/130 exception. P0 remains highest priority. The prepared H20064 basic stage is not permission to duplicate an old successful basic run before inspecting its retained evidence.

No formal package was sealed without genuine prerequisite evidence. Missing evidence deliberately blocks formal release only. Source/native correctness and measured-profile admission remain prerequisites, not claims established by CPU tests. Budget projections are timeout-based reservation limits (qualification60min; canonical90min), not measured native runtime predictions. No failed or unfinished delegated implementation item remains.
