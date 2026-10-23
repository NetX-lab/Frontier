# D29 CPU performance component acceptance archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive three explicitly accepted CPU components and preserve separate review/test provenance and pending workflow gates. |

Task ID: D29-PERFORMANCE-COMPONENT-ARCHIVE.

## Outcome

Primary explicitly accepts all three completed CPU performance components.
Standards and Spec each close No findings,0 findings/worst none. The group
records-only P2 coverage overclaim is corrected by37c7832e and closed by the
same Spec reviewer. This is **component acceptance,not final D29 acceptance**.

| Component | Code-only commit | Records delivery | Step gross / ceiling |
| --- | --- | --- | --- |
| Fixed-model group reuse | 3763e5b87a845eac186cfd481f0b56cffb2dd045 | 88fe6c65,then37c7832ee5ee2e9588ba6fed411a9e9894949ab8 | 71 / 105 |
| Once-per-loader family derivation | ee7e66ee439054978897c91fa32598fc13804883 | 35d2f74ed60855e48e5af4ea97c49721b9aba7ea | 38 / 45 |
| Exact-current-text artifact parse reuse | 51e1f7205d07b269db2291c5f8076431a86d29e5 | 0fa22e49417b80ee315518f9ea6843b77663fe09 | 43 / 50 |

Group current-layer/binding/proof gates remain live;loader validates every row
before filtering;parser reads current text on every call and retains live gates
and read/parse errors. Native numerical cache hashes are unchanged.

## Deliverables

All paths are relative to `/data/ycfeng/Frontier/.worktrees/step5-preview`:

- `task_memory/task_2026-10-22_step5_position_routing/resumed_performance/primary_acceptance.json`
- `task_memory/task_2026-10-22_step5_position_routing/resumed_performance/review.md`
- `task_memory/task_2026-10-22_step5_position_routing/resumed_performance/test_report_2026-10-23_components.md`
- `task_memory/task_2026-10-22_step5_position_routing/resumed_performance/summary.md`
- `task_memory/task_2026-10-22_step5_position_routing/resumed_parent/review_2026-10-23_group_reuse.md`
- `task_memory/task_2026-10-22_step5_position_routing/resumed_parent/test_report_2026-10-23_group_reuse.md`

No primary requirements/plan/progress/summary,source,test or script was edited.
Raw command/log/diff evidence in owner records is unchanged. New Markdown
documents carry Modification History;the existing group reports retain their
original raw commands and add current coverage/review closure.

## Observed validation

Archive reparses and SHA256-rehashes all five existing parent JUnit receipts:
group81PASS/1existing Torch SKIP164.431s;separate actual trainer/preflight and
independent fit/cache2PASS10.327s;parser128PASS6.171s;loader29PASS32.316s;
Python3.12 governance52PASS19.366s. Every supplied hash/count/time matches.
No overlapping selections are summed. No tests,probes or source scans ran.

Group parent point timing0.924303847/0.354695124s and88/0 descriptions preserve
payload SHA256d2c141f…. Parser0.380598 to0.144797s and complete payload
SHA256c2aa250d… remain owner evidence until the integrated parent probe.
The test report records exact hashes,environment and command provenance.

Three-step gross152/200. Expected combined retained global gross9254 is only
an arithmetic expectation;actual merged audit and final accounting acceptance
remain pending. Primary source-clean/byte-equality assertions are attributed
to the primary and were not independently rescanned during archival.

## Open work and boundaries

Primary owns integration,actual merged per-path audit,final freeze and the
integrated performance probe. Full CLI,fresh strict77/full-unit,T2/native/GPU/
G3/E2E remain pending for the new combined candidate. Historical e41 regression
and T2 receipts are not relabelled as fresh checks. No extra early-init repair
is approved;partial startup diagnosis cannot justify changed cache hashes.

No unresolved component source/review finding remains. This archive grants
no additional implementation,workflow or resource authority.
