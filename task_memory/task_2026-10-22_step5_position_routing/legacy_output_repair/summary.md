# D29 legacy output and fixture repair summary

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive original four-path legacy repair and exact output evidence. |
| 2026-10-23 | Preserve proposed fixture edits uncommitted during root authority correction. |
| 2026-10-23 | Archive explicit YC approval,separate fixture commit,fresh results and final budget. |

TaskID D29-LEGACY-OUTPUT-REPAIR;isolated worktree
`/data/ycfeng/Frontier/.worktrees/step5-d29-legacy-output`;branch
`feat/step5-d29-legacy-output`;immutable source
`9974acdb88f5c47166fee28dec564b7cc782ca1a`.

## Immutable delivery receipt

- Original production/regression commit:
  `b60dfddd1df09a8b048da31240988531ce7412e0` (four original approved files).
- Separate approved fixture-only commit:
  `047f7313387cb68f45e2214c61ed160864459343` (two new authorized test paths).
- Original records/pause handoff: `b32ad29d`,`f451d554`.
- Root-reported explicit YC scope answer: "批准，类似问题都授权".
  Effective111 authorized global paths,owner3956/global10000 ceilings unchanged.

Both scopes are complete and available for parent integration. Two test edits
were preserved uncommitted during the authority pause;no edit/commit occurred
after pause until explicit YC approval. No production edits in resumed fixture slice.

## Outcome and validation

Metadata emission reuses MoE applicability and real profiling/source inventory:
legacy runs omit Step-only prepare,actual registered prepare retains FP8 instead
of default BF16. Ledger omits absent collective_proofs;actual nonempty plan/cost
evidence retained. No timing,precision lookup,defaults,admission/comparator changed.
Original RED11=7 FAIL→11 PASS;whole affected194 PASS. Actual Llama2 CSV is
byte-identical to original baseline. Full strict77 not launched by this owner.

Fixture-only correction explicitly adds the fifth MoE operator to all catalogue
attributes,updates local stubs to real operators/profiling_target interface,and
derives V4 inventory from real applicability with explicit Step prepare exclusion.
All assertions retained,no skips or production fallback. Exact six initial
RED6 FAIL→approved fresh GREEN6 PASS;both whole files53 PASS,0 SKIP/FAIL/ERROR.
Original194 plus approved53 whole-file cases=247 distinct PASS;focused reruns
already included,not counted twice. No global/GPU qualification claim.

## Deliverables

All records in
`task_memory/task_2026-10-22_step5_position_routing/legacy_output_repair/`:

- requirements.md:raw decisions,authority pause,latest explicit YC approval.
- plan.md:two independent scoped slices and separate commit boundaries.
- progress.md:observed results,final checkpoint and parent-owned gates.
- test_report_2026-10-23_legacy_output.md:exact commands,JUnit,raw evidence,
  environment/prerequisite and latest approved validation.
- summary.md:this completion archive and immutable delivery/budget receipt.

## Exact source and retained budget

| Path | Step added | Step deleted | Delta gross | Retained gross | Ceiling |
| --- | ---: | ---: | ---: | ---: | ---: |
| frontier/config/quantization_manager.py |8|0|8|10|29|
| frontier/metrics/ep_wave_metrics.py |3|1|4|22|60|
| tests/unit/test_comm_operator_families.py |22|0|22|48|50|
| tests/unit/test_ep_wave_trace_context.py |3|0|3|49|65|
| tests/unit/test_moe_share_expert_operator_families.py |12|2|14|14|50|
| tests/unit/test_dsv4_precision_metadata.py |2|0|2|2|10|

Original legacy+36/-1=37/80 gross;fixture+14/-2=16/60;combined+50/-3=53/140.
All caps PASS. Legacy retained increment35 (replacement line already in D29
retained diff),fixture16,total51. Root confirmed3805 baseline plus51=3856/3956,
remaining100. 45 changed/46 authorized owner paths;global111 authorized paths,
no increase to aggregate10000 ceiling. Do not add raw53 churn to3805.

Independent sub checkout `3dac53c544e546574cb6e8078cd14053d0916113` clean;
gitlink unchanged,no new sub commit/build/network or peer worktree edit.
Parent integrates source/test commits preserving independent changes.

## Remaining work and limitations

No unresolved owner implementation/fixture issue or approval blocker remains.
Parent owns full strict77/global-unit/CLI integration and independent review.
Dependency-bound aiohttp/torch cases stay parent cross-env responsibilities.
No GPU/native qualification or global D29 acceptance is claimed.
