# D29 approved implementation sequence

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Separate native image/source publication from full Frontier integration; retain qualification gates. |
| 2026-10-23 | Allocate the approved complete scope to three disjoint streams. |

Decision: **D29-SCOPE-107-10000**. Existing D29 correction records/requirements
are the task issue; no new ticket files are created. The human review receipt
in this directory binds the approval and analysis. Native19 design remains at
`/data/ycfeng/Frontier/.worktrees/step5-d24-stage/task_memory/task_2026-10-22_step5_d29_native_balanced/native_design.md`
(commit c9e9274); the other three designs are in this task.

## Sole-writer assignments

Use the retained rows of `../scope.json` (138 rows after removing H11,107 unique
paths). Sum each path's retained allowances once; do not subtract H11 again.

| Stream / change marker | Files | Gross quota | Assignment |
| --- | ---: | ---: | --- |
| D29-NATIVE-IMPLEMENTATION | 19 | 1582 | All native-owner rows, including four native-image paths. |
| D29-CORE-SOURCE-IMPLEMENTATION | 44 | 4404 | Any path with a core/source row, except H04/H17. |
| D29-PRICING-IMPLEMENTATION | 44 | 3956 | Remaining pricing paths, plus H04/H17 with their contributor allowances. |

H04=`frontier/execution_time_predictor/moe_operator_times.py`;
H17=`frontier/profiling/moe/fp8_block_experts.py`. Each stream implements all
contributors' already reviewed responsibilities for its files. No simultaneous
writers. No writer borrows another quota or expands paths without root allocation.
Stop extending when its allocated quota is exhausted; report actual diff and
remaining work. Parent owns the<=12-line existing size-note and2-line gitlink
integration within the global10000 ceiling; planned total9956 leaves44 unallocated.

Native-image paths denote edits against the locked native source, not new full
files charged as zero. Use a fresh derived-source candidate, preserve the old
export/captures, retain four-file before/after hashes and reviewable generated
patch. Root owns image build/publication and all GPU/platform operations.

## Sequence and contracts

Common approval/source baseline -> {native context/router; CORE count descriptor,
source_frame_tensors and source receipt verifier; PRICING selected operator
bindings and finite-plan schema} -> respective nonshared consumers -> sole-writer
shared integrations -> combined fixture/main/reporting -> parent regressions and
independent review -> package readiness -> resource/run decision -> fresh evidence.

Parallel source branch: native four-file source/image-context checks and review
-> derived image publication/installed-source receipts -> profiling/serving source
binding. A local image candidate may be built while source review runs; publish
after that review and affected checks pass. The image does not depend on complete
Frontier fixture/regression work; package admission still requires all applicable
integration and evidence checks. This corrects S1 in the native-delivery checkpoint.
After native source/image acceptance, controlled T2 inputs and their existing
execution/resource proposal can be prepared in parallel with remaining Frontier
implementation. This preparation neither qualifies runtime nor authorizes a run;
package readiness and the separate resource/run decision still apply.

Native publishes set_routing_positions and the actual factory contract first.
CORE/SOURCE publishes immutable lane/source descriptors, source_frame_tensors,
and callable origin verification. PRICING publishes the sole
get_moe_profiling_operators(context) selector, native quantization/layout facts,
and finite-plan/model input contract. Provide early interface commits and their
checks before dependents need them. A missing peer implementation is not a
reason for a duplicate registry, fallback or fabricated timing/source evidence.

Use dedicated worktrees, scoped code commits after affected checks, and local
task records. Root merges every accepted code/document commit into
feat/step5-preview; no pushes or branch/worktree deletion. Workers may read peers'
interfaces but only root authorizes merge/import. Read applicable AGENTS/skill
rules; current Step authority overrides historical generic vLLM0.10.2/H200/round-
robin site assumptions. No further agent delegation from these streams.

## Verification and acceptance

- Each stream: actual source/consumer tests, exact commands/environment and
  before/after results; diff/path/budget/AST/whitespace and existing size analysis.
  Preserve D30 source coverage and finish synchronization in native fixtures.
- Combined: supplied logical positions/masks/count conservation and P/D/chunk
  invariance; native selector/weights; actual source-local/gathered scopes;
  paired exact duration/count keys including applicable empty physical scopes;
  source/callable identities; all selected native collection keys; finite-plan
  byte/group/rail/accounting and reporting. Missing inputs fail explicitly.
- Parent: compare full unit failure/error IDs to the retained pre-D29/D30-equal
  baseline; run strict77 against reusable certified artifacts, including902
  hash checks; unaffected default/model outputs remain identical.
- Fresh CPU-only fixture: retain old fixtures; full6144/1024 offline and online
  actual-main within2400s with READ/readiness/release/output and final metrics.
  CPU fixture values prove plumbing only. Native CPU-gap campaigns remain last.
- Native graph T2/eager module/source/pins, independent G3, pressure and clean
  E2E metrics require actual evidence; no static/CPU receipts substitute. Four
  E2E metrics individually<=10%; numerical-state tolerance remains separate.
- Root prepares additional collection resource/run decisions at package
  readiness, rather than allowing workers to initiate GPU/auth/cloud operations.

Current status: **in-progress implementation; no D29 runtime qualification**.
