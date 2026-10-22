# D29 Frontier position-routing progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | All design reviews closed; parent scope107/9942 and sole-writer allocation complete; model/scope decision pending. |
| 2026-10-22 | Receive pricing proposal, identify an unapproved communication-model choice and arithmetic error, and assign callable-provenance correction. |
| 2026-10-22 | Receive profile-source and revised core subtotals; require explicit adapter ownership and output-frame protocol. |
| 2026-10-22 | Receive six design findings and start disjoint core, pricing and profile-source corrections. |
| 2026-10-22 | Receive the first design and request independent scope/correctness review. |

## Current frontier

- completed: native19/source16/core33/pricing60 design reviews, including both
  final producer seams and the shared source-frame materializer.
- completed: parent parses all scope rows and removes only H11's34-line duplicate;
 107 unique code/test/config paths,9942 gross. Plus existing gitlink and at most12
  size-note lines, proposed10000 total. `shared_scope.md`/`scope.json` bind details.
- pending: YC decision on the fixed collective model and complete implementation
  scope. Implementation has not started; actual collection resources/run approval
  follow package readiness. D30 is independently integrated4ec5b23.

## Earlier execution record

- completed: `design.md` traces actual Batch context/query snapshots, source
  lanes, shared materializers and profile consumers. Author reports 200
  mathematical cases and path/budget/format checks passing; these are not
  production tests and the parent has not independently rerun them.
- completed: initial estimate is 29 code/test files / 1556 gross changed lines,
  plus 3 required fixture files / 143 lines. Communication/pricing corrections
  are not included, so this is not a complete implementation budget.
- completed: reviewer `ses_e9e0793b2ffe5YWSjciM0uRzna` identifies six findings;
  the initial estimate is not acceptable as complete/minimal. `review.md`
  records source references, required corrections and unnecessary new-policy scope.
- in-progress: three disjoint design corrections cover count/schema/cache/exact
  coverage, source-local/physical collective pricing, and derived profile-source
  entry. The actual-main fixture is mandatory. No source implementation started.
- completed: `profile_source_design.md` supplies 13 paths / 673 gross lines;
  parent independently checks path existence, uniqueness and +633/-40 arithmetic.
   Independent review resolves the launcher omission but finds an actual timed
   callable provenance gap; correction is in progress. The old 10-file/662-line
  increment is relative to the superseded core table, not the revised union.
- received: core correction subtotal 31 paths / 2068 gross lines. Parent found
  two unresolved integration details: Frontier native-router adapter ownership
  was assigned mutually to the other owner, and actual CSV rows still need an
  explicit source-local versus gathered timing/width protocol. Core author is
   reconciling these with the delivered source design.
- received: `pricing_design.md` traces grouped unequal DP Broadcast/Reduce and
  proposes measured exact-plan lookup. Parent's first arithmetic check failed:
  the 37 unique paths sum to +1802/-185 =1987, not the reported 2087. The table
  summary is corrected; source/compute pricing review is running separately.
- in-progress: compare a faithful existing-simulator extension with proposed
  measured-plan lookup. Prior requirements explicitly retain collective_sim
  rather than measured CC. This is a modeling decision, not numerical-scope
  bookkeeping; the proposed variant has not been selected or implemented.
- in-progress: pricing review confirms source/phase direction but finds four
  remaining interfaces: precision-specific targets, routed prepare contract,
  actual metrics/trace consumers, and preallocated/registered native combine.
  The pricing author now owns their correction and the simulator alternative;
  the parent will reconcile the resulting interface and budget with other owners.
- in-progress: source-binding owner must cover actual GateLinear, RMSNorm and
  FP8/BF16 oracle callables using generated source receipts, in addition to the
  53 serving pins. Producer-specific import maps need not be identical.
- received: core follow-up assigns native adapter integration to core and
  specifies one existing dict per source-local/gathered cell; author subtotal
  is35 paths /2676 gross lines. Pricing's now-confirmed precision-specific
  target sets still require a focused core CSV/coverage reconciliation.
- received: source follow-up proposes16 paths /879 gross lines and generated
  installed-source inventory for actual callable checks. Delta review now checks
  receipt producer ownership, final image binding and serialization; the subtotal
  and the claim of complete receipt plumbing are not final union acceptance.
- completed (design only): parent specifies the full-source receipt emitter as
  a retained post-build command, 51 Python lines including the nested payload;
  both bodies compile without execution. Reviewer closes the emitter P2 with no
  new findings. Source design's16 maintained files remain its own subtotal;
  actual image generation/export and runtime source validation are still pending.
- completed: phase checkpoint finds D4/S1: D30's complete scope decision was
  unnecessarily waiting for D29 design. The dependency is removed from plans;
  request the independent D30 decision now while core/pricing work continues.
- completed: YC approves D30-SCOPE-7-750 directly with "批准，继续". D30
  implementation can resume independently. Core delta review requires three
  specified constructor/trainer/timer seams; original author owns the correction.
- received: pricing preserves collective_sim and explicitly stops a complete
  algorithm estimate at missing native channel-schedule/GPU-local overlap inputs.
  Known interface subtotal50 paths/2782 is not a complete implementation scope;
  measured lookup46 paths/2595 remains unapproved.
- completed: pricing scope review closes the four interface findings but rejects
  exact NCCL microarchitecture reconstruction as an unsupported pre-code gate.
  Parent applies the plan correction and requests finite model alternatives,
  explicit assumptions and a complete bounded implementation budget. Existing
  collective_sim and subsequent native/E2E qualification remain required.
- received: core author closes the constructor, trainer pre-loader and timer
  singleton seams. Parent independently checks33 unique paths, +2625/-341=2966
  gross ceiling and existing/new path declarations; receipt
  `/data/ycfeng/tmp/d29_core_budget_2966_parent01.json`. Narrow three-seam review
  is running. These are design checks, not implementation/native acceptance.
- completed (design only): narrow three-seam reviewer closes all three findings
  with no new delta. Parent updates stale source overlap annotations against
  final core33: three shared core paths, S16 pricing-owned; source879 unchanged.
  Finite pricing model and complete union remain open.
- received: finite pricing proposal54 paths/+3518/-294=3812, independently
  parsed by parent. Model review and shared-hunk ownership reconciliation are
  running independently. The preliminary102-path/9239 summed ceiling excludes
  gitlink/documentation and is not an accepted complete scope; receipt:
  `/data/ycfeng/tmp/d29_complete_scope_union_candidate01.json`. Model assumptions,
  numeric inputs, collection authorization and final scope remain undecided.
- pending: settle required companions, combine corrected native design and
  user scope approval, implement, verify and collect fresh evidence.

No simulator/profiler source was changed and no GPU/platform operation was run.
