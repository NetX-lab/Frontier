# D29 Frontier position-routing progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | YC approves D29-SCOPE-107-10000; allocate native19/1582, core-source44/4404 and pricing44/3956. |
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
  size-note lines, approved10000 total. `shared_scope.md`/`scope.json` bind details.
- completed: YC selects `批准完整方案 (Recommended)` for D29-SCOPE-107-10000;
  requirements and human review receipt retain the exact decision.
- in-progress: three implementation streams from the common approval commit;
  native19/1582, core-source44/4404, pricing44/3956, no shared-file writers.
  Native19 includes four native-image files changed against locked base bytes.
- pending: interface integration, focused checks, independent review, parent
  unit failure/error-set equality, strict77, full offline/online request, image
  receipts and fresh GPU/native qualification. Additional collection resources
  and run approval follow readiness. D30 remains integrated4ec5b23.

## Active implementation dispatch — 2026-10-23

Common baseline1aee0af. Disjoint worktrees and background sessions:

| Stream | Worktree under /data/ycfeng/Frontier/.worktrees | Session |
| --- | --- | --- |
| native19/1582 | step5-position-native | ses_e9b3f65b5ffeMAo4OpbcxD9LIV |
| core-source44/4404 | step5-position-core | ses_e9b3efeebffem2dcbXlVGNr8O2 |
| pricing44/3956 | step5-position-pricing | ses_e9b3e99a6ffe3r2973GDLkFpEm |

Root supplies integration/review/image operations; no peer file writes or extra
agents. Early interface commits precede dependent consumers. CORE/SOURCE and
PRICING remain active. Native19 is delivered at2c4cc9e; root review
ses_e9afdb7c4ffejb3hbkjZCeHFKS returns four concrete findings, now assigned to the
original native session within1582. Repair74d5c48/records6827507 now delivers
19 paths/1269 gross, unchanged native4 patch. Root independently passes485 checks
and verifies all12 backend schema sources against the actual local-image export.
Reviewer closes N1–N4 without new findings. Root integrates accepted native19
source/CPU scope at9495cfb; image publication/source receipts and runtime qualification follow.
The internal target tag is absent by direct registry inspection; proxy111 on the
first inspection is resolved per-command without changing host proxy settings.
Publication completes: remote index10cdd7af… and linux/amd64 manifestf037d958…
hashes are verified. New bindings use the platform reference. Actual neutral-cwd
source receipt and Python-tree export agree on2907 files, with53 installed serving
pins and all12 backend schema modules. See image_publication/ and the publication
test report. Native runtime/GPU/G3/E2E qualification remains open.
The completed CORE/SOURCE+PRICING foundation receives independent review in
ses_e9abd8364ffee5sQKdpiQh3yH4 from frozen sourcebd97ac5: F1 actual routing
implementation provenance and F2 nested sequence immutability are assigned to
their respective owners. Exact37-path diff is supplied for the reviewer's missing
deleted-hunk/baseline check. Remaining implementation continues in parallel.

Parent independently reproduces418 PASS/35.45s at2c4cc9e and verifies all19 paths,
1017 gross lines, four native before/after hashes and exact sparse context02.
Receipts/logs: `/data/ycfeng/tmp/d29_native_parent01{.log,.xml}` and
`/data/ycfeng/tmp/d29_native_parent_source01.json`. First image build exits125
because Buildx is absent. Ubuntu provides0.30.1; default APT proxy127.0.0.1:51015
is unavailable, so the package download is retried directly from the configured
company mirror. `check_progress_2026-10-23_native_delivery.md` identifies an
unnecessary image-after-full-CPU edge; repair_plan.md now separates that branch.

Buildx dependency resolved using the extracted Ubuntu0.30.1 package after
per-command direct access to the company APT mirror; package SHA256 matches APT.
Local build02 succeeds. Candidate image index10cdd7af… has platform manifest
f037d958…; all2907 installed Python files equal the base plus exactly the4 approved
changes, all40 base layers retained,47 total layers, image Config unchanged and
installed derivation-manifest hash verified. In that actual image (Python3.12.3,
Torch2.13.0+cu130), the installed-source CPU suite passes11/11. This supplies no
CUDA/full-import/T2/native serving acceptance and is not a published image.

### CORE/SOURCE partial delivery and development integration

CORE/SOURCE delivers c995fdc/ac48363/04294dc/02f061d,29 of44 paths and1565/4404
gross. Its source/count/registry/launcher/provenance/timer consumers remain
distinct from unfinished selected trainers/main/fresh fixture. Exact remaining
dependencies are PRICING's committed native adapter/schema/ranges and finite-plan
config/fixture payload, plus root permission to import peer APIs.

Root grants development-branch imports PRICING7b86614/NATIVE2c4cc9e to CORE and
CORE02f061d/NATIVE2c4cc9e to PRICING. These are interface-integration permissions,
not final acceptance: N1–N4 stay open and corrected native commits must follow.
Both original sessions receive the shared APIs and keep sole-writer ownership
and quotas. CORE resumes independent consumers; PRICING receives exact required
next interface deliveries, not a request for status.

Root freezes the combined API state atbd97ac520fd0ba796ecd9d1905fdff352e859680 in
`/data/ycfeng/Frontier/.worktrees/verify-step5-position-contracts-01`. Python3.10
combined checks:283 PASS/19 SKIP/4 deselected. Python3.12 actual CPU timer/router/
provenance subset:18 PASS. Their testcase-ID union is292 PASS. A targeted run of
the10 unresolved torch skips and4 torch deselections in Python3.12 yields11 PASS
and3 FAIL: that host vLLM lacks fused_moe.activation and router.gate_linear. The
three actual-native-dependency cases pass in the built image after the existing
isolated profiler-dependency recipe and explicit pytest importlib mode resolve
pandas and the image's tests namespace collision. Source is unchanged. Full
testcase-ID union is306 PASS. Details/receipts are recorded in the verification
worktree's repairs/test_report_2026-10-23_contract_integration.md for integration.
Full-unit/strict77 and independent source review remain pending final delivery.

Earlier parent prerequisite check: Docker29.1.3 works; locked base digest is pulled under
the exact mirror reference and all2907 installed Python files equal the frozen
image_src export. Actual interpreter is python3/3.12.3; the design command's
python entrypoint failed127 and is corrected to python3. A broader binary/package
inventory timed out after50s and supplies no such evidence; the narrowed Python
source check passes. No new derived image or GPU job exists from these checks.
Receipt `/data/ycfeng/tmp/d29_base_python_sources01.json`; detailed prerequisite
report is retained under repairs. Approval arithmetic/chronology/hashes PASS;
normal git-add rejected ignored task records, then exact-path force-add committed
only the authorized records at1aee0af. Scope arithmetic uses python3 because this
host has no unversioned python command.

Parent regression preparation PASS: reparse retained full+affected unit outcomes
(7647 pass/267 baseline failures/31 baseline errors/179 skipped), preserve exact
failure/error IDs, and re-audit the clean D30 strict77 candidate with451 artifact
hash checks and matching dependencies/definitions/cache inventory. Runtime,
matrix/examples/data inputs are unchanged from that certified source to1aee0af.
Full commands, failed image attempts and current evidence limits are recorded in
`repairs/test_report_2026-10-23_parent_prerequisites.md`; selected raw receipts are
retained in `repairs/evidence/`. D29 candidate verification remains pending.

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
