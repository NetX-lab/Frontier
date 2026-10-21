# D23-MRV2-LOAD-DIAG Progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record evidence, hypotheses, and bounded profiling plan. |

- completed: isolated worktree/branch from ec5de15; clean initial Git status.
- completed: AGENTS, GLOSSARY, existing ADR, diagnosing-bugs workflow, actual-main
  requirements/plan, timeout receipt and production loader/manager paths inspected.
- completed: relevant loader/trainer/registry/producer files are byte-identical
  between actual05 26db654 and ec5de15 (`git diff` empty).
- completed: actual loader/role-union profiles and KERNEL_ONLY warm-cache seam;
  `diag02` runs to the expected RED_REPEATED_LOAD verdict in seconds. Original
  cache SHA256 values and source field equality checks pass. No fit occurs.
- completed: diag03 plain-loader control0.255748/0.453526s; profiler materially
  increases wall and is not used as full-wall attribution.
- completed: diag04 RED repeats; cache/source unchanged; current event root
  cache0.529611s/71.15MB/94392 exact keys. Zero estimator fits.
- completed: durable receipts, source multiplicities, full-wall limits and
  smallest source proposal archived in the dated test report/English summary.
- completed: source/script verification and diagnostic code commit `c655305`.

## Ranked hypotheses (before probes)
1. Fixed typed metadata rebuilding repeats per row. Prediction: typed parsing,
   expected contract construction and metadata validation counts scale with rows.
2. Role union revalidates identical inputs per role and again before signature
   deduplication. Prediction: PREFILL/DECODE each invoke the same loader twice per
   measurement, including the role whose attention signature is already trained.
3. Warm cache does not bypass training-input loading/hash construction.
   Prediction: loader and full DataFrame hashing precede cache lookup; warm-cache
   API shows zero estimator fits but substantial pre-lookup work.
4. CSV/pickle I/O or actual fitting dominates. Prediction: read_csv/pickle/fit
   boundaries dominate the short profile instead of semantic validation.

## Existing full-run observation (not a new profile)
Parent receipt measures command-file creation to first scheduler event at
998.503711 seconds offline and 1004.875261 seconds online. This includes all
initialization and is not function-level attribution. Fixture source is CPU
synthetic evidence, not native admission or predictive accuracy evidence.

## Inspection limit
A streaming CSV census exceeded an initially too-short 10-second command limit;
it was terminated without output or writes. Do not repeat a full CSV parse to
obtain subset timing. Keep subset selection bounded and use existing row census.

## Key profile evidence
- Source CUDA_EVENT: 5,606,492,481 bytes, 125 columns. First row quant_signature
  contains 53,767 characters (of 59,209 value characters); source metadata is
  preserved, not compacted or replaced by a hash.
- `event_load_100/200`: cProfile wall 0.564803/1.044887 seconds, one CSV read each;
  200/400 feature calls, 900/1800 typed metadata validation calls. Layer metadata
  is already cached per layer inside one load (one description call each).
- Full small KERNEL_ONLY184: 2.229518 seconds; 92 layer descriptions.
- Warm PREFILL KERNEL_ONLY: 4.971523 seconds, loaders 4.472816 seconds,
  two cache hits 0.003508 seconds, zero estimator fits.
- Already-deduplicated DECODE KERNEL_ONLY: 4.596507 seconds; two full loader calls
  (4.442424 seconds), one hash, no models returned, no cache read or fit.
- Actual05 retained logs are 3.2KB each and have no MODEL CACHE MISS / training
  warnings. The unchanged fit branch emits these at warning level on every miss.
  This supports warm initialization; it is not a full function-level profile.

## Iteration note
`diag01` stopped at config identity assertion before profiling. Cause was Python
tuple versus JSON list representation, not changed model semantics. Comparing the
model's actual JSON serialization fixes the harness; diag02 proves complete
model equality to recorded actual config before API calls.

## Final acceptance boundaries
- Diagnosis complete; no production patch, full setup/CLI, fit, fixture edit,
  cache invalidation, timing substitute or runtime-seam investigation.
- Parent retains original2400s full acceptance. The remaining wall interval is
  not exactly allocated; approximately747s loader arithmetic is prefix-only.
- Proposal: reuse identical validated union/hash within one initialization with
  all role/source/shape/evidence records and unchanged estimator hashes; retain
  per-row validation and distinct-source rejection. Feature reuse is secondary.
- Exact commands and source checks: dated test report. Raw artifacts remain
  under `/data/ycfeng/tmp/d23_mrv2_load_diag03/` and `diag04/`.
- completed: final AST/JSON/receipt checks PASS; expected RED2loadercalls and
  source-preservation flags confirmed. `git diff ec5de15 -- frontier data
  tests/integration` is empty. No production/source edit in the worktree.
- completed: self-check after archive/code commit: plan1–5 PASS for deliverables;
  D1–D7 and S1–S6 CLEAR. BranchA; record commit follows, no further profiling.
- completed: diagnostic code committed separately as `c655305`; only
  `tests/performance/profile_step5_load.py` changed. Evidence records are committed
  separately. Parent owns source decisions, integration and full acceptance.
