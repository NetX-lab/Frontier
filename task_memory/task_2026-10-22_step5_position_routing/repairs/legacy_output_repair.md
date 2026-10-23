# Legacy output repair allocation

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record strict77 RED and allocate bounded PRICING repair using existing four paths. |
| 2026-10-23 | Propose two regression-fixture paths within unchanged budgets; correct premature root authorization and request YC approval. Record merge deduplication. |

TaskID D29-LEGACY-OUTPUT-REPAIR. Source9974acdb, baseline713e6a8. Strict candidate
completed77 cases without failures; comparison has77 mismatches, zero definition,
provenance, missing-evidence or cache-name differences. Raw evidence stays intact
at `/data/ycfeng/tmp/d29_workflow_fidelity_20261023/{comparison.json,compare.log}`.

## Diagnosis

Ranked predictions: (1) the new supported operator enters legacy metadata fallback
without applicability evidence; (2) empty collective proofs are written for every
legacy phase; (3) a changed configuration or numerical run explains the differences.
The original comparator is the RED feedback loop. A minimized direct invocation
of the real MetricsStore writer and existing EP-wave reporting fixture on9974
also fails in seconds: a real Llama-2 configuration emits
`moe_input_quantize,FP16,profiling`, and all five legacy phases gain
`collective_proofs={}`. The same isolated calls need no Step position mode.
The first two predictions are confirmed. Existing comparator differences cover
only these artifact types; acceptance still requires an unchanged full comparison.

## Scope and acceptance

Original PRICING owner receives a separate repair delta ceiling80 against9974,
inside its remaining151 and unchanged3956 total. Effective109 paths/global10000
remain unchanged. Only these four already-approved paths may be edited:

| Path | Current gross | Existing ceiling |
| --- | ---: | ---: |
| frontier/config/quantization_manager.py | 2 | 29 |
| frontier/metrics/ep_wave_metrics.py | 20 | 60 |
| tests/unit/test_comm_operator_families.py | 26 | 50 |
| tests/unit/test_ep_wave_trace_context.py | 46 | 65 |

Preserve the legacy metadata catalogue when no actual source inventory admits the
new operator, while retaining the correctly inventoried Step prepare row. Reuse
the existing operator/applicability/source-inventory mechanisms. Omit absent
collective proof data on legacy paths while preserving complete actual Step plan
and cost evidence. No comparator filters, baseline regeneration, fake precision,
new flags, timing changes or validation relaxation.

Add regression coverage at the real metadata writer/manager and EP-ledger seams,
observe RED then GREEN, retain positive source-inventory and nonempty-plan evidence,
and run affected existing metadata/trace/consumer suites. Parent owns full strict77
rerun with original baseline. Preserve original artifacts and source-frozen runs.
Request root reallocation before exceeding the four-path or80-gross scope.

## Full-unit fixture amendment

The9974 full unit run contains8005 PASS,285 FAIL,281 SKIP,32 ERROR. Compared with
the recorded b8be287 baseline there are18 new failure IDs and one new collection
error: twelve newly-added HTTP cases lack aiohttp, the new native-position module
lacks torch, and six existing tests have stale catalogue/stub assumptions.
Parent verifies dependency-bound cases in the existing Python3.12 environment.

The six test-only repairs are proposed for the same PRICING owner, additional
delta<=60, in two added paths (111 paths only after YC approval):

| Path | New ceiling |
| --- | ---: |
| tests/unit/test_moe_share_expert_operator_families.py | 50 |
| tests/unit/test_dsv4_precision_metadata.py | 10 |

Preserve explicit catalogue assertions for the newly declared operator; make
the existing family stubs satisfy the real OperatorFamilySpec interface, and
derive V4 inventory expectations from applicable operators rather than the
unfiltered global catalogue. Do not admit the Step-only preparation scope on V4,
introduce production mock fallbacks, skip tests, or remove assertions. Verify RED
then GREEN for the six exact IDs and both complete affected files. The four-path
legacy repair remains<=80 and this two-path repair<=60, combined<=140.

Final source assembly initially duplicated an identical T2 test from both sides.
Root verified matching AST SHA256a0ab621486f0c06005e171d810efaf1d4dc161acd89a63b1e3d6f3174d6676b8,
removed only the duplicate in d0dfcdb5, and verified the resulting full file is
byte-identical to9974 plus the two parameterized cases pass. This is a root merge
correction, not loss of an accepted test. PRICING returns to3805/3956, leaving151;
both assigned repairs fit. Native cumulative1350, CORE3833 remain unchanged.
Final per-path ceiling redistribution would fund these two entries from existing
PRICING slack after actual repair accounting, with no aggregate ceiling increase.

Root initially dispatched this two-path addition before re-reading the explicit
file limit in requirements.md28/44-47. That authorization was premature. The owner
has been instructed to pause only those additions, preserve any local work and
continue the original four approved legacy-output paths. No scope expansion is
effective pending YC approval. Read-only diagnosis and RED tests remain allowed.
