# D29 CLI Cloud Write Proposal

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-25 HKT | Update accepted2ca5e571 ledger behavior and finite diagnosis closure; execution environment/throughput/memory remain unverified. |
| 2026-10-25 | Accept YC's sufficient-quota confirmation and update the committed-but-unadmitted ledger candidate and handoff execution boundary. |
| 2026-10-24 | Link the verified135-gross ledger allocation;implementation and worker readiness remain pending. |
| 2026-10-24 | Record corrected storage accounting, timing boundaries and unresolved worker/memory prerequisites. |

## Verdict and evidence ownership

**BLOCKER: execution readiness is not established.** This is a static proposal,
not a job approval, capacity reservation, throughput measurement or CLI result.
Quota capacity itself is USER_CONFIRMED_SUFFICIENT by YC on2026-10-25; do not
repeat quota discovery as a blocker. Worker environment, append throughput and full-workload memory/runtime remain
unverified; production source admission is complete at2ca5e571.
Read-only analysis: D29-CLI-CLOUD-WRITE-PLAN, session
`ses_e91b47e39ffeRn6lPWVUegoZQ6`. Primary owns acceptance and subsequent allocation.
Source inspected by that historical analysis: `b0277287`. Current accepted source
is2ca5e571,9682 source / 9696 global, with the116-path/126-binding parent audit
complete. This document allocates no source change.

## Existing execution route

`tests/integration/run_step5_measured_main.py` accepts an output root and puts
both mode directories, metrics, communication cache and execution receipts below
it. The existing flags can therefore address
`/mnt/codesign-exp/ycfeng/frontier/step5_position_routing/` on a worker without
adding a simulator output mode. The CPU host must not mount the cloud volume.

Candidate: a codesign worker with zero GPUs. H800/codesign is an alternative only
if a verified image/environment requirement justifies a GPU. Neither resource
shape nor image compatibility is established for this full CLI. The 30 GiB
shape-probe limit is not a full-run memory allocation.

The established single-source mount can expose `/data/ycfeng`, including the
worktree, parent Git metadata, Python environment and fixture. A narrower staged
tree is an unimplemented alternative; environment portability, absolute fixture
paths, source identity and cache bindings need verification. No new mount or
worker allocation follows from the existing test-storage approval.

## Corrected storage projection

The earlier approximately 261 GB/mode estimate covered operator traces only.
It did not include the EP lane ledger written at successful termination.

| Item | Representative basis | Projection per mode |
| --- | --- | --- |
| EP-wave multiplicity | 88 MoE layers × 1025 forwards | 90,200 waves |
| EP lane ledger | 16 rows/wave; retained 16-row JSON-array serialization 2,309,545 B | approximately 208.3 GB |
| Operator traces | retained FP8 one-token decode JSONL sample | approximately 260.7 GB |
| Combined | representative sample scaling | approximately 469 GB/mode, 938 GB/both |

These are historical FP8 decode sample extrapolations, not measured completed
files, RSS estimates, lower bounds or upper bounds. Array framing is not identical
to final JSONL framing. BF16/prefill size differences, stage-batch ledger,
event trace, filesystem overhead and other non-wave output are not included.
The dated local capacity and older arithmetic remain in `evidence.json` as
history, not current readiness calculations.

## Memory and timing are independent prerequisites

Production2ca5e571 writes each complete EP lane row and releases historical row
objects; actual-main validation reads the complete ledger one row at a time.
These two EP growth patterns are removed and their source gates are accepted.
Full multiplicity remains1,443,200 ledger rows per mode. Stage-batch/event-trace
memory and actual whole-workload RSS remain unmeasured; source admission alone
does not qualify the execution environment or establish a full-run memory bound.

| Timing boundary | Contract |
| --- | --- |
| `frontier.main` subprocess | 2400 seconds separately for offline and online |
| `validate()` after subprocess | outside that subprocess limit |
| Retained outer launcher | 5100 seconds for both modes, validation and receipts |

Historical main08 runtime projections are not current-source timing. Pricing
0.088167 s and emission 0.038156 s come from different source/control runs and
must not be added into a measured full-wave or whole-CLI result. Current-source
cohesive pricing/emission/flush timing and cloud append cost remain unverified.

## Cloud evidence and cleanup

Existing evidence establishes a 2406 B write/read/delete control and a
40,581,120 B selected-shape archive with hash-identical readback. Both use the
personal alias from the CPU host; neither proves worker-to-volume append
throughput. The approximately 39 TB shared free-space observation is dated,
unreserved and does not prove personal project quota. Quota sufficiency is settled
instead by YC's explicit2026-10-25 decision, not by that observation.

Retain selected commands, manifests, execution/validation receipts, hashes and
decisive failure evidence. After analysis, clean only completed disposable
outputs owned by this task in the personal directory; preserve active fixtures,
cited inputs and other-session files. Never interpret a partial ledger/file as
a successful run; completion requires terminal execution and validation receipts.

## Dependencies and next actions

1. Allocation is verified in ledger_memory/ and
   repairs/ep_lane_ledger_scope_amendment.json: four CORE files,135gross,
    116allocated paths,unchanged budgets; amended caps36/20/65/14 in886945f3.
    Final candidate1461573d and production2ca5e571 pass caller, independent
    audit/reviews, mandatory frozen suites and production parent guard. Source
    admission is complete; full workflow remains pending.
    Standing authority is requirements.md119–134.
2. Verify supported successful output bytes/order, snapshots and failure behavior;
   re-freeze and run required regressions after any accepted source repair.
3. Choose concrete task-required worker resources, image and the current-machine
    source mount under the recorded H800/codesign authority; qualify environment,
    creator, worker cloud access/append throughput and lifecycle. Quota sufficiency
    is user-confirmed. This records update does not launch a worker.
4. Copy/init diagnoses are complete: NO_ALLOC static copy and ALLOC init
   recommendation only, both with zero new controls and no source allocation.
   Obtain cohesive current-source timing before another full6144/1024 run; do not
   extend old diagnostic ceilings. Retain2400 seconds per main subprocess and
   full output. Measure full-run memory rather than substituting configured limits.

Static references: run_step5_measured_main.py:206–258,274–323;
metrics_store.py:380,525–538,4995–5017; ep_wave_metrics.py:119–134;
immutable_cost_fields_repair/evidence/baseline_result.json:20;
workload_multiplicity/report.md; storage_readiness/{evidence.json,cloud_control.json};
workload_shape_validation/evidence/supplemental/cloud_archive.json.
