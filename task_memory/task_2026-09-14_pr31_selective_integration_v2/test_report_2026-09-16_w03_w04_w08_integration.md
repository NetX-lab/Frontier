## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Recorded integrated runtime, numerical reuse, reporting and complete CPU validation. |

# W03/W04/W08 integration verification

## Execution

Repository: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`. Python `/usr/bin/python`3.12.3; no active conda environment. Pinned main0515589; candidate5a1cc2af plus `/data/ycfeng/tmp/pr33-w10-final-source-2.diff` and the explicitly staged new modules/tests. NumPy2.4.6,pandas3.0.3,scikit-learn1.9.0; exact installed versions are also recorded in the final validation manifest.

```bash
env PYTHONPATH=. TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/unit -q -p no:cacheprovider
```

Final log `/data/ycfeng/tmp/pr33-w10-final-unit-2.log`: **3584 PASS / 18 FAIL / 25 SKIP / 96.78s**,3627 nodes. Root directly compared exact failed node sets with fresh baseline `/data/ycfeng/tmp/pr33-main-unit-baseline.log`: candidate-only failures **zero**,18 common failures; one original profiling-documentation assertion now passes. Causes and removed/added test nodes are recorded in the W10 CPU report. The first45FAIL/3551PASS result is retained, not overwritten.27 candidate-only failures were resolved through contract-aware fixtures and explicit quantization-state isolation without weakening numerical oracles.

## Criteria and observed evidence

- `ExecutionTime` is always one layer; `StageExecutionTime` aggregates ordered physical IDs and publishes isolated finalized numerical state. Single-layer probes reject multi-layer inputs; source/sibling mutation, repeated reads, PP offsets, model-time and diagnostic-time ownership are checked independently.
- Stage-owned CPU/PP/proposer/MTP values are constructed once. MTP full8-layer and offset5/count3 replay pass without invalid layer8. Homogeneous dense/disaggregation predicts and freezes once; per-layer MoE/EP work remains independent.
- Attention reuse is one synchronous stage-local dictionary keyed by validated family/variant. Cross-request cache probes and repr-derived keys were removed. Direct missing GDN artifacts fail before dense lookup.
- Requested operation metrics and completed ledgers work with utilization disabled. Stage fields project the shared ownership contract. Actual EP lane operators retain five phase starts and barrier gaps; the independent two-lane oracle has13/12ms work within a15ms shared wave. Stage owner is not charged again in lane reports.
- Layer expansion decisions are shared by source-batch identity. Both EP-before-attention and attention-before-EP preserve the first batch and aggregate the next quota-excluded batch. Reporting tests14PASS; actual hybrid/reporting integration18PASS before the two extra order tests; full CPU covers the final versions.
- Final-source non-dummy8 deployments **8PASS/37.57s**. Seven homogeneous baseline request/system pairs MATCH with unchanged rel1e-12/abs1e-9 comparator. Final payload optimization preserves all106 artifacts across8 deployments. See W10 non-dummy report for family/lane cardinality and fixed-target arithmetic.
- Native integration19 cases explicitlySKIP due unavailable GPU/runtime. No hardware or production-profile numerical fidelity is inferred from CPU synthetic fixtures.

## Performance evidence boundary

Separate cProfile of the exact W00 small dense PDD workload observed component snapshots1440→36 while1440 identity records and36stage calls remain. `_assemble_stage` cumulative profile time.114594→.025278s; `_clone_mutable_components` .065055→.001986s. These profiled times explain removed work and are not unprofiled acceptance samples. Profiles: `/data/ycfeng/tmp/pr33-w11-small-dense.profile` and `.../pr33-w11-small-dense-final.profile`. W04's prior controlled attention ablation independently established18.26% paired reduction on the trained synthetic hybrid fixture; the final paired baseline/D02 decision belongs to `performance_rca.md`.

Raw58-case comparison remains separately classified, because D01-approved uniform scope and corrected reporting cause real differences. No golden or comparison tolerance was changed. Final-source rerun and detailed arithmetic classification remain tracked by W10 fidelity evidence.
