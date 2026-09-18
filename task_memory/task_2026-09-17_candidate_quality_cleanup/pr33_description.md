## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-17 | Prepared local PR description with reviewed corrections and explicit evidence boundaries. |

# AMD profiling, hybrid GDN simulation, and ownership cleanup

This branch adds standard ROCm profiling and hybrid GDN simulation support, with physical-layer timing and explicit request-state ownership. The review follow-up fixes the boundaries between scheduler batches, GDN prediction/training artifacts, native producer initialization, timer provenance and documented profiling commands.

Concurrent co-location traffic can combine running decode with a new prefill chunk. Under the user's explicit temporary decision, GDN prediction now handles the complete mixed batch with prefill estimators and emits a RuntimeWarning. The scheduler and request state retain their actual phases. Native mixed-kernel accuracy is unvalidated; full-attention layers retain their own mixed-profile requirements. Native GDN profiling and training still require pure phases.

The remaining corrections use the canonical runtime attention-family resolver for hybrid ROCm full-attention; publish immutable six-estimator generations through one atomic manifest switch; preserve ragged query features across producer/CSV/runtime; reject requested-model dtype mismatches; validate effective timer ownership before native initialization; resolve shell postflight paths through the canonical measurement helper; and normalize one-shot graph workloads once. MI355X recipes explicitly select VLLM_ROCM and launch TP8 through eight torchrun processes. GDN capacity checks avoid owner-tuple allocation, and reporting-only prefill prediction runs only when an enabled consumer needs its payload, including ledger summaries.

## Revision and comparison scope

- Reviewed revision: `d43ae93240444bd4eff9bd99f296d2e751370514`.
- Reviewed-fix production/test checkpoint: `35ac95eb`; final evidence below identifies any subsequent source correction.
- Frozen pre-cleanup candidate: `c288a19f59bec09529ee18d782fa57218da2c781`.
- Frozen main: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`.
- Exact cleanup/fidelity comparisons use the frozen candidate. They do not claim equivalence to main: the earlier main-to-candidate comparison includes accepted D01 physical-layer/stage accounting corrections. D02 is limited to its recorded measured workload; it is not a universal performance budget. D03 preserves supported lifecycle cleanup without introducing cancellation.
- Historical checkpoints, including the original 3,274-pass/19-failure result and later 3,813-pass/18-failure/25-skip cleanup result, remain archived. They are not relabeled as fresh results for the reviewed fixes.

## Current verification

Fresh verification at production/test checkpoint `35ac95eb`:

- Full unit: **3,928 PASS / 18 existing FAIL / 25 existing SKIP**, 167.43 s. Failure and skip node sets match the prior frozen validation exactly; failure messages also match after scratch-directory normalization. The suite is not all green.
- Non-dummy: **9 PASS**, 81.28 s, including the new staggered mixed-batch/slot-ownership regression. The eight existing cases preserve **90 stable artifacts**, a symmetric **106-file metrics inventory**, and **16 nonempty supplemental JSON pairs** against retained frozen-candidate outputs.
- Fidelity against `c288a19f`: **58 PASS / 0 FAIL**, 304 artifacts, 122 completed requests per revision. Independent traversal finds zero drift across 902,628 finite numeric leaf pairs (maximum absolute/relative difference 0.0).
- Final isolated timing: **18 successful measurements**, all nine event/completion-preserving pairs. Median paired Simulator.run changes: **−12.07% / −8.84% / −0.62%** for small dense / longer dense / representative MoE; subprocess changes **−3.62% / −6.90% / −2.92%**. Small-dense attempt 2 is 16.62% slower and remains in the evidence. Three reporting-disabled dummy pairs do not establish general or native performance.

Commands, failed-node classification and evidence are in `task_memory/task_2026-09-17_candidate_quality_cleanup/test_report_2026-09-17_review_final.md`; durable verdicts and observations are in `review_final_evidence.json` in the same directory. The 18 existing failures concern legacy optimizer/debug assets, older public-documentation assertions and analysis CLI dependencies.

## Limits and handoff

CPU orchestration and synthetic profiles establish contracts and simulator preservation, not native CUDA/ROCm/SGLang/GDN correctness or production-profile timing accuracy. The native acceptance suite now includes real hybrid Qwen3.8 ROCm prefill/decode with causal-reference and timing checks, but no native hardware run is claimed. TP8 requires eight visible compatible devices and the documented runtime dependencies. GDN output-gate variant equivalence is not established by the dtype check.

The review report, individual R01–R10/C01–C02 test reports, decisions, final evidence and completion summary are retained under `task_memory/task_2026-09-17_candidate_quality_cleanup/`. This file is a local, reviewable PR body; no remote description update, push or merge has been performed.
