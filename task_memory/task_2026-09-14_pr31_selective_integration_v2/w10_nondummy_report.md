## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Final c9f8f904 rerun: 8 PASS in 35.43 s; 106 artifacts and 7 baseline metric pairs PASS. |
| 2026-09-16 | Latest-source rerun: 8 PASS in 37.57 s; 106 prior-final artifacts PASS under the established comparator. |
| 2026-09-16 | Completed eight normal-constructor synthetic CPU cases, seven fixed-baseline controls, and independent operator/lane/capacity oracles. |

# W10 non-dummy acceptance campaign

Status: synthetic non-dummy acceptance PASS for the eight cases below. Native timing parity, the full CPU unit suite, and the 58-case fidelity lane remain separately owned gates.

Owned harness: `tests/integration/test_pr33_nondummy_acceptance.py`.
Agent: `/root/w03_fixture_migration`.

## Scope and criteria

The campaign uses normal `Simulator`, profile loading/training, predictor construction, scheduler/event processing, and metric export. No predictor `object.__new__` or numerical runtime hooks are installed. The only model substitution registers a typed, explicit synthetic configuration at the existing model-name loading boundary. Each case runs in an isolated Python process, output directory, and cache directory, matching the established fidelity runner's deployment isolation.

Synthetic CPU CSVs adapt the existing hybrid production-constructor fixture. Constant targets support independent numerical assertions. `CUDA_EVENT` and `KERNEL_ONLY` labels select simulator profile routes; they are synthetic observations, not native measurements. Hybrid GDN uses the existing checked-in synthetic CSV and real GDN trainer/artifact loader.

| Case | Family | Architecture | PP | EP | Mode |
| --- | --- | --- | --- | --- | --- |
| 0 | Dense | co-location | 1 | 1 | offline |
| 1 | Dense | sequential PDD | 1 | 1 | offline |
| 2 | Dense | sequential PD-AF | 1 | 1 | offline |
| 3 | MLA | co-location | 2 | 1 | offline |
| 4 | MoE | co-location | 1 | 2 | offline |
| 5 | MoE | sequential PDD | 1 | 1 | offline |
| 6 | MoE | sequential PD-AF | 1 | 2 | offline |
| 7 | Hybrid GDN/dense | co-location | 1 | 1 | online |

Every case requests and checks nonempty request metrics, system metrics, per-role operation CSVs, operator traces, and stage ledgers. Two requests must complete with 18 tokens each. Trace events must have finite, nonnegative durations and retain all physical layer identities through either expansion or explicit aggregate membership. A family-native trace scope must equal the independent 0.01 ms target. Dense and MLA co-location operation CSVs must equal `physical_layers * 0.01 ms` for that scope.

Hybrid additionally checks the independent G/G/G/A/G/G/G/A schedule, GDN core prefill/decode targets 0.22/0.05 ms per physical layer, capacity one, no remaining state-slot owners, and complete KV release.

## Environment and execution

- Python: `/usr/bin/python`, 3.12.3; no active conda environment.
- Simulator subprocesses use `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `TMPDIR` under `/data/ycfeng/tmp`.
- Baseline worktree: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr33-r12-baseline-20260915`.
- Verified baseline commit: `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2`.
- Each standalone run writes `invocation.json` with its actual imported `frontier.simulator` path, avoiding accidental candidate imports in baseline runs.

Collection command:

```bash
python -m pytest tests/integration/test_pr33_nondummy_acceptance.py --collect-only -q > /data/ycfeng/tmp/pr33-w10-nondummy-collect.log 2>&1
```

Observed: **8 tests collected in 2.70 s**.

Standalone baseline execution pattern, run from the baseline worktree:

```bash
PYTHONPATH=/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/pr33-r12-baseline-20260915 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp python /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn/tests/integration/test_pr33_nondummy_acceptance.py --case CASE_INDEX --output OUTPUT_DIRECTORY > LOG_PATH 2>&1
```

All seven homogeneous controls returned exit 0. Case 0 output/log use `/data/ycfeng/tmp/pr33-w10-baseline-dense-coloc[.log]`; case 3 uses `/data/ycfeng/tmp/pr33-w10-baseline-mla[.log]`; cases 1, 2, 4, 5, 6 use `/data/ycfeng/tmp/pr33-w10-baseline-caseN[.log]`. Hybrid is deliberately outside baseline comparison because main does not support its runtime contract.

## Intermediate findings

Harness setup failures were visible and corrected at their source: dense co-location legitimately uses direct predictor training without a shared manager; process-global `IS_MOE` requires deployment process isolation; PD-AF requires explicit AF microbatch fields and the Orca FFN scheduler; its manager requires explicit KERNEL_ONLY paths; MLA imports require all registered timing-stat columns. None was hidden by skips or fixture-only production behavior.

All seven homogeneous candidate cases have identical request/system metrics; dense and MLA cases additionally have identical stage ledgers to baseline under `run_scheduler_refactor_fidelity.compare` (`rel_tol=1e-12`, `abs_tol=1e-9`). Full operation CSV comparison has explicit schema and numerical differences: inactive-family columns disappear, and baseline repeats a layer multiplier. For a 0.01 ms scope, four-layer dense baseline reports 0.16 ms versus candidate/oracle 0.04 ms; MLA PP2 baseline reports 0.08 ms versus candidate/oracle 0.04 ms. These are tracked uniform-ownership differences, not a full-artifact parity PASS. Final comparison evidence is recorded below.

MLA omits generic input norm/model projection names from traces on both baseline and candidate. The campaign therefore checks the registered MLA-native cache scope rather than claiming those baseline-absent trace names are newly supported. This limitation was reported to the parent for W08 assessment.

During concurrent W08 integration, MoE/hybrid runs exposed an intermediate `on_prefill_sync(metrics_store=...)` signature mismatch and then a missing `MetricsStore.ep_wave_reporting_enabled`. These failures were reported immediately and disappeared after the parent completed W08. The complete eight-case campaign then passed.

## Independent numerical and ownership oracles

- Every expanded dense input norm or MLA-native cache event equals the synthetic 0.01 ms target, with absolute and relative errors at or below the established comparison tolerance. The four-layer dense and MLA co-location operation total is 0.04 ms; baseline's 0.16/0.08 ms is deliberately retained as a DIFFERENCE, not treated as a golden value.
- Every positive future EP lane grouped GEMM equals the synthetic 0.12 ms target. Each of the five phase dictionaries has finite nonnegative operator values whose sum equals its phase duration. PD-AF's existing nonempty FFN stage lanes independently satisfy the same 0.12 ms grouped GEMM target.
- Two requests, four physical layers, EP2, and two model passes imply 32 co-location MoE lane records. Sequential PDD retains one prefill plus two decode passes, producing `2 * 4 * 3 * 1 = 24` records. Sequential PD-AF records `2 * 4 * 2 = 16` PREFILL lanes in the new ledger and another 16 DECODE_FFN lanes in the existing stage ledger. These are explicit existing architecture semantics; this campaign does not revise first-token behavior.
- Eight-layer hybrid, EP1, two requests, two model passes imply 32 lane records. Exact GDN membership is `{0,1,2,4,5,6}` and dense membership `{3,7}`. Every positive GDN prefill/decode core event matches 0.22/0.05 ms per represented physical layer. Capacity one is exhausted/reused during execution and empty at the end; no state-slot owners or KV blocks remain.

The eighth campaign's stronger lane assertion initially failed two cases because its provisional formula incorrectly assumed two passes and a single ledger owner for every architecture. The corrected formula above is independently tied to request counts, requested decode tokens, physical layers, and EP width, and retains checks of both PD-AF ledger domains. The ninth complete run passed **8 tests in 36.50 s**. After the parent's predictor freeze, the final complete run passed **8 tests in 36.78 s**; there were no skips or collection differences.

## Baseline differences retained for review

Every common request and system artifact passes `run_scheduler_refactor_fidelity.compare` at `rel_tol=1e-12`, `abs_tol=1e-9`. No tolerance was loosened and no golden output was rewritten.

| Artifact domain | Observed comparison | Explanation and independent check |
| --- | --- | --- |
| Dense/MLA stage ledger | PASS | Same stage timing and discrete rows |
| MoE stage ledger | DIFFERENCE | Uniform stage ownership: for example 0.002109227 -> 0.008436908 ms TP attention collective, or 0.01 -> 0.04 ms cache scope, covering all four physical layers |
| Dense/MLA operation CSV | DIFFERENCE | Inactive-family columns removed; repeated stage layer multiplier removed, independently checked 0.04 ms total |
| Dense/MoE PD-AF decode operation common columns | PASS | Existing decode attention/FFN common values unchanged; inactive-family columns removed |
| MoE operation CSV | DIFFERENCE | Future EP lane rows now present: co-location 4 -> 36 rows, PDD prefill 2 -> 10/decode 4 -> 20, PD-AF prefill 2 -> 18 |
| Operator traces | DIFFERENCE | Family membership metadata, corrected PP physical-layer expansion, and actual EP lane operators. All durations/layer membership independently checked |
| Future EP lane ledger | ADDED_ARTIFACT | Nonempty real schedule reporting checked by explicit cardinality, lane IDs, phases, and numerical GEMM oracle |

These differences have arithmetic explanations under the planned uniform layer/stage and W08 reporting contracts; they do not establish full raw-artifact equality with main. D01 explicitly accepted two isolated historical corrections and did not authorize blanket acceptance of other discrepancies. The parent must classify these additional reporting differences in its final decision audit; this campaign does not extend that authorization. The existing MLA generic norm/model-projection omission remains a known coverage limit on both baseline and candidate. There are no unexplained request/system numerical differences in this campaign.

## Final observed artifact inventory

| Case | Requests | Stage rows | EP wave rows | AF lane rows (within stage ledger) | Operator events | Operation CSVs | Request E2E ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 2 | 4 | 0 | 0 | 100 | 1 | [2.56, 5.12] |
| 1 | 2 | 6 | 0 | 0 | 182 | 2 | [5.22262144, 7.6626214400000014] |
| 2 | 2 | 18 | 0 | 0 | 186 | 3 | [2.9960913066666675, 4.549561173333334] |
| 3 | 2 | 8 | 0 | 0 | 92 | 1 | [2.1621160533333335, 4.324232106666667] |
| 4 | 2 | 4 | 32 | 0 | 320 | 1 | [3.3929557333333342, 6.785911466666665] |
| 5 | 2 | 6 | 24 | 0 | 266 | 2 | [6.822621440000006, 10.062621440000004] |
| 6 | 2 | 26 | 16 | 16 | 306 | 3 | [3.837074346666666, 5.814653440000003] |
| 7 | 2 | 4 | 32 | 0 | 304 | 1 | [9.780000000000006, 19.559974671160965] |

Final log: `/data/ycfeng/tmp/pr33-w10-nondummy-final.log`. Final output root: `/data/ycfeng/tmp/pr33-w10-nondummy-final`. Per-case roots are `test_nondummy_simulator_acceptN/run`; each contains `invocation.json`, `acceptance_evidence.json`, profiles/cache, and metrics. Full temporary comparison: `/data/ycfeng/tmp/pr33-w10-comparison-final.json`.

Persistent exact commands and selected comparison evidence are in `test_report_2026-09-16_w10_nondummy_acceptance.md`. The owned implementation and validation are complete; no production edits or commits were made by this agent. Remaining W10 work owned by the parent is the full CPU suite, 58-case control, and final integrated decision audit.

## Latest-source final rerun

After the parent changed homogeneous disaggregation to predict/finalize once, removed discarded ExecutionTime copies, and added direct-query GDN missing-artifact fail-fast, the entire eight-case campaign was rerun on that source. **PASS: 8 tests in 37.57 s**, no skips. This supersedes the earlier 36.78 s run as the latest-source execution evidence.

Exact command:

```bash
python -m pytest tests/integration/test_pr33_nondummy_acceptance.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-w10-nondummy-final-2 > /data/ycfeng/tmp/pr33-w10-nondummy-final-2.log 2>&1
```

All eight new case outputs were compared with the prior final run using the same established comparator and trace-header normalization. **PASS: 106 artifacts**, per-case counts `[10,14,17,10,11,15,18,11]`. Selected artifact names are exactly identical. The comparison includes all CSV/JSONL outputs, system metrics, stage-ledger summaries, and exact acceptance cardinalities. No new numerical/discrete differences occurred. Previous baseline differences remain explicitly documented; this comparison does not change their decision status.

Evidence: `/data/ycfeng/tmp/pr33-w10-nondummy-final-2.log`, `/data/ycfeng/tmp/pr33-w10-final-2-vs-final.json`, and case roots under `/data/ycfeng/tmp/pr33-w10-nondummy-final-2`. No production or test code changed during this rerun.

## Final revision c9f8f904 verification

**Latest-source PASS: 8 tests in 35.43 s** on commit `c9f8f904e3550c11aad3cc5d851d75d648cef6e1`. This revision reuses a homogeneous sum only when the finalized timing payload is shared safely. This run supersedes final-2 as the latest-source evidence; all previous observations are preserved above.

Environment remains `/usr/bin/python` 3.12.3, no conda environment, CPU only; isolated simulator processes use OMP/OpenBLAS threads one and TMPDIR under the case root. The exact executed command was:

```bash
python -m pytest tests/integration/test_pr33_nondummy_acceptance.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/pr33-w10-nondummy-final-3 > /data/ycfeng/tmp/pr33-w10-nondummy-final-3.log 2>&1
```

Observed: `8 passed in 35.43s`, no skips or collection changes. Compared all **106 artifacts** with final-2 using the established comparator (`rel_tol=1e-12`, `abs_tol=1e-9`) and existing trace-header normalization: **106/106 PASS**, artifact-name sets identical, per-case counts `[10,14,17,10,11,15,18,11]`. Rechecked baseline `0515589ac7f49ac5288a5f55b0ce38b0ede29bb2` request and system metrics for the seven homogeneous cases: **7/7 request PASS, 7/7 system PASS**. The independent physical-layer, lane-count, timing, and capacity assertions all passed. There is no new numerical/discrete difference to classify.

Evidence: `/data/ycfeng/tmp/pr33-w10-nondummy-final-3.log`, `/data/ycfeng/tmp/pr33-w10-final-3-vs-final-2.json`, `/data/ycfeng/tmp/pr33-w10-final-3-baseline-metrics.json`, and isolated output roots below `/data/ycfeng/tmp/pr33-w10-nondummy-final-3`. No production or test edits accompanied this rerun. CPU correctness and comparison work is complete; this agent holds further CPU executions for the parent's exclusive paired window.
