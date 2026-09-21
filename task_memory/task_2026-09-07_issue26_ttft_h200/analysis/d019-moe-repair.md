## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Validated fresh H200 numerical/profile evidence, corrected missing export metadata without changing measurements, and committed isolated repair5dd5ee39. |
| 2026-09-08 | Recorded the approved activation/reduction correction and exact-layout profiling preparation. |

# D019 MoE profiling correction

Status: isolated activation/reduction correction PASS and committed5dd5ee39; paired operator integration, first-forward CUDA and clean E2E numerical calibration pending. Earlier preparation/pending statements below are historical and superseded by the fresh-validation section.

Approval: YC explicitly approved `允许将已证明的 activation／reduction 覆盖问题纳入具体修复评审（ gated SiLU）。按照上述计划执行。` This lifts the D013 deferral for the demonstrated grouped-expert activation/reduction coverage repair. The approved D019 first-forward sequence supersedes the skill's broader batch-completion gate; it does not establish numerical completion.

Change marker: D019-MOE-ACTIVATION-REDUCTION-01. Owner: `/root/moe_profile_repair`. Sources inspected: `frontier/profiling/moe/moe_vllm_kernel.py` and diagnostic vLLM `vllm/model_executor/layers/fused_moe/fused_moe.py` at 8453dd342. Existing profile provenance and shape findings are in `profile_coverage_plan_d019.md`.

The Frontier timed iteration invokes W1, copies only the first half of its output, and invokes W2. vLLM executes gated SiLU between those GEMMs and `ops.moe_sum` after W2. W2 already applies routing weights on both sides. Replace the slice-copy with the real activation, add the reduction, and allocate buffers before timing with the same W1/W2 workspace reuse as vLLM. Preserve the existing grouped-GEMM profile identity, which owns the expert FFN compute path. No additive timing constant is introduced.

The affected production module is 688 lines, below the 2,000-line module-size threshold. Its existing timed-iteration seam is the appropriate scope; no functional split or broad refactor is necessary.

The separate bounded profiler will reuse `routing_inputs` for deterministic runtime `uniform_topk`, keeping global input rows, local expert assignments, and padded kernel blocks distinct. This does not add a public traced-routing interface. Existing sampled-uniform profiles remain evidence of the old producer and are not silently relabeled.

Validation planned: direct numerical equality of the repaired BF16 gated expert path against the current vLLM `fused_experts` on identical weights, routing and EP maps; targeted H200 measurements at global M=4096/4097, EP0 and another rank, plus nearby M=4095 holdout. Record exact kernel configuration and alignment output before timing. GPU execution is centrally scheduled by root; this lane does not launch a GPU job.

## Prepared implementation and direct checks

Production patch: `frontier/profiling/moe/moe_vllm_kernel.py`, 22 insertions and 12 deletions. Regression: `tests/unit/test_moe_fused_expert_numerical_parity.py`, four real CUDA cases at M4096/4097 and EP0/1, using model-derived hidden/expert dimensions and identical BF16 operands. Acceptance is bitwise output equality against the currently installed vLLM `fused_experts`.

GPU profiler: `tests/performance/issue26_moe_exact_profile.py`. It writes `moe.csv` via existing MoEWrapper load-feature and typed-contract producers, with exact deterministic routing supplied through `routing_inputs`. Independent `receipts.json` measurements cover full expert path and W1, activation, W2 and reduction in separate loops with both standalone and prefill-hot context. These component medians are not summed as an additive full-span reconstruction. The CSV's grouped profile uses the unchanged public profiler's standalone execution context; the separate contextual experiments remain diagnostic evidence.

The actual vLLM `moe_grouped_gemm` event scope ends before `moe_sum`. The repaired Frontier grouped-expert cost must be compared with same-run vLLM grouped parent plus sum; the historical 18.180160ms parent alone omits reduction.

CPU checks, Python3.13.13 in conda `dev-vidur-v03-hopper-e2e`: AST parsing of all three changed/added Python files PASS; `git diff --check` PASS. This CPU environment has neither torch nor triton, so numerical CUDA tests are pending, not passed by syntax inspection. No GPU job was launched by this lane, and code is not committed before GPU validation.

Exact uniform load-feature arithmetic used the shared `frontier.moe_load_imbalance.MoELoadImbalanceInput`: M4096 all examined ranks have 16x256 local assignments and entropy4.000000000017834; M4097 EP0 has eight257 and eight256, total4104, entropy3.9999972590111907; EP1 remains16x256. Initial diagnostic code queried nonexistent `total_tokens` and raised AttributeError; corrected it to the existing `total_routed_tokens`/sum of counts. No production behavior was altered for that diagnostic typo.

M4095 EP0/1 retain exactly the same local-feature key as M4096. Therefore add EP7 to the exact profile command when testing an unseen local-load feature: its M4095 counts include eight255 and eight256. Merely calling M4095 EP0 a holdout does not demonstrate RF generalization.

```bash
PYTHONPATH="$PWD:/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908" \
  /local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python -m pytest \
  tests/unit/test_moe_fused_expert_numerical_parity.py -q -p no:cacheprovider

PYTHONPATH="$PWD:/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908" \
  /local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python \
  tests/performance/issue26_moe_exact_profile.py --output-dir "$D019_MOE_OUTPUT" \
  --model qwen3-a3b-30b-moe --ep-size 8 --tokens 4095 4096 4097 \
  --ep-ranks 0 1 7 --samples 20 --seeds 0 1 2
```

Run from the active worktree in the approved H200 image, with `$D019_MOE_OUTPUT` bound to the new immutable run directory by root. Commands above are prepared and have not yet executed on H200.

## Fresh H200 validation and measurements

Production correction and bounded checks committed as `5dd5ee39`. H200 runtime `analysis/h200-d019-profiles-01/runtime/numerical.log`: **4 passed in14.64s**. The exact profiler also passed27 bitwise comparisons against current vLLM. Artifact checks passed27 rows (M4095/4096/4097 x EP0/1/7 x seeds0/1/2), four typed contracts per row, actual expert counts/padded blocks, and5,400 positive finite component samples. Standard deviations are finite and nonnegative.

Values below are the median of three seed medians, milliseconds per layer. The CSV grouped cost includes the repaired activation and final reduction. The standalone/hot diagnostic full-path values use independent operands/loops; they are context measurements, not an additive timing reconstruction.

| Global M | EP | Local assignments | Local padded assignments | CSV complete expert path | Diagnostic standalone full | Diagnostic prefill-hot full |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4095 | 0 | 4096 | 4096 | 0.445376009 | 0.450191990 | 0.393519998 |
| 4095 | 1 | 4096 | 4096 | 0.457471997 | 0.461728007 | 0.408736005 |
| 4095 | 7 | 4088 | 4096 | 0.473760009 | 0.477247998 | 0.423920006 |
| 4096 | 0 | 4096 | 4096 | 0.449375987 | 0.450400010 | 0.392399997 |
| 4096 | 1 | 4096 | 4096 | 0.461183995 | 0.462880000 | 0.408751994 |
| 4096 | 7 | 4096 | 4096 | 0.475775987 | 0.477632001 | 0.424191996 |
| 4097 | 0 | 4104 | 4608 | 0.463360012 | 0.458335996 | 0.399632007 |
| 4097 | 1 | 4096 | 4096 | 0.457792014 | 0.453167990 | 0.401840001 |
| 4097 | 7 | 4096 | 4096 | 0.472416013 | 0.465759993 | 0.413039997 |

Every receipt selected BLOCK_SIZE_M64, BLOCK_SIZE_N64, BLOCK_SIZE_K32, GROUP_SIZE_M8. W1 shape[16,1536,2048], W2[16,2048,768], BF16. M4097 EP0 crosses eight expert blocks from256 to320, adding512 padded assignments (local4608). This is measured alignment output, not a one-token cost estimate.

For M4096 EP0, independent standalone component medians are W1=.178847998ms, activation=.138608001ms, W2=.134928003ms, reduction=.060432000ms. The corresponding hot-context medians are .124095999/.137167998/.081999999/.059776001ms. Their sum is not used as a full-span estimate. Activation/reduction show material device execution; GEMM event durations depend on the measured context.

Old P grouped-expert estimate was .325748ms/layer (15.635904ms across48 layers), while the repaired M4096 EP0 CSV measures .449375987ms/layer. This is **a changed implementation and exact input layout**, not a same-implementation P-minus-S prediction error. The new measurement includes gated SiLU and reduction and removes the old slice copy. Neither the difference nor independently measured component medians establish how much clean TTFT will change; fresh simulator and paired V scopes remain necessary.

Even with identical local counts, M4096 EP0/1/7 CSV medians differ (.449376/.461184/.475776ms). The existing predictor features do not encode global expert-map position. These measurements establish residual map/context variation, not its unique causal mechanism. Preserve the rank receipts rather than silently collapsing them into a claim of exact physical equivalence.

### Export validation failure and scoped correction

The first exact script exported wrapper-level rows but omitted the canonical CLI metadata postprocessor. Actual `_get_profiling_metadata` rejected raw moe.csv: `profiling_precision column is missing`. The script now invokes existing `moe.main._attach_moe_output_metadata` with model-resolved fields. No timing/production code changed after GPU collection. Preserved raw evidence; generated `analysis/d019-moe-normalized/moe.csv` using measured BF16 and model identity (generic/generic/none), with all original columns exactly unchanged under round-trip CSV parsing. Actual predictor metadata validation PASS: BF16, generic architecture/profile, quant_signature none, CUDA_EVENT. Existing metadata tests: **6 passed in0.61s**. One scratch validation initially tried to instantiate abstract SklearnExecutionTimePredictor and raised TypeError; corrected only the check to call its real metadata method with the required model context.

### Integration requirements

1. All new rows label gating_runtime_context=prefill_hot, routing_runtime_path=uniform_topk, assignment=round_robin_uniform and uniform_1_over_topk weights. Actual counts agree with that policy. The old loader requires standalone_legacy rows for base gating models (`sklearn_moe_execution_time_predictor.py:1256`) in addition to hot pseudo-model rows. The new27-row CSV cannot replace the complete runtime source by itself.
2. Retain current-task measurements of unchanged gating/shuffling only through explicit per-op target selection. The changed expert-path targets must come exclusively from corrected measurements. Do not concatenate old incomplete grouped targets into the new training table. The ordinary trainer drops NaN target rows per operation (`sklearn_execution_time_predictor.py:3048`, `training/moe_trainer.py:471`); any composed sparse table must be validated with the actual runtime/training path, not falsely presented as a newly emitted canonical dense profile grid.
3. Exclude M4095 labeled holdout from training for the relevant holdout test. M4095 EP0/1 local feature keys duplicate M4096; only EP7 supplies the distinct partial-load key. Corrected27 rows do not cover decode/small-batch grouped work; they are first-forward anchors, not full-case profiling completion.
4. Default pandas CSV parsing changes a few low bits of fractional feature columns. For M4097 EP0, max_load_ratio becomes1.0019493177387917 from1.0019493177387915 and entropy3.999997259011191 from3.9999972590111907. Feature checks pass1e-14 tolerance but exact-key equality does not necessarily pass. Perfect-uniform4096 features are exactly equal. Preserve this distinction in new P-query receipts; no rounding/scaling workaround is added here.

Fresh operator mapping, predictor integration and first-forward/full-case CUDA and official TTFT gates remain pending; this commit closes the demonstrated expert-kernel coverage defect and its isolated correctness check.
