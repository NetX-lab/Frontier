# V4 graph and measured-scope CPU verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded source evidence,686 affected passes, real isolated consumer comparisons, full-unit baseline comparison and clean strict77 IDENTICAL. |
| 2026-10-21 | Applied H800-priority steering:688 affected passes,102 scopes/14 negatives and hardware-mismatch rejection; runtime code remains621b198. |
| 2026-10-21 | Review correction:CPU red/green evidence, exact layout, typed coverage causes and padded embedding collective;208 affected passes/102 scopes/68 negatives. |
| 2026-10-21 | Archived correction282cb11 sole clean strict77 IDENTICAL77/77, zero cache/provenance differences and clean retained checkout. |

## Scope and result

Implemented native-descriptor compute widths and exact isolated measured scope
consumption. This is not native serving admission or a calibrated E2E prediction.
No GPU, platform, authentication, SSH, cloud, network, Docker, native runtime,
profiler runner, frozen package, B-owned allocator/transfer or H5 file was changed.

## Pinned source evidence

All native reads used `git show 2cf0a6915ce544dc493a0990f2ea38d81601128a:<path>`
inside `/data/ycfeng/Frontier/.real-engine/vLLM-BS`; its working HEAD was not used.

- `vllm/v1/worker/gpu/cudagraph_utils.py:138–143,382–438`: independent breakable
  graph selection, native descriptor dispatch, FULL replay and PIECEWISE wrapper.
  Mode0 is not graph-disabled. Parent G0-B records FULL_AND_PIECEWISE and19 FULL
  captures at TP8/DP4/EP32; these do not qualify timing/replay parity.
- `vllm/v1/worker/gpu/model_runner.py:1106–1111,1259–1315,1510–1545,1624–1672`:
  actual scheduled width stays distinct from descriptor.num_tokens; input IDs and
  positions use padded slices; slot mappings invalidate padded positions; FULL
  dummy prepares attention metadata and differs from no-metadata eager dummy.
- `vllm/models/deepseek_v4/nvidia/model.py:961–967,1149–1157`: SP all-gather is
  sliced to positions.shape[0], then reduce-scatter; initial SP sharding follows
  the graph-padded tensor width. Existing CC payload code already uses the
  effective batch width and therefore needs no second payload implementation.
- `vllm/models/deepseek_v4/attention.py:895–965`: short-context index fill is
  disabled during stream capture; FULL graph workloads retain scoring preparation.
- `vllm/models/deepseek_v4/compressor.py:345–399`: FULL cannot branch on current
  CPU c128_boundary and retains the compressor store invocation even with zero
  actual compressed writes. PIECEWISE HCA and short-context CSA capture-branch
  equivalence remain explicit coverage errors rather than guessed behavior.

## Environment and commands

Worktree `/data/ycfeng/Frontier/.worktrees/v4-graph-measured`, Python
`/data/ycfeng/envs/frontier-py310/bin/python`3.10.6. No environment changes.
All commands use `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1` and
`TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode`.

```bash
/data/ycfeng/envs/frontier-py310/bin/python -m pytest -q tests/unit/test_dsv4*.py tests/unit/test_dsa_predictor.py tests/unit/test_glm53_attention_cost_contract.py tests/unit/test_glm_sp_moe_contract.py tests/unit/test_glm_sp_moe_comm.py tests/unit/test_moe_grouped_gemm_padded_features.py tests/unit/test_shared_prediction_model_manager*.py tests/unit/test_profiling_runtime_boundary.py --junitxml=/data/ycfeng/tmp/v4_graph_measured_affected.xml
/data/ycfeng/envs/frontier-py310/bin/python tests/integration/check_dsv4_measured_scopes.py --attention-root /data/ycfeng/tmp/glm53_dsv4/v4_attention_cwd02_results/P6144 --compute-root /data/ycfeng/tmp/glm53_dsv4/h200_dsv4_compute_results_20261021/9f58dbe --output task_memory/task_2026-10-21_v4_graph_measured/real_scope_receipts.json
/data/ycfeng/envs/frontier-py310/bin/python -m pytest -q -p no:logging --continue-on-collection-errors tests/unit --junitxml=/data/ycfeng/tmp/v4_graph_measured_unit.xml > /data/ycfeng/tmp/v4_graph_measured_unit.log 2>&1
```

Expected affected suite: existing supported V4/GLM/reader behavior remains passing;
new graph widths preserve request/context and distinguish FULL capture branches;
bad measurement/graph/TP/tile-planning/context/phase/scope identity and relabelled
CUDA-event rows produce typed coverage errors. Final actual:
**688 PASS/8 SKIP** (11.59s).
Earlier failures were obsolete blanket-rejection expectations, changed only where
Q10 authorizes typed coverage errors. Existing test node IDs were preserved.

Expected real rows: every applicable scope of the12 accepted P6144 rows, three
shared kernels per method, and a matched MARLIN histogram per method return the
exact selected CSV median, retaining provenance and `serving_admission=false`.
Final actual: **102 positive exact scope comparisons and14 typed negatives PASS**;
`real_scope_receipts.json` contains each input path, source revision, runtime
fingerprint, physical/target TP, measured value and retained metadata.

The102 comparisons comprise attention50 CUDA_EVENT/44 KERNEL_ONLY and
compute4 CUDA_EVENT/4 KERNEL_ONLY. The enclosing attention-layer scope exists
only in the CUDA-event table and is not fabricated for kernel-only rows.

H800-priority follow-up uses the existing source interfaces at TP8/DP2/EP16,
with explicit P16/D16 requests. P16 prefill PIECEWISE width32 and decode-at-C16
FULL width8 preserve real16/1 rows and contexts0/16; per-rank compute4/1 and
two-source total40 are CPU geometry fixtures only, not observed native capture
sizes or hardware capacity. A test-only mistaken helper method name was corrected
to the existing `source_physical_rows` function; no runtime code change resulted.
Both actual H200 measurement methods are additionally rejected by the H800
consumer with `DeepSeekV4CoverageError` for device mismatch. The original H200
tables/provenance are not rewritten or admitted as H800 evidence. No H800 timing,
capacity, compiled/phase equivalence or calibration result was produced. Root's
common68f891b workload, platform actions, gates and B integration remain root-owned.

The MARLIN check uses an explicit reachable synthetic lane: source32/topk6,
local8 experts with counts[1,1,0,0,0,0,0,0], global192 assignments. It matches the
measured32-token/seed1 feature vector exactly. It is not a recovered native route.
The default pandas parser changed serialized entropy1.0000000002294083 to
1.0000000002294085; pandas `float_precision='round_trip'` through the existing
MoE reader preserves exact equality. Other callers keep their original parser
default. No tolerance, fitted interpolation, extrapolation or timing fallback was
introduced. Failed diagnostic output is retained in the session transcript.

## Measurement limits

- Shared FP8 DeepGEMM projections/native clamped activation and routed FP4 MARLIN
  grouped GEMM retain source9f58dbe and their isolated eager identity.
- Attention retains source5e051e4, P6144/C0, three variants/two tile roles,
  physicalTP1/target per-rankTP8, eager, maxlen6145/budget6144/maxseq128 and the
  measured metadata. SWA64, attention256 and state4/8 are kept distinct; they are
  not promoted to full G0-B physical allocation geometry.
- Composite scopes are not split into incidental operators, and overlapping
  auxiliary-stream scopes are never summed into serving attention latency.
- Missing real hash/score routing, mHC/endpoints/zeroing/CPU scopes, real native
  assignments, decode/first-tail, FULL dummy capture metadata, PIECEWISE branches,
  compiled/graph replay and physical TP/phase equivalence remain explicit gaps.
- Base family and standalone serving-training admission remain closed. No four
  metric or ≤10% accuracy claim follows from these CPU fixtures/isolated rows.

## Final regression receipt

Full unit sweep:5581 PASS/108 FAIL/72 SKIP/20 ERROR (238.32s). This is not an
all-green unit result. Historical baseline
`/data/ycfeng/tmp/w3_review_fixes_20261020/unit_base.xml` has4928 PASS/100 FAIL/
51 SKIP/19 ERROR. Among baseline-passing IDs present in the new sweep, one failed
the import-boundary allowlist: the new V4 reader explicitly reuses the existing
CPU schema, scope ownership plan and provenance format. These three pure helpers
were added to the narrow allowlist; profiler runners/native main imports remain
forbidden. The affected rerun passed686 tests, including the boundary check.
There are **zero unresolved regressions among those historical passing IDs**.

Seven additional pre-A HTTP harness failures and one pre-A collection error lack
`aiohttp`; no dependency installation or network workaround was attempted.64
absent historical passed IDs reflect earlier parent changes, including formerly
frozen DSA/GLM and H200 admission tests, not deletions by this slice. Original
full-sweep failures and targeted correction are both preserved in
`unit_baseline_comparison.json`; raw XML/log paths are recorded above. No broad
rerun is needed for a helper allowlist-only correction.

The sole strict77 invocation completed on committed code
`621b198476c677ae32464387149402c1c4952c9c`:

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/frontier-py310/bin/python tests/e2e/refactor_fidelity/measure_commit.py --sha 621b198 --output-root /data/ycfeng/tmp/glm53_dsv4/v4_graph_measured_20261021/fidelity --baseline-label /data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/baseline --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 > /data/ycfeng/tmp/v4_graph_measured_strict77.log 2>&1
```

Expected all77 cases execute and compare, with no missing evidence, differences,
failures or predictor cache filename differences. Actual **PASS/IDENTICAL**:
77/77 executed and compared;77 identical;0 mismatched, missing evidence, missing
cases, differing definitions, baseline failures or candidate-only failures;
predictor cache filenames0 baseline-only/0 candidate-only. Provenance findings0.
Manifest `source_dirty=false`, empty git_dirty_paths, clean cache and identical
driver/source/harness revision. Retained checkout was also clean after the run.

- Selected durable receipt: `strict77_receipt.json`.
- Original comparison: `/data/ycfeng/tmp/glm53_dsv4/v4_graph_measured_20261021/fidelity/comparison.json`.
- Manifest/results: `/data/ycfeng/tmp/glm53_dsv4/v4_graph_measured_20261021/fidelity/candidate_621b198/{manifest.json,results.jsonl}`.
- Retained checkout: `/data/ycfeng/Frontier/.worktrees/fidelity-candidate-621b198`.
- Review diff: `/data/ycfeng/tmp/v4_graph_measured_final.diff`.

This verifies legacy artifact/cache fidelity, not native V4 timing accuracy.
Only task records and H800 CPU regression checks change after the measured code
commit. Runtime and strict77 case definitions remain byte-identical to621b198;
the requested sole strict invocation was not repeated.

## Independent-review correction (supersedes the prior runtime freeze)

Authority: primary's REQUEST CHANGES from reviewer
`ses_ea22254bfffesOGp3CPziZRXZL`. Only the owned V4 predictor, focused tests,
qualified-input fixture and task records change. No timer/reference-median audit,
B edits, other development-worktree edits, broad unit sweep or GPU operations.

### Reproduction and correction

| Finding | Expected red / observed | Final expectation / observation |
| --- | --- | --- |
| Layout omissions/mixed median | 5 failed; BF16 cache/padded128/SWA128/maxlen8192 accepted; median0.3125 rather than0.125 | Complete exact43-field isolated contract required; valid scope remains0.125 and invalid layout excluded. |
| Contract disagrees with source | 2 failed for changed replica cache and contract/row padded128 | Reuse source attention workload's padded heads and require replica cache dtype; both reject. |
| Missing shared/MARLIN coverage | 8 failed,1 expected control passed; FileNotFoundError, ValueError or required-column KeyError escaped | Only known coverage/read errors become DeepSeekV4CoverageError with original __cause__; unrelated ValueError remains unchanged. |
| Embedding collective | 1 failed: output40x4096, all-reduce17x4096 | All-reduce40x4096, source/request17 unchanged. |

The explicit layout includes existing engine/backend/dtype/head/stream fields,
engine bounds, compression and every flattened metadata field, including null
nonapplicable values. The fixture is an independently retained snapshot of the
already accepted reduced H200 P6144/C0 scope, not an inferred contract copied from
the input under test. `attention_layout` in receipts documents the selected
isolated geometry; attention metadata256 never asserts full physical capacity.
Source consistency uses existing workload geometry, not a second sizing formula.
Native authority remains2cf0a6915ce544dc493a0990f2ea38d81601128a:
`vllm/models/deepseek_v4/nvidia/model.py:1137–1157` embeds before SP and
`vllm/model_executor/layers/vocab_parallel_embedding.py:506` all-reduces that output.

### Exact commands and affected checks

Same Python3.10.6 environment and command-prefix environment as above. Each red
command ran before its respective source correction; raw XML/log prefixes are
`/data/ycfeng/tmp/v4_review_{layout_red,layout_source_red,reader_red,embedding_red}`.

```bash
/data/ycfeng/envs/frontier-py310/bin/python -m pytest -q tests/unit/test_dsv4_graph_measured.py -k 'unqualified_isolated_layout or median_across_isolated_layouts' --junitxml=/data/ycfeng/tmp/v4_review_layout_red.xml > /data/ycfeng/tmp/v4_review_layout_red.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -m pytest -q tests/unit/test_dsv4_graph_measured.py -k 'layout_must_match_replica_cache_dtype or cannot_override_source_padded_heads' --junitxml=/data/ycfeng/tmp/v4_review_layout_source_red.xml > /data/ycfeng/tmp/v4_review_layout_source_red.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -m pytest -q tests/unit/test_dsv4_graph_measured.py -k 'missing_table or known_component_coverage or unexpected_reader' --junitxml=/data/ycfeng/tmp/v4_review_reader_red.xml > /data/ycfeng/tmp/v4_review_reader_red.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -m pytest -q tests/unit/test_dsv4_graph_measured.py -k 'layer0_embedding_collective' --junitxml=/data/ycfeng/tmp/v4_review_embedding_red.xml > /data/ycfeng/tmp/v4_review_embedding_red.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python -m pytest -q tests/unit/test_dsv4_graph_measured.py tests/unit/test_dsv4_source_execution.py tests/unit/test_dsv4_structural_execution.py tests/unit/test_dsv4_attention_workload.py tests/unit/test_dsv4_attention_profiling.py tests/unit/test_dsv4_linear_op_profiling.py tests/unit/test_dsv4_moe_profiling.py tests/unit/test_profiling_runtime_boundary.py --junitxml=/data/ycfeng/tmp/v4_review_affected.xml > /data/ycfeng/tmp/v4_review_affected.log 2>&1
/data/ycfeng/envs/frontier-py310/bin/python tests/integration/check_dsv4_measured_scopes.py --attention-root /data/ycfeng/tmp/glm53_dsv4/v4_attention_cwd02_results/P6144 --compute-root /data/ycfeng/tmp/glm53_dsv4/h200_dsv4_compute_results_20261021/9f58dbe --output task_memory/task_2026-10-21_v4_graph_measured/real_scope_receipts.json > /data/ycfeng/tmp/v4_review_real_scope.log 2>&1
```

Final affected checks: **208 PASS/8 SKIP**,216 cases, zero failures/errors.
Intermediate entire graph-measured module checks passed30/39/40 as fixes landed.
The first layout integration exposed model ratio0 versus the schema's ratio1
for uncompressed attention; matching now uses the producer's canonical ratio1.
Final real rows: **102 exact positive scopes/68 typed negatives PASS** plus two
mixed-layout controls. The latter preserve the true CSA sparse medians exactly:
CUDA_EVENT0.9165120124816896ms and KERNEL_ONLY0.904703ms despite a padded128 row
with tenfold timing. These compare consumer selection only, not timer validity.
Known errors preserve FileNotFoundError/ValueError/KeyError causes; a missing
canonical contract rejected directly has no prior exception to wrap. The original
CUDA_EVENT/KERNEL_ONLY files and raw data are unchanged. All native/compiled,
physical-TP, decode/first-tail and H800 measurement limitations above still apply.

### Exact recommendation to primary for B assertions

In parent `tests/unit/test_dsv4_structural_pd.py:333–337`, replace only the
conditional old regex at336 with the following constant. Keep both dummy values,
the autouse NIXL fixture and NotImplementedError; do not weaken the guard:

```python
with pytest.raises(NotImplementedError, match="V4 predictor PD phase/graph equivalence remains unqualified"):
    serving_context(predictor, None, 0)
```

The owned two-case regression verifies that exact behavior. This is a
recommendation, not a claim that merged A+B tests were run here. B stays unedited.

### Review clean strict77

The sole invocation for committed correction
`282cb110800b5641e9ab21d51ecf77dba601e233` completed:

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/frontier-py310/bin/python tests/e2e/refactor_fidelity/measure_commit.py --sha 282cb11 --output-root /data/ycfeng/tmp/glm53_dsv4/v4_graph_measured_review_20261021/fidelity --baseline-label /data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/baseline --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 > /data/ycfeng/tmp/v4_review_strict77.log 2>&1
```

Expected:77 executed and compared, all identical, no missing evidence,
definition/failure/cache/provenance differences, clean committed source and cache.
Actual: **PASS/IDENTICAL77/77**. Executed77, compared77, identical77; zero
mismatches, missing cases/evidence, differing definitions, unaccounted cases or
baseline/candidate failures. Predictor cache filenames:0 baseline-only,
0 candidate-only; clean population and zero findings. Manifest source_dirty=false,
empty git_dirty_paths, cache_clean_before_run=true, no case filter or stale
drops, all77 listed executed, and matching source/driver/harness revision282cb11.
Retained measured checkout is also clean after the run. No second invocation,
dirty replay, source/test edits or new metric-admission claim followed.

- Durable selected result: `strict77_receipt.json` under `review_correction`;
  the initial621b198 result is explicitly historical.
- Original comparison: `/data/ycfeng/tmp/glm53_dsv4/v4_graph_measured_review_20261021/fidelity/comparison.json`.
- Manifest/results: `/data/ycfeng/tmp/glm53_dsv4/v4_graph_measured_review_20261021/fidelity/candidate_282cb11/{manifest.json,results.jsonl}`.
- Retained checkout: `/data/ycfeng/Frontier/.worktrees/fidelity-candidate-282cb11`.
- Review source/test diff: `/data/ycfeng/tmp/v4_graph_measured_review.diff`.

Only task records are archived after correction282cb11. The review repair is
complete; primary acceptance/integration and all native/measurement limits remain
outside this CPU source correction. Inner-event reference timing validity was not
audited or approved by these median-selection checks.
