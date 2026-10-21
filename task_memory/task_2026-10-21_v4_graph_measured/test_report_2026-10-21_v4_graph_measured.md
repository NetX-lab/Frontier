# V4 graph and measured-scope CPU verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded source evidence,686 affected passes, real isolated consumer comparisons and full-unit baseline comparison; strict77 pending. |

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
CUDA-event rows produce typed coverage errors. Actual: **686 PASS/8 SKIP**.
Earlier failures were obsolete blanket-rejection expectations, changed only where
Q10 authorizes typed coverage errors. Existing test node IDs were preserved.

Expected real rows: every applicable scope of the12 accepted P6144 rows, three
shared kernels per method, and a matched MARLIN histogram per method return the
exact selected CSV median, retaining provenance and `serving_admission=false`.
Actual: **102 positive exact scope comparisons and12 typed negatives PASS**;
`real_scope_receipts.json` contains each input path, source revision, runtime
fingerprint, physical/target TP, measured value and retained metadata.

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

One clean committed strict77 is pending.
