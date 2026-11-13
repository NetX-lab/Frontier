# Selected binding consumers and source-context integration

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record actual shared/independent fit/cache, exact pair and executor/source checks. |

TaskID: D29-CORE-SOURCE-IMPLEMENTATION. Workdir: /data/ycfeng/Frontier/.worktrees/step5-position-core. Environment: frontier-py310 Python3.10.6 and asyncopd-py312 Python3.12.13; no dependency installation. Every command uses PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD, Python -B, pytest -q -p no:cacheprovider. Logs/caches/pytest temporary files remain under /data/ycfeng/tmp. Scope evidence is CPU plumbing only.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_grouped_gemm_padded_features.py tests/unit/test_step5_measured_ffn.py tests/unit/test_moe_routing_runtime.py tests/unit/test_moe_profiling_output_metadata.py tests/unit/test_moe_position_routing.py tests/unit/test_moe_routing_conservation.py tests/unit/test_step5_fixed_probe.py tests/unit/test_shared_prediction_model_manager_mixed_layer_moe.py > /data/ycfeng/tmp/d29-core-selected-consumers-final-02.log 2>&1
```

Observed111 PASS/1 SKIP,25.76s. Expected and observed: old helper traps are not reached by selected preflight/loader/trainers; actual ROUTED prepare TP1/EP=ANY and FP8 identity survive fit; source physical8/live0 still has duration1/count3; actual shared and independent sklearn fitting and cache reload preserve those exact pairs. Changing bound native context produces a different actual model hash. BF16 selector omits prepare; source gating contexts remain distinct. Missing duration/count/typed/frame/physical/context coverage and missing/duplicate cell inventory fail. Runtime dispatch rejects in-bound missing exact keys and nonselected shuffle before parent prediction. Old source/conservation/cache tests pass.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_step5_moe_profiling.py -k 'position_adapter or real_device_timer or masked_expert or oracle_callable or lazy_norm or actual_source_context' > /data/ycfeng/tmp/d29-core-selected-source-final-01.log 2>&1
```

Observed19 PASS/11 deselected,9.86s. Actual singleton annotation/parser/restore/event preservation and callable receipt checks remain passing. Explicit source gating context does not mutate wrapper context. Native routing transport/source fixtures do not prove native CUDA correctness.

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_step5_moe_profiling.py > /data/ycfeng/tmp/d29-core-selected-legacy-source-py310.log 2>&1
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest -q -p no:cacheprovider tests/unit/test_moe_share_expert_operator_families.py tests/unit/test_shared_prediction_model_manager_mixed_layer_moe.py tests/unit/test_step5_moe_pricing.py tests/unit/test_moe_finite_prediction_lookup.py > /data/ycfeng/tmp/d29-core-selected-peer-legacy-02.log 2>&1
```

Observed12 PASS/18 SKIP in minimal source environment; peer/legacy union40 PASS/5 FAIL. Five unchanged tests expect the old four-entry raw inventory or mock only profiling_ops while7b uses raw operator inventory in families/quantization_manager. Those are peer registry/test prerequisites, not suppressed here. A sixth failure introduced by always forwarding operator_facts=None to an old fake validator was diagnosed and corrected to retain the legacy invocation shape; rerun eliminated it. Full installed-vLLM source suite previously had3 FAIL/26 PASS/1 SKIP because the existing environment lacks fused_moe.activation and router.gate_linear; raw evidence retained at /data/ycfeng/tmp/d29-core-selected-source-01.log and /data/ycfeng/tmp/d29-core-source-baseline-module-evidence.log. Imports already existed at02f061d. Root's actual image/receipt delivery now permits independent source checks, not relabeling these environment failures.

Checks: git diff --check PASS. Python AST parsing, exact44-owned path check and every retained path ceiling PASS. Baseline1aee0af code/test/config diff only:37/44 changed paths,+2056/-133=2189/4404 gross;2215 remain. Task records and peer imports are excluded. tests/unit/test_moe_profiling_output_metadata.py reaches its85-line allowance; no further additions there without root allocation. Exact accounting in summary.md.

Limitations: production per-group native row serialization/schema/layout and finite-model fixture integration remain pending. Authentic source inputs remove the root image blocker; they do not authorize Docker/platform/GPU runs or qualify native timing/NONE/PW/FULL/G3. No synthetic coefficients, image hashes or missing-scope zeros were introduced.
