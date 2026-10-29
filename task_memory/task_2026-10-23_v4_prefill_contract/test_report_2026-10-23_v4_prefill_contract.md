# Isolated accepted prefill contract verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded final scoped source, exact CPU commands, real-table medians and preserved legacy behavior. |

Task ID: `W128-V4-PREFILL-CONTRACT-DQ10`. Worktree `/data/ycfeng/Frontier/.worktrees/v4-prefill-contract`, base `b62c155abd981a71e12e62aa0c0fe3990741d2f8`. CPU-only. No GPU/platform/SSH/cloud/Docker/network/delegation, producer edit, main-worktree source edit or strict77 run.

## Result

| Check | Expected passing condition | Observed | Verdict |
| --- | --- | --- | --- |
| Final affected unit module | All existing/new tests pass; no skips/errors | 144 passed in69.46s,0fail/0skip | PASS |
| Final prefill subset | Missing/incorrect accepted provenance/layout, variant/role, graph/PD and historical unbalanced q1 reject with DeepSeekV4CoverageError | 27passed in5.30s,117deselected;24 new typed-negative cases | PASS |
| Explicit accepted-input checker | Independently build contract; every real accepted scope returns exact CSV median, no serving admission | 94/94exact: CUDA_EVENT50,KERNEL_ONLY44;12rows across both tables | PASS |
| Legacy checker | H200/TP8/DP4/EP32 absent-selector defaults, counts and errors unchanged | 102positive exact comparisons,72typed negatives,2mixed-layout controls | PASS |
| Accepted original integrity | CSV/raw/summary and frozen package manifest/model/plan/job bytes match independently supplied pins | All selected pins match | PASS |
| Source bounds | <=3 maintained files/200added+deleted;core<=150 | 3files/197total;core50added+5deleted=55 | PASS |
| Source whitespace | git diff --check clean | no findings | PASS |

All94 predicted reader durations equal their actual CSV medians: maximum absolute error0ms and relative error0. Every actual value is positive, so relative error is defined. Individual measured values and comparisons are in `comparisons.json`. No duration is summed, inferred, scaled or substituted. The expected50 count was event-only; kernel-only has44 because the producer intentionally excludes enclosing attention-layer timings.

## Independent contract and provenance

- Core `frontier/execution_time_predictor/deepseek_v4.py:579` builds per-layer contracts from non-timing engine/step records; excludes call records before `index_records`. Reuses existing `_model_columns` and `_variant_metadata`; unions observed non-timing metadata keys and fills variant-inapplicable fields with null. Planner/model/engine fields and native layer variant/planning role are not read from candidate CSV rows.
- Checker `tests/integration/check_dsv4_measured_scopes.py:33` receives explicit acceptance and frozen-package paths. It verifies package manifest against primary acceptance; verifies model, plan and job against manifest; parses the frozen producer command using the existing parser/plan builder and binds model input to the verified supplied package. Resulting plan equals the frozen plan after JSON serialization.
- Accepted aggregate summary supplies source/runtime/model identity independently; primary acceptance pins its bytes. Accepted CSV hashes come from primary acceptance, not computed as new expectations from candidate timing rows. The actual file bytes are compared to those independently supplied hashes by the core reader. Raw and summary pins are verified before building layouts.
- Accepted source `39b2bf8088dff40dc8b695b6927d6e6322c337e9`, runtime fingerprint `80b7753fb935a8e0`, model hash `b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818`; balanced policy `6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea`.
- CSV hashes: event `52571225ce8ddd0d2e87615f80f2aaadd980cbd8b2c10d460a1c75dff7c2af5f`; kernel `fe22f22cf4c21bcad93c8ccd463115a607ace65193d900925b8eaa2e6cf69b43`.
- Wrong source/runtime/model/hash, missing hash, missing/wrong native metadata, variant and planning-role mutations use original immutable accepted tables. Historical warmed unbalanced q1 uses its own valid legacy decode contract and independent old accepted identity, then adds explicit balanced routing selection; absence of balanced columns causes typed rejection. It is not forced to fail in legacy non-balanced mode.

## Environment and exact commands

Interpreter `/data/ycfeng/envs/frontier-py310/bin/python`: Python3.10.6,GCC11.3.0; pandas2.3.3; CONDA_PREFIX unset. Commands run from the worktree root. Unit and checker commands use `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_graph_measured.py -q -p no:cacheprovider --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_prefill_contract_final_units.xml > /data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_prefill_contract_final_units.log 2>&1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_graph_measured.py -k 'prefill' -q -p no:cacheprovider --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_prefill_contract_final_prefill.xml > /data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_prefill_contract_final_prefill.log 2>&1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/frontier-py310/bin/python tests/integration/check_dsv4_measured_scopes.py --accepted-prefill /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json --prefill-package /data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill_balanced --output task_memory/task_2026-10-23_v4_prefill_contract/accepted_comparison.json > /data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_prefill_contract_accepted.log 2>&1
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/frontier-py310/bin/python tests/integration/check_dsv4_measured_scopes.py --attention-root /data/ycfeng/tmp/glm53_dsv4/v4_attention_cwd02_results/P6144 --compute-root /data/ycfeng/tmp/glm53_dsv4/h200_dsv4_compute_results_20261021/9f58dbe --output task_memory/task_2026-10-23_v4_prefill_contract/legacy_comparison.json > /data/ycfeng/tmp/opencode-v2-tmp/opencode/v4_prefill_contract_legacy.log 2>&1
git diff --numstat
git diff --check
git diff -- frontier/execution_time_predictor/deepseek_v4.py tests/integration/check_dsv4_measured_scopes.py tests/unit/test_dsv4_graph_measured.py > task_memory/task_2026-10-23_v4_prefill_contract/source.diff
```

Both checker logs include the existing experimental V4 architecture warning; it correctly states runtime limitations and is not a test failure. Initial inspection used nonexistent raw key `record_type`; corrected to actual `kind`. Initial checker plan comparison failed because dataclass payload contained tuples while decoded JSON contains arrays; normalized the comparison through JSON and reran successfully. Neither failure altered raw data, model values or prices. All final command exits0.

## Exact changed-source hashes and regression decision

| Artifact | SHA-256 |
| --- | --- |
| frontier/execution_time_predictor/deepseek_v4.py | 45d30210ffc34a1538e86eb3305a8cd5ef9960122c4b3c5d4c6b26df12f4429c |
| tests/integration/check_dsv4_measured_scopes.py | 8921d66ab53d3758fc9f3f3ea7f9f2de6e31bdf8ede27834b2e727cb63158855 |
| tests/unit/test_dsv4_graph_measured.py | 561ed21d7731a697cdfa90683a59e042e1e1f2c46a2b937cc5a0a2dee9b13ad6 |
| source.diff | aa3450a980a046184cb2e4410cae9552b6646d22d19b2c98b977e6e3924e3fbf |

Only isolated reader admission and a new independent builder changed. `grep` found no production caller of either function: only their definitions under `frontier/`; the checker and unit tests drive them. Serving/synthetic execution, scheduler, allocator, training/cache and shared price paths are unchanged. The complete affected unit module verifies decode provenance, first-tail/graph refusal, typed component errors and preservation of unrelated exceptions; the actual legacy checker verifies default prefill/shared/MoE medians. Therefore strict77 would not exercise this seam and was not rerun. Root decides final integration regression based on any further source change.

## Limits

This proves independent-contract consumption of accepted isolated physicalTP1/eager/six-layer dummy-model observations only. TP8-local dimensions are not physicalTP8 execution. Planner/effective128 is pinned evidence for this case, not a newly measured capacity model. Full non-dummy constructor/family, serving, PD, graph/compiled, first-tail, balanced decode, independent expert histogram, CPU-overhead and E2E qualification remain open. No numerical accuracy or full predictor-bundle acceptance is claimed; root review/acceptance remains pending.
