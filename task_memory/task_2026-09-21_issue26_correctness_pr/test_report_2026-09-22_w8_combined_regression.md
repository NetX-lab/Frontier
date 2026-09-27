# Step 8 — Combined regression on the integrated branch

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-22 | Initial record of the Step 8 combined regression (plan.md §14.1). |

## 1. Scope

This report covers `plan.md` §14.1: running the selected suites together on the
integrated branch rather than package by package, exercising the cold
(first-load) predictor-cache path deliberately instead of inheriting a warm
cache, and confirming that the supported co-location, sequential PDD, and
sequential PD-AF paths still run.

- Branch: `fix/issue26-correctness-pr`
- Revision under test: `d881357`
- PR base: `refactor/oversized-module-split` @ `6ef0a3c` (32 commits ahead)
- Python: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10, no `torch`)
- Environment: `PYTHONPATH=<worktree>`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`

`origin/main` was re-fetched before the run and was still `1f694f7`, which is an
ancestor of `HEAD`. No integration merge was needed and none was performed.

## 2. Unit suite

```bash
python -m pytest tests/unit -q --continue-on-collection-errors
```

`--continue-on-collection-errors` matches the method used for the recorded
baseline: 11 modules import `torch` or `matplotlib`, which the simulator
environment does not carry, so collection of those modules fails on both sides.

Result on this branch:

```
84 failed, 3782 passed, 49 skipped, 11 errors in 88.59s
```

The gate is the FAILED **set**, not the counts. This branch's list was compared
against the recorded `origin/main` @ `1f694f7` baseline list (`base_failed.txt`,
84 entries):

```bash
diff <(sort base_failed.txt) <(sort now_failed.txt)   # empty
```

Empty in both directions: no test that passes on the base fails here, and no
baseline failure was silently repaired. The baseline's passed/skipped counts
were not separately recorded, so this report does not compare them; the 3782
passed here includes this branch's own new tests.

## 3. Integration suite

```bash
python -m pytest tests/integration -q --continue-on-collection-errors
```

| Metric | Baseline | This branch |
| --- | --- | --- |
| passed | 15 | 15 |
| skipped | 21 | 22 |
| errors | 5 | 5 |

The one added skip is `tests/integration/test_moe_fused_expert_numerical_parity.py`,
this branch's W6 module, which skips without a GPU and a matching vLLM build. Its
passing result on real hardware is recorded in
`test_report_2026-09-22_w6_fused_expert_arithmetic.md`.

The 5 errors are all `tests/integration/test_pdaf_reference_lifecycle_observer.py`
and all report the same cause:

```
FileNotFoundError: [Errno 2] No such file or directory:
PosixPath('/data/ycfeng/stepfun-performance-optimization/Frontier/worktrees/ref-afd-readonly')
```

That pinned read-only PD-AF Reference checkout is not present on this host. The
errors are identical on the base and are environmental, not code defects.

## 4. Architecture examples

All 16 release-supported architecture example scripts were run end to end with
task-owned metrics directories:

| Family | Scripts | Result |
| --- | --- | --- |
| co-location offline | `dense_model_basic`, `moe_model_basic`, `thinking_mode_basic`, `moe_spec_dec`, `moe_prefix_caching` | 5 passed |
| co-location online | `dense_model_basic_online`, `moe_model_basic_online` | 2 passed |
| PDD offline | `dense_model_basic`, `moe_model_basic` | 2 passed |
| PDD online | `dense_model_basic_online`, `moe_model_basic_online` | 2 passed |
| PD-AF offline | `dense_model_basic`, `moe_model_basic`, `moe_model_ep`, `moe_cuda_graph` | 4 passed |
| PD-AF online | `moe_cuda_graph_online` | 1 passed |

**16 passed, 0 failed.** Every run wrote `request_metrics.csv` and
`system_metrics.json` under its run id.

## 5. Pipeline-parallel and dense/MoE coverage

§14.1 asks for pipeline and heterogeneous/dense-layer cases where the changed
shared code applies, even though the new DP placement strategy itself stays
PP1-only. The changed cluster-scheduling, stage-dispatch, and metrics code runs
on every pipeline stage, so `PP=2` exercises it on the multi-stage path.

Driver: `w8_pp_and_cache.sh` (scratchpad), logs under `/data/ycfeng/tmp/w8_pp_cache/`.

| Case | Configuration | Result |
| --- | --- | --- |
| `pp2_dense_tp2` | co-location offline dense, `TP=2, PP=2` | PASS (2.6 s) |
| `pp2_moe_tp2ep2` | co-location offline MoE, `Attn_TP=4, MoE_TP=2, MoE_EP=2, PP=2` | PASS (7.1 s) |
| `pp2_pdd_dense` | sequential PDD offline dense, `PREFILL_PP=2, DECODE_PP=2` | PASS (2.9 s) |
| `pp2_dense_online` | co-location online dense, `TP=2, PP=2` | PASS (3.5 s) |

Each script echoes its resolved topology, and the logs confirm the intended
values (`Parallelism: TP=2, PP=2`;
`Parallelism: Attn_TP=4, MoE_TP=2, MoE_EP=2, PP=2`;
`Prefill parallelism: ... PP=2` / `Decode parallelism: ... PP=2`).

Dense-layer coverage comes from two directions: the dense examples above, and
the MoE examples, whose shared-expert work uses the separate
`dense_mlp_hidden_dim` width through the ordinary linear-op path while routed
work uses `routed_mlp_hidden_dim`. Frontier has no "first `k` dense layers then
MoE" model field, so there is no third heterogeneous shape to exercise.

## 6. Trained-predictor cache: cold then warm

The two checked-in CSV smokes (`examples/profiling/smoke_simulator_*_csv.sh`)
disable dummy mode, so they run the trained-predictor path. Against the
repository `cache/`, which is already populated, they only ever measure the
cache-hit path. To exercise first-load behavior without moving or deleting the
repository cache, the same simulation was run twice against an empty scratch
cache directory via `--metrics_config_cache_dir`.

| Run | Cache entries before | Wall time | Cache entries after |
| --- | --- | --- | --- |
| `csv_dense_cold` | 0 | 26.9 s | 63 |
| `csv_dense_warm` | 63 | 2.1 s | 63 |

The cold run trained and persisted 63 predictor artifacts; the warm run loaded
them back, wrote nothing new, and finished in 8% of the time. Their
`request_metrics.csv` outputs are byte-identical:

```bash
diff csv_dense_cold/request_metrics.csv csv_dense_warm/request_metrics.csv   # empty
```

so the persisted-cache path reproduces the freshly trained path exactly
(`request_e2e_time = 16.77818517187422 ms`, `ttft = 8.430025150867172 ms` in
both).

`csv_moe_repo_cache` then ran the MoE smoke against the repository cache and
passed (2.5 s). The repository `cache/` was neither moved nor deleted at any
point; it holds 0 tracked files and is a generated artifact.

## 7. Working-tree hygiene

After the run, `git status --porcelain` is empty. The example runs were pointed
at scratch metrics directories, and the one leftover `outputs/examples/` tree
from an earlier iteration was removed after confirming it contained 0 tracked
files; the 110 tracked files under `outputs/` are all still present on disk.

## 8. Pre-existing defect found and deliberately not repaired

`AGENTS.md` §Tests lists two entry points that do not exist:

```
bash tests/debug/e2e-level/monolith_mode/scripts/test_dense_tp2_pp2_dummy.sh
bash tests/debug/e2e-level/monolith_mode/scripts/test_moe_tp2_ep2_pp2_dummy.sh
```

`tests/debug/` is absent from this branch **and from `origin/main`**, as are the
`comm_backend_tests/` directory the same section names. The published `tests/`
tree contains `analysis/`, `comparison/`, `e2e/`, `fixtures/`, `integration/`,
`performance/`, and `unit/`.

The same missing tree is the cause of 10 of the 84 baseline unit failures, in
`tests/unit/test_colocation_release_review_contracts.py`, which resolves paths
under `tests/debug/e2e-level/monolith_mode/scripts/`. A further reference
survives in a docstring at
`frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:16`.

This is one pre-existing defect class inherited from the release scrub that
removed `tests/debug/`, and it is unrelated to Issue 26. Repairing it properly
means either restoring the tree or retargeting the contract test, which is a
decision about the published test surface, not a documentation tweak. It is
recorded in `future.md` and left untouched here. The PP2 coverage §14.1 asked
for was obtained through the existing example scripts instead, as §5 shows.

## 9. Limits of this validation

- CPU only. No native profiling suite and no vLLM serving or TTFT comparison was
  run; §14.1 explicitly excludes both.
- The PD-AF Reference-checkout integration tests could not run on this host.
- The W6 GPU parity module is skipped here; its hardware result lives in its own
  report.
- The example runs use dummy execution time except for the CSV smokes, so they
  validate structure, lifecycle, and conservation rather than latency accuracy.
