# Test report: G1h `package_01` failure in the worker image, the import fix and `package_02`

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: G1h-01 failure, root cause, fix commit `a689ed05`, package plan `c101e34e`, `package_02` build and checks, G1h-02 submission. |

## G1h-01 failure (observed)

- Job `exp-1029-010007-750668` (1 H200, `step_main`, non-preemptible), package `g1h/package_01` (manifest `7cdc51ae…ae11`, 546 files, source `32d3d08b`), Pi `g1h-submit-1028f`, local copy `/data/ycfeng/tmp/glm53_dsv4/h200/g1h_01/`.
- All three engine runs (e0, e1, e2) exited 1 after about 21.6 s with `ModuleNotFoundError: No module named 'fasteners'`; 0 of the planned 686 / 296 / 7 cases were measured. GPU time about 0.03 GPU-hours.

## Root cause

- `frontier/profiling/kernel_gap/dsv4_replay_gaps.py:35` imports `frontier.execution_time_predictor.dsv4_composition`.
- The old `frontier/execution_time_predictor/__init__.py` imported the predictor registry and the shared prediction-model manager, which load `sklearn` and `fasteners`.
- The worker image `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e853…6635` has neither package.
- B2 had accepted the profiling-to-predictor import as a side effect with no runtime cost; that acceptance did not check the worker image and is withdrawn by this fix.

## Fix (commit `a689ed05`)

| Change | Reason |
| --- | --- |
| `frontier/execution_time_predictor/__init__.py` holds only a docstring | A profiler can load one predictor module without the sklearn stack. |
| The four scheduler bases import `BaseExecutionTimePredictor` from `base_execution_time_predictor`; `simulator.py` and `tests/unit/test_gdn_hybrid_e2e_increment14ab.py` import `ExecutionTimePredictorRegistry` from `execution_time_predictor_registry` | They used the removed package re-exports. |
| `frontier/operators/families.py` and `frontier/operators/sequence_parallel_moe.py` import their `frontier.config` names inside the payload functions | The old `__init__` hid an import cycle that predates this change: `frontier.config` → `quantization_manager` → `model_config` → `model_architectures` → profile `__post_init__` → `frontier.operators.families` → top-level `frontier.config` imports. |
| `tests/unit/test_profiling_runtime_boundary.py::test_profiler_entry_points_import_without_simulator_only_packages` | Imports every profiler `main` / `*_main` and the two worker entry modules with `fasteners` and `sklearn` blocked. |
| `tests/unit/test_moe_rank_plan.py::test_builder_checks_all_four_plans_without_building` | Compares the destination before and after the plan check; the submitted G3h package now exists. |

## Checks

Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `ulimit -v 62914560` (suites) or `31457280` (other checks). Scratch `/data/ycfeng/tmp/g1h_fix/`.

| # | Check | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 1 | Full unit suite on the fixed tree (`unit_full.log`) | failures classified | 131 failed, 6,931 passed, 135 skipped, 101 errors | input to check 2 |
| 2 | The 36 files holding a failing id (`failed_files.txt`), run by path on the fixed tree and on a `895f61af` snapshot (`rerun_files.sh`; `files_current.log`, `files_base.log`) | no new failure caused by the fix | the only new ids are 61 setup errors in `test_native_groundtruth_check.py` (`TarFile.extractall()` has no `filter` argument on Python 3.10.6; in the snapshot without Git the file fails at collection instead) and the state-bound `test_moe_rank_plan.py` check (fixed in the same commit); one id disappears (the snapshot's collection error) | PASS |
| 3 | Fresh import of every module under `frontier/`, fixed tree against the snapshot (`fresh_import_sweep.sh`; `sweep_current.txt`, `sweep_baseline.txt`) | no module that loaded before fails | none; four modules that failed before now load | PASS |
| 4 | `bash examples/architecture/run_dummy_smoke_matrix.sh` (`smoke_current.log`) | all 12 cases complete | exit 0, 12/12 | PASS |
| 5 | `pytest tests/unit/test_profiling_runtime_boundary.py tests/unit/test_moe_rank_plan.py tests/unit/test_v4_attention_coverage.py -q` on HEAD `c101e34e` | all pass | 155 passed | PASS |

## `package_02` (plan commit `c101e34e`)

`g1h/package_plan.json`: destination `package_02`, job tag `v4-h200-g1h-attention-20261028-02`, cloud output `.../h200_v4_attention/g1h_02`, `source_revision` `a689ed05f462eb269b9dffe640da151e6614a5ef`; timeout 10,800 s unchanged.

Build: the first attempt ran from the `g1h` directory and failed at the plan step because the model config path `data/config/models/...` resolves against the working directory. The lead removed that partial `package_02` (created by the same command seconds earlier, 542 copied files, no manifest) and rebuilt from the repository root:

```sh
cd /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028
(ulimit -v 31457280; PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp timeout 1800 \
  /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g1h/prepare_package.py)
```

| # | Check | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 6 | Build exit and manifest | exit 0 | exit 0; `manifest.json` SHA-256 `7f9fecbab75cbbd0b3264db29f55ccc9fdc1227fd0e51878885f830dbc632dd0`, 546 files | PASS |
| 7 | `compare_packages.py package_01/manifest.json package_02/manifest.json` | same plans; only the fixed sources and the job identity differ | same 546 paths; per-run planned steps equal (e0 6,098, e1 3,016, e2 20 estimated engine steps; same case lists); changed files: the 7 fixed sources, `bridge_config.json`, `overlay.json` and `jobs/G1H_ATTENTION.json` (package path, tag, output) | PASS |
| 8 | `blocked_import_probe.py fasteners,sklearn <9 profiler entry points> dsv4_graph_replay_runner vllm_collective_benchmark dsv4_replay_gaps graph_replay_trace` with `PYTHONPATH=<package_02>` (`probe_package_02.txt`) | every module imports | 13/13 OK; modules resolve from `package_02` | PASS |
| 9 | The two package scripts `dsv4_graph_bridge_check.py` and `profile_dsv4_prefill.py`, loaded with the same packages blocked | both load | both OK | PASS |

## G1h-02 submission

Pi `g1h-submit-1028i` (prompt `/data/ycfeng/tmp/ds4_runs/g1h-submit-1028i/prompt.md`, derived from the G1h-01 prompt with the new package, tag, output directory and an early `run.log` check): RJob `exp-1029-020008-235508` submitted at 18:00:08 UTC, phase `Running` with every identity field matching, worker on NVIDIA H200 with vLLM 0.28.0. Results are recorded here when the job ends.

## Limits

- The probe blocks only `fasteners` and `sklearn`; other packages missing from the image would surface only on the worker. The early `run.log` check in the Pi prompt covers that.
- `package_01` and its cloud output stay as evidence of the failure.
