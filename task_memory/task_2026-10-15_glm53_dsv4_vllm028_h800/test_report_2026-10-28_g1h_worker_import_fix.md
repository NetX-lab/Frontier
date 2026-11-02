# Test report: G1h `package_01` failure in the worker image, the import fix and `package_02`

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Added the e0 raw-record copy (check 17) and the CPU gap reduction of the G1h-02 records, which gives the four V4 gap values in place of a separate G4h job (check 18). |
| 2026-10-28 | Added the G1h-02 result: succeeded, bridge PASS, 989 of 989 cases, 0.85 GPU-hours; the zeroing segment-count observation. |
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

Pi `g1h-submit-1028i` (prompt `/data/ycfeng/tmp/ds4_runs/g1h-submit-1028i/prompt.md`, derived from the G1h-01 prompt with the new package, tag, output directory and an early `run.log` check): RJob `exp-1029-020008-235508` submitted at 18:00:08 UTC, phase `Running` with every identity field matching, worker on NVIDIA H200 with vLLM 0.28.0. The result follows below.

## G1h-02 result (observed)

Pi report `/data/ycfeng/tmp/ds4_runs/g1h-submit-1028i/output.md`; local copy `/data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/download/` (1,006 files, every size equal to the cloud listing; `raw/dsv4_raw_e0.jsonl`, 342,060,399 bytes, was copied separately by Pi `g1h-copy-e0-1028j`, check 17). Cloud `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/g1h_02`.

| # | Check | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| 10 | RJob terminal phase | Succeeded | `exp-1029-020008-235508` Succeeded at 18:51:00 UTC; creator `i-fengyicheng`, non-preemptible, `step-main-default`, image and mount as submitted; the launcher stopped the job itself after `FINAL_STATUS succeeded` | PASS |
| 11 | Worker runtime | H200, vLLM 0.28.0, no import error | NVIDIA H200 (driver 570.124.06), vLLM 0.28.0, torch 2.13.0+cu130; no `ModuleNotFoundError`; the only traceback per engine is the non-fatal chat-template warm-up warning under `skip_tokenizer_init=True` | PASS |
| 12 | Bridge verdict (`bridge/bridge_report.json`) | PASS, no reasons | `PASS`, `reasons` empty, `artifact_errors` empty on all three runs; `logs/bridge_report.log`: `PASS: 3 runs, FULL=342, PIECEWISE=645` | PASS |
| 13 | Measured cases per engine against the plan (686 / 296 / 7) | equal case sets | e0 686 (FULL 342, PIECEWISE 342, NONE 2), e1 296 PIECEWISE, e2 7 PIECEWISE; case-id sets equal to the manifest (0 missing, 0 extra); 989 traces | PASS |
| 14 | Capture | as `package_01` planned | each engine: FULL_AND_PIECEWISE, 35 capture sizes 1-256, 19 graph and 420 breakable-segment captures, 25,985 nodes, 385 eager breaks, all owned by `dsv4_attention_layer` | PASS |
| 15 | KV-cache groups of e1 and e2 (`raw/dsv4_raw_e{1,2}.jsonl` engine record) | the 11-layer B4 prediction [14, 6, 5, 10, 4] | [14, 6, 5, 10, 4] on both (chunk budgets 8,192 and 24,576) | PASS |
| 16 | GPU time | within the G1h-G4h cap | job wall 3,055 s = 0.85 GPU-hours (phase 2,991 s; engine runs 2,711 s); G1h-G4h total about 0.92 of 7 | PASS |
| 17 | e0 raw-record copy (Pi `g1h-copy-e0-1028j`, `output.md`): destination absent before, `ssh cat` to `.part`, size and SHA-256 compared with the cloud file, then renamed | equal size and SHA-256 | 342,060,399 bytes on both sides; SHA-256 `bdaed9b8544859a9c7991e00694464cbb5bbfc4584a7a72d8e39ea62c16404a6` on both; 511,146 lines; cloud access read-only | PASS |
| 18 | Gap reduction on CPU: `python -m frontier.profiling.kernel_gap.dsv4_replay_gaps --raw_files raw/dsv4_raw_e{0,1,2}.jsonl --output_dir /data/ycfeng/tmp/g1h_fix/gaps_g1h02/out --device h200 --model DeepSeek-V4 --diagnostics_path /data/ycfeng/tmp/g1h_fix/gaps_g1h02/diagnostics.json` (HEAD `4f54d928`, `ulimit -v 31457280`) | exit 0; four finite non-negative values from host-ahead samples only; every hold covers its step | exit 0; µs (host-ahead samples): `cuda_graph` 0.4977 (169,232), `cuda_graph_piecewise` 0.6250 (289,730), `eager` 0.5446 (30,243), `eager_preparation` 1.2209 (91,170); 0 host-bound exclusions; minimum hold slack 189,344 µs (PIECEWISE), 195,042 µs (FULL), 150,935 µs (NONE); 0 launches after a hold ended | PASS |

Observation for later stages: the 11-layer replay engine zeroes KV blocks with 30 segments (`_zero_kv_blocks_kernel` grid (2520, 30, 10) in `traces/g1-e0-0000-full.json`), while native runs use 139 segments (grid [2520, 139, 10]). The replay's block-table writes, block-table gathers and slot mappings use the native 5 groups. B13r measures the zeroing on an engine with the native segment table.

Gap values by step mode (from check 18; observation): the `eager` value pools PIECEWISE (29,780 samples, mean 0.381 µs, median 1.184 µs) and NONE (463 samples, mean 11.06 µs, median 2.27 µs, p90 37.75 µs, maximum 147.2 µs). The NONE samples come from the two eager prefill and dummy cases; their large gaps are not yet explained (a cross-stream wait that the stream-order rule does not see is one candidate). The stage 3 composition check against the native prefill steps decides whether the pooled value prices prefill correctly; no change is made now. `eager_preparation` is the same in FULL (1.192 µs) and PIECEWISE (1.237 µs) steps.

## Limits

- The probe blocks only `fasteners` and `sklearn`; other packages missing from the image would surface only on the worker. The early `run.log` check in the Pi prompt covers that.
- `package_01` and its cloud output stay as evidence of the failure.
