# D20/D21 parent integration and native execution

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record parent-run suites, actual bundle output, initial fidelity audit, prioritized submissions and remaining actual-main defects. |
| 2026-10-21 | Reverify the native API-front-end correction and explicit request-owned profile identity. |
| 2026-10-21 | Record integrated-source77-case execution and separate schema/provenance disposition. |

## CPU integration checks

Parent ran the combined measured suite in `step5-d20-integration` with Python3.10.6:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/frontier/cc_backend/backends/collective-sim/python" \
  TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
  /data/ycfeng/envs/frontier-py310/bin/python -m pytest \
  tests/unit/test_step5_measured_cost.py tests/unit/test_step5_measured_training.py \
  tests/unit/test_step5_model_bundle.py tests/unit/test_step5_measured_replay.py \
  tests/unit/test_step5_measured_ffn.py tests/unit/test_step5_measured_ffn_replay.py \
  tests/unit/test_step5_measured_manager.py tests/unit/test_step5_measured_interfaces.py \
  tests/unit/test_step5_nixl_delegates.py tests/unit/test_step5_measured_public_predictor.py \
  tests/unit/test_step5_attention_input_union.py tests/unit/test_step5_measured_stage_boundaries.py \
  tests/unit/test_nixl_read_cost.py tests/unit/test_step5_measured_pd.py \
  tests/unit/test_pd_transfer_prompt_recompute.py -q -p no:cacheprovider --tb=short \
  --junitxml=/data/ycfeng/tmp/d20_parent_complete01.xml \
  > /data/ycfeng/tmp/d20_parent_complete01.log 2>&1
```

Expected zero failures/errors; observed **274 passed,0 failed/errors/skipped,227.291s**. This verifies delivered compute/PD inputs, not complete CLI or native accuracy.

The same environment executed:

```bash
/data/ycfeng/envs/frontier-py310/bin/python tests/integration/step5_measured_compute_fixture.py \
  --output-dir /data/ycfeng/tmp/d20_measured_compute_fixture_h80032_parent01 \
  --device h800 --gpus-per-role 16
```

Observed four bundles with34 artifacts each through the real public manager. Inputs are explicitly synthetic CPU fixtures. Log `/data/ycfeng/tmp/d20_parent_fixture01.log`.

Parent ran D21/native/client integration in `step5-priority-runs` with the Python3.12 environment and explicit simulator Python3.10:

```bash
env PYTHONPATH="$PWD" TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1 \
  FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python \
  /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest \
  tests/unit/test_step5_priority_runs.py tests/unit/test_step5_client_metrics.py \
  tests/unit/test_canonical_client_metrics.py tests/unit/test_step5_canonical_native.py \
  tests/unit/test_step5_native_pd.py tests/unit/test_step5_native_pd_r2_stage.py \
  tests/unit/test_e2e_gate.py -q -p no:cacheprovider --tb=short \
  --junitxml=/data/ycfeng/tmp/d21_parent_runs01.xml > /data/ycfeng/tmp/d21_parent_runs01.log 2>&1
```

Observed **204 passed,0 failed/errors/skipped,91.298s**. Subsequent staged-package correction42dda4c passed38 checks (`/data/ycfeng/tmp/d21_package_green01.log`) and actual StepMind Python dry-runs of new H20032/H80064 generations. The original local import error was `No module named tests.comparison`; the installed regular `tests` package shadowed the staged namespace. The fix supplies explicit package markers; old sealed packages are unchanged.

## Initial legacy fidelity

Both sides used `tests/e2e/refactor_fidelity/run_matrix.py run`, the existing77 cases, `--jobs 4 --clean-cache --continue-on-failure` and `/data/ycfeng/envs/frontier-py310/bin/python`. Output root `/data/ycfeng/tmp/step5_d20_fidelity_20261021`; baseline worktree `step5-d20-baseline-b65609b`, candidate `step5-d20-integration` at371a992. Original comparison is `comparison.json`; original run logs are `/data/ycfeng/tmp/step5_d20_fidelity_{baseline,candidate}.log`.

- Both sides77/77 executed; no failures, missing evidence, definition differences, cache findings or provenance findings.
- Strict comparison is **not identical**:77 cases differ in a new null `nixl_read_profile` config field and nine registered Step metadata rows. The comparator reports32→41 CSV rows including the header; those are31→40 operator records.
- Parent inspected every case: all31 existing operator records are preserved; nine additions are `step5_*`; the new config input is null. All297 other artifact pairs have identical raw bytes/hashes, and predictor-cache filenames are unchanged.
- Receipt: `d20_fidelity_schema_audit01.json`. This separate schema-aware disposition does not rewrite the original comparison. Final-source affected verification remains needed after subsequent actual-main code changes.

## Prioritized native execution

### Final integrated-source legacy run

Parent executed from `step5-preview` at `bc8ae27`:

```bash
env PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/step5-preview --label candidate --output-root /data/ycfeng/tmp/step5_d20_final_fidelity_20261021 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 --clean-cache --continue-on-failure
env PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py compare --output-root /data/ycfeng/tmp/step5_d20_final_fidelity_20261021 --baseline-label /data/ycfeng/tmp/step5_d20_fidelity_20261021/baseline --candidate-label /data/ycfeng/tmp/step5_d20_final_fidelity_20261021/candidate
```

Expected no changed legacy simulation results. Actual execution **77/77 PASS**; strict comparison **FAIL/exit1** with77 config/metadata mismatches, no failed/missing cases or cache differences, and one dirty-tree provenance finding. Logs: `/data/ycfeng/tmp/step5_d20_final_fidelity_candidate.log` and `step5_d20_final_fidelity_compare01.log`. The manifest lists six changed documentation files and pre-existing `.pi/`/`.cortexkit/`, not changed source. It remains a dirty-tree result.

Parent separately compared actual files for all77 cases: **297 non-schema pairs byte-identical**;31 prior metadata records preserved plus exactly nine `step5_*` records; normalized configs equal after removing only the new null `cluster_config.execution_time_predictor_config.nixl_read_profile`. The existing comparator's path substitutions were reused; no comparator or retained result was changed. Full per-case receipt: `d20_fidelity_schema_audit02.json`. This explains the differences without promoting the strict or provenance verdict. Final ticket29 must use the completed clean selected source; no additional unchanged-code matrix is started here.

Parent also revalidated **79/79** retained H80064/H20032 evidence hashes, **12/12** generation03 inputs/source hashes and **99/99** raw-log runtime row addresses. A scoped Python link check over the public guide and current plan/progress/summary/checkpoint/report verified **133 existing local targets**; `git diff --check` passes. The campaign checkpoint records four complete tickets, eight partial tickets, fourteen blocked tickets and four intentionally retired tickets; it does not promote CPU implementation to native or accuracy acceptance.

Each command uses the existing persistent personal RJob launcher:

```bash
bash /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification01/stage/submit.sh
bash /data/ycfeng/tmp/step5_d21_priority_20261021/h200_32_qualification02/stage/submit.sh
bash /data/ycfeng/tmp/step5_d21_priority_20261021/h800_64_qualification02/stage/submit.sh
```

The parent used `setsid`, redirected each output to the enclosing run directory's `launcher.log`, and checked personal nonterminal reservations and quota before each submission. H80064 was submitted only after P0 finished; it is basic validation, not gated formal calibration. H200 non-preemptible reservations stayed32; the separate D22 probe used one preemptible GPU.

| Track | Job | Platform result | Business scope |
| --- | --- | --- | --- |
| P0 H80032 | `exp-1021-195639-166258` | Succeeded12:10:22UTC |24/24 exact requests and both-lane/role idle PASS; actual FULL rows/G3/reference missing, basicPARTIAL |
| P1 H80064 | `exp-1021-202325-579469` | Succeeded12:38:18UTC |`CLIENT_WORKLOAD_VERIFIED`; detailed basic judgment pending |
| P2 H20032 | `exp-1021-201544-965435` | Succeeded12:27:23UTC |`CLIENT_WORKLOAD_VERIFIED`; detailed basic judgment pending |
| P2 H20064 | retained `exp-1021-125354-923219` | Historical Succeeded |Assess existing basic evidence before any repeat; future jobs wholly preemptible |

Original staged identity/config/output files remain under the exact run directories. Selected P0 judgment is `../task_2026-10-21_step5_priority_runs/p0_judgment/`; parent source/identity/status receipts are retained beside each run. D22 strict native PASS is separately documented in `../task_2026-10-21_step5_d22_timing/test_report_2026-10-21_d22_native.md`.

## Remaining work and limits

### Parent follow-up verification

- Native replay correction `6838bc1`: parent independently ran the five-file suite from `native_replay/test_report_2026-10-21_native_replay.md` in `step5-priority-runs`, using `asyncopd-py312`, `STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm`, and `FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python`. Expected no failures, errors or skips; observed **125 PASS, 75.373s**. Logs/JUnit: `/data/ycfeng/tmp/d21_native_replay_parent01.{log,xml}`. The real StepMind Python also executed qualification03 `submit_native_pd.py --dry-run`, preserving its resource/spawn arguments in `h800_32_qualification03/parent_stepmind_dryrun01.json`.
- The native profile producer now writes `has_attention_metadata=1` for its existing request-owned cells. This keeps them distinct from measured request-free idle-DP input (`0`), without changing timings or old sealed tables. Parent command from `step5-preview`: `env PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu PYTHONPATH=$PWD:/data/ycfeng/tmp/step5_profiling_20261020/extra_site TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/oneshot-opd-py312/bin/python -m pytest tests/unit/test_step5_attention_profiling.py -q --tb=short -p no:cacheprovider`. Expected complete typed-row validation and explicit identity; observed **99 PASS, 3.67s**. Log: `/data/ycfeng/tmp/d20_profile_metadata_parent01.log`.
- Before integration, parent independently ran `test_step5_measured_replay.py`, `test_step5_measured_training.py`, and `test_step5_measured_public_predictor.py` at `acc3a88` in `step5-d20-integration`, with the Python3.10 environment above: **41 PASS, 22.50s**. Log: `/data/ycfeng/tmp/d20_request_free_parent01.log`. Complete CLI execution and combined-tree checks remain pending.
- Combined-tree verification after merging `acc3a88`: parent reran the full15-file measured suite above plus `tests/unit/test_step5_structural_execution.py`, using the same Python3.10 environment and `--junitxml=/data/ycfeng/tmp/d20_merged_complete02.xml`. Expected zero failures/errors/skips; **323 PASS, 243.780s**, log `/data/ycfeng/tmp/d20_merged_complete02.log`. The obsolete unconditional measured-mode rejection assertion now verifies a deliberately absent measured CSV raises `FileNotFoundError` instead of falling back to dummy timing.
- The merge preserves both D22 reference cohorts and D20 profile precision/architecture fields, plus explicit request-owned identity. Parent ran `tests/unit/test_step5_attention_profiling.py tests/unit/test_step5_attention_timing.py` with the profiling environment above: **197 PASS, 4.25s**, log `/data/ycfeng/tmp/d20_merged_profile03.log`. An earlier command named a nonexistent `test_step5_timing_reference.py` and collected no tests (exit4); its log remains `/data/ycfeng/tmp/d20_merged_profile02.log`. No production change was needed for that command error.

The real CLI first rejected `full_decode_only` at a phase-specific mapping. Correction7e41b80 reuses the existing policy (pure decode FULL; prefill/tail NONE) and passes100 affected checks plus an initialization-only CLI regression. Full execution then reached a separate missing capability: request-free idle-DP attention coverage. The same owner is implementing its source-derived physical scopes; no regular decode substitution or zero-cost fallback is accepted. Exact CLI/error/config artifacts are in `../task_2026-10-21_step5_d20_integration/actual_main/` in the integration worktree.

Independent compute/native source reviews failed due to model limits; no independent review PASS is claimed. Parent checks and executable evidence above are distinct from those unfinished reviews. None of these results supplies admitted complete profiles, G3/G4, three clean128-request repetitions or four-metric accuracy. The30% same-hardware32→64 formal release and final four10% gates remain unchanged.
