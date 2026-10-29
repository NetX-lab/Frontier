# Test report: B22, the G1h writer-count amendment and the supplement package

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created: B22 (Codex `7c19fa56`) review, the lead changes, the merge, the census closure checks, the supplement package build and its submission. |

## Scope

Assignment: [h3o_operator_costs/codex_assignment_B22_g1h_writer_supplement.md](h3o_operator_costs/codex_assignment_B22_g1h_writer_supplement.md), decision 37 (D-G1H-WRITER-EXTREMES) and decision 39 (D-G1H-SUPPLEMENT-IDENTITY) in [requirements.md](requirements.md). Codex handoff: [codex-implement-7c19fa56-…/implementation.md](h3o_operator_costs/codex-implement-7c19fa56-228f-478a-b7a1-4216223394fc/implementation.md); Codex scratch `/data/ycfeng/tmp/codex-implement/7c19fa56-228f-478a-b7a1-4216223394fc/b22/`. Working tree `.worktrees/b22-g1h-writers`, branch commit `60dfafe7`, merged into `feat/glm53-dsv4-vllm028` as `59838675`. No file under `frontier/` outside `frontier/profiling/attention/` changed.

## Delivered

| Area | File | Behavior |
| --- | --- | --- |
| Grid | `frontier/profiling/attention/v4_attention_windows.py` | Mixes (0, 0), (k4, 0), (k128, k128), (own, 0), (own, own), each raised to the first-tail floor, duplicates removed: 1,473 windows (1,041 before; added W8 72, W16 216, W24 144). `repeat_windows()`: 18 windows (3 categories × W 1, 2, 4, 8, 16, 24, the largest own of the class, high profile, (k4, 0) floored) |
| Cases | `frontier/profiling/attention/v4_attention_cases.py` | `build_attention_cases(limits, *, windows=None, case_prefix="g1")`; a window subset builds only the engines it needs |
| Coverage | `tests/analysis/v4_attention_coverage.py` | Several package plans (duplicate case ids rejected); `--predict-plan` marks windows as predicted, apart from measured |
| Replay closure | `tests/analysis/v4_attention_replay_coverage.py` (new) | Projects the supplement case features with the measured scope inventory, computes the group ranges through the trainer, and lists every census request outside them |
| Package difference | `tests/analysis/v4_attention_package_difference.py` (new) | Signed relative median differences of the 18 repeated windows between the two packages; undefined at a zero denominator |
| Promotion | `h3o_operator_costs/h200_tables/promote_tables.py` | Optional `--supplement-raw-dir` and `--supplement-plan`: repeat cases excluded by window geometry; attention rows are the union; V4 kernel gaps from G1h-02 only; the manifest records the supplement revision and the excluded repeat cases |
| Supplement | `h3o_operator_costs/g1h_supplement/` | `build_cases.py`, `prepare_package.py`, `pilot_cases.json` (450 cases), `coverage_grid.json`, `gpu_time_estimate.json`, `package_plan.json` (source revision `a689ed05`), `placement_examples.json`, `README.md` |
| Tests | `tests/unit/test_v4_attention_coverage.py` (counts), `test_v4_attention_supplement.py`, `test_v4_attention_package_difference.py`, `test_v4_attention_promotion.py` (new) | Amendment counts, repeats, placements, closure of the probe-c coordinate, difference arithmetic, repeat exclusion |

## Lead review

- **Core refactor reverted.** Codex moved the group-range computation of `dsv4_predictor.py` into a shared helper for the closure tool. The trainer already returns the fitted models with their ranges, so the tool now calls `train_dsv4_attention_models(rows, lambda **_: SimpleNamespace())` and reads `_frontier_dsv4_workload_group_ranges`; `dsv4_predictor.py` is unchanged and the helper's test is removed.
- **Loader blocker resolved by identity, not by a new rule.** Codex reported that the merged table fails `validate_measured_identity` (`frontier/execution_time_predictor/deepseek_v4.py:683-706`) because its placeholder supplement revision gives a second runtime fingerprint. Decision 39 builds the supplement from `a689ed05`, the G1h-02 revision, so the union carries one fingerprint. The supplement README's provenance section was corrected to say this.
- **Fixture limit.** `test_package_plan_cli_rebuilds_pilot_and_summary` needs the ignored `p1h_bridge/package_02` fixture, which exists only in the main working tree; it was deselected in the B22 tree and passes on main.

## Checks

Lead checks use `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6), `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, BLAS/OMP/MKL threads 1, `ulimit -v 31457280`. Lead scratch: `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/b22_review/`.

| # | Check | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 1 | Codex: original G1h and GLM DSA plans rebuilt | byte-identical | package-03 plan SHA-256 `8bb82d07…6011` identical; 64 of 64 DSA plans identical | PASS |
| 2 | Codex: amended grid counts | 1,473 windows, +432 | FULL 495, PIECEWISE 495, first tail 462, multi-tail 7, dummy 13, prefill 1; +432 (W8 72, W16 216, W24 144) | PASS |
| 3 | Codex: supplement plan | every added window and repeat placed | 450 cases (e0 300, e1 150), 0 infeasible windows | PASS |
| 4 | Codex: coverage of G1h-02 measured plus the supplement predicted, against the C1 native classes | 0 missing | 1,041 measured and 432 predicted windows; P2-32 43/43, P2-64 54/54, P0 48/48 classes | PASS |
| 5 | Codex: no-supplement promotion | 4 tables byte-identical to the repository | attention 4,945 rows `34c4f3a8…321f`, gaps, linear and MoE identical | PASS |
| 6 | Lead: synthetic merge (4 genuine G1h-02 cases renamed as a supplement, 2 of them repeats) built at `a689ed05`, loaded by `load_dsv4_attention_dataframe` | union loads; one fingerprint; base rows unchanged | 4,955 rows loaded, 1 fingerprint, revision `a689ed05`; the 4,945 base rows are an unchanged prefix; both repeats excluded; gaps, linear and MoE tables identical | PASS |
| 7 | Lead: replay closure on census copies at 17:54 (`b22_review/replay.json`) | 0 union misses | P2-32: 16,000 distinct requests, 318 outside the current ranges, 0 outside the union; P2-64: 15,000, 0 and 0; recomputed current ranges equal the recorded ones | PASS (snapshot) |
| 8 | Lead: replay closure on census copies at 18:11 (`/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/census_snapshots/1811/replay.json`) | 0 union misses | P2-32: 17,000 requests, 480 outside the current ranges, 0 outside the union; P2-64: 16,000, 0 and 0 | PASS (snapshot) |
| 9 | Lead: B22 tree, 14 test files (pilot cases, graph replay, attention and DSA profiling, features, coverage, predictor, memo, supplement, difference, promotion, H200 tables) | all pass but the fixture test | 950 passed, 1 deselected | PASS |
| 10 | Lead: main after the merge, the same 14 files plus `test_dsv4_a10_tools.py` and `test_v4_replay_census.py` (`--basetemp b22_review/pytest_main`) | all pass | 980 passed in 178.9 s | PASS |
| 11 | Lead: supplement package built in `.worktrees/g1h-supplement-build` (HEAD `a689ed05`) | plan equals main's `plan_from_spec`; sources equal G1h-02's | 546 files, manifest SHA-256 `8af5eeddd813cdb7dcb883f2dcea46bfda878834859014bb31cde159657d0fc8`; plan equal; against `g1h/package_02` only the case list, `overlay.json`, `pilot_cases.json`, the plan, the output root and the paths differ; no path to the main working tree inside the package | PASS |

The census runs are still in progress, so checks 7 and 8 cover only the requests recorded at each snapshot. The census writes in steps of 1,000 new distinct requests. The final closure check runs on the finished censuses and again with the measured supplement rows.

## Official A10 runs

The official P2-32 cold and warm runs (B21, started 15:41 on main `d53bab0b`) failed at 16:44 and 16:50 with the probe-c error: `DeepSeekV4CoverageError` for `dsv4_compressor_fused_compress_store` (csa, PIECEWISE, first tail, padded rows 16), `num_compressed_writes=8 outside [1.0, 7.0]`. They run again after the supplement tables are promoted. P2-64 cold and warm are still running.

## GPU measurement

Supplement job: 1 H200, non-preemptible, `step_main`, timeout 10,800 s; tag `v4-h200-g1h-supplement-20261029-01`; cloud output `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/g1h_s01`. Estimate 0.608 GPU-hours (e0 0.358, e1 0.250), envelope 0.655, against about 3.07 GPU-hours left of the 7-hour G1h-G4h cap. Submitted through `/ds4-exec` (Pi run `g1h-supplement-submit-1029a`); results go to `/data/ycfeng/tmp/glm53_dsv4/h200/g1h_s01`.

## Limits

- The supplement rows are CPU projections until the GPU job measures them; the closure checks above say which coordinates the job must cover, not what it will time.
- The repeat comparison measures the difference between two packages on one GPU type, not kernel stability across hosts.
