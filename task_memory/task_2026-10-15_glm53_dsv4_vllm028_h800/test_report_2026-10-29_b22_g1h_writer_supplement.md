# Test report: B22, the G1h writer-count amendment, the supplement package, its measurement and promotion

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Added the supplement measurement, the post-job CPU checks, the package difference and the routing-origin loader failure (B23). |
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

## Supplement measurement

Job `exp-1030-021110-248027` (tag `v4-h200-g1h-supplement-20261029-01`), 1 H200, `step_main`, non-preemptible: Succeeded. Pi run `g1h-supplement-submit-1029a` (exit 0) downloaded the cloud output to `/data/ycfeng/tmp/glm53_dsv4/h200/g1h_s01/download/` (10,569,782,264 bytes, of which `traces/` 10,293,872,888).

| # | Check | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 12 | Bridge report | PASS, no artifact errors | PASS; e0 FULL 150 + PIECEWISE 150, e1 PIECEWISE 150; all traces `error: null` | PASS |
| 13 | Measured steps against the plan | e0 300, e1 150 | e0 300 step rows (300 distinct cases), e1 150 (150) | PASS |
| 14 | Runtime and device | as G1h-02 | vLLM 0.28.0, torch 2.13.0+cu130, FlashInfer 0.6.16.post3, Triton 3.7.1, CUDA 13.0, NVIDIA H200 sm90, driver 570.124.06 | PASS |
| 15 | GPU time | estimate 0.608 h, envelope 0.655 h | measured phases e0 1,538.1 s + e1 403.4 s = 0.539 h; G1h-G4h cap usage about 4.52 of 7 h | PASS |

Pi observed two cases with 14,056 unattributed kernel launches against a typical 673-734: `g1hs-e0-0437-full` and `g1hs-e0-0443-piecewise`. Both are repeat windows (W24, own 24, mix (7, 0)), so promotion excludes them; their summed window times differ from G1h-02 by -1.11% and -0.56% (check 19), so the launches do not show in the attributed scope times. One run-log traceback per engine is the caught vLLM warning `Chat template warmup failed` (`skip_tokenizer_init=True`); the engines continued.

## Post-job CPU checks

Script `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/g1h_s01_post.sh census_snapshots/1930` on main `8b9e2016`; log `g1h_s01_post.log` beside it; outputs in `g1h_s01_post/`. The census copies at 19:30 hold P2-32 17,000 and P2-64 16,000 distinct requests.

| # | Check | Expected | Actual | Verdict |
| --- | --- | --- | --- | --- |
| 16 | Engine identity, 21 fields of each engine record against G1h-02 engine 0 | SAME | SAME (`identity.json`) | PASS |
| 17 | Measured coverage of both packages against the amended grid and the C1 native classes | 1,473 windows, 0 missing | 1,473 covered, 0 missing, 0 mismatches | PASS |
| 18 | Supplement reduction with repeats | 450 cases reduced | 2,250 rows in `supplement_full/` | PASS |
| 19 | Package difference on the 18 repeats | report | 2,448 rows compared, 0 missing. Scope time: abs(relative) p50 2.78%, p90 11.2%, max 42.8%; abs(absolute) max 0.0087 ms. Summed time per window: supplement lower by 0.25% to 3.04% (median 1.11%). Only `dsv4_compressor_fused_compress_store` csa PIECEWISE has a scope median beyond 10%: -32.8% over 24 rows, abs at most 0.0087 ms. Gap rows: abs(relative) p50 10.4% on medians near 0.0001 ms | REPORTED |
| 20 | Scratch promotion | attention union, other tables unchanged | 7,105 attention rows (4,945 + 2,160; 90 repeat rows excluded); `kernel_gap.csv`, `linear_op_kernel_only.csv`, `moe_kernel_only.csv` byte-identical to the repository | PASS |
| 21 | Loader check of the merged table | loads, 1 fingerprint | `DeepSeekV4CoverageError: V4 balanced replay attention missing/ambiguous moe_routing` | FAIL |
| 22 | Measured replay closure | 0 misses | not run (needs check 21) | BLOCKED |

Every sample in check 19 is one measured step per package, so the difference mixes run-to-run spread with any package effect. The supplement is lower in all 18 windows, by at most 3.04%.

### Check 21: cause

`validate_balanced_routing` (`frontier/execution_time_predictor/dsv4_predictor.py:78-94`) requires one `moe_routing` value. The two packages record different host paths of their `balanced_routing.py` copy in the descriptor's top-level `origin`; every other identity column and descriptor field is equal, including `sha256`, routers, runners, native-source digests and the runtime fingerprint `5237f2bd076fe132`. The B22 synthetic merge (check 6) renamed G1h-02 cases, so both halves carried one path and the case did not arise.

Lead diagnostic (scratch monkeypatch, `g1h_s01_post/replay_measured_diag.json`) that compares the descriptors without the top-level `origin`: the merged table loads with 7,105 rows, 1 fingerprint, revision `a689ed05`, 554 trained models; measured replay closure gives 0 misses for P2-32 (480 against the committed table) and P2-64. Decision 40 (D-ROUTING-ORIGIN-PROVENANCE) adopts this rule; Codex implements it as B23 ([assignment](h3o_operator_costs/codex_assignment_B23_routing_origin.md), request `ce6d9226`, effort ultra).

## Limits

- The supplement rows are CPU projections until the GPU job measures them; the closure checks above say which coordinates the job must cover, not what it will time.
- The repeat comparison measures the difference between two packages on one GPU type, not kernel stability across hosts.
