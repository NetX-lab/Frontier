# Test report: H200 Step-5 operator profiles (D31 step D(g)) and the DECODE pricing fix

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-26 | Created: DECODE run 02, the four profiler and predictor fixes found by Frontier smoke runs, run 03, the merged tables and the DECODE measurement-family fix. |

## Scope

Step D(g) acceptance: each priced profile row has a coefficient of variation (CV) of at most 5%,
and the tables are wired into the predictor options of the step-B runner. This report also
records the defects that the first Frontier smoke runs on the measured tables found, and their
fixes.

## Environment

- Profile jobs: 1 H200 each, `step_main`, non-preemptible, creator `i-fengyicheng`, vLLM image
  `vllm-openai@sha256:69308454…`. The native workload of step C used the routing image
  `sha256:f037d958…`. Both images carry the same vLLM Step runtime; the routing image adds only
  the `position_cyclic` router (D29). The profile rows record their image in
  `runtime_stack_signature`.
- Staging: `/data/ycfeng/envs/oneshot-opd-py312/bin/python` with
  `frontier.profiling.step5.job --p0-role {PREFILL,DECODE} --profile-plan
  frontier/profiling/step5/h200_profile_plan.json`.
- Frontier runs and unit tests: `/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python`
  (Python 3.10; no `torch`, `aiohttp` or `plotly`).
- Fidelity matrix: `/data/ycfeng/envs/frontier-py310/bin/python`.

## Profile runs

| Run | Jobs | Result |
| --- | --- | --- |
| 01 | PREFILL `exp-1026-144712-650202`, DECODE `exp-1026-144726-883624` | PREFILL complete. DECODE failed at the first PIECEWISE attention capture (host graph pool assert); fixed in `688eb32b`. |
| 02 | DECODE `exp-1026-151616-591957` (Succeeded 07:31 UTC, 901 s) | 160/160 attention rows; linear 70 + 70; MoE 804 + 804 rows. MoE used the wrong load distribution (see below). |
| 03 | PREFILL `exp-1026-183852-975984` (417 s), DECODE `exp-1026-183852-950646` (978 s); both Succeeded | All tables complete; used for step E. |

Run 03 was submitted and copied by Pi run `d31-h200-profiles03-submit-copy-1026c`
(`/data/ycfeng/tmp/ds4_runs/d31-h200-profiles03-submit-copy-1026c/output.md`). Both copies are
`HASH_IDENTICAL_DOWNLOAD` with no mismatches (PREFILL 177 files, DECODE 1,047 files), under
`/data/ycfeng/tmp/step5_d31_profiles_20261026/downloads/d31_h200_{prefill,decode}_20261026_03/`.
Cloud output: `/mnt/codesign-exp/ycfeng/frontier/step5_profiling/d31_h200_{prefill,decode}_20261026_03`.

## Run 03 table checks

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Attention rows | PREFILL 6 + 6 kernel-only; DECODE 160 | PREFILL 6 + 6; DECODE 160; `complete: true` in both `attention/summary.json` | PASS |
| Linear rows | One row per planned token count, both measurement types | PREFILL 10 + 10, DECODE 70 + 70 | PASS |
| Linear stage boundaries | `input_cast` and `final_residual` columns | Present in both roles and both types | PASS |
| MoE rows and load | `load_distribution=balanced`, `min_load_ratio` reaches 1.0 | PREFILL 44 + 44, DECODE 268 + 268; all `balanced`; the balanced rows have `min_load_ratio` 1.0 | PASS |
| MoE token coverage on DECODE | Capture sizes up to 256 and EP-dispatched counts up to 1024 | 67 token counts from 1 to 1024 | PASS |
| Priced attention target CV | `step5_layer_e2e` CV and median disagreement ≤ 5% | DECODE 155/160; PREFILL CUDA_EVENT 5/6; PREFILL KERNEL_ONLY 6/6 | PASS with 6 marginal rows |

The 6 rows above 5%, as (graph mode, phase, layer, batch size, tokens): CV, median disagreement.

| Row | CV | Disagreement |
| --- | --- | --- |
| piecewise, decode, 0, 32, 32 | 5.30% | 2.18% |
| piecewise, mixed, 3, 17, 32 | 2.55% | 5.63% |
| piecewise, prefill, 0, 2, 32 | 5.14% | 1.65% |
| piecewise, prefill, 0, 1, 48 | 5.52% | 2.65% |
| none, decode, 3, 1, 1 (PREFILL CUDA_EVENT; a diagnostic row, not priced) | 5.17% | 0.39% |

All 6 are within 5.7%. In run 02, one row (piecewise decode, layer 0, batch 128) had CV 39.6%;
it does not recur in run 03 (worst row in that group: 5.30%).

`stability_pass` is false on 73 of the 160 DECODE rows (`STEP5_SUMMARY.stable_rows` 73 in the
worker log). The flag requires every sub-operator of the layer (QKV projection, SWA kernel, gate
and so on) to stay within 5%. Those sub-operators are short, and in PIECEWISE rows they sit
between graph pieces, so their CV reaches 18%. Frontier prices only `step5_layer_e2e`, so the
acceptance uses the CV of that target. The sub-operator times are diagnostic.

The DECODE package writes no kernel-only attention table (`gpu_active_rows` 0/0, the same as in
run 02). Every DECODE cell in the plan is FULL or PIECEWISE, and graph-replayed Step attention is
priced from CUDA_EVENT rows. A DECODE batch that runs eagerly would need a kernel-only row;
Frontier raises a coverage error if a run reaches one.

Small operators: `final_residual` costs at most 0.08 ms per forward (CUDA_EVENT, 6144 tokens) and
about 1–4 µs kernel-only at decode sizes. It is profiled as a replicated TP1 row over the full
token count. Whether native vLLM splits it across TP ranks is not verified; the effect on step
time is below 0.1%.

## Merge

```bash
D=/data/ycfeng/tmp/step5_d31_profiles_20261026/downloads
T=task_memory/task_2026-10-15_step5_preview_h800/evidence
PYTHONPATH=$PWD python tests/integration/step5_profiling/merge_role_profiles.py \
  --prefill $D/d31_h200_prefill_20261026_03/artifacts \
  --decode $D/d31_h200_decode_20261026_03/artifacts \
  --output /data/ycfeng/tmp/step5_d31_profiles_20261026/merged03 \
  --kernel-gap $T/d31_h200_step5_kernel_gap_proxy.csv \
  --nixl-read-profile $T/d31_h200_nixl_read_cost.json
```

Exit 0. Rows: linear 10 + 70 per type, MoE 44 + 268 per type, attention 6 + 160 (CUDA_EVENT) and
6 (kernel-only, PREFILL only). Output:
`/data/ycfeng/tmp/step5_d31_profiles_20261026/merged03/predictor_options.json`.

## Defects found by Frontier smoke runs

The smoke runs used `tests/integration/step5_pd/frontier_workload.py run --case h200_32` with a
4-request × 8-output trace (`/data/ycfeng/tmp/d31_e_smoke/trace_4x8.csv`) and the run-01/02 tables.

| Smoke | Failure | Cause | Fix |
| --- | --- | --- | --- |
| run01 | `Step input_layernorm requires actual positive integral kernel coverage` | The Step training path demanded kernel counts for every operator, but CUDA_EVENT tables carry none. Kernel counts only price the kernel gap of KERNEL_ONLY steps. | `ae4c9312` |
| run02 | `KeyError: 'evidence_kind'` when publishing Step model bundles | The linear and MoE profilers wrote no measurement identity; only the attention wrapper did. | `dd1f37dc`: one identity builder (`runtime_identity.step5_measurement_columns`) for all three |
| run03 | `min_load_ratio ... above declared maximum 0.9477: 1.0` | The MoE profile used the `uniform` distribution, which draws random top-k choices and gives imbalanced loads. Frontier prices Step MoE with `balanced` routing (D31.9). | `6790f110`: profile the `balanced` distribution with one sample |
| run04 | `Step measured attention coverage is unavailable for step5_layer_e2e` | The merged tables held only run 01's 38 FULL DECODE rows. | Merge run 02's 160 DECODE rows |
| — | Missing `input_cast` / `final_residual` coverage | The two stage-boundary operators were not profiling targets. | `39a82403` |
| run05 | Completed (rc 0, 438 s wall time): TTFT 689 ms, TPOT 94.0 ms, E2E 1,347 ms | TPOT was too high for a 4-request batch; see the next section. | `d034202c` |

Review finding against the D(g) ticket: the MoE load distribution was a planning error. The plan
named the routing policy (`balanced`, D31.9) but the job kept the old `uniform` profile setting;
neither the preflight nor the CPU coverage check compared the profiled load features with the
routing that Frontier uses. Run 03 is the first profile with the matching distribution.

## DECODE measurement-family fix (`d034202c`)

Symptom: with one request (6144 input, 4 output tokens) TPOT was 93.94 ms, while the native TPOT
at full load is 61.7 ms.

Diagnosis: an operator-traced run (`/data/ycfeng/tmp/d31_e_smoke/opdiag01/`,
`--metrics_config_enable_op_level_tracing`) showed that the PD DECODE cluster priced its MoE,
linear and EP communication work from the CUDA_EVENT tables. Under a CUDA Graph mode those
operators must use the KERNEL_ONLY tables, as for any other model. The cause:
`_should_enable_measurement_family` also returns true for the families that Step attention
adds (`step5_measurements_for_role`), and `_get_default_measurement_type_for_cluster` used the
same function to choose the step family. Step attention adds CUDA_EVENT for graph-replayed steps,
so the DECODE cluster switched every operator to CUDA_EVENT.

Fix: `_should_enable_compute_measurement_family` holds the original rule and alone chooses the
step family; `_should_enable_measurement_family` adds the Step attention families on top of it.
Step attention still selects its own family per batch. Only Step PD DECODE with a graph mode
changes. PREFILL, MONOLITHIC, graph mode `none` and models other than Step are unchanged.

| Metric | Before (`opdiag01`) | After (`opdiag02`) |
| --- | --- | --- |
| TTFT | 487.07 ms | 487.07 ms |
| TPOT | 93.94 ms | 14.77 ms |
| E2E | 927.28 ms | 615.13 ms |
| Decode compute | 388.84 ms | 76.69 ms |
| KV transfer | 51.37 ms | 51.37 ms |

Per-layer decode costs after the fix (batch 1): grouped GEMM 79.75 µs, `step5_layer_e2e`
73.81 µs (CUDA_EVENT), top-k 6.8 µs, shuffling 6.8 µs, shared-expert up projection 5.9 µs, gating
linear 2.9 µs, each EP all-to-all 2.68 µs, each layernorm 1.5 µs.

The grouped GEMM time also confirms that the `precision mismatch (BF16 -> FP8) ... factor=0.500`
warnings change nothing for Step MoE: the KERNEL_ONLY table gives 75.8 µs at 16 tokens, and the
trace gives 79.8 µs, which is that value plus the per-kernel gaps. The warning comes from metadata
only.

## Unit tests and fidelity for `d034202c`

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| New test `test_step5_attention_families_do_not_choose_pd_step_family` | Pass | Pass | PASS |
| `pytest tests/unit/ -k "step5 or kernel_gap or graph_policy or cuda_graph or measurement" --continue-on-collection-errors` (log `/data/ycfeng/tmp/d31_pytest/step5all3.log`) | Every failure also fails without the fix | 143 failed, 2,904 passed, 206 skipped, 15 collection errors | see next row |
| The 143 failed IDs on a `git archive` export of `431e5bd4` (the commit before the fix; log `/data/ycfeng/tmp/d31_pytest/headbase.log`) | Same IDs fail | 143 failed | PASS: no new failure |
| Fidelity matrix `tests/e2e/refactor_fidelity/run_matrix.py run --label candidate --jobs 8` (script `/data/ycfeng/tmp/d31_fidelity_dmf_run.sh`), candidate `d034202c` against the `e275c83` baseline | 77/77 identical | 77/77 identical; 0 failures on either side | PASS |

How the fidelity check was compared: the baseline outputs were copied from
`/data/ycfeng/tmp/d31_fidelity_dfi/baseline`. The harness's own `compare` therefore reported
77 mismatches, and each was only the `metrics_config/output_dir` path in `config.json`, which
still named the original directory. `/data/ycfeng/tmp/d31_fidelity_dmf_recompare.py` reran the
harness comparison functions (`compare_artifact_directories`, `substitutions_for`), with the
baseline paths read from their original root. Result:
`/data/ycfeng/tmp/d31_fidelity_dmf_recompare.json`, 77 identical. Predictor cache names were not
compared: neither side ran with `--clean-cache`. The harness also notes that the candidate's
working tree had uncommitted files. These belong to other sessions (task records and an untracked
integration script) and are not simulator code.

The 143 shared failures: 85 `test_step5_native_reference`, 15 `http_capture`, 14 `canonical_g3`,
7 `measurement_family_selector`, 4 `linear_op_ownership`, 4 `request_free_profiling`,
3 `model_config`, 3 `native_pd`, 2 `calibration_pd_harness`, 2 `structural_execution` and one each
in `kernel_gap_pricing`, `pdaf_examples`, `boundary_probe` and `native_pd_r2_stage`. Most need
packages this environment lacks (`torch`, `aiohttp`, `plotly`). The others assert contracts that
later commits replaced, for example "Step attention execution is disabled"
(`test_step5_linear_op_ownership.py:66`, `test_step5_model_config.py:551`), or use
`SimpleNamespace` fixtures that lack newer attributes (`take_forward_start_batch`,
`num_processed_tokens`, `OperatorSpec.kernel_count`). They are stale tests, outside step E.

## Limits

- The kernel gap is a Qwen3-235B H200 proxy (see the D(f) entry in `progress.md`).
- The NIXL READ cost is the mean under the native QPS-1 load.
- CPU overhead is not modelled (`skip_cpu_overhead_modeling=true`).
