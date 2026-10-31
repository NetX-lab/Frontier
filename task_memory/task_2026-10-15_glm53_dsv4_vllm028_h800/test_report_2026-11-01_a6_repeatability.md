# A6 repeatability probe verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Verified terminal business/source/128-case evidence, analyzed all repeats and retained uncertainty about any new measurement generation. |
| 2026-11-01 | Recorded pre-submission geometry, source, payload and reservation checks. |

## Environment and commands

Workspace: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, HEAD `2b4da71e`; frozen measurement source `a689ed05`. CPU Python `/data/ycfeng/envs/frontier-py310/bin/python`, 3.10.6; platform Python `/data/ycfeng/tmp/stepmind-env/bin/python`, 3.10.6. Temporary paths use `/data/ycfeng/tmp`. Shell is bash for proxy initialization, zsh otherwise.

- Geometry/source: `env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/glm53_dsv4/verify_a6_repeatability_20261101.py`. Retained exact script: `h3o_operator_costs/a6_repeatability_20261101/verify_probe.py`. It uses the existing streaming reader with a 30-GiB RLIMIT_AS; systemd scope requires unavailable interactive authentication as recorded in the earlier A6 report. Expected: every clone preserves the original case and native measured geometry, with source/hash equality. Actual: exit0, 11.632083s; 546/546 manifest files, 527 source files, 128/128 clones, 8/8 original raw steps, zero geometry mismatches. PASS. This is not full-grid coverage. The retained geometry JSON omits only the unrelated full-grid `windows` and `classes` catalogues; every observed/predicted step result remains.
- Actual ResourceConfig dry run: `env PYTHONPATH=/data/ycfeng/steptron PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/glm53_dsv4/a6_repeatability_payload_20261101.py`. Exact script retained as `verify_payload.py`. Only `spawn_tasks` is replaced with a capture callback. Actual: exit0, one H200/step_main, non-preemptible,16CPU/128GB, local frozen source, explicit company image, cloud mount,5400s payload timeout. PASS, zero platform operations. Existing Paramiko TripleDES deprecation warnings are retained in the tool output; they do not change the payload.
- Personal reservation query: handbook proxy initialization followed by `timeout 60s /data/ycfeng/tmp/stepmind-env/bin/python /data/ycfeng/tmp/glm53_dsv4/a6_repeatability_quota_20261101.py`. Actual exit0; one complete page, six existing personal jobs, all preemptible, zero personal non-preemptible reservations. Quota-group API403 means free capacity UNKNOWN; no platform access or capacity claim is inferred.

Evidence and frozen hypothesis: `h3o_operator_costs/a6_repeatability_20261101/`. `probe_plan.md` records selection, limits, analysis and budget before launch. The package manifest SHA-256 is `c7082cef1d4c6ff32fe8fb106dbef363ced1fbb3c4c747c4eca4007b67e37822`.

## Completed probe and artifact verification

RJob `exp-1101-015820-503227` and its sole replica are Succeeded; container exit0/restart0 and launcher exit0. Personal API confirms creator `i-fengyicheng`, non-preemptible H200/step_main, the frozen image, local NFS `100.96.128.195:/data/ycfeng/Frontier/.worktrees/g1h-supplement-build`, and actual container memory125Gi. The worker ran on `gpu-h200-0819.lgcm.sh.istep.fun`: NVIDIA H200143771MiB, driver570.124.06, Python3.12.3, torch2.13.0+cu130, vLLM0.28.0, FlashInfer0.6.16.post3. Frozen measurement source remains `a689ed05f462eb269b9dffe640da151e6614a5ef`.

Both engine runs exit0 and the frozen bridge report is PASS with no reasons. All128 expected trace identities have no trace error and exactly one hold interval. Engine0 has64 FULL and32 PIECEWISE steps; engine1 has32 PIECEWISE first-tail steps. The archive contains all140 listed nonempty files, including128 traces totaling4,009,671,911 bytes. Selected download:11 files,66,013,651 bytes; every local SHA256 equals its remote SHA256. Full traces remain at `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_attention/a6_repeatability_20261101_01/A6_REPEATABILITY/bridge/FULL_AND_PIECEWISE/traces/`; names/sizes were checked remotely, without rehashing the entire trace archive.

The first success-only download stopped before creating its target because the SSH view still returned `state=running` after platform completion. A separate retained observation subsequently saw `succeeded` and complete reports; the unchanged download then passed. This establishes delayed visibility in that read, not its storage-layer cause. The business verifier initially assumed engine1 FULL; its actual and previously predicted first-tail dispatch is PIECEWISE. Correcting that verifier assertion produced PASS; no workload, plan or worker output changed. Evidence: `archive_observation_01.json`, `business_initial_failure.json`, `platform_03.json`, `worker_status_final.json`, `download.json`, `business_verification.json`.

Commands (all from the current worktree):

- `/data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/glm53_dsv4/collect_a6_repeatability_20261101.py`; exact source retained as `collect_selected.py`, remote commands/hashes in `download.json`.
- `/data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_repeatability_20261101/verify_business.py`; final PASS, including exact remote archive stat commands. No new GPU job or gate.

Observed timing: container01:58:58–02:17:23 on November1 UTC+8,1,105s =0.306944GPUh; phase1,084.3s; launcher1,151.371131s =0.319825GPUh as a time window. The platform finish timestamp is02:17:25 UTC+8. Using the larger launcher window conservatively moves the approximate G1h–G4h ledger from4.52 to4.839825 of7GPUh, leaving about2.160175GPUh. These are measured duration windows plus an approximate prior ledger, not an exact platform bill.

## Repeatability analysis

`tests/analysis/v4_attention_repeatability.py` reuses production `build_dsv4_replay_rows`, existing exact geometry prediction and the existing source/workload identity comparator. Source locations are retained as provenance; only the already-decided routing top-level `origin` is excluded from identity. All128 planned cases and640 layer-group rows pass, with zero identity/count mismatches. All544 original medians reconstruct to the published table within1e-12ms, agreeing with the earlier standalone reference check. The probe is a selected subset, so `grid_complete=false` is expected; it is not full qualification.

Exact analysis command (declared environment and source/hash timestamps are retained in `analysis_command.json`):

```bash
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028:/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/analysis PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 60s /data/ycfeng/envs/frontier-py310/bin/python tests/analysis/v4_attention_repeatability.py --package-plan /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_repeatability_20261101/package_plan.json --case-mapping /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_repeatability_20261101/case_mapping.json --reference-raw /data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/download/G1H_ATTENTION/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e0.jsonl /data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/download/G1H_ATTENTION/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e1.jsonl /data/ycfeng/tmp/glm53_dsv4/h200/g1h_s01/download/G1H_SUPPLEMENT/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e0.jsonl --raw /data/ycfeng/tmp/glm53_dsv4/a6_repeatability_20261101_download/A6_REPEATABILITY/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e0.jsonl /data/ycfeng/tmp/glm53_dsv4/a6_repeatability_20261101_download/A6_REPEATABILITY/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e1.jsonl --published-table data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv --output /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_repeatability_20261101/result.json
```

Actual: exit0, analyzer15.833276s, status `PASS_IDENTITY_AND_GEOMETRY`. `result.json` retains all16 values for every measured group of the four selected scopes (104 measured scope/group combinations), sample standard deviations, CV, inclusive p10/p90, first/second eight medians, original values, absolute/relative differences and CSA signed contrasts. Raw records and their source hashes preserve all other scopes and kernel/sample counts. No measurements or extremes are dropped. Durations below are microseconds; relative difference uses the original duration as denominator and is undefined for zero (none of these durations is zero).

Selected failing cases, first CSA layer2 and reuse layers4/6/8/10:

| Case / scope | Layer group | Original µs | Repeat median µs | Absolute difference µs | Relative difference | Sample SD µs | CV | p10 / p90 µs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| g1-e0-0263-full / index_compressor_fused_compress_store | first | 7.583984 | 8.080078 | 0.496094 | 6.5413% | 1.742157 | 19.1238% | 7.679932 / 11.887939 |
| g1-e0-0263-full / index_compressor_fused_compress_store | reuse | 11.439941 | 9.567871 | 1.872070 | 16.3643% | 1.201548 | 12.6519% | 8.040039 / 11.079956 |
| g1-e1-0729-first_tail / compressor_fused_compress_store | first | 11.681152 | 20.031494 | 8.350342 | 71.4856% | 0.400650 | 2.0013% | 19.551514 / 20.400391 |
| g1-e1-0729-first_tail / compressor_fused_compress_store | reuse | 20.304443 | 20.367920 | 0.063477 | 0.3126% | 1.309351 | 6.5589% | 18.495972 / 20.736084 |
| g1hs-e0-0101-full / q_projection | first | 7.169922 | 11.329102 | 4.159180 | 58.0087% | 0.663054 | 5.9939% | 10.095703 / 11.695801 |
| g1hs-e0-0101-full / q_projection | reuse | 10.159912 | 11.096069 | 0.936157 | 9.2142% | 0.458993 | 4.1750% | 10.552246 / 11.416138 |
| g1hs-e0-0254-piecewise / index_fused_query_prepare | first | 4.255859 | 2.656494 | 1.599365 | 37.5803% | 0.156742 | 5.8571% | 2.479980 / 2.927979 |
| g1hs-e0-0254-piecewise / index_fused_query_prepare | reuse | 2.687988 | 2.623535 | 0.064453 | 2.3978% | 0.073663 | 2.8063% | 2.527710 / 2.720093 |

First-layer split and adjacent controls (diagnostic; no new threshold):

| Case | Original µs | Repeat median µs | First8 median µs | Last8 median µs | Absolute split difference µs | Relative split difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| g1-e0-0263-full | 7.583984 | 8.080078 | 7.808105 | 8.239502 | 0.431396 | 5.5250% |
| g1-e0-0262-full | 12.575195 | 8.095947 | 9.600098 | 7.632080 | 1.968018 | 20.5000% |
| g1-e1-0729-first_tail | 11.681152 | 20.031494 | 20.127686 | 19.904053 | 0.223633 | 1.1111% |
| g1-e1-0730-first_tail | 19.520020 | 19.520020 | 19.247803 | 19.648193 | 0.400391 | 2.0802% |
| g1hs-e0-0101-full | 7.169922 | 11.329102 | 11.376953 | 11.280029 | 0.096924 | 0.8519% |
| g1hs-e0-0100-full | 11.295898 | 11.312012 | 11.423828 | 11.216064 | 0.207764 | 1.8187% |
| g1hs-e0-0254-piecewise | 4.255859 | 2.656494 | 2.671875 | 2.640137 | 0.031738 | 1.1879% |
| g1hs-e0-0253-piecewise | 2.527832 | 2.736084 | 2.752441 | 2.687988 | 0.064453 | 2.3417% |

The CSA first-minus-reuse contrast shrinks from−3.855957 to median−0.383667µs for index compression,−8.623291 to−0.264648µs for compression,−2.989990 to+0.097290µs for query projection, and+1.567871 to+0.031494µs for query preparation. Only9/16,12/16,6/16 and10/16 signs agree with their original contrasts. The two query-projection case medians converge to11.329102 and11.312012µs despite originals7.169922 and11.295898µs. The index-compression control still has19.96%CV and20.50%half-median difference;16 observations do not establish a stable population median for every scope.

## Decision and verification limits

The frozen hypothesis is supported for these selected cases: single-step timings and contrast magnitudes change while the recorded geometry, source identity and ownership/counts agree. This is diagnostic evidence of insufficient repeatability of individual measurement inputs. It does not identify prompt content, GPU cache, GPU clocks, kernel content dependence or instrumentation as the cause. Prompt contents vary through the original seeded generator, and hardware state is not reset. The selected cases do not estimate whole-grid repeatability.

Under D-NO-INTERRUPT, keep the estimator, layer/planner features and every formal table unchanged. A repeat-based measurement generation deserves consideration, but the unstable index-compression control prevents asserting that any fixed count tested here solves qualification. Stop repeating the same selected probe or expanding its analyzer. Complete the already-running F comparisons, then decide a uniform measurement correction from all retained evidence. A recipe must preserve the full1,421-case qualification population and every original measurement; selecting only favorable repetitions or replacing only failing rows is not supported.

`reconstruction_costs.py/json` maps all1,421 cases to the two existing plans and14,468 engine steps per reconstruction. Conditional costs from the frozen timing model are1.307903/2.210720/4.016353/7.627620/14.850153GPUh for1/2/4/8/16 reconstructions each, before timeout/platform reserve. No option is selected or submitted. Even two full reconstructions exceed the approximate remaining2.160175GPUh before reserve; a larger campaign would need a concrete recipe and additional resource approval. One extra full pass fits arithmetically but is not a demonstrated solution to the observed variability.

A6 four attention and seven collective failures remain, as do tickets05/09 and the formal hardware/payload/coverage gates. Formal measured CC, K19, ingress10.3ms/handoff11.4ms, three clean references, folds and thresholds are unchanged. This report completes the diagnostic probe deliverable only.
