# Primary R4 integration and native06 acceptance

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded independent review, merged regression, comprehensive-gate binding, freeze and complete re-evaluation. |

## Outcome

`ACCEPTED_NATIVE_REFERENCE_R4` for existing `exp-1023-043123-830898`, original source b62c155. Root integrated the two-file/223-line correction at d278659f8469a1cde6b4d5113a23ad1084587e59. Independent Standards and Spec reviews found no actionable issues (`review_2026-10-23.md`). Historical routing953/dispatch491 and isolated-consumer197 are separate deltas; this correction adds223 gross changed lines and one new maintained code path, with zero core edits. No new GPU run occurred.

The original R3 evaluation remains FAIL. The new result is bound to R4 SHA256 `67f3192302b6d12c0b8296581ee8b2788a772bce80b158ad45bc349eacb32a7b`, frozen R3 `ca8ddcea83c852c27295c8a73453746d755c51055e703bd94df48e8f1af7d917`, and frozen R2 `29523700ac8492fa6e192ce272f8ab5a7a48d1528bd4064417e26e811533d10a`.

## Environment and commands

Primary cwd: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Python: `/data/ycfeng/envs/asyncopd-py312/bin/python`, 3.12.13, Torch2.9.1+cu128; no conda activation or dependency changes. CPU-only checker execution; CUDA stayed uninitialized in root diagnostics.

```bash
env PYTHONDONTWRITEBYTECODE=1 FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 NATIVE_READER_BINDING_CHECKER=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_reader_binding.py tests/unit/test_native_v2_serving_inventory.py tests/unit/test_native_serving_inventory.py -q -p no:cacheprovider --junitxml=/data/ycfeng/tmp/glm53_dsv4/root_native_r4_merged_d278659.xml
```

Expected208tests with no failure/error/skip; observed208PASS/0/0/0. Log uses the same prefix `.log`. The preceding independent29790c9 run also passed208/0/0/0 in52.396seconds (`root_native_r4_29790c9.xml`). Root separately called committed `R4.check()` before freeze:8PASS,32ranks/six phases,123original paths unchanged before/after; `/data/ycfeng/tmp/glm53_dsv4/root_native_r4_29790c9_diagnostic.json`.

## Comprehensive default-matrix gate

No new matrix capture was claimed. Root revalidated the actual accepted576d278 capture:77/77identical,451artifact files rehashed,519cache filenames unchanged, every result bound to clean576d278, comparison complete without provenance findings. Root then independently hashed777current runtime/matrix input files against that commit, checked the unchanged collective-sim gitlink and the matrix driver's actual Python/package environment. The only changed non-task path is the native-checker unit test; every captured shell entrypoint was checked to exclude pytest/native-checker execution. These checks and exact capture hashes are in `matrix_accounting.json`. Native-reader behavior is covered separately by208tests and the complete actual-data checker. This closes the existing comprehensive gate through explicit capture/source/input/reachability correspondence, not a waiver or a relabelled capture.

## Frozen re-evaluation

Freeze: `../task_2026-10-22_v4_native_dispatch/FINAL_FREEZE_R4.json`. Both source writers stopped; it binds source, dependencies, reviews, merged XML, matrix accounting and123original files. Root created a fresh analysis context `/data/ycfeng/tmp/glm53_dsv4/native06_case_r4_20261023` with byte/nanosecond-mtime-preserved manifest/inputs and a `runs` symlink to the original retained run. This reuses the existing checker CLI/output mechanism while preserving the original R3 analysis directory.

After rehashing every freeze dependency/evidence/original, root invoked the following through `runpy.run_path(..., run_name='__main__')`:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028 FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/asyncopd-py312/bin/python -B task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py /data/ycfeng/tmp/glm53_dsv4/native06_case_r4_20261023 dsv4-h800-p0-canonical-20261021-06 clean
```

Expected all8predicates PASS and no original-file changes; observed both. Exact stdout: `/data/ycfeng/tmp/glm53_dsv4/native06_r4_frozen_evaluation_20261023.log`. Durable result and root decision: parent `native06_acceptance_20261023/{frozen_r4_result.json,primary_acceptance_r4.json}`. The standard analysis status is also present in the fresh evaluation context for downstream case consumers.

| Observed runtime coverage | Value |
| --- | ---: |
| Physical ranks / phases per rank |32 /6 |
| Independent owners / routers / runners per rank |43 /43 /43 |
| Prefill NONE forwards / replays |3456 /0 |
| Decode PIECEWISE forwards / replays |6760 /297440 |
| Decode FULL forwards / replays |442064 /442064 |
| Client requests, warmup / formal |432,48 /384 |
| Containers exit0 / restarts |4 /0 |
| Actual GPU hours / original cap |12.002222 /32 |

These are actual host-return/replay counters under the approved observer protocol, not independently measured GPU kernel durations. Source/state/DP operands, padding, graph phases, canonical populations/idle boundaries and terminal checks all pass. Prefill NONE is the pinned native dispatch result, not a waived replay requirement.

## Limits and remaining work

The native reference is admitted for the original three-round6144/1024/QPS1 balanced P16+D16 TP8/DP2/EP16/PP1 case. Frontier prediction and four-metric calibration are not admitted: missing operator-cost/physicalTP8-profile/graph-cost/balanced-decode coverage and full consumer wiring remain. No independent expert histogram, zero-observer-overhead claim or CPU-overhead analysis is added. Native05 and the original R3 failure stay historical. Resource estimates in the separate remaining-measurements proposal are preliminary and authorize no run.
