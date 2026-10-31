# B30b ex-situ probes: scoped implementation verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Verified the existing B30b patch, retained fresh plans and CPU evidence, and prepared the authorized scoped commit. |

## Outcome and authority

The required CPU suite passed **163 tests in 6.02 seconds**. Both fresh jobs were generated successfully. Independent inspection of the generated arrays and retained timing inputs passed **74 checks**. All eight successful historical G5h/K5 listings are byte-identical to `b0e568f5`; five historical failures retain the same exit codes and error categories. Bounded source review found no demonstrated defect requiring a repair. No source code was changed during this continuation.

Decision 46 in the main requirements controls this package: measured CC remains the formal input; F1b/F2/F3 supply diagnostic inputs only. F1a, F4, B29, ingress option A, A10 comparisons, and main task-record corrections remain with the integration owner. The latest user instruction explicitly authorizes this scoped commit and supersedes the assignment's earlier instruction to leave committing to the lead.

Read references included the applicable AGENTS instructions, main requirements/plan/progress, decision 46, the current Spec v2 at `/data/ycfeng/Frontier/.local-draft/new-models/Frontier_GLM53_DeepSeekV4Flash_vLLM028_H80064_Spec_v2.md`, tickets 05/06/09/19, `../codex_assignment_B30b_ex_situ_link_probes.md`, and the existing B30b README. Ticket 06 and the K5/runtime helpers were read from the integration workspace where they exist. These references were not edited.

## Scope and retained state

Worktree: `/data/ycfeng/Frontier/.worktrees/b30b-ex-situ-link-probes`; branch: `b30b-ex-situ-link-probes`; initial HEAD: `b0e568f55ed5ade0c234949697bf22a84cd2f184`.

Before continuation edits, the scoped process/descriptor check found no active writer. The original diff and file hashes are retained in `evidence_20261031_initial/initial_worktree.patch` and `initial_state.json`. The final scoped writable-descriptor census also found **0 writers**. All **114 retained paths** checked against that snapshot remain byte-identical, including protected main records, READMEs, original B30b source/stages, and old dry-run artifacts. The final checks are in `validation_20261031/scope_and_retention.json`.

The original visible implementation/test diff contains 8 paths, +677/-11 = **688 changed lines**. Its frontier portion contains exactly **5 files**, +235/-8 = **243 changed lines**. Four original, ignored B30b-local Python tools add 679 lines when included in Git; all 12 implementation/test paths therefore account for +1,356/-11 = **1,367 changed lines**. This broader count was reported before packaging. There was no implementation expansion or source repair. The largest reviewed Python module is 562 lines. Generated plans and durable evidence are counted separately in `commit_scope.json`.

The scoped commit includes the original profiling/test changes, shared G5h checker extension, existing B30b tools/stages and retained local evidence. No README, main requirements/spec/plan/progress/ticket, other worktree, simulator behavior, measured backend, backend interface, or configuration schema was changed. No file was deleted or moved and no GPU/platform/cloud operation was performed.

## Exact environment and validation commands

All commands used the absolute CPU interpreter `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6). The host was Linux 5.10.25-nvidia-gpu with glibc 2.35. Scratch/log/cache root: `/data/ycfeng/tmp/b30b_sidecar_20261031_2257`.

The required suite explicitly adds the supplied torch locations to PYTHONPATH. Actual module imports under that setting are **pytest 9.1.1, numpy 1.24.3, scipy 1.10.1, pandas 2.0.1, torch 2.1.0+cu121**, with torch at `/usr/local/lib/python3.10/dist-packages/torch/__init__.py` and CUDA availability false. The base CPU environment with only the repository in PYTHONPATH instead imports numpy 2.2.6, scipy 1.15.3 and pandas 2.3.3. These are distinct import environments; `validation_20261031/environment.json` records both sets and their exact paths. No dependencies were installed or changed.

```bash
cd /data/ycfeng/Frontier/.worktrees/b30b-ex-situ-link-probes
ulimit -v 31457280
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages"
export TMPDIR=/data/ycfeng/tmp/b30b_sidecar_20261031_2257/tmp
export XDG_CACHE_HOME=/data/ycfeng/tmp/b30b_sidecar_20261031_2257/cache
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

timeout 240s /data/ycfeng/envs/frontier-py310/bin/python -m pytest \
  tests/unit/test_b30b_ex_situ_probes.py \
  tests/unit/test_collective_benchmark_helpers.py \
  tests/unit/test_vllm_dsv4_forward_collectives.py \
  tests/unit/test_vllm_forward_collectives.py \
  tests/unit/test_collective_kernel_evidence.py \
  task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g5h/launch/tests \
  -q -p no:cacheprovider \
  --basetemp=/data/ycfeng/tmp/b30b_sidecar_20261031_2257/pytest_initial \
  --junitxml=/data/ycfeng/tmp/b30b_sidecar_20261031_2257/pytest_initial.xml

timeout 60s /data/ycfeng/envs/frontier-py310/bin/python \
  task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b30b/planner.py \
  --output task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b30b/fresh_dry_run_20261031_01

timeout 60s /data/ycfeng/envs/frontier-py310/bin/python \
  /data/ycfeng/tmp/b30b_sidecar_20261031_2257/verify_plan_preservation.py

timeout 60s /data/ycfeng/envs/frontier-py310/bin/python \
  /data/ycfeng/tmp/b30b_sidecar_20261031_2257/audit_fresh_plan.py

git diff --check
```

`validation_20261031/commands.json` records exact argv, working directory, results and raw log paths. Final scoped staging verified all 53 selected paths and every selected JSON. The staged whitespace command is recorded there; it excludes only the immutable original unified-diff snapshot, whose 14 blank context lines must preserve their single-space patch format. All source/document paths passed and the snapshot bytes were preserved. Preserved copies of the test log, JUnit XML, preservation log, independent arithmetic program and original preservation-program text are in the same evidence directory. The original preservation command is a one-shot check that creates scratch baseline and evidence directories with `exist_ok=False`; a later rerun must use fresh directories. The independent arithmetic program re-reads existing fresh arrays and CSV inputs without invoking any benchmark or planner again.

## Required test results

| Check | Expected | Actual | Status |
| --- | --- | --- | --- |
| B30b unit tests | Grid/admission, native layout, placement, phase continuation, depth, synthetic fits and CLI checks pass | 28 passed | PASS |
| Collective benchmark helpers | Existing helper behavior preserved | 18 passed | PASS |
| V4 forward collectives | Existing V4 planner/shape checks pass | 58 passed | PASS |
| Existing forward collectives | Existing planner behavior preserved | 15 passed | PASS |
| Collective kernel evidence | Existing evidence handling preserved | 3 passed | PASS |
| G5h launch tests | Existing lock, worker command and launch behavior preserved | 41 passed | PASS |
| Full required command | No failures/errors/skips; exit 0 | 163 passed in 6.02 s; exit 0; 0 failures/errors/skips | PASS |
| Fresh B30b planner | Both jobs generated; exit 0 | JSON/text plans for both jobs; exit 0 | PASS |
| Independent fresh artifact audit | Counts, input hashes, service arithmetic, wall/GPUh and timeout checks agree | 74 checks passed; exit 0 | PASS |
| Original source/protected-file retention | Initial paths retain their bytes | 114/114 paths unchanged; 0 active writable descriptors | PASS |
| Whitespace check | No source/document diff whitespace errors | Unstaged and final scoped staged checks: no output; exit 0 | PASS |

Existing completed CLI fixtures were inspected without rerunning the tests. Recovered synthetic parameters were: ASTRA intra latency 2.5000000000000804 us / bandwidth 319.9999999999682 Gbit/s against 2.5 / 320; ASTRA inter 11.499999999998899 us / 124.99999999997488 Gbit/s against 11.5 / 125; NIXL 74.99999999999991 us / 109.99999999999999 Gbit/s against 75 / 110. All pass the existing `pytest.approx` tolerance. Exact absolute and relative parameter errors are recorded in `validation_20261031/synthetic_fit_results.json`. The synthetic launch-analysis CLI recovers its constructed K=128 and reads the retained native count 2,240; this is only a synthetic analysis check.

## Fresh cases and budget arithmetic

The stage inputs are `stage_b30b_links_kv.json` and `stage_b30b_launch_depth.json`. Complete fresh arrays and listings are:

- `fresh_dry_run_20261031_01/b30b_links_kv.json`
- `fresh_dry_run_20261031_01/b30b_links_kv.txt`
- `fresh_dry_run_20261031_01/b30b_launch_depth.json`
- `fresh_dry_run_20261031_01/b30b_launch_depth.txt`

The audit checks the actual array lengths and unique entries, rather than accepting a summary count.

| Phase | Actual cases per copy | Copies | Actual total case instances | Planned pass rows/trials | Estimated phase wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_intra, concurrent nodes | 132 | 2 | 264 | 792 | 252.7005644989252 |
| B_same_rail | 142 | 1 | 142 | 426 | 342.8460085073471 |
| B_ring_boundary | 142 | 1 | 142 | 426 | 342.8460085073471 |
| C_nixl | 80 | 1 | 80 | 240 | 199.48209916539935 |
| F2_launch_depth | 84 | 1 | 84 | 84 | 358.44728710937494 |

A contains 7 graph sizes × 6 calls + 14 eager sizes × 6 calls + 6 logits cases = **132 per node**. Each B placement contains 7 graph sizes × 2 EP calls + 8² ordered eager lane pairs × 2 EP calls = **142**. C contains 8 prompt sizes × 2 active-pair choices × 5 requests-in-flight choices = **80**. F2 contains (13 graph sizes + one plain-launch reference) × 2 sleep durations × 3 trials = **84**, comprising 78 graph trials and 6 plain trials. The two-node job therefore contains **628 case instances and 1,884 planned pass rows**. The two A nodes execute concurrently, so A's wall time is charged once while its case count includes both nodes.

| Job | Allocated GPUs | Estimated wall seconds | Estimated GPUh | Worker timeout seconds | Timeout GPUh exposure |
| --- | ---: | ---: | ---: | ---: | ---: |
| b30b_links_kv | 16 | 1976.8120210185284 | 8.785831204526794 | 4500 | 20 |
| b30b_launch_depth | 1 | 807.6709306640623 | 0.22435303629557288 | 3600 | 1 |

Combined estimate: **9.010184240822367 GPUh**; combined configured worker-time exposure: **21 GPUh**. These are planning estimates, not consumed GPU time.

The exact requested stage resources are H200 / `step_main`, with the assignment's non-preemptible placement:

| Stage | Workers | Per-worker CPU | Per-worker GPUs | Per-worker mem_gb | Total CPU / GPUs / mem_gb |
| --- | ---: | ---: | ---: | ---: | --- |
| b30b_links_kv | 2 | 64 | 8 | 512 | 128 / 16 / 1024 |
| b30b_launch_depth | 1 | 16 | 1 | 128 | 16 / 1 / 128 |

These resource values come directly from the unchanged stage JSONs. Configured phase timeouts are A=900 s, each B=1200 s, C=600 s, F2=3000 s; job payload timeouts are 4500 s and 3600 s. The 20+1 GPUh exposure bounds the configured worker payload timeout; it excludes platform initialization and billing overhead and is not an absolute billing cap.

Both jobs reuse `../g5h/launch/submit.py` with `--config stage_b30b_links_kv.json` or `--config stage_b30b_launch_depth.json`, resolved inside the integration workspace. The shared checker is `../g5h/launch/runtime_check.py`; both stages set `node_entry` to this directory's `phase_runner.py`. The two-node runner invokes the existing `frontier.profiling.collectives.vllm_collective_benchmark` for A/B and `frontier.profiling.collectives.nixl_kv_read_benchmark` for C. The one-GPU runner invokes `frontier.profiling.collectives.cuda_launch_depth`.

Declared output roots are `/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/b30b_links_kv_01` and `/mnt/codesign-exp/ycfeng/glm53_dsv4/comm/h200/b30b_launch_depth_01`. These are stage declarations only; no cloud directory was accessed. The lead reported mainline `058e4ba0` while this sidecar was finishing; this branch was not rebased and no mainline comparisons were duplicated.


Collective service uses the per-op/per-mode maximum measured G5h time, scaled only above its largest measured payload. A uses `3*(5+4*100)*1+1 = 1216`; each B uses `3*(5+4*30)*1+1 = 376`. Resulting service seconds are 88.9005644989252 for A and 170.04600850734712 for each B. Case coordination overheads are 118.8 and 127.8 seconds respectively, with 45 seconds of process startup per phase.

NIXL service is `sum(reader_bytes)*max(K5_complete_us/reader_bytes)*3*(2+30)/1e6 = 106.48209916539936 s`, plus 48 seconds of coordination and 45 seconds of process startup. F2 estimates 63 seconds of sleep plus 44,040,192 worst-case enqueued kernels at the minimum G5h call proxy 5.500797... us, yielding 305.25628710937497 seconds of service; capture contributes `8191*0.001 = 8.191 s`, then 45 seconds of process startup. The synthetic add-kernel time is not independently measured.

Two-node wall: `(180 + 252.7005644989252 + 342.8460085073471 + 342.8460085073471 + 199.48209916539935)*1.5 = 1976.8120210185284 s`; GPUh = wall × 16 / 3600. F2 wall: `(180 + 358.44728710937494)*1.5 = 807.6709306640623 s`; GPUh = wall / 3600. All phase estimates with the 1.5 safety factor and both job estimates are below their configured timeouts.

The two G5h summary CSVs and K5 NIXL summary CSV are used only as budget proxies. Their exact paths and SHA256 values are retained in each fresh JSON and verified by the independent audit. The fitting tools take future B30b A/B/C outputs; they do not fit parameters from G5h/K5 target curves.

The old README and `dry_run/` still state B=496 per placement and combined 29.648560 GPUh. Those artifacts were deliberately preserved. Current stage generation uses B=142 per placement and 9.010184240822367 combined GPUh; only this report and the new evidence describe the current budget.

## Native KV geometry

`kv_layout_evidence.json` binds 12 retained P2-32/P2-64 log lines, the native `base_worker.py` SHA256, and the retained `native_retention.md` SHA256. The planner checked all cited lines and both source hashes and rejected missing or inconsistent evidence in the unit tests.

The confirmed layout is **one packed region, 1,002,240 bytes per physical block**. Logical live-ID groups are `(256, full cache)`, `(64, window 128)`, `(64, window 128)`, `(4, window 8)`, `(8, window 128)`. Each selected physical block transfers the complete packed region; the request count concatenates live IDs across the groups after the native sliding-window pruning rule.

Expected/actual live-ID counts: T64=13, T6144=46 and T8192=54, all PASS. T6144 is a unit geometry check and is not inserted into the power-of-two C grid. The maximum fresh case uses 54 blocks × 16 requests = **864 blocks**, admitted by the **1,024-block scratch pool**. These imply 46,103,040 packed bytes per T6144 request, 54,120,960 per T8192 request, and 865,935,360 bytes per reader for the largest planned case. The scratch pool models registration/descriptor geometry, not native capacity, allocation fragmentation or traffic contention. Native packed-region bytes and simulator component-page accounting remain separately scoped.

## Historical preservation

The existing `plan_preservation.json` contract was executed against baseline/current actual stdout bytes and its recorded SHA256 values. The new result is `preservation_20261031/plan_preservation_20261031.json`; all 28 source checks in that contract match the existing patch. Generated successful listings are retained beside it.

| Existing stage | Expected and actual cases | Expected and actual planned measurements | Status |
| --- | ---: | ---: | --- |
| stage_g5h16_isolated_01 | 35 | 105 | PASS, byte-identical |
| stage_g5h16_tp8_dp2_ep16 | 82 | 246 | PASS, byte-identical |
| stage_g5h32_isolated_01 | 46 | 138 | PASS, byte-identical |
| stage_g5h32_tp8_dp4_ep32 | 46 | 138 | PASS, byte-identical |
| stage_k5_tp8_dp4_ep32 | 324 | 972 | PASS, byte-identical |
| stage_k5_tp8_dp2_ep16 | 324 | 972 | PASS, byte-identical |
| stage_k5_nixl_read_tp8 | 62 | 186 | PASS, byte-identical |
| stage_k5_ep_nixl_overlap | 14 | 42 | PASS, byte-identical |

Four older non-K5 plans already fail with `KeyError: 'passes'`; an older NIXL plan fails with `TypeError: string indices must be integers`. Baseline/current exit code 1, empty stdout bytes, hashes and error categories agree for all five. Their preservation status is PASS; the plans themselves remain historical failures and were not repaired.

## Bounded source review and validation limits

Standards review: the package reuses the existing collective/NIXL benchmarks, K5 rendezvous/atomic-output helpers and G5h lock/checker mechanism. Reviewed modules remain below 2,000 lines, and no simulator/backend code was modified. No concrete in-scope standards finding remained.

Spec review: default native SP admission remains strict; the explicit V4 link-probe opt-in retains EP/PP/dtype/backend validation. Physical placement uses the allocation UUID census and physical inventory, with no GPU7 substitution. Unit fault injection verifies phase preparation/failure continuation and process-group cleanup after timeout. Launch analysis distinguishes graph-launch and kernel-node capacities, retains censoring and refuses insufficient or sleep-dependent evidence. ASTRA residuals preserve native grouped dispatch and operation latency differences; NIXL fits admit only verified eight-pair rows with complete notifications and use W7 per-request normalization. Flat output fields are existing configuration fields. No demonstrated source defect required a repair.

The stage workspace remains `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. K5 `w5_glm_comm_20261020/launch/node_entry.py` and runtime-lock `probe_nixl_read.py`/`probe_runtime_inventory.py` exist there but are absent from this sidecar worktree. Integration and committed mounted-source preparation remain prerequisites for the lead's execution; they do not block this CPU/scoped-commit deliverable. No actual submission lock, authentication, quota, job inspection or cloud access was attempted.

CPU checks do not establish physical H200 placement, rail/NIC routing, measured link parameters, NIXL throughput, CUDA launch capacity, native stream equivalence, or any formal four-metric accuracy gate. Same-index GPUs alone do not prove the same rail. The retained native dummy artifact records 2,240 dispatch kernels in 93 lane0 FULL dummy forwards; this continuation tested its selector and consumed the artifact without repeating the heavy trace census. Real F2 prediction against native K=19 remains pending hardware measurements.

An existing ASTRA full-pod limitation remains: `_rank_to_physical_address` returns `(gpu, server)` and `_get_first_differing_dimension` compares the GPU coordinate first. Thus physical ranks 7→8 in an eight-GPU-per-server pod select its intra dimension. B30b's inter fit uses two servers with one participating GPU each and retains the two placement fits; applying those fields to a full pod retains that backend limitation. No backend repair or per-operation correction was introduced.

All sidecar acceptance checks are complete; scoped staging contains exactly 53 paths and no protected path. The integration owner owns the remaining hardware execution and diagnostic A10 comparisons. The verified commit containing this report and `commit_scope.json` is the scoped handoff; its commit ID is reported in the CLI response.
