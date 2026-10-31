# Issues: failures and resolutions

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | I-008: B28 merged (`ee29d548`); check of the uniform eager EP rows that B28 also replaced, against the nine uniform forwards of the native P2-64 profile. |
| 2026-10-30 | I-008 measured: the isolated G5h-32 run (`exp-1030-234535-245357`, 8.01 GPU-hours) and the native critical paths show that a single aligned call reproduces the eager EP calls on EP32 and EP16 (within 1-18 µs), while back-to-back timing fits the decode-graph and TP calls. Repair assigned as B28. |
| 2026-10-30 | I-008 opened: the back-to-back collective timing in `data/profiling/network/h200_dgx/summary.csv` gives the period of pipelined calls, not the time of one call started on aligned ranks. At 6,144 rows the EP combine is priced 164.7 µs per layer low on EP32 and 16.6-45.6 µs high on EP16. |
| 2026-10-30 | I-007 opened: the scheduler-lifecycle runs of 2026-10-25/26 are slower on the prefill side than the canonical runs in every repeat (13.8-34.4 ms); the 300 ms native prefill step taken from them is overstated. |
| 2026-10-30 | I-006 resolved by B26b (`414fdd03`, merge `49e7e5fa`): a test-fixture timing problem caused by `b0bf0c78`; the fixture's third arrival moved from 8 ms to 10 ms; no simulator change. |
| 2026-10-30 | I-002 resolved by B27 (`f00859be`, merge `3834bab6`): 128 passed. I-006 assigned with B26b (request `a92368f9`). |
| 2026-10-30 | I-006 opened: `test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward` fails on main `88fbef52`; found during the B26 review. |
| 2026-10-30 | I-002 assigned to Codex as B27 (request `7653c18e`); reproduced on main `591d7c54`: 126 passed, 2 failed. |
| 2026-10-30 | I-005 opened: the B25 wave observer wrote the engine identity as bytes into JSON, so the START_DP_WAVE records of the P2-64 wave run were lost (896 error records); the forward-level records were complete and carried the diagnosis. |
| 2026-10-30 | I-004 opened and worked around: the vLLM 0.28 environment `oneshot-opd-py312` was removed at 05:22 UTC outside this task; the B25 installed-wheel check became a source-only check. |
| 2026-10-30 | I-003 opened and resolved: `/data/ycfeng` filled at 20:58 on 2026-10-29 and every running simulation failed with ENOSPC; space was recovered by YC-approved deletions and the runs were restarted. |
| 2026-10-29 | I-002 opened: two `test_balanced_routing.py` cases fail on main since B16b; the module is skipped without torch, so earlier suites did not run it. |
| 2026-10-28 | Created: unit-test fixtures wrote into this worktree's Git directory through an exported `GIT_DIR`. |
| 2026-10-28 | I-001 resolved: repair applied after YC's approval and verified. |

## I-001: unit-test fixtures wrote into the worktree's Git directory (2026-10-28)

Status: resolved 2026-10-28. YC approved the repair ("批准，继续"); the lead applied it with exit 0, and every check below passed. Owner: lead.

**Failure.** During the B8 review, the lead ran the whole unit suite twice in parallel: before (on a `git archive dd4c4317` snapshot) and after (on the working tree). The runner was `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b8_review/checks/run_unit_pair.sh`. The before half exported `GIT_DIR=/data/ycfeng/Frontier/.git/worktrees/glm53-dsv4-vllm028` to give the snapshot a Git identity. Fixture tests that build temporary repositories then sent `git init`, `git config` and `git commit` to that directory instead of their temporary directories:
- `tests/unit/test_moe_ep_baseline_replay.py:167`;
- `tests/unit/test_moe_ep_non_dummy_matrix.py:5615`;
- `tests/unit/test_native_pd_manifest.py:35`;
- `tests/unit/test_refactor_fidelity_gate.py:464`.

**Observed state** (writes between 10:43:04 and 10:45:31 UTC):
- `refs/heads/feat/glm53-dsv4-vllm028`: `dd4c4317` (B7) → `80aacd8e` ("baseline", Frontier Tests, parent `dd4c4317`, deletes 4994 files and adds `tracked.txt`).
- This worktree's HEAD is detached at `4a107a47` ("initial", author `t`), after 29 fixture commits. Its index holds the 9 fixture files.
- The shared `/data/ycfeng/Frontier/.git/config`:
  - `core.bare = true`, written by the reinitializing `git init`; the main checkout is a normal checkout, so the value must be `false`;
  - an appended `[user]` section, `name = t` and `email = t@example.com`, which would give every repository commit without its own identity the author `t`.
- Not affected:
  - the working-tree files (the fixtures' work tree was their temporary directory), so the B8 edits are on disk;
  - the main checkout's `HEAD` and `index`;
  - every other branch and every remote-tracking ref;
  - other worktrees' HEADs.

**Backups.** In `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b8_review/git_repair/`: `config.polluted`, `worktree_HEAD.polluted`, `worktree_index.polluted`, `branch_ref.polluted`.

**Repair.** The permission check first blocked it as a destructive Git action; it was applied after YC's approval. It restores the earlier state exactly, and the reflog keeps the old values:
1. `git config --file /data/ycfeng/Frontier/.git/config core.bare false`
2. `git config --file /data/ycfeng/Frontier/.git/config --remove-section user`
3. `git update-ref refs/heads/feat/glm53-dsv4-vllm028 dd4c4317606789e5cafcba148f6c1955f3b96f7f 80aacd8e282bd1ae2d877caeb794e5718dbf7686`
4. In `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`: `git symbolic-ref HEAD refs/heads/feat/glm53-dsv4-vllm028`, then `git read-tree dd4c4317606789e5cafcba148f6c1955f3b96f7f`. This touches the index only; the working tree is not changed.

**Checks after the repair:**
- `git -C /data/ycfeng/Frontier rev-parse --is-bare-repository` prints `false`;
- `git config user.name` resolves to the global identity;
- in this worktree, `git branch --show-current` prints `feat/glm53-dsv4-vllm028` and `git log -1` shows `dd4c4317`;
- `git status --short` lists only the B8 changes, the untracked task records and the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/`.

Observed after the repair:
- every check above as expected; the main checkout's `git status` works on branch `main`;
- `git status --short` in this worktree lists exactly the B8 changes and the known untracked paths;
- the whole unit suite rerun on the working tree without `GIT_DIR`: 133 failed, 6650 passed, 133 skipped, 109 errors. It differs from the first after-run only by the R4 test (failure → passed), so the 91 s overlap changed no result.

B8 was then committed.

**Prevention.** Never export `GIT_DIR` or `GIT_WORK_TREE` around a test suite or any command that runs `git init`. A before-snapshot that needs a Git identity gets its own worktree; otherwise its Git-reading tests are accepted as failing in the comparison.

## I-003: `/data/ycfeng` full; all stage 4 simulations lost (2026-10-29)

Status: resolved for now; the external writer is unidentified. Owner: lead.

Observed: at 20:58 on 2026-10-29, `df` showed `/dev/rbd1` (XFS with project quota, mounted at `/data`) with 1 MB free. Every simulation that was still running failed with `OSError: [Errno 28] No space left on device`:

- official A10 runs on the promoted table: `stage4/a10/p2_32_cold_g1hs`, `p2_32_warm_g1hs`, `p2_64_cold_g1hs` and `p2_64_warm_g1hs` (all mid-run; the observer stream close and the receipt write failed);
- old-table A10 runs `stage4/a10/p2_64_cold` and `p2_64_warm`;
- the mid-run profile `stage4/probe/p2_64_mid` (it had reached 61.3 simulated s after 8,460 wall s; no `profile.pstats` was written);
- the census runs under `stage4/census/`.

Probe `stage4/probe/p2_64_c` (old table, no observer) had already ended with `EXIT 0` at 20:37; its gate output is kept.

Cause (inference): the simulations were not the writer. Each run root is about 300 MB including its 1,758-file cache. `find -mmin`/`-cmin` over the readable trees showed no matching growth. Free space also moved both ways while no lead process was writing (14,256 MB, then 14,972 MB two minutes later on 2026-10-30). The likely writers are processes outside this sandbox, such as GPU workers writing as uid 65534 through NFS into directories this user cannot read.

Resolution: YC approved deleting three groups (2026-10-30):

- the G1h supplement traces `/data/ycfeng/tmp/glm53_dsv4/h200/g1h_s01/download/G1H_SUPPLEMENT/bridge/FULL_AND_PIECEWISE/traces/` (9.6 GB, 450 files; the raw records and the promoted table do not depend on them);
- the temporary directories `stage4/promote_real/scratch` (830 MB) and `stage4/g1h_s01_post_dryrun` (615 MB);
- the old-table caches under `stage4/a10/{p2_32,p2_64}_{cold,warm}/cache` and `stage4/probe/p2_32_{a,b,c,perf,perf2,prof}/cache` (1,758 files each).

The G1h-02 traces were not approved and were kept. Free space after the deletions: 14,257 MB. A background `df` logger now writes `stage4/disk_watch.log` every 300 s so a repeat can be timed.

## I-002: `test_balanced_routing.py` cache test fails on main since B16b (2026-10-29)

Status: resolved on 2026-10-30 by B27 (commit `f00859be`, merged as `3834bab6`): the test now trains and reads the V4 models under `KERNEL_ONLY` and asserts the empty `eager` view; lead rerun with CPU torch: 128 passed (log `/data/ycfeng/tmp/b27_lead.log`). Assigned as B27 (request `7653c18e`, [assignment](h3o_operator_costs/codex_assignment_B27_balanced_routing_v4_test.md), worktree `.worktrees/b27-balanced-routing-v4-test` from main `b1cf3447`). Reproduced on main `591d7c54` with the command of the assignment: 126 passed, 2 failed (both parametrizations). Owner: lead.

Observed during the B23 review. With CPU torch on the path (`PYTHONPATH=...:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages`, `VLLM_BS_SOURCE_ROOT=/data/ycfeng/tmp/opencode/vllm-v0.28.0`), `tests/unit/test_balanced_routing.py` gives 126 passed and 2 failed, the same on main `2da2c186` and on the B23 tree:

```text
tests/unit/test_balanced_routing.py:316: KeyError: 'moe_gating_routing_topk'
    assert manager.get_models_for_cluster(role)["eager"][op] is cold[role]
```

Cause (from the code): B16b (`61c2f9d3`) makes `_get_measurement_types_for_cluster` return only the V4 measurement type, `KERNEL_ONLY`, for every DeepSeek-V4 role, so the `eager` view of `get_models_for_cluster` is empty for V4. The test still trains V4 gating models from `CUDA_EVENT` rows and reads them from the `eager` view. Without torch the module is skipped by `pytest.importorskip("torch")`, so suites run without torch do not report it.

Next action: update the test to the B16b contract (V4 roles read `KERNEL_ONLY`), in a later assignment; no simulator behavior is involved.

## I-004: vLLM 0.28 environment removed during the B25 implementation (2026-10-30)

Status: worked around; the environment is not restored. Owner: lead.

**Observed.** Codex's B25 CPU check exited 127: `/data/ycfeng/envs/oneshot-opd-py312/bin/python` no longer existed. The `df` logger (`stage4/disk_watch.log`) shows used space on `/data/ycfeng` falling from 986,494 MB at 05:18 UTC to 934,741 MB at 05:23 and 851,306 MB at 05:28, which matches the removal time. No command from this task removed the environment (inference from the session history; the deleting process is not known).

**Impact.** B25 planned to check the observer hooks against the installed vLLM wheel. Without the environment, the attributes could be checked only against the NATIVE source tree (`/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`).

**Resolution.**
- `check_native_dp_wave_source.py` checks every hooked attribute against the NATIVE source; it passed on the B25 branch and on the composed harness branch `p2-64-wave-composed` (`6488ea15`).
- The H4b check had already found the installed wheel byte-identical to NATIVE for the engine-core, scheduler and output files, and the lead verified each hooked attribute in the NATIVE source.
- The soft checks of the instrumented run itself serve as the installed-path check: an attribute that differs in the image appears as an error record, not as a silent gap.
- Recorded in `h3o_operator_costs/` B25 design section 9.

Next action: none for B25. A later task that needs a local vLLM 0.28 Python must rebuild the environment first.

## I-005: B25 wave observer lost the START_DP_WAVE records (2026-10-30)

Status: open; fix only before a future run (no rerun is approved). Owner: lead.

**Observed.** The instrumented native P2-64 run `exp-1030-144848-164721` (requirements item 42) succeeded, but its observer files hold 896 error records instead of the wave-start messages. `tests/comparison/calibration/native_dp_wave.py` (`enqueue_fields`, line 105, and `handle_begin`, line 136) copies `excluded_engine_index` into the JSON record as received. In vLLM-BS `2cf0a691` that value is the engine identity, a `bytes` value: the front end sends `("FIRST_REQ", chosen_engine)` (`vllm/v1/engine/core_client.py:1422`), and the coordinator forwards it as `exclude_engine_index` in START_DP_WAVE (`vllm/v1/engine/coordinator.py:354` and `466`). `json` cannot encode `bytes`, so every such record failed.

**Impact.** The wave-start times of the coordinator and the engines are missing. The forward-level records (worker `dp_cpu_all_reduce` and `dp_padding_sync`, scheduler `schedule`/`update`/`add`) are complete on all eight hosts, and they carry the diagnosis (A10 report section 3.5). The run is not repeated for this.

**Resolution (pending).** Encode the identity before writing it (for example as hex) in the observer, with a unit test that feeds a `bytes` identity. Apply it only when a further wave run is approved.

## I-006: placement test of the late-joining DP lane fails on main (2026-10-30)

Status: resolved 2026-10-30 by B26b (request `a92368f9`, [assignment](h3o_operator_costs/codex_assignment_B26b_placement_key_and_i006.md), [report](h3o_operator_costs/b26b/report.md); commit `414fdd03`, merged into main as `49e7e5fa`). Owner: lead.

**Observed.** `tests/integration/test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward` fails on main `88fbef52` (18 passed, 1 failed; log `/data/ycfeng/tmp/main_placement.log`) and in the same way on the B26 branch. The assertion at line 481 expects lane 2 to start forward group 0 with a dummy forward, `{2: (0, True)}`, and receives a real batch, `{2: (0, False)}`. The test file last changed in `5f907f66` (H4b-(b) item 1); 143 commits follow it on main, so the commit that broke the test is not known.

**Impact.** The test checks the four-lane MoE shape of issue W9-04 under both placement policies: a lane that is idle when a peer reaches the first MoE layer joins that forward with a dummy forward, and a request that arrives during it starts the next forward. Either the simulator now places the request's batch into the dummy forward's group, or the test's setup no longer reaches that shape. A10 is not known to be affected, because its prefill requests arrive 1 s apart, but this is not verified.

**Cause.** The test setup, not the simulator. Commit `b0bf0c78` (2026-10-08) added the forward preparation time (schedule time plus input preparation time) before the first Prefill/Decode sync event of a forward. With the fixture's 1 ms operator durations, the first lanes now reach the first MoE layer at 9 ms and 11 ms instead of 7 ms and 9 ms. The fixture's third request arrives at 8 ms, before the idle lanes are woken, so lane 2 correctly forms a real forward 0. Codex checked exported trees: the parent `2c9d625d` passes the test and `b0bf0c78` fails it. Main first received this test in merge `9a650fba` (2026-10-14), and that tree already fails. The earlier premise that the test passed at `5f907f66` was wrong: it already failed there.

**Native check.** vLLM's busy loop (`core.py:2153`, `2169`, `2178-2182`) runs the dummy forward before the next input poll, so a request that arrives during the dummy forward starts the next forward. The test's intended assertion still matches native behavior.

**Resolution.** `LATE_JOIN_TRACE`'s third arrival moved from 8 ms to 10 ms, and the test now asserts that the arrival falls inside lane 2's dummy forward (dummy start 9 ms, arrival 10 ms, dummy end 65 ms). Both placement policies keep placement `[0, 1, 2]`. No production source changed. The lead's run of the 13 affected test files passed (269 passed; log `/data/ycfeng/tmp/b26b_lead/verify.log`). A10 uses `round_robin` placement and does not run this path.

## I-007: the scheduler-lifecycle runs slow down the native prefill step (2026-10-30)

Status: open (cause not identified; the affected readings are corrected). Owner: lead.

**Observed.** On the proxy clock, without first-contact requests, every repeat of `dsv4-h200-p2-lifecycle-20261025-01` (P2-32: 260.7 / 255.4 / 256.4 ms) and of `dsv4-h200-p2-64-lifecycle-20261026-01` (P2-64: 570.3 / 581.6 / 574.0 ms) has a longer prefill side than every repeat of the canonical runs (P2-32r: 234.4 / 236.0 / 236.1 ms; P2-64r: 556.5 / 552.4 / 547.2 ms). The P2-64 wave run of 2026-10-30 (551.5 ms) falls inside the canonical range. Evidence: `h3o_operator_costs/a10/k19/p_side_per_repeat_stdout.txt`; A10 report section 5.5.

**Impact.** The added time sits inside the engine step: the P2-32 lifecycle step averages 244.6 ms against about 222 ms left by the canonical prefill side, and the lifecycle run's ingress is not larger. Readings that took the native prefill step from the lifecycle runs overstate it: the 300 ms compute of A10 report sections 3.2 and 3.5 and the "about 25 ms of compute" difference of section 3.2. On the wave run, a prefill served at once takes 289.6 ms (n 6). The H4b handoff measurements are not affected: the lifecycle runs' handoff is not longer than the canonical one (A10 report section 3.6).

**Next.** Native step values come from the wave run or from canonical-timed runs. The lifecycle observer's code is not inspected yet; the cause matters only if another lifecycle run is planned.

## I-008: back-to-back timing misprices the 6,144-row EP combine (2026-10-30)

Status: open (measured on 2026-10-30; repair B28 merged on 2026-10-31 as `ee29d548`; A10 replays running). Owner: lead.

**Observed.** The P2-64 native lane-0 prefill takes 2,176.5 µs per layer in the EP combine `ncclDevKernel_Reduce_Sum_bf16_RING_LL` on TP rank 0. Frontier prices that call at 2,011.6 µs, the back-to-back median of G5h-32 `_01` (`eager_t6144_1_1_1`), which is 7.08 ms per forward low. Evidence: A10 report section 5.6; `h3o_operator_costs/a10/k19/prefill_compute/`.

**Cause.** In the forward, all ranks start the call together; lanes and TP ranks arrive within 3.5 µs and 21.4 µs (medians), and no other stream overlaps the Reduce. The root rank 0 ends last. In the benchmark's 10 back-to-back calls, ranks 1-7 start each call 450-720 µs before rank 0, because they finished the previous call earlier. Rank 0's kernel is therefore shorter, about 2,000 µs. On EP16 (G5h-16 `_02`) that head start is only 60-260 µs, and the benchmark is above native: 764.4 µs priced against 718.8 µs (lane 0) and 747.8 µs (lane 1). No single factor fits both topologies.

**Impact.** P2-64 prefill step −7.08 ms; through the FIFO queue, about −10.6 ms of TTFT. P2-32 step +0.7 to +2.0 ms. Decode-width calls use the same timing mode. Whether they are also affected is not checked; the G5h report recorded decode biases of −12 to +15 µs per call.

**Next (done 2026-10-30, below).** Measure every EP collective from an aligned start (a barrier on the group's stream before each timed call; per-rank end times from the kernel trace; price = the latest end in the timed lane) on EP32 and EP16. Then compare with the native kernels of every group before the consumer changes.

**Measured (2026-10-30).** G5h-32 isolated `_01` (RJob `exp-1030-234535-245357`, 32 H200, 8.01 GPU-hours; G5h 25.32 of 36) repeats the G5h-32 `_01` grid with one aligned call per repeat; its kernels equal the native P2-64 kernels in 9 of 9 groups. The native side is now the critical path of each layer over all EP ranks (`ep_critical.py`), not one rank's kernel. Per layer, at 6,144 rows:

| Call | Native critical path | Isolated | Back-to-back |
| --- | --- | --- | --- |
| EP32 combine (P2-64) | 2,171.2 µs | 2,172.0 µs | 2,011.1 µs |
| EP32 dispatch (P2-64) | about 1,048 µs | 1,043.0 µs | 1,031.9 µs |
| EP16 combine (P2-32) | 717.0 µs | 734.6 µs | 763.9 µs |
| EP16 dispatch (P2-32) | 541.8 µs | 560.2 µs | 801.8 µs |

The isolated mode is 43-93 µs above back-to-back for the decode-graph EP calls and 4-36 µs above for the TP calls, where back-to-back matches native. Details and the reading: [G5h report](test_report_2026-10-28_g5h_collective_kernel_comparison.md), section "I-008" and "Revised decision".

**Repair (B28).** (1) `data/profiling/network/h200_dgx/summary.csv` takes the EP rows of every eager scenario with a 6,144-token lane from the isolated runs (G5h-16 isolated `_01`, G5h-32 isolated `_01`); all other rows stay back-to-back. (2) `MeasuredCCBackend` prices an uneven EP dispatch from its exact measured `AllGatherV` row, as it prices the uneven combine. Expected change per 6,144-token prefill forward: P2-64 +5.50 ms, P2-32 −11.64 ms (`evidence/collective_kernels/expected_prices.txt`). With this repair alone both cases are expected to fail the TTFT gate (about −10.5% and −11.1%) until the ingress and the host part of the step are modeled.

**Uniform rows (2026-10-31).** B28 (merged as `ee29d548`) also replaced the rows where every lane carries 6,144 tokens. On EP16 the uniform (6,144, 6,144) prices fell: combine 768.5 → 454.3 µs, dispatch 801.7 → 567.5 µs. On EP32 the uniform (6,144 × 4) prices rose by 3.5% and 5.8%: combine 865.2 → 895.8 µs, dispatch 1,076.0 → 1,138.8 µs. The native P2-32 profile cannot test the EP16 rows: each of its three prefill forwards pairs one prefill lane with a dummy lane. The native P2-64 profile has nine uniform forwards (98-106): all four DP lanes are in prefill, and the EP kernels are the uniform `AllGather_RING_LL` and `ReduceScatter_Sum_bf16_RING_LL`, not the uneven `AllGatherV` and `Reduce`. Readings on TP rank 0 of dp1 and dp3 (the dp0 trace ends at forward 95):

- Dispatch: the 43 EP calls are exactly the 43 AllGather kernels above 700 µs; the next one is 545-751 µs. Their shortest per forward is 1,052-1,386 µs, close to both prices (1,076.0 and 1,138.8 µs). Their medians (1,132-4,040 µs) include the wait for later ranks.
- Combine: duration does not separate the 43 EP calls from the TP ReduceScatter calls (44-47 kernels above 700 µs per forward). The 43 longest have minima 809-849 µs and medians 834-852 µs (one forward 1,211 µs): 5-10% below the new price 895.8 µs and 1.5-6.5% below the old 865.2 µs.

One rank's kernel durations are not critical paths, so these readings check consistency only. For EP16 the support is an inference from scaling: the EP32 uniform call moves twice the data of the EP16 one, and the isolated rows scale by 1.97 (combine) and 2.01 (dispatch), while the old back-to-back rows scale by 1.13 and 1.34. That fits the old EP16 uniform rows carrying the back-to-back queueing that I-008 describes. Evidence: `h3o_operator_costs/b28/lead/uniform/` (`uniform_ep.py` → `uniform_ep_p2_64.txt`, `uniform_ep_ranks.py` → `uniform_ep_ranks.txt`, forward lists `forwards_p2_{32,64}.txt`).
