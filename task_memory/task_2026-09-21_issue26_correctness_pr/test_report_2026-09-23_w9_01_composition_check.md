# Test report — W9-01 merge-forward composition check (2026-09-23)

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-23 | Created. K1–K4 of plan §18.14 on the merged tree `03d5f24`. |

## Environment

| Item | Value |
| --- | --- |
| Host | `kun-workspace-vgen2` |
| Worktree | `/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr`, branch `fix/issue26-correctness-pr` |
| Merge | `dd9b8d9` merges `origin/main` `4ab1964` (PR 36, squash) into `8315d9b`; no conflict, no overlapping file |
| Revision under test | `03d5f24` (the merge plus the drain-reader fix below) |
| Interpreter | `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6; distributions SHA-256 `ecd50ea8…902620` |
| Environment variables | `PYTHONPATH=<worktree>`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `FRONTIER_TMP_ROOT=/data/ycfeng/tmp` |
| Matrix root | `/data/ycfeng/tmp/stage_admission_ordering` (sets `c-merged`, `c-pr35`; PR 36 sets `base`, `after-r2`) |
| Untracked, not ours | `outputs/metrics/meta_llama_llama_2_7b_hf/` (recorded in each set's provenance) |

## Commands

```bash
# c-merged: the merged tree, clean
python -m tests.e2e.stage_admission_matrix run --set c-merged --group G3b --group G9 --group G10 --jobs 16
# c-pr35: the same tree with the rule file of 1f694f7, restored afterwards with git checkout
git show 1f694f7:frontier/scheduler/replica_stage_scheduler/stage_execution_context.py > frontier/scheduler/replica_stage_scheduler/stage_execution_context.py
python -m tests.e2e.stage_admission_matrix run --set c-pr35 --group G3b --group G9 --group G10 --jobs 16
git checkout -- frontier/scheduler/replica_stage_scheduler/stage_execution_context.py
python -m tests.e2e.stage_admission_matrix compare --before c-pr35 --after c-merged --output <scratch>/compare_c-pr35_c-merged.json
python task_memory/.../w9_01_stage_admission_ordering/composition_check.py c-pr35 c-merged after-r2 <compare json> <output json>
# K4: both trees, full suites with JUnit output
w9_01_stage_admission_ordering/composition_run_suites.sh <worktree> <out>          # merged tree
w9_01_stage_admission_ordering/composition_run_suites.sh <git archive 8315d9b> <out>  # pre-merge tree
python w9_01_stage_admission_ordering/composition_compare_junit.py <before.xml> <after.xml> <output>
```

`c-pr35` equals this branch's source before the merge. The merge brings one
source file, the rule file, so swapping it back reproduces `8315d9b`'s
`frontier/` tree. After the run the file equals `origin/main` again.

## Harness fix found on the way (`03d5f24`)

The first `c-pr35` run classified two drained cells, `G9-moe-dp2-pp3-n8` and
`G9-moe-dp4-pp3-n8`, as `other_failure`. The drain reader raised
`KeyError: 'batches'`: `enter_layer_sync` pops the fields of a room it
dispatches but keeps the room's key, and the reader indexed every room. The
same code is on `main` (`sync_entry.py:140`, `:277`); it was first reached
here, because W2 lets Poisson PDD cells drain after some dispatches. The
reader now skips rooms without batches. Both cells then classify as
`admission_deadlock` (probe in an exported tree, then the full reruns below).
Both sets were rerun at `03d5f24` so that they share one harness revision.

## Results

| Id | Check | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| K1 | `c-merged` outcomes and conservation, 51 cases | all succeed; requests and prefill/decode tokens equal the workload | 51/51 success (G3b 12, G9 11, G10 28); conservation holds in all 51 | PASS |
| K2 | `compare --before c-pr35 --after c-merged` | 0 STOP; U identical; L complete and conserved; each EXPLAIN only moves start times | U 15 PASS; L 16 PASS; T 12 PASS (identical) and 8 EXPLAIN; 0 STOP. EXPLAIN: 7 have the same ordered batches and component durations on every (cluster, replica, stage, lane); 1 (`G10-dense-dp2-pp3-n8`) does not — see below | PASS under the amended rule |
| K3 | lanes used by Poisson cells with `attn_dp > 1` | every MONOLITHIC/PREFILL stage uses all lanes in `c-merged` | 22/22 cells use all lanes on every stage; `after-r2` (main plus the rule, no W2) uses lane 0 only in all 22 | PASS |
| K4 | unit and integration suites, merged vs pre-merge tree | 0 regressions; targeted modules pass | unit: 0 regressions, 0 new failures, 0 skip changes; integration: same, 3 new tests pass; 25 targeted modules, 487 tests, all pass | PASS |

Path L (16 cells that drain under the pre-merge rule): G3b 6, G9 burst 4,
G10 MoE burst 4, and two cells that do not drain on `main`,
`G9-moe-dp2-pp3-n8` and `G9-moe-dp4-pp3-n8`. W2 spreads their Poisson
arrivals over the lanes, which makes the W9-01 pattern reachable there. The
drained cluster is the unified DECODE (`simulation_time` 0.4161 s in both
cells). Every L cell completes on the merged tree with requests and tokens
conserved.

### K2: the one EXPLAIN cell whose batches differ

`G10-dense-dp2-pp3-n8` (online Poisson, dense, `attn_dp=2`, PP=3):

| Item | Value |
| --- | --- |
| Identical lanes | lane 0 on stages 0, 1 and 2 |
| First divergence of the time-ordered ledger (position 12) | MONOLITHIC stage 0, lane 1, batch of request `1`, same duration and components, starts at 0.16027 s on the merged tree against 0.18366 s before |
| Why it waited before | request `1` left stage 2 at 0.16027 s and stage 0 of lane 1 was idle, but lane 0's queued ticket headed the FIFO while lane 0's stage 0 was busy (0.1477–0.1837 s): the W9-01 coupling |
| Consequence | lane 1 then forms later batches at different times, so Poisson arrivals `5` and `7` share batches (36 → 33 lane-1 ledger rows) |

Lane 0 is byte-for-byte the same, and the first difference is the rule's
intended effect: an identical ready batch admitted earlier. The pre-fixed K2
wording ("only start times differ") holds for offline and burst cells, where
batch contents are fixed at t=0, but not for an online cell. There, a lane
admitted earlier meets later arrivals at a different point in its schedule.
The check therefore accepts a batch difference when the first divergence is
the same batch on the same stage and lane, admitted earlier. This amendment
was made after measuring; it is recorded in plan §18.14 with this cell.

### K4 detail

| Suite | Pre-merge `8315d9b` (export) | Merged `03d5f24` (worktree) |
| --- | --- | --- |
| `tests/unit` | 172 failed, 3706 passed, 51 skipped, 10 errors | 84 failed, 3814 passed, 50 skipped, 10 errors |
| `tests/integration` | 16 passed, 22 skipped, 5 errors | 19 passed, 22 skipped, 5 errors |

The 88 extra unit failures on the pre-merge side all come from the export,
which has no `.git`: `git rev-parse HEAD` and `git ls-files` fail in
`test_sim_walltime_scaling_sweep` (40), `test_moe_ep_non_dummy_matrix` (30),
`test_sim_walltime_scaling_run_case` (14), `test_moe_ep_h800_profile_backfill_script`
(3) and `test_pd_disaggregation_naming_guard` (1). They pass in the worktree.
The one skip difference is `test_collective_sim_zero_payload`: it is skipped
at module level in the export, which lacks the built collective-sim binary,
and its 4 tests run in the worktree. The 20 unit tests and 3 integration tests
that exist only on the merged side are PR 36's and the 4 zero-payload tests;
all pass. The 84 common failures and the 10 collection errors (no `torch` or
`matplotlib`) are the same set on both sides. The 5 integration errors are
the absent PD-AF Reference checkout.

Targeted modules (all pass on the merged tree): PR 36's
`test_stage_execution_context`, `test_shared_forward_group_admission`,
`test_mixed_layer_decode_ffn_scheduling`, `test_stage_admission_pp_tools` and
`test_stage_admission_pipeline_lanes`; W2 `test_cluster_scheduler_dp_lanes`;
W3 `test_monolithic_mixed_forward_sync` and `test_monolithic_mixed_forward_runtime`;
the forward-sync regression set of the W3 report; Step 9's
`test_vllm_dp_load_balancer`, `test_dp_placement_reference_loop` and
`test_vllm_dp_placement_runtime`.

## Evidence

- `w9_01_stage_admission_ordering/composition_evidence/`:
  - `compare_c-pr35_c-merged.json` (harness paths);
  - `composition_check.json` (K1, K2 EXPLAIN detail, K3 lanes for both sets);
  - `unit_compare.json`, `integration_compare.json`.
- Scripts: `composition_check.py`, `composition_compare_junit.py`,
  `composition_run_suites.sh`.
- Raw sets and suite logs: `/data/ycfeng/tmp/stage_admission_ordering/{c-merged,c-pr35,composition}`.

## Limits

- The Frontier side uses the dummy predictor; the check concerns admission,
  placement and completion, not latency.
- The pre-merge suite baseline ran from a `git archive` export. Its
  git-dependent failures are attributed from their messages, not by a rerun
  in a checkout.
- `c-pr35` reproduces the pre-merge `frontier/` tree by swapping the rule file
  in the worktree. Its provenance records the modified file.
