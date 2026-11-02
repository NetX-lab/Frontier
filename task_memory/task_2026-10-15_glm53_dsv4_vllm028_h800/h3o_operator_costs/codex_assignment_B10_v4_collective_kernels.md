# Task: the native DeepSeek-V4 per-forward collective kernel table and its comparator against the G5h benchmark kernels (stage 2 assignment B10, CPU-tested)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: Codex assignment B10 of stage 2 (G5h, [g5h/design.md](g5h/design.md) §1 and §6). |

## Goal

G5h measures the V4 collectives with the vLLM collective benchmark (assignment B9). Before G5h-32 spends 32 GPUs, the lead must know that the benchmark issues the same collective kernels as the native serving runs, for each kind of forward that native runs.

B10 builds two things, both CPU-only:

1. **Native table:** for each native forward kind, the collective kernels per forward, with their launch shape and duration spread, from the native P2 torch-profiler traces. B10 runs it now on the P2-32 and P2-64 traces and saves the result as evidence.
2. **Comparator:** for the same forward kinds, the collective kernels that the benchmark issued for the forward's calls, read from a G5h output directory, compared with the native table. B10 tests it on synthetic benchmark outputs; the lead runs it on the real G5h-16 output when that job ends.

## Approach and decisions

### Evidence (lead probe on the P2-32 traces)

Probe script `/data/ycfeng/tmp/g5h_b10_probe/kernel_probe.py`, output `kernel_probe.txt` beside it; lane pairing probe `pair_probe.py` / `pair_probe.txt`. Read them first. Native traces:

- P2-32 (each role TP8, DP2, EP16): `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces`
- P2-64 (each role TP8, DP4, EP32): `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/dsv4-h200-p2-64-profile-20261025-01/artifacts/torch_traces`

Layout `<role>/dp<N>/dp<N>_pp0_tp<k>_*.pt.trace.json.gz`, roles `prefill` and `decode`. B10 reads the TP-rank-0 trace of each lane, as the census does (`lane_traces` in `tests/analysis/v4_runtime_census.py`).

Observed collective kernels per forward (P2-32, TP-rank 0; the name prefix, then `g` = grid x):

| Forward kind | Kernels per forward |
| --- | --- |
| FULL or PIECEWISE, own rows 1 (decode, dummy) | 43 `ncclDevKernel_AllGather_RING_LL` g 4 (EP dispatch); 43 `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` g 2 (EP combine); 1 `vllm::cross_device_reduce_1stage` (embedding all-reduce); 43 `vllm::mnnvl_lamport_all_gather` g 4 (SP attention gather); 1 `vllm::mnnvl_lamport_all_gather` g 16 (final mHC gather); 43 `vllm::mnnvl_lamport_reduce_scatter_kernel` g 16 (SP reduce-scatter) |
| NONE, own rows 6144, other lane 1 (prefill) | 3 × 43 `ncclDevKernel_AllGatherV_RING_LL` g 4, 9, 16 (EP dispatch, uneven lanes); 44 `ncclDevKernel_AllGather_RING_LL` g 24 (SP gathers above 2 MiB fall back to NCCL); 43 `ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL` g 24 (SP reduce-scatter above 16 MiB falls back); 43 `ncclDevKernel_Reduce_Sum_bf16_RING_LL` g 16 (EP combine, uneven lanes); 1 `multimem_all_reduce_kernel` (embedding all-reduce, symmetric memory) |
| NONE, own rows 1, other lane 6144 (prefill dummy) | the same EP kernels as the row above; the custom and Lamport TP kernels of the 1-row case |
| real forwards (logits) | 1 `ncclDevKernel_AllGather_RING_LL` in the `post` segment (g 7 for 1 request, g 13 for 2) |

Further facts:

- The own-lane rows of a forward are grid x of its first `_fused_q_kv_rmsnorm_kernel` (as in `tests/comparison/calibration/dsv4_kernel_identity_check.py`, `_ROW_COUNT_KERNEL`): 1 or 2 in the graph forwards, 6144 or 1 in the eager ones. For real forwards it must equal the step's scheduled tokens after graph padding; check it.
- **Lane pairing.** The lanes' traces have different windows (P2-32 prefill: dp0 96 forwards, dp1 128), so forward indices do not pair. The host clocks of the lanes agree: paired dispatch starts differ by under 1 ms (`pair_probe.txt`), while graph forwards are about 8 ms apart and eager ones about 350 ms. Pair a forward with the other lane's forward of the same mode whose dispatch start is nearest, when the difference is below half the smaller neighbor gap of the first lane; each forward pairs at most once. A forward without a partner on every lane is left out of the lane-vector groups and counted. Report the median and maximum start difference of the accepted pairs.
- Native collective kernel durations include each rank's wait for its peers. The first collective of a forward absorbs the lanes' arrival skew (for example `cross_device_reduce_1stage` with a median of 8.0 ms in decode FULL), so native durations are upper bounds and their minimum is the closest native estimate of the collective's own time.

### Native table: new `tests/analysis/v4_collective_kernels.py`

- **Kernel classification table.** One module-level table maps a kernel name prefix to a collective family: `ncclDevKernel_` → `nccl`; `void vllm::cross_device_reduce_` → `custom_all_reduce`; `void vllm::mnnvl_lamport_` → `mnnvl`; `void (anonymous namespace)::multimem_` → `symm_mem`. A kernel is a collective kernel when its name starts with a table prefix. Add a prefix only from a kernel seen in the traces.
- **Review list.** Over all forwards, list the kernel names outside the table that match `nccl|lamport|cross_device|multimem|symm|all_?reduce|all_?gather|reduce_?scatter|one_shot|two_shot` (case-insensitive), with counts. The lead reads this list once; the table changes only after that review.
- **Per forward:** reuse the census segmentation. Add one public function to `v4_runtime_census.py` that returns, per forward, its record, its segment bounds and its kernel events per segment (`prepare`, `dispatch`, `post`), and rebuild `dispatch_kernel_multisets` on it. The census JSON and text outputs and `dispatch_kernel_multisets` results must not change (Acceptance 2). From it, per forward:
  - mode, step class, `real`, the step's requests and scheduled tokens;
  - own rows (above);
  - the collective kernels of the `dispatch` and `post` segments as a multiset of `(segment, name, grid, block)`, and their durations per key.
- **Groups.** Key `(role, mode, real, own rows, lane vector)`, where the lane vector is the tuple of own rows of the paired forwards in lane order, and `requests` (the logits rows, below) for real forwards. Per group: the forward count, the number of distinct kernel multisets with each multiset and its count, and per kernel key the count per forward and the duration min, p10, median and p90 over the group's forwards.
- **CLI:** `python tests/analysis/v4_collective_kernels.py native --trace-dir <torch_traces> --output <json> --summary <txt>`. Read one trace at a time (the census pattern).
- **Run now** on P2-32 and P2-64. Outputs: `h3o_operator_costs/evidence/collective_kernels/p2_32_native.json`, `p2_32_native.txt`, `p2_64_native.json`, `p2_64_native.txt`, and the run logs `p2_32_native.log`, `p2_64_native.log` (command, wall time, peak memory).

### Comparator: in the same module

- **CLI:** `python tests/analysis/v4_collective_kernels.py compare --native <native json> --benchmark <G5h output dir> [--benchmark <dir> ...] --output <json> --summary <txt>`.
- **Benchmark side.** A G5h output directory holds `plan.json` (its `config` gives the stage, the parallel sizes, the grid and `settings.calls_per_repeat`), `kernels.jsonl` (one record per `(scenario, call)` with `per_rank`) and `traces/<scenario>/<call>/rank<r>.json` (Chrome traces of the untimed kernel pass, one `run()` of `calls_per_repeat` calls). Resolve the trace paths relative to the given directory; the paths inside `kernels.jsonl` are worker paths.
  - Per `(scenario, call, rank)`: the multiset of `(name, grid, block)` over the trace's kernel events divided by `calls_per_repeat`. A count that does not divide evenly is an error for that case, reported, not rounded. Per key, the per-call kernel durations (median, min, max).
  - Also record per case the device activities outside the classification table (copies, pads, `elementwise_kernel`, memcpy and memset), as auxiliary work. They are reported, not compared: the benchmark times them, and native runs them too (for example the `F.pad` before the reduce-scatter and the eager custom all-reduce input copy).
- **Rank of a lane.** The benchmark rank of lane `l`'s TP-rank 0 follows the rank layout that `CollectiveRunner` enforces (`_check_layout` and the groups recorded in `plan.json`); use that, do not derive a second layout.
- **Expected kernels of a native group** (lane `l`, own rows `T`, lane vector `v`), from the V4 plan of B9 (`frontier/profiling/collectives/vllm_dsv4_forward_collectives.py`: call names, `calls_per_forward`, scenario names):
  - TP calls (`embedding_all_reduce`, `sp_attention_all_gather`, `sp_attention_reduce_scatter`, `sp_final_mhc_all_gather`): FULL and PIECEWISE use `graph_t{T}`; NONE uses the uniform eager scenario `eager_t{T}_{T}…` (TP calls are planned only on uniform vectors and depend on the own lane only).
  - EP calls (`moe_dispatch_all_gatherv`, `moe_combine_reduce_scatterv`): FULL and PIECEWISE use `graph_t{T}`; NONE uses `eager_t{v}`.
  - `logits_all_gather` for real forwards: `logits_b{requests}`, in the `post` segment.
  - Expected multiset = Σ over calls of `calls_per_forward` × that call's per-call multiset at lane `l`'s rank.
  - With several `--benchmark` directories, a call is taken from the directory that planned it; EP calls only from a directory whose DP size equals the native lane count. This lets the G5h-32 comparison take the TP calls from G5h-16, because G5h-32 plans only the EP calls.
  - A group whose scenario is not in any directory's plan is listed as unmatched with the missing `(scenario, call)`.
- **Verdict per matched group:**
  - `kernels`: `equal` when every distinct native multiset of the group equals the expected multiset (segment ignored for the comparison, kept in the report); else `different`, with the per-key count difference.
  - `durations`, per kernel key present on both sides: `within` when the benchmark per-call median lies in the native `[min, p90]`; else `below` or `above`. Durations do not change the `kernels` verdict; they are reported for the lead.
- **Summary text:** one block per group with the forward count, both multisets side by side (count, family, name prefix up to 60 characters, grid, block), the verdicts, the unmatched groups, the auxiliary work and the review list.

### Unit tests: new `tests/unit/test_v4_collective_kernels.py`

Synthetic inputs in `tmp_path`, written with the field layout the census reads (gzip Chrome traces with `user_annotation` ranges, launch events and kernel events joined by `correlation`, as in the existing census tests if any; otherwise build the minimum the census parser needs) and a synthetic benchmark directory (`plan.json`, `kernels.jsonl`, `traces/…/rank<r>.json`). Cover:

- classification and the review list;
- own rows from `_fused_q_kv_rmsnorm_kernel`;
- lane pairing: nearest start accepted, a forward outside the other lane's window left out and counted, a candidate beyond half the gap rejected;
- division by `calls_per_repeat`, and the uneven-count error;
- the expected multiset for a graph group, a mixed eager group (TP calls from the uniform scenario, EP calls from the mixed one) and a real group with logits;
- the multi-directory selection (EP calls only from the matching DP size);
- `equal`, `different` and unmatched verdicts; `within`, `below`, `above`.

## Out of scope

- Any change to the benchmark, the V4 planner, the GLM path, the census outputs or the simulator.
- Running the comparator on real G5h data (the lead does this after G5h-16).
- Any GPU, platform or network action.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028` (Git worktree). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Concurrent work:** assignment B9 may still be writing in `frontier/profiling/collectives/`, `tests/unit/test_vllm_dsv4_forward_collectives.py` and `h3o_operator_costs/g5h/launch/`. Do not edit those; read `vllm_dsv4_forward_collectives.py` for the call names, `calls_per_forward` and the scenario names. If it is not in its final form yet, take the names from [codex_assignment_B9_dsv4_forward_collectives.md](codex_assignment_B9_dsv4_forward_collectives.md) and say so in the report.
- **Writable paths:**
  - `tests/analysis/v4_collective_kernels.py` (new) and `tests/analysis/v4_runtime_census.py` (the shared function only);
  - `tests/unit/test_v4_collective_kernels.py` (new);
  - `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/evidence/collective_kernels/` (new; the native outputs above).
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b10/`.
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE` ([../issues.md](../issues.md) I-001).
  - No GPU, network, job submission or package installs. No deletions or moves.
  - Python 3.10. Keep modules at or below 2,000 lines.
  - No magic literals in code paths: model and parallel sizes come from `plan.json` and the V4 plan; the classification prefixes are the one table above. No guards for states the code cannot reach.
  - Memory: `ulimit -v 31457280` and a `timeout` on every command. The P2 traces are 21 MB gzip each; read one at a time.
- **Read-only data:** the native traces above; the probe files; `tests/analysis/v4_runtime_census.py` (before your change); `tests/comparison/calibration/dsv4_kernel_identity_check.py`; `frontier/profiling/collectives/vllm_collective_benchmark.py` and `vllm_dsv4_forward_collectives.py`; [g5h/design.md](g5h/design.md); `evidence/runtime_census/p2_32.txt`, `p2_64.txt`.
- **Environment:** `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6); `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`; `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`.

## Acceptance

1. **Native tables.** The four output files exist for P2-32 and P2-64. For P2-32 the groups include the kinds of the evidence table above with the same kernels and counts; any difference is explained with the forward index and trace path. Report the group list (key, forwards, distinct multisets), the pairing statistics (accepted pairs, left-out forwards per lane, median and maximum start difference) and the review list for both runs.
2. **Census unchanged.** Rerun `tests/analysis/v4_runtime_census.py` on the P2-32 traces into scratch; its `--summary` text is byte-identical (`cmp`) to `evidence/runtime_census/p2_32.txt`, and its JSON equal to a run of the pre-change module on the same input (keep the pre-change copy in scratch). The existing tests that use `dispatch_kernel_multisets` pass (find them with `grep -rln dispatch_kernel_multisets tests`).
3. **Unit tests.** `tests/unit/test_v4_collective_kernels.py` passes; report the count.
4. **Comparator dry run.** Run `compare` with the P2-32 native JSON and a synthetic benchmark directory built from the native multisets (so every matched group is `equal`), then with one kernel removed from one case (one group `different`); report both summaries' verdict lines.
5. **Hygiene:** line counts; no unused imports (`/data/ycfeng/tmp/unused_imports.py`); a cold import of the new module exits 0.

## Expected return

- Report language: English.
- Changed and new paths, each with a summary.
- The Acceptance 1 group lists, pairing statistics and review lists; the Acceptance 2 `cmp` result; the test count; the Acceptance 4 verdict lines.
- Scratch paths, deviations and their reasons, and limits.
- If the native traces contradict a decision above (for example a lane pairing that is not unique, or collective kernels in the `prepare` segment), stop that part and report the evidence and options instead of choosing a different design.
