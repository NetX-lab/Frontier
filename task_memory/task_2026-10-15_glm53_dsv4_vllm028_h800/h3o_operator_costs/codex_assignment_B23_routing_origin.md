# Task: let the V4 attention loader accept a table merged from two packages whose routing descriptors differ only in the package path (stage 4 assignment B23, CPU only)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-29 | Created after the merged G1h-02 + supplement attention table failed the loader on `moe_routing` (requirements item 40, D-ROUTING-ORIGIN-PROVENANCE). |

## Goal

The G1h supplement GPU job (`exp-1030-021110-248027`, H200, succeeded) measured the writer-count windows that G1h-02 lacked. The lead's CPU steps after the job (`/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/g1h_s01_post.sh`, log `g1h_s01_post.log` beside it) passed engine identity, measured coverage of both packages (1,473 of 1,473 windows), the package difference, and the scratch promotion. The loader check then failed:

```text
frontier.attention.deepseek_v4.DeepSeekV4CoverageError: V4 replay .../g1h_s01_post/promote/output/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv:
V4 balanced replay attention missing/ambiguous moe_routing
```

`validate_balanced_routing` (`frontier/execution_time_predictor/dsv4_predictor.py:78-94`) requires one distinct `moe_routing` string. The merged table has two (4,945 G1h-02 rows and 2,160 supplement rows). The lead compared every identity column of the two row sets: only `moe_routing` differs, and inside it only the top-level `origin`, the host path of the package's `balanced_routing.py`:

- G1h-02: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/.../h3o_operator_costs/g1h/package_02/balanced_routing.py`
- supplement: `/data/ycfeng/Frontier/.worktrees/g1h-supplement-build/task_memory/.../h3o_operator_costs/g1h_supplement/package_01/balanced_routing.py`

The routing code itself is identified by `sha256` (equal, and `routing_columns` checks it against `BALANCED_ROUTING_SHA256`), the routers, the runners and the native vLLM source digests, which are all equal. `runtime_fingerprint` is one value (`5237f2bd076fe132`) for both.

A lead diagnostic that compared the descriptors without the top-level `origin` loaded the merged table: 7,105 rows, 1 fingerprint, revision `a689ed05`, 554 trained models; the measured replay closure on the 19:30 census copies gave 0 misses for P2-32 (480 before) and for P2-64.

## Decision (requirements item 40, D-ROUTING-ORIGIN-PROVENANCE)

The top-level `origin` of the routing descriptor is provenance: where the package copy of `balanced_routing.py` lived on the host. It is not part of the routing identity. The loader accepts several `moe_routing` values when each passes `routing_columns` and all are equal after the top-level `origin` is removed. Everything else stays as it is:

- `routing_runtime_path` and `routing_strategy_sha256` stay single-valued and equal to `balanced_topk` and `BALANCED_ROUTING_SHA256`;
- `origin` must still be present and end with `/balanced_routing.py` in every value (`routing_columns` already checks this);
- the `origin` fields inside `native_sources` stay part of the identity;
- the reducer rule that one package's engines share one routing descriptor (`dsv4_attention_rows.py:224`, `dsv4_replay_rows.py:130`) does not change;
- the `moe_routing` column of the table is not rewritten; each row keeps the provenance it was measured with.

## Work

1. Change `validate_balanced_routing` for the decision. Keep the existing error messages for the existing failure modes; a set of descriptors that still disagree after removing `origin` raises the existing "missing/ambiguous moe_routing" error.
2. Tests in `tests/unit/test_dsv4_predictor.py`: two descriptors that differ only in the top-level `origin` load; descriptors that differ in a native source digest, in a native source `origin`, or in the routers are rejected; a missing value is rejected; one descriptor loads as before.
3. Extend the B22 promotion test (`tests/unit/test_v4_attention_promotion.py`) so the synthetic supplement carries a different package path in its engine's `moe_routing.origin`, and check that `load_dsv4_attention_dataframe` loads the merged table with one runtime fingerprint. This is the case the B22 synthetic merge did not cover.
4. Run the loader and the measured replay closure on the real merged scratch table, read-only:
   - table `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/g1h_s01_post/promote/output/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv`;
   - `tests/analysis/v4_attention_replay_coverage.py --table <that table> --supplement-plan task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g1h_supplement/package_plan.json --census /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/census_snapshots/1930/p2_32 /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/census_snapshots/1930/p2_64 --output <scratch>/replay_measured.json`.

## Acceptance

1. The merged table loads with `load_dsv4_attention_dataframe` (replica: `attn_tensor_parallel_size=8`, `device="h200"`, `kv_cache_dtype="fp8_ds_mla"`, `moe_gating_routing_runtime_path="balanced_topk"`; block size 256; `MeasurementType.KERNEL_ONLY`): 7,105 rows, 1 runtime fingerprint, revision `a689ed05f462eb269b9dffe640da151e6614a5ef`.
2. The committed table `data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv` loads as before: the same rows, and every model trained from it has equal `_frontier_dsv4_workload_group_ranges` before and after the change.
3. Replay closure on the real merged table: `recomputed_current` misses 0 for both census copies (P2-32 17,000 and P2-64 16,000 requests).
4. Tests: `tests/unit/test_dsv4_predictor.py`, `tests/unit/test_v4_attention_promotion.py`, `tests/unit/test_v4_attention_supplement.py`, `tests/unit/test_v4_attention_heldout.py`, `tests/unit/test_dsv4_attention_runtime.py`, `tests/unit/test_dsv4_operator_consumer.py`, `tests/unit/test_v4_prediction_memo.py`, `tests/unit/test_balanced_routing.py` and `tests/unit/test_dsv4_replay_rows.py` pass; report the counts.
5. Hygiene: line counts of the changed files (each at or below 2,000); no unused imports (`/data/ycfeng/tmp/unused_imports.py`); `git diff --check` clean.

## Scope and context

- **Working root:** `/data/ycfeng/Frontier/.worktrees/b23-routing-origin` (Git worktree, branch `b23-routing-origin` from main `8b9e2016`). Leave the untracked `.cortexkit/`, `.pi/` and `remaining_measurements/` alone.
- **Task records (read in the main tree `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`):** `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/requirements.md` (items 37, 39, 40), `test_report_2026-10-29_b22_g1h_writer_supplement.md`, `h3o_operator_costs/codex_assignment_B22_g1h_writer_supplement.md`.
- **Writable paths:** `frontier/execution_time_predictor/dsv4_predictor.py` (`validate_balanced_routing` only), `tests/unit/test_dsv4_predictor.py`, `tests/unit/test_v4_attention_promotion.py`.
- **Read-only inputs:** the scratch merged table and census copies above; `/data/ycfeng/tmp/glm53_dsv4/h200/g1h_02/` and `/data/ycfeng/tmp/glm53_dsv4/h200/g1h_s01/download/` (raw records only; do not read the `traces/` directories, about 27 GB together).
- **Scratch:** `/data/ycfeng/tmp/codex-implement/<request-id>/b23/`. The `/data/ycfeng` disk has about 8 GB free: keep scratch output small (no table copies above a few hundred MB).
- **Restrictions:**
  - No Git commands that change state; the lead commits. `git diff`, `git status`, `git show` and `git rev-parse` are fine. Never export `GIT_DIR` or `GIT_WORK_TREE`.
  - No GPU, network, job submission or package installs. No deletions or moves. Do not write under `data/profiling/`.
  - Do not stop, signal or read with heavy tools the running simulations under `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/`; no recursive `grep` or `find` over `/data/ycfeng/tmp/glm53_dsv4/`.
  - Python 3.10. No magic literals in code paths. No guards for states the code cannot reach. No second way to do what the code already does: reuse `routing_columns` and `canonical_json`.
  - Memory: `ulimit -v 31457280` and a `timeout` on every command.
- **Environment:** `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6, no torch or vLLM); `PYTHONPATH=$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis`; `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, BLAS/OMP/MKL threads 1.

## Expected return

- Report language: English.
- Changed paths, each with a summary of the change and the diff of `validate_balanced_routing`.
- Acceptance 1-5 with the commands, the expected and actual values, and the counts.
- Scratch paths, deviations and their reasons, and limits.
