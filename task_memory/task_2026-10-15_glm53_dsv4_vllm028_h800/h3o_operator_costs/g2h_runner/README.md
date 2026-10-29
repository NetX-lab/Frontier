# G2h V4 runner operators

CPU plan and builder for B13s, correcting the missing sampling annotations and the second-bootstrap failure observed in G2h-01 (`exp-1029-113750-923871`). The G1h-02 replay grid and the B13r direct grid remain unchanged. Claude reviews/commits these files, replaces `source_revision` with that commit, builds a fresh package, then submits its jobs. Keep `package_01` and `package_02` intact. CPU implementation does not build or submit a package.

`package_02` ran G2h-02. Its replay job is the path A result. Its block job (`exp-1029-135037-016686`) failed in 5.8 s: the block entry runs the prefill entry `profile_dsv4_prefill.py`, which requires `VLLM_USE_V2_MODEL_RUNNER=1` (line 20). The bridge driver sets that variable for its replay children, but nothing set it for the block job. The block job command now sets it with `env`, as the accepted base package's prefill job does. `package_03` carries this fix; only its block job is submitted, because the replay job of `package_02` already ran.

## Plan check (CPU only)

From the repository root, with Python 3.10:

```bash
ulimit -v 31457280
export PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis"
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
timeout 120s /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h_runner/prepare_package.py --check-plan
```

`--dry_run` is an alias. This checks the exact planned coordinates/case IDs against `expected_rows.json`, plans the native-layer engine using the existing attention builder, and cold imports both worker entry paths with fasteners, sklearn, torch, vLLM and cuda blocked. It lists both jobs, their tags, output directories and one bootstrap command per job, and creates no package. Direct grid-only check:

```bash
timeout 120s /data/ycfeng/envs/frontier-py310/bin/python -m frontier.profiling.model_runner.dsv4_block_operators --grid task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h_runner/block_operator_grid.json --model_config_path data/config/models/deepseek-ai__DeepSeek-V4-Flash.json --dry_run
```

The builder reuses `p1h_bridge/prepare_package.py`'s committed source snapshot, accepted bootstrap, base-package verification, model assets, image, CUDA compatibility path and mount. No separate engine builder or vLLM method implementation is introduced. G1h `plan_args`, pilot/coverage files and the three case lists are reused verbatim.

## Separate worker jobs and outputs

Each job runs in its own container on one H200, charged to step_main, non-preemptible. The accepted bootstrap checks the installed base before applying its overlay; running it a second time in one container caused path B's G2h-01 base-hash mismatch. The bootstrap and base package are unchanged. Both jobs collect their local `/tmp/glm_producers/out` tree into distinct destinations:

| Job / phase | Tag | Output directory |
| --- | --- | --- |
| `G2H_V4_RUNNER_REPLAY` / `G2H_RUNNER_REPLAY` | `v4-h200-g2h-runner-20261029-03-replay` (`package_02`: `-02-replay`, `g2h_runner_02/replay`) | `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_runner_operators/g2h_runner_03/replay` |
| `G2H_V4_BLOCK_OPERATORS` / `G2H_BLOCK_OPERATORS` | `v4-h200-g2h-runner-20261029-03-block` | `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_runner_operators/g2h_runner_03/block` |

Replay uses the same 989 IDs (e0 686, e1 296, e2 7). Direct calls use the native 43-layer allocation and grid-specified eight experts: 33 zero and 164 write coordinates per method, 394 rows total. Local outputs are `G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/{raw,traces}` and `G2H_BLOCK_OPERATORS/` respectively.

After committing and pinning the revision, the lead builds and runs these commands from this directory (not executed during CPU implementation):

```bash
timeout 600s /data/ycfeng/envs/frontier-py310/bin/python prepare_package.py
timeout 7200s bash package_03/launch/launch_glm_producers.sh package_03/jobs/G2H_V4_BLOCK_OPERATORS.json
```

The generated jobs run the following worker commands in **separate containers**:

```bash
python -I package_03/native_dummy_bootstrap.py --config package_03/bridge_config.json
python package_03/dsv4_graph_bridge_check.py --config package_03/bridge_config.json --report
```

```bash
env VLLM_USE_V2_MODEL_RUNNER=1 python -I package_03/native_dummy_bootstrap.py --config package_03/block_config.json
```

Replay produces its existing raw JSONL, captures, traces and attention CSVs plus adjacent `dsv4_raw_eN.ownership.json` and `dsv4_raw_eN.runner.csv`. The ownership file classifies **every recorded-step kernel, memcpy and memset** as attention scope, step scope, model forward or unowned. Hold kernels are calibration and excluded from the census. Each step has counts and each observed kind (FULL, PIECEWISE, first tail, multi-tail, dummy, prefill) has unowned names, grids and counts. There is no name-based ownership inference. Model-forward boundaries use CPU launch correlations within `frontier_v4_model_forward|<step>` ranges; graph nodes without attention ownership are model-forward work.

Step owners use reserved integer layer **-1** and a top-level scope; physical attention layers remain >=0. `dsv4_step_scopes.is_step_scope` validates both fields. `dsv4_attention_rows.build_dsv4_rows` filters these call rows; `dsv4_replay_rows` continues to read existing attention layer calls through the gap reducer; the bridge reads capture owners/interval kinds without layer pairing assumptions. The gap reducer excludes step owners from attention dependency chains and preserves their previous outside-work/graph classes and hold diagnostics. Thus new ownership does not reclassify preparation as attention eager work.

`sample()` has the `dsv4_sampling` owner, and `model.compute_logits` inside it is the child scope `dsv4_compute_logits` (diagnostic). Graph-replay ranges may nest: the trace reader gives each kernel to the innermost range, so the row gather and sampler stay with sampling and the LM head goes to the logits scope. A logits call outside `sample()` fails the parent check. Ordinary nested step scope calls fail the same parent check as attention scopes. The execute range opens before `execute_model` and closes after `sample_tokens`, including AsyncOutput's unpriced output copy. `_wrap_execute_model` retains `state.active` while that existing range is open, and `close_step_range` calls the existing `end_step` once before clearing it. GLM DSA and eager V4 open no range and retain their execute-return lifetime. A failed execute or sample call ends the engine run, so no failure path continues recording.

Graph replay selects only `KERNEL_ONLY` (`frontier/profiling/attention/dsa_vllm_runner.py`, `_TIMING_MEASUREMENTS`); CUDA_EVENT sampling handling is unnecessary. FULL, PIECEWISE, first-tail, multi-tail and prefill all return `None` at pinned `vllm/v1/worker/gpu/model_runner.py:1703`, then the engine samples at `vllm/v1/engine/core.py:602-603`. **Dummy steps do not sample**: `gpu_worker.py:1210-1212` calls `_dummy_run`, which executes at `gpu/model_runner.py:695`, consumes its state at 709-715, gathers hidden rows at 767 and returns at 768. The existing profiler `run_dummy_step` closes that range after the native call. No synthetic sampling call is added to dummy steps.

Direct output includes `layout.json`, `block_operators.csv`, all Chrome traces and adjacent `.trace.intervals.json` files with kernel and memcpy intervals kept separately. Rows use **`dsv4_model_runner.v1`**, constructed in `frontier/profiling/model_runner/dsv4_runner_rows.py`: existing DSA `time_stats.<operator>` statistics in ms, separate `kernel_stats`/`memcpy_stats`, coordinates, method, repeat count and engine layout JSON. `cuda_event`'s headline sample spans the native call; `record_function`'s headline sample sums owned kernel durations. The future operator loader reads these producer CSVs; no current consumer, training path or execution flag is activated here. Column `row_role` states each row's use: path A input preparation and sampling rows and all path B rows are `cost`; path A write rows are `validation` (kernels and shapes against path B); path A zeroing and full TP1 compute_logits rows are `diagnostic`.

## Direct grids and timing

All coordinates, expert/layer counts, warm-up (3), repeats (20), hold (20000 µs), first chunk (6144) and expected values live in `block_operator_grid.json`. Zeroing uses `n=1..32,2520`. Writes use `w_c=0..32` and `r=0..4`, omit `(0,0)`, and launch one fused call after native staging. Table entries come from actual group block sizes `[256,64,64,4,8]`, checked against `[24,96,96,1536,768]`. At most 11 distinct request rows are needed by the write grid; 128 are planned. Consecutive free IDs start at 1. The run checks actual free-block capacity, request rows/table widths, native B4 group members, **139 segments / 10 chunks**, and unsplit table entries before timing.

Each measured call has the replay's calibrated host-ahead device hold, both events and trace ranges, and a post-call synchronization. Staging/restoration happens before the measured range. Each positive grid call must own one kernel; missing ownership or late launches fail with a coordinate/repeat diagnostic. Kernel and HtoD copy times are not merged into a policy decision. See `native_kernel_notes.md` for pinned source lines, allocation caveats, hash/dummy-finalizer evidence and native/G1h kernel tables. Exact segment derivation needs allocated native tensor views; the CPU plan does not fabricate aliases.

## Lead check after the GPU run: four gaps against G1h-02

Set `G2H_RESULTS` to the downloaded replay job output tree containing `G2H_RUNNER_REPLAY`, and `G2H_GAPS` to a writable report directory. The block job has its own download tree. Run the existing reducer:

```bash
timeout 600s python -m frontier.profiling.kernel_gap.dsv4_replay_gaps --raw_files "$G2H_RESULTS/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e0.jsonl" "$G2H_RESULTS/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e1.jsonl" "$G2H_RESULTS/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e2.jsonl" --output_dir "$G2H_GAPS" --device h200 --model deepseek-ai/DeepSeek-V4-Flash --diagnostics_path "$G2H_GAPS/diagnostics.json"
timeout 60s python - <<'PY_COMPARE'
import json, os
from pathlib import Path
reference = {
    "cuda_graph": (0.49773414267747085, 0.4801221185543367, 169232),
    "cuda_graph_piecewise": (0.6249537992538311, 0.6106290602187813, 289730),
    "eager": (0.5446049595929223, 0.5065229453923222, 30243),
    "eager_preparation": (1.2208671118771937, 1.2117857340253784, 91170),
}
diag = json.loads((Path(os.environ["G2H_GAPS"]) / "diagnostics.json").read_text())
for name, (before, g2h01, count) in reference.items():
    row = diag["classes"][name]
    after = row["mean"]
    print(name, "G1h02_us", before, "G2h01_us", g2h01, "G2h02_us", after, "delta_us", after-before,
          "delta_pct", 100*(after-before)/before, "G1h02_count", count,
          "G2h_count", row["host_ahead_count"], "host_bound", row["host_bound_excluded_count"])
PY_COMPARE
```

Inspect `kernel_gap.csv` and report the four new values, sample counts, deltas and host/hold diagnostics against both references. Compare the 4,945 attention rows for the same 989 cases: identity fields and kernel counts must match, while timings are compared between GPU runs. CPU before/after reduction of both saved runs preserves their replay rows, runner rows, ownership summaries and gap diagnostics byte for byte.

Inspect every engine's `dsv4_raw_eN.ownership.json` and `.runner.csv`. Expect `dsv4_sampling` and nested diagnostic `dsv4_compute_logits` for all 976 real steps, with the same step index and boundaries as preparation. The 13 dummy steps have no sampling rows. Expected row counts across engines: preparation 989, write 976, zero 499, sampling 976, logits 976 (4,416 total); verify from the new records rather than fabricating missing costs. The sampling kernels and LM-head GEMM must leave the unowned list; LM-head kernels must belong to the logits child. Check direct `layout.json`, grids and all 394 rows separately; path B has not run successfully on a GPU yet.

## G2h-01 unowned review (CPU evidence)

Read the three ownership summaries and only `g1-e0-0000-full.json` and `g1-e0-1040-prefill.json`. Group (a) is work inside the existing sampling/logits hooks, (b) is runner work outside every wrapped call, and (c) is intentional output-copy work. Counts include kernels, memcpy and memset:

| Case kind | (a) newly owned | (b) stays outside hooks | (c) output copies | Total |
| --- | ---: | ---: | ---: | ---: |
| FULL | 3,624 | 0 | 672 | 4,296 |
| PIECEWISE | 3,624 | 0 | 672 | 4,296 |
| first tail | 3,184 | 0 | 592 | 3,776 |
| multi-tail | 77 | 0 | 14 | 91 |
| dummy | 0 | 273 | 0 | 273 |
| prefill | 11 | 6 | 2 | 19 |

Group (a) includes the logits-row gather, LM-head `nvjet_sm90_tst_*` grid `[2,66,1]` and its memset, temperature/position indexing, `_gumbel_sample_kernel` `[b,127,1]`, argmax and gather, `new_ones` Fill, `_get_num_sampled_and_rejected_kernel` and `_post_update_kernel` `[b,1,1]`. The two selected traces confirm the memset is under `aten::linear -> aten::matmul -> aten::mm`; it belongs to logits, not output copies. Group (c) comprises the two `Memcpy DtoH (Device -> Pinned)` activities per real step, from `AsyncOutput` (`gpu/async_utils.py:140,149,253-254`).

Group (b) is left for the lead's scope decision; this change adds no wrappers for it:

- Prefill: three `_apply_write_kernel` `[1,1,1]` and three HtoD copies. `execute_model` calls `add_requests` at `gpu/model_runner.py:1422`; `add_requests` calls `req_states.apply_staged_writes` at 1008. `gpu/states.py:122-124` writes `total_len`, `all_token_ids` and `num_computed_tokens`; each uses `gpu/buffer_utils.py:183-196` (contents HtoD at 184-186 and kernel at 189). These are distinct from the hooked fused block-table writer.
- Dummy: 26 arange kernels `[1,1,1]` at `gpu/input_batch.py:128,169`; 13 scalar-add kernels `[1,1,1]` for `logits_indices` at 168; 26 CUB scan kernels `[1,1,1]` plus 13 conversion-copy kernels `[1,1,1]` for `torch.cumsum` at 155-157. `InputBatch.make_dummy` is called outside the preparation hooks at `gpu/model_runner.py:1503-1508`.
- Dummy: 182 Fill kernels in total (169 vectorized, 13 unrolled). Fill<int> grids have first dimensions `1,2,3,4,7,14,21,28,42`; Fill<long> uses `[1,1,1]` and `[40,1,1]`, and Fill<bool> uses `[1,1,1]`. Sites are `gpu/input_batch.py:130,145-146,148,154,159,162-163,165`, dummy table zeroing at `gpu/block_table.py:174-175` (called by `gpu/model_runner.py:1327`), and full slot-buffer fill at `gpu/block_table.py:213` (via `gpu/model_runner.py:1328-1330` and `gpu/pcp_manager.py:635-637`). The summaries aggregate identical Fill names across these sites; individual per-site counts require a dummy trace, which was outside the two-trace allowance.
- Dummy: 13 hidden-row gathers after `execute_model`, consisting of eight `index_elementwise_kernel` `[8,1,1]` and five `vectorized_gather_kernel` grids `[2,2,1]`, `[4,2,1]`, `[8,2,1]`, `[16,2,1]`, `[24,2,1]`. Site: `_dummy_run`, `gpu/model_runner.py:767`, outside `sample` and `sample_tokens`.

## GPU time and remaining admission

Observed G2h-01 path A took **3,089 s / 0.858 GPU h (about 0.86)**. Retaining a conservative 120-360 s allowance for the new annotations gives replay **0.891-0.958 GPU h**. The separate block job needs its own startup (180-480 s), planned holds `394 * (3+20) * 0.020 = 181.24 s` as a conservative allowance (warm-up does not hold), and 60-180 s for other work/tracing: **0.117-0.234 GPU h**. Total estimated additional time is **1.008-1.192 GPU h**; direct startup and GPU/native matching remain unmeasured.

The G1h-G4h cap is **7 GPU h**, with about **1.95 used** (1.89 before G2h-02 plus 0.06 for its failed block job, Job Create 13:50:37 to Job Stop 13:54:12). With the G2h-02 replay and the `package_03` block job, estimated usage is **2.958-3.142 GPU h**, leaving **3.858-4.042 GPU h**. These are planning estimates, not measured path-B results.

## Registry impact audit

`kv_cache_block_zero` and `dsv4_block_table_write` use disabled `KV_CACHE_FAMILY` (memory operations writing cache/block metadata). Preparation/sampling use new disabled `DEEPSEEK_V4_RUNNER_FAMILY` (indexing/compute outside a layer). All four are profiling targets only; predictor/e2e flags and execution admission remain false. No model binds the new runner family. Registry code is unchanged.

The complete production-reader search of `profiling_target`, `profiling_ops`, registry iteration and family lookup found these effects:

- `operators/spec.py:332-357` validates/filters profiling targets; `operators/families.py:161,934-963` exposes new names through explicit family queries. `operators/binding.py:46-61` recognizes the new physical/profiling keys; execution still fails on the disabled families. Its frozen attention copies at 326 still clear flags. `entities/time_components.py:33-40` expands the canonical-name vocabulary by the three new keys; execution/e2e timing maps remain unchanged.
- `model_architectures.py:491-494`, `config/quantization_manager.py:34-50`, `execution_time_predictor/prediction_model_identity.py:112`, `layer_contract_resolution.py:188-239`, `sklearn_execution_time_predictor.py:2918`, and `profiling/linear_op/profiling_plan.py:239-281` inspect model-owned families. No existing non-V4 model contract selects cache/runner families here. Serialized registry comparison preserves all 11 other families; 50 concrete GLM/dense/MoE profiling plans (five models, TP 1/2/4/8/16, replicated enabled/disabled) are identical against HEAD's registry.
- Fixed-family callers are unchanged: `training/moe_trainer.py`, `execution_time_predictor/{prediction_model_identity,moe_operator_times,moe_dataset_training,prediction_family_trainers,sklearn_execution_time_predictor}.py`'s MOE/FFN/shared helpers; `profiling/moe/{main,moe_wrapper}.py`; `profiling/linear_op/{profiling_plan,deepseek_v4_plan}.py`'s memory/FFN/shared/mHC helpers. They query families whose fields did not change.
- Attention-only readers (`attention/profiling_mapping.py`, `config/quantization_manager.py`'s GDN/DSA helpers, `profiling/experimental/sglang/gdn_trace.py`, and attention-family flag declarations) query the separate unchanged attention registry. Their plans and admission are unchanged.

GPU execution/layout validation, package generation/submission, native comparison, final acceptance and future loader integration remain with Claude.
