# G2h V4 runner operators

CPU plan and builder for B13r, paired with the unchanged G1h-02 replay grid. Claude commits/reviews these files, replaces `source_revision` with the committed revision, builds a fresh `package_01`, then submits it. Implementation did not build a package, install anything or run GPU/platform commands.

## Plan check (CPU only)

From the repository root, with Python 3.10:

```bash
ulimit -v 31457280
export PYTHONPATH="$PWD:$PWD/tests/unit:$PWD/tests/comparison/calibration:$PWD/tests/analysis"
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
timeout 120s /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h_runner/prepare_package.py --check-plan
```

`--dry_run` is an alias. This checks the exact planned coordinates/case IDs against `expected_rows.json`, plans the native-layer engine using the existing attention builder, and cold imports both worker entry paths with fasteners, sklearn, torch, vLLM and cuda blocked. It creates no package. Direct grid-only check:

```bash
timeout 120s /data/ycfeng/envs/frontier-py310/bin/python -m frontier.profiling.model_runner.dsv4_block_operators --grid task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/g2h_runner/block_operator_grid.json --model_config_path data/config/models/deepseek-ai__DeepSeek-V4-Flash.json --dry_run
```

The builder reuses `p1h_bridge/prepare_package.py`'s committed source snapshot, accepted bootstrap, base-package verification, model assets, image, CUDA compatibility path and mount. No separate engine builder or vLLM method implementation is introduced. G1h `plan_args`, pilot/coverage files and the three case lists are reused verbatim.

## Worker phases and outputs

Job `G2H_V4_RUNNER_OPERATORS`, tag `v4-h200-g2h-runner-20261028-01`, uses one H200, step_main, non-preemptible. Output destination is `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/h200_v4_runner_operators/g2h_runner_01`. The accepted worker uses `/tmp/glm_producers/out` locally; the job worker collects that tree.

| Phase | Work | Local output |
| --- | --- | --- |
| `G2H_RUNNER_REPLAY` | Same G1h-02 engines and 989 IDs: e0 686, e1 296, e2 7. New scopes, unchanged attention rows. | `/tmp/glm_producers/out/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/{raw,traces}` |
| `G2H_BLOCK_OPERATORS` | Native 43-layer allocation, grid-specified 8 experts. 33 zero and 164 write coordinates per method, 394 rows total. | `/tmp/glm_producers/out/G2H_BLOCK_OPERATORS/` |

Built-package worker commands (lead/GPU run only; not executed in implementation):

```bash
python -I package_01/native_dummy_bootstrap.py --config package_01/bridge_config.json
python -I package_01/native_dummy_bootstrap.py --config package_01/block_config.json
```

Replay produces its existing raw JSONL, captures, traces and attention CSVs plus adjacent `dsv4_raw_eN.ownership.json` and `dsv4_raw_eN.runner.csv`. The ownership file classifies **every recorded-step kernel, memcpy and memset** as attention scope, step scope, model forward or unowned. Hold kernels are calibration and excluded from the census. Each step has counts and each observed kind (FULL, PIECEWISE, first tail, multi-tail, dummy, prefill) has unowned names, grids and counts. There is no name-based ownership inference. Model-forward boundaries use CPU launch correlations within `frontier_v4_model_forward|<step>` ranges; graph nodes without attention ownership are model-forward work.

Step owners use reserved integer layer **-1** and a top-level scope; physical attention layers remain >=0. `dsv4_step_scopes.is_step_scope` validates both fields. `dsv4_attention_rows.build_dsv4_rows` filters these call rows; `dsv4_replay_rows` continues to read existing attention layer calls through the gap reducer; the bridge reads capture owners/interval kinds without layer pairing assumptions. The gap reducer excludes step owners from attention dependency chains and preserves their previous outside-work/graph classes and hold diagnostics. Thus new ownership does not reclassify preparation as attention eager work.

`sample()` has the `dsv4_sampling` owner, and `model.compute_logits` inside it is the child scope `dsv4_compute_logits` (diagnostic). Graph-replay ranges may nest: the trace reader gives each kernel to the innermost range, so the row gather and sampler stay with sampling and the LM head goes to the logits scope. A logits call outside `sample()` fails the parent check. Ordinary nested step scope calls fail the same parent check as attention scopes. The execute range still opens before `execute_model` and closes after `sample_tokens`, including AsyncOutput's unpriced output copy.

Direct output includes `layout.json`, `block_operators.csv`, all Chrome traces and adjacent `.trace.intervals.json` files with kernel and memcpy intervals kept separately. Rows use **`dsv4_model_runner.v1`**, constructed in `frontier/profiling/model_runner/dsv4_runner_rows.py`: existing DSA `time_stats.<operator>` statistics in ms, separate `kernel_stats`/`memcpy_stats`, coordinates, method, repeat count and engine layout JSON. `cuda_event`'s headline sample spans the native call; `record_function`'s headline sample sums owned kernel durations. The future operator loader reads these producer CSVs; no current consumer, training path or execution flag is activated here. Column `row_role` states each row's use: path A input preparation and sampling rows and all path B rows are `cost`; path A write rows are `validation` (kernels and shapes against path B); path A zeroing and full TP1 compute_logits rows are `diagnostic`.

## Direct grids and timing

All coordinates, expert/layer counts, warm-up (3), repeats (20), hold (20000 µs), first chunk (6144) and expected values live in `block_operator_grid.json`. Zeroing uses `n=1..32,2520`. Writes use `w_c=0..32` and `r=0..4`, omit `(0,0)`, and launch one fused call after native staging. Table entries come from actual group block sizes `[256,64,64,4,8]`, checked against `[24,96,96,1536,768]`. At most 11 distinct request rows are needed by the write grid; 128 are planned. Consecutive free IDs start at 1. The run checks actual free-block capacity, request rows/table widths, native B4 group members, **139 segments / 10 chunks**, and unsplit table entries before timing.

Each measured call has the replay's calibrated host-ahead device hold, both events and trace ranges, and a post-call synchronization. Staging/restoration happens before the measured range. Each positive grid call must own one kernel; missing ownership or late launches fail with a coordinate/repeat diagnostic. Kernel and HtoD copy times are not merged into a policy decision. See `native_kernel_notes.md` for pinned source lines, allocation caveats, hash/dummy-finalizer evidence and native/G1h kernel tables. Exact segment derivation needs allocated native tensor views; the CPU plan does not fabricate aliases.

## Lead check after the GPU run: four gaps against G1h-02

Set `G2H_RESULTS` to the downloaded local output tree containing the two phases and `G2H_GAPS` to a writable report directory. Run the existing reducer:

```bash
timeout 600s python -m frontier.profiling.kernel_gap.dsv4_replay_gaps --raw_files "$G2H_RESULTS/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e0.jsonl" "$G2H_RESULTS/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e1.jsonl" "$G2H_RESULTS/G2H_RUNNER_REPLAY/bridge/FULL_AND_PIECEWISE/raw/dsv4_raw_e2.jsonl" --output_dir "$G2H_GAPS" --device h200 --model deepseek-ai/DeepSeek-V4-Flash --diagnostics_path "$G2H_GAPS/diagnostics.json"
timeout 60s python - <<'PY_COMPARE'
import json, os
from pathlib import Path
reference = {
    "cuda_graph": (0.49773414267747085, 169232),
    "cuda_graph_piecewise": (0.6249537992538311, 289730),
    "eager": (0.5446049595929223, 30243),
    "eager_preparation": (1.2208671118771937, 91170),
}
diag = json.loads((Path(os.environ["G2H_GAPS"]) / "diagnostics.json").read_text())
for name, (before, count) in reference.items():
    row = diag["classes"][name]
    after = row["mean"]
    print(name, "G1h02_us", before, "G2h_us", after, "delta_us", after-before,
          "delta_pct", 100*(after-before)/before, "G1h02_count", count,
          "G2h_count", row["host_ahead_count"], "host_bound", row["host_bound_excluded_count"])
PY_COMPARE
```

Inspect `kernel_gap.csv` and report all four new values, counts, deltas and host/hold diagnostics against the references. This is a comparison, not an equality assertion between separate GPU runs. Compare the attention rows for the same 989 cases and inspect unowned summaries and direct layout/kernel grids. During CPU implementation, old raw-record gap diagnostics and all 4945 attention rows were reproduced byte-for-byte; bridge equivalence was checked on the two permitted selected traces, not all 989 trace files.

## GPU time and remaining admission

G1h-02 engines took 2711 s; its job took 3055 s (0.849 GPU h). Allow 120-360 s for new replay scope/report overhead. Direct startup is estimated at 180-480 s; planned holds total `394 * (3+20) * 0.020 = 181.24 s` as a conservative allowance (warm-up does not actually hold), with 60-180 s for other work/tracing. Total estimate is **1.00-1.19 GPU h**. Actual startup/trace size remain uncertain until the run. Prior G1h/G4h usage 0.92 plus B13 0.23-0.60 plus this plan gives **2.15-2.71 of the 7 GPU h cap**, leaving 4.29-4.85. The lead's newly available B13 report (`../../test_report_2026-10-28_b13_v4_standalone_operators.md:62`, recorded at HEAD `4ed287fc`) now reports 407 s / 0.113 GPU h for that job. Using this observed B13 usage gives **2.03-2.22 of 7 GPU h** after this runner plan, leaving about 4.78-4.97. The older 2.15-2.71 envelope above is retained as the original planning comparison. New B13r GPU timing/native matching remain unmeasured.

## Registry impact audit

`kv_cache_block_zero` and `dsv4_block_table_write` use disabled `KV_CACHE_FAMILY` (memory operations writing cache/block metadata). Preparation/sampling use new disabled `DEEPSEEK_V4_RUNNER_FAMILY` (indexing/compute outside a layer). All four are profiling targets only; predictor/e2e flags and execution admission remain false. No model binds the new runner family. Registry code is unchanged.

The complete production-reader search of `profiling_target`, `profiling_ops`, registry iteration and family lookup found these effects:

- `operators/spec.py:332-357` validates/filters profiling targets; `operators/families.py:161,934-963` exposes new names through explicit family queries. `operators/binding.py:46-61` recognizes the new physical/profiling keys; execution still fails on the disabled families. Its frozen attention copies at 326 still clear flags. `entities/time_components.py:33-40` expands the canonical-name vocabulary by the three new keys; execution/e2e timing maps remain unchanged.
- `model_architectures.py:491-494`, `config/quantization_manager.py:34-50`, `execution_time_predictor/prediction_model_identity.py:112`, `layer_contract_resolution.py:188-239`, `sklearn_execution_time_predictor.py:2918`, and `profiling/linear_op/profiling_plan.py:239-281` inspect model-owned families. No existing non-V4 model contract selects cache/runner families here. Serialized registry comparison preserves all 11 other families; 50 concrete GLM/dense/MoE profiling plans (five models, TP 1/2/4/8/16, replicated enabled/disabled) are identical against HEAD's registry.
- Fixed-family callers are unchanged: `training/moe_trainer.py`, `execution_time_predictor/{prediction_model_identity,moe_operator_times,moe_dataset_training,prediction_family_trainers,sklearn_execution_time_predictor}.py`'s MOE/FFN/shared helpers; `profiling/moe/{main,moe_wrapper}.py`; `profiling/linear_op/{profiling_plan,deepseek_v4_plan}.py`'s memory/FFN/shared/mHC helpers. They query families whose fields did not change.
- Attention-only readers (`attention/profiling_mapping.py`, `config/quantization_manager.py`'s GDN/DSA helpers, `profiling/experimental/sglang/gdn_trace.py`, and attention-family flag declarations) query the separate unchanged attention registry. Their plans and admission are unchanged.

GPU execution/layout validation, package generation/submission, native comparison, final acceptance and future loader integration remain with Claude.
