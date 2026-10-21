# D20 compute integration handoff

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record tested source revisions,executable H80032 fixture command and exact input ownership. |

## Source and execution

Merge/cherry-pick the branch containing `1ca09be` and `83af0a7`;primary integration already
owns PD9625984 and NIXL loader/config8215196. No new scheduler,event,Request,Batch,
comparator,native-client or transfer-loader implementation is present in this slice.

From the integrated repository root,choose a fresh output path:

```bash
env PYTHONDONTWRITEBYTECODE=1 \
 PYTHONPATH="$PWD:$PWD/tests/unit:/data/ycfeng/Frontier/.worktrees/step5-preview/frontier/cc_backend/backends/collective-sim/python" \
 TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 \
 /data/ycfeng/envs/frontier-py310/bin/python \
 tests/integration/step5_measured_compute_fixture.py \
 --output-dir /data/ycfeng/tmp/d20_measured_compute_fixture_h80032_parent01 \
 --device h800 --gpus-per-role 16
```

This actually fits/caches/publishes/reloads models. Expected:`fixture_manifest.json`
with `evidence_kind=cpu_fixture`,native_admitted=false,TP8/DP2/EP16,92 layers,
4 role/method receipts and34 artifacts per receipt. It does not submit any job.
Existing verified output is `/data/ycfeng/tmp/d20_measured_compute_fixture_h80032_03/`.
Pass `--nixl-read-profile <explicit-primary-file>` only when supplied;there is no default.
Use `predictor_config` and `cache_dir` from the manifest in the primary actual-main input,
preserving supplied method-specific files. The same command supports explicit h200 or
32 GPUs/role without changing production cost code;these are CPU fixtures,not quotas.

## Input contract

- `atten_kernel_only_input_file`:actual KERNEL_ONLY eager GPU-active rows;root target
  `time_stats.step5_layer_e2e.median` and explicit per-replay `.kernel_count`.
- `atten_input_file`:actual CUDA_EVENT graph rows;root target `.mean` is the clean-window
  candidate. `.median` remains the independent diagnostic/admission check. The model
  marker and runtime validate statistic and actual fitted target column.
- Both require typed source geometry,variant/global layer,query/context arrays,physical
  cache/table extents,model/profile/quant/runtime/backend/device/dtype/weight/timing metadata.
  One39-feature schema serves loader/query. Root counts are never inferred from stats.count
  or aggregate trace counts. Serving-capacity tables must match;live-length widths are
  not padded or silently substituted by the consumer.
- `linear_op_input_file`/`moe_input_file`:actual event tables;the corresponding
  `*_kernel_only_input_file` fields must contain separately labelled actual GPU-active
  measurements when that family is selected. Preserve routed-group/typed contracts and
  true duration/count columns. Overall input profiling precision does not relabel FP8
  expert weights or FP32 router;source layer contracts select those estimator domains.
- Outside-root input norm and optional `input_cast`/`final_residual` targets use their
  explicit existing stats/count fields. Stage targets require correct source tensor width
  and dtype. Absent or malformed targets remain errors,not0;no producer expansion was added.
- `kernel_gap_input_file`:actual eager/cuda_graph rows with TP and kernel_gap_us,as used by
  the existing kernel-gap mechanism. Bound path/SHA changes bundle identity even if models
  can reuse immutable artifacts. Event spans do not receive an additional gap charge.
- `nixl_read_profile`:primary-owned exact JSON input (`nixl_read_cost.v1`),supplied explicitly.
  Same bundle input binding is checked before first lazy read. Predictor delegates only to
  `NixlReadCost.from_file/get_coverage/get_time_ms`;absence raises Missing measured NIXL profile.
  Geometry/payload/READ-ready validation belongs to primary,not this compute module.

Fixture coverage is batch1,max_model_len8192,block16 and6144+0/16+6128/1+6144/1+7167
cases. The tail and values are labelled code fixtures,not observed serving facts.
Other reached distributions/shapes need actual matching measurements;do not extrapolate
the fixture into native admission. Proper CC backend and CPU overhead inputs remain
primary integration requirements;unit tests explicitly isolate communication/CPU cost.

## Interfaces and evidence

Normal manager auto-publishes registered domains and exposes `_step5_model_bundles`.
`publish_registered_model_bundle(identity,family,role)` and
`load_model_bundle(path,expected_identity)` use existing files and restore count pairs.
Prediction APIs remain unchanged. `AttentionOperatorTimes.costs` records raw duration,
actual count,method,model hashes and gap inclusion;priced op_times/current kernel-gap
ledger show the one-time charge. FFN and supplied stage proof maps are retained on
execution objects. No generic attention projection is added to the inclusive root.

Run the exact192-case command in the test report for compute acceptance. Primary then
runs actual-main,combined PD/NIXL and77-case fidelity. No native/profile/four10% PASS is
claimed. Remaining concrete production gap:qualified H80032 actual count/timing tables
with serving physical extents;old H200 probe data is still unadmitted. Final independent
re-review could not run due advisor quota;the legacy default-count assertion remains
documented,not patched with an invented kernel count.
