# Parent verification of CPU reuse changes

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Record input-reuse verification, exact frame/hash comparison, merged tests and frozen-checkout prerequisites. |
| 2026-10-22 | Record complete77-case strict comparison of both integrated CPU reuse changes. |

## Scope and results

Runtime fixed bindings: `ee28bf7`, independently accepted and integrated at
`97c2234`; its parent evidence is in `cpu_performance_review.md` and
`contract_parent_receipt.json`. Input reuse: `02f566a`, records `dfb6c96`, merged
with runtime reuse at `7e8bea3cd2362f8e46c4b7cd0e712d2f2fe392fb`.
All changed production/test bytes were compared with the corresponding tested
branches and match. No full CLI result is inferred from these checks.

| Parent check | Observed result |
| --- | --- |
| Input/union/model-bundle/manager tests | 187 PASS, 53.285s, no failures/errors/skips |
| Request-free profiling helpers in existing torch environment | 19 PASS, 7.956s, no failures/errors/skips |
| Combined loader/runtime/public-predictor/root tests | 224 PASS, 96.594s, no failures/errors/skips |
| Frozen-checkout integration fixture tests | 8 PASS, 5.529s |
| Original repeated-load probe | GREEN_NO_REPEATED_LOAD; DECODE 0 loader calls, 0.028414s; warm PREFILL 2.574255s |
| Original input/config checks | PASS; no changed source fields/configuration |
| Baseline frame, attrs and JSON | Exact equality for event100 and kernel184 |
| Root/count hashes | All four unchanged; reverse-role lookup reuses frame/hash |

These suites overlap and are not summed into a unique-test total. The separate
runtime suite's two baseline-identical structural failures remain recorded in
`test_report_2026-10-22_actual_main.md`. The 19 helper tests run on CPU; they do
not establish GPU/native qualification.

## Commands and environments

Input tests and probe cwd: `/data/ycfeng/Frontier/.worktrees/step5-mrv2-load`.
CPU environment is the existing `frontier-py310`, Python3.10.6. Helper tests use
the existing `oneshot-opd-py312`; the exact interpreter version is retained in
`load_parent_receipt.json`. No dependencies were installed.

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_attention_input_union.py tests/unit/test_step5_measured_training.py tests/unit/test_step5_mrv2_inputs.py tests/unit/test_step5_model_bundle.py tests/unit/test_step5_measured_manager.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_load_reuse_parent02.xml > /data/ycfeng/tmp/d23_load_reuse_parent_tests02.log 2>&1
env VLLM_TARGET_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:/data/ycfeng/tmp/step5_profiling_20261020/extra_site TMPDIR=/data/ycfeng/tmp STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm /data/ycfeng/envs/oneshot-opd-py312/bin/python -B -m pytest tests/unit/test_step5_request_free_profiling.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_load_native_helpers_parent01.xml > /data/ycfeng/tmp/d23_load_native_helpers_parent01.log 2>&1
timeout 60s env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/profile_step5_load.py --config /data/ycfeng/tmp/d23_main_actual05_offline/offline/metrics/stepfun_ai_step_5_preview/offline_batch/step5_measured_cpu_main_offline/config.json --output /data/ycfeng/tmp/d23_load_reuse_parent01 --rows 100 --event-cache-hash 23b7026a --assert-no-repeated-loads > /data/ycfeng/tmp/d23_load_reuse_parent01.log 2>&1
```

The first parent pytest attempt incorrectly named a nonexistent
`test_step5_measured_model_bundle.py`; it exited4 without collecting tests.
`d23_load_reuse_parent_tests01.log` retains the error. The corrected command
above uses the actual `test_step5_model_bundle.py`.

Combined check cwd: parent `step5-preview` at merged source `7e8bea3`:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_step5_attention_input_union.py tests/unit/test_step5_measured_training.py tests/unit/test_step5_mrv2_inputs.py tests/unit/test_step5_model_bundle.py tests/unit/test_step5_measured_manager.py tests/unit/test_step5_contract_reuse.py tests/unit/test_step5_measured_public_predictor.py tests/unit/test_measured_attention_root_metrics.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_perf_merged01.xml > /data/ycfeng/tmp/d23_perf_merged01.log 2>&1
```

## Exact frame/hash reproduction

The parent independently loaded the original saved DataFrames from
`/data/ycfeng/tmp/d23_load_reuse_baseline01/{CUDA_EVENT,KERNEL_ONLY}.pkl` and the
actual05 offline configuration using `profile_step5_load.configuration`.
The original recipe keeps `predictor.atten_input_file` as the string
`/data/ycfeng/tmp/d23_mrv2_load_diag03/cuda_event_prefix_100.csv` for **both**
measurements. The first parent attempt reset it before the kernel check, which
correctly changed that hash-relevant configuration and failed the comparison.
This was a reproduction setup error, not a production hash change. The corrected
recipe retains the original path, uses the original kernel path, and passes:

```python
pd.testing.assert_frame_equal(before, after, check_exact=True)
assert before.attrs == after.attrs
assert before.to_json() == after.to_json()
hashes = {
    name: manager._get_model_hash(
        name, after, fit_config,
        manager._get_profiling_precision_from_df(after), measurement,
    )
    for name in ("step5_layer_e2e", "step5_layer_e2e_kernel_count")
}
```

The parent then invokes the union API for DECODE and checks object identity and
`step5_attention_input_hash` against the first role's frame and root hash.
No fitting or original source/cache writes occur.

| Source | Rows | Root hash | Count hash |
| --- | ---: | --- | --- |
| CUDA_EVENT diagnostic prefix | 100 | `9688c187` | `008badcf` |
| KERNEL_ONLY complete table | 184 | `b1bedbbb` | `664eca75` |

Evidence: `/data/ycfeng/tmp/d23_load_frame_parent01.log` (setup error),
`/data/ycfeng/tmp/d23_load_frame_parent02.{log,json}` (PASS), and the selected
results and artifact hashes in `load_parent_receipt.json`.

## Review and run prerequisites

Independent reviewer `ses_e9f8b6cffffeSZiBfBvZkmXBQa` inspected the complete
`32a1f60..02f566a` diff, relevant contracts and tests and returned **No findings**.
Same-manager content/model/shape/fit changes invalidate reuse; every distinct
source is still fingerprinted per lookup. Both role proofs remain. Initialization
success, training exception and publication exception release retained frames.
The reviewer did not execute the parent's separate frame comparison or later
torch-helper checks; those results above are parent observations.

An immutable checkout was created at
`/data/ycfeng/Frontier/.worktrees/fidelity-candidate-7e8bea3`. The submodule was
initialized locally at `d28fe917370c2d55230c3ea226e5660785b44aca`, with no download.
The parent's existing `htsim_ndp` binary was copied to the new checkout and both
hashes match `06b3a5fb9b7a37e1517536cb55d761553e04365b2073606e511fc9e90a45cdca`.
The checkout was clean before launching actual06; its HEAD will remain fixed.

Commands from that immutable checkout:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$PWD:$PWD/tests/unit TMPDIR=/data/ycfeng/tmp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/integration/test_step5_measured_main.py -q --tb=short -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d23_actual06_fixture_check.xml > /data/ycfeng/tmp/d23_actual06_fixture_check.log 2>&1
timeout 60s env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B tests/performance/step5_mrv2_collective.py --output /data/ycfeng/tmp/d23_actual06_collective_check > /data/ycfeng/tmp/d23_actual06_collective_check.log 2>&1
```

The collective control executes actual backend misses/hits: TP8 allreduce
predicts0.122465ms on both, EP16 all-to-all predicts0.0199104ms on both. Four
requests yield two misses/two hits. These are CPU simulator predictions and
prerequisite checks, not native GPU measurements. Full CLI results remain pending.

## Strict legacy comparison

The candidate77 run and comparison commands in `progress.md` completed with exit0.
The baseline is the retained clean26db654 matrix; candidate is clean7e8bea3.
Observed:77 of77 compared and identical; zero failures, missing evidence/cases,
definition differences or un-compared cases. Both cache-file inventories match,
with no provenance findings. The original report is retained byte-for-byte at
`cpu_reuse_fidelity.json`, SHA256
`5753dc5399065bed197e879047caae448f55c51de7a9b180c7cb53e020d29488`.
Raw comparison: `/data/ycfeng/tmp/step5_d23_perf_fidelity_20261022/comparison.json`;
log: `/data/ycfeng/tmp/d23_perf_fidelity_compare01.log`. This passes the legacy
preservation criterion; actual06 measured-input full-request acceptance remains
independent and unfinished.
