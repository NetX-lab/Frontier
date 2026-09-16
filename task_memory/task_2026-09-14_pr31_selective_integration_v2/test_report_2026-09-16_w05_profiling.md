## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Recorded preflight/timer/sample fixes, retained native numerical tolerance, and collected hardware acceptance lanes. |

# W05 profiling boundary and timer verification

## Requirement and scope

W05 / R04 / R10 / N05 / N06 / N09. This independent boundary sub-step proceeds while W03 D01 remains pending; it does not settle timing semantics or claim W03 completion.

## Implementation and ownership

- `frontier/profiling/gdn/main.py`: validate the complete workload campaign, method, counts, GDN topology/TP, dtype, physical capacity and context+query lengths before native construction. Existing output files remain untouched on failure.
- `frontier/profiling/gdn/inputs.py`: typed input owns phase/capacity validation; shared iteration validator serves CLI and direct wrapper.
- `frontier/profiling/common/device_timer.py` and `frontier/profiling/utils/singleton.py`: narrow `get_existing()` owner accessor; unnamed standalone timer creates no disabled singleton; existing store owns method/disabled state. All five canonical methods retained. Failed contexts unwind without contributing successful timing samples.
- `frontier/profiling/gdn/vllm_wrapper.py`: require actual active/E2E samples, expected count and finite nonnegative values; only inactive operators receive schema zeros. TP ranks exchange validation/schema before tensor aggregation. `ExitStack` closes independently created model-parallel and process-group resources even if later construction fails; externally owned groups are retained. Metadata model dtype comes from actual hidden-state dtype. Added full-prefix versus carried-prefix output/state check using identical prefix/current inputs; BF16 tolerance remains rtol=atol=2e-2, outside measured work.
- Collected native lanes invoke the real wrapper/decomposition and prefix-continuation check when capable hardware/checkpoint are available. Native timing tests are separate from AMD GDN tests.

No numerical fallback, temporary scalar factor, GPU environment installation, native support expansion, or CSV success artifact on invalid input was introduced. Existing GDN family operator phases, timer store, typed inputs and contextlib.ExitStack are reused.

## Execution

Interpreter `/usr/bin/python` 3.12.3; no conda environment active. Baseline production revision before this sub-step: `2e8ab0e1` plus uncommitted W03 migration (separate files except task records). This report binds to the accompanying W05 code commit.

```bash
python -m pytest tests/unit/test_gdn_campaign_preflight.py tests/unit/test_gdn_profiler_cpu_increment7.py tests/unit/test_device_timer_contract.py tests/unit/test_timer_owner_lifecycle.py tests/unit/test_gdn_profile_samples.py tests/integration/test_gdn_native_acceptance.py tests/integration/test_device_timer_native.py -q -rs -p no:cacheprovider --tb=short > /data/ycfeng/tmp/pr33-w05-final2.log 2>&1
```

Native worker recipe when hardware/checkpoint exist:

```bash
FRONTIER_GDN_MODEL_PATH=/path/to/existing/qwen-checkpoint python -m pytest tests/integration/test_gdn_native_acceptance.py -q -rs -p no:cacheprovider
python -m pytest tests/integration/test_device_timer_native.py -q -rs -p no:cacheprovider
```

Read `/data/ycfeng/stepfun-env-handbook/guidence.md` before local GPU capability inspection. `nvidia-smi -L` produced no device rows; actual pytest capability fixtures report no available GPU. No worker allocated.

## Criteria and evidence

- Before repair, independent campaign regressions: 25 FAIL / 2 PASS, all invalid cases hit the forbidden native constructor recorder. Exact nodes and logs are in `w00_profiling_review.md`.
- After repair, combined command: **111 PASS / 9 SKIP in 5.12s**, no failure. Includes invalid campaigns with unchanged output sentinels, positive one-token continuation at limits, all timer methods/owner precedence/exception paths, complete sample statistics, invalid sample/count failures and TP schema mismatch before tensor collective. `git diff --check` PASS.
- Four native GDN cases: **SKIP: AMD/MI355X hardware and ROCm PyTorch unavailable**.
- Five native timer cases: **SKIP: Native device timer requires an available CUDA or ROCm GPU**.

## Practical limits

CPU tests prove boundary ordering, ownership and sample validation. Native constructor cleanup, kernel output/state equivalence, per-rank tensor aggregation and actual device timing remain UNVERIFIED until the collected hardware lanes execute. No native PASS is inferred. W07 will continue broader profiling adapter review. Required final-source integrated rerun remains W10.
