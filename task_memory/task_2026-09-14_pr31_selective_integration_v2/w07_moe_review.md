## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Reviewed fused MoE API/MXFP4 changes; repaired DEVICE_EVENT admission and sample statistics; recorded 23 passing CPU checks and native-evidence limits. |

# W07 fused MoE profiling review

## Scope and result

- Target Component/Phase: W07 fused MoE event timing and relevant MXFP4/version-compatibility hunks.
- Reviewer Agent Identity: `/root/w03_contract_review`.
- Inspected Artifacts: complete `frontier/profiling/moe/moe_vllm_kernel.py`; its additions in commit `3b20d702`; wrapper `_profile_with_vllm_kernel()` and grouped-GEMM dispatch; CLI measurement labels; shared accelerator/platform/profile-method helpers; `DeviceTimer`, `TimerStatsStore`, and `vllm_config_context`; existing MXFP4 CPU tests.
- Remediation/Verification Code Actions Taken: changed only `frontier/profiling/moe/moe_vllm_kernel.py`, added `tests/unit/test_moe_fused_event_contract.py`, and created this report. Existing MXFP4 tests and shared timer/helper files were preserved. No commit by the reviewer.

The real failure was a boundary mismatch: `moe_wrapper.py` and CLI permit `device_event`, but `profile_fused_moe_kernel()` admitted only `cuda_event` and `record_function`. This rejected the ROCm DEVICE_EVENT lane before reaching either supported vLLM call path. The public profiler now normalizes existing aliases, admits DEVICE_EVENT, and uses the existing `validate_profile_method_platform()` with `accelerator_platform(torch)` to enforce platform provenance.

PyTorch's `torch.cuda.Event` interface remains the event primitive for both CUDA and HIP. The profiler does not rewrite `device_event` to `cuda_event`; `profile_method_to_measurement_type()` still maps it to `MeasurementType.DEVICE_EVENT`. `moe/main.py` owns exported measurement labels and uses that mapping. The legacy-named private event collector now documents its CUDA-compatible HIP use.

## Additional verified timing defects and correction

The old collector used `torch.tensor(times).std()` with the sample correction default. One active sample produced NaN and a degrees-of-freedom warning. It also used `torch.median()`, which selects the lower middle sample for even sample counts; the shared profiling store uses the midpoint median. These differing definitions made fused MoE output inconsistent with other profiling paths.

The collector now delegates numerical summaries to `TimerStatsStore.get_stats_from_times()`, preserving the existing five returned statistic keys. This uses population standard deviation, including zero for a singleton, and the shared midpoint median. The oracle `[1.0, 3.0] ms` produces min 1, max 3, mean 2, median 2, and std 1. Samples that are negative or non-finite are rejected; non-finite computed summary values are also rejected.

The public profiling boundary now validates `active_steps` as a positive integer and `warmup_steps` as a non-negative integer before tensor allocation. Boolean and fractional iteration counts are rejected rather than interpreted accidentally or failing later inside `range()`/empty reductions.

## Changed-hunk and contract adjudication

| Area | Evidence and judgment |
| --- | --- |
| vLLM imports / API selection | Existing import boundary first attempts the 0.10.x low-level symbols and then imports functional `fused_experts`. These are explicit dependency paths; this task adds no automatic backend selection. CPU tests separately exercise the public profiler branches using mock native calls. Installed-package compatibility is not established because vLLM is absent. |
| Quantization mode | Existing `validate_moe_quantization_mode()` rejects FP8 and MXFP4 together. Existing tests retain this guard. Functional FP8 still fails explicitly because its descriptor adapter is not configured. |
| MXFP4 physical plan | Existing CPU planner enforces positive integer dimensions, group size 32, even packed dimensions, and group divisibility. For `(experts=2, hidden=64, intermediate=96, gated=True)`, packed weight shapes are `(2,192,32)` and `(2,64,48)`; E8M0 byte-scale shapes are `(2,192,2)` and `(2,64,3)`. Existing tests pass. These are planning assertions, not inspected physical tensors from a native vLLM run. |
| MXFP4 materialization | `_get_functional_mxfp4_state()` reuses `vllm_config_context`, creates online quantization state, and invokes `process_weights_after_loading()`. The cache key includes device, expert dimensions, top-k, and dtype; old shapes are released before retaining the next shape. On HIP, a backend label lacking AITER fails explicitly. No runtime packing, scale dtype, workspace, or numerical correctness claim is made without native dependencies. |
| Functional iteration | Existing code dispatches either complete `fused_experts` or MXFP4 `moe_kernel.apply()` with the supplied global expert count and expert map. This change modifies only profile-method admission and event-statistics collection. |
| Low-level iteration | Existing code allocates two GEMM intermediate buffers and uses the vLLM 0.10.x alignment/configuration path. The new CPU test proves DEVICE_EVENT reaches this branch and invokes the requested warmup/active iteration count. It does not validate kernel buffers or output numerics. |
| Event statistics and provenance | Fixed as described above; shared normalizer, platform validator, accelerator detector, and statistics helper are reused. No common timer files were changed. |
| Routing | Supplied `topk_weights`, `topk_ids`, global expert count, and expert map retain their existing call contract. No routing generator or parallel implementation was introduced. |

An observed scope concern is documented without claiming a proved supported-runtime defect: this fused entry point always builds gated `w13` shapes and has no `use_gated` argument, whereas the wrapper retains model-level `use_gated`. This review did not establish a reachable supported non-gated MoE model or a failing real configuration, so no unrequested shape/API expansion was made. Similarly, the low-level branch times two GEMM invocations while the functional branch calls a complete fused operation; native evidence is needed before asserting numerical or timing equivalence between those version paths.

## Execution and evidence

Environment: `/usr/bin/python`, Python 3.12.3, local CPU host; no conda activation. Torch CPU tensors are used for tiny fixtures. Event creation, synchronization, vLLM contexts, and native kernel calls are replaced with scoped mocks, so no GPU initialization or native profiling occurs.

Before the production edit, the new 16-case suite produced **15 failed, 1 passed, 4 warnings in 3.06 seconds**. The failures included DEVICE_EVENT rejection, absent platform restrictions, invalid iteration handling, singleton NaN, and invalid samples accepted. The singleton warning identified `torch.std()`'s degrees-of-freedom behavior. The one passing case was ordinary CUDA_EVENT on the CUDA mock runtime.

After the correction, two additional checks exercised midpoint/population statistics and the legacy low-level API branch. Final exact command:

```bash
PYTHONPATH=$PWD WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_LOG_LEVEL=ERROR python -m pytest tests/unit/test_moe_fused_event_contract.py tests/unit/test_moe_mxfp4_increment10.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/pr33-w07-moe-events
```

Observed result: **23 passed in 3.11 seconds**. This includes 18 new boundary/statistics cases and five existing MXFP4/quantization/backend tests. `git diff --check -- frontier/profiling/moe/moe_vllm_kernel.py tests/unit/test_moe_fused_event_contract.py` passed.

Pass criteria include correct method-to-family identity, rejection before allocation for platform/iteration errors, three mock kernel calls for one warmup plus two active steps, finite singleton std=0, rejection of NaN/Inf/negative elapsed samples, and exact shared statistics for `[1,3]`.

## Remaining evidence

No unresolved CPU-test failures remain. Native NVIDIA/vLLM and AMD/MI355X/AITER/HIP validation remain unexecuted and are not represented by these CPU checks. No GPU command, allocation, package installation, or remote command was issued. Broader W07 collectives/experimental adapter work belongs to other agents/root.
