## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Added one real native acceptance module with 10 collected attention/MXFP4/NCCL/RCCL/SGLang cases; CPU execution recorded 10 capability SKIPs without native claims. |
| 2026-09-16 | Added standard MoE admission/provenance, physical storage checks, serialized replay validation, shared collective dtype ownership, 119 CPU regression passes, and all-hunk closure/limitations. |
| 2026-09-16 | Fixed routed sorting block ownership and routed replay correctness/reset/trace handling using existing orchestration; recorded CPU RED/GREEN evidence and native limits. |

# W07 SGLang Routed Profiling Repair

## Scope and ownership

- Reviewer/implementer: `/root/w00_review_inventory`.
- Specification: W07 in `Frontier_PR33_New_Execution_Plan_2026-09-16_EN.md`.
- Owned production files: `frontier/profiling/experimental/sglang/moe.py`, `routed_moe_replay.py`, and `graph_replay.py`.
- The parent explicitly approved extending ownership to `graph_replay.py` to reuse the existing normalization and replay orchestration. Ordinary `profile_graph` callers retain their return contract.
- Tests: existing `tests/unit/test_sglang_graph_replay_orchestration.py` and new `tests/unit/test_sglang_routed_sorting.py`.
- No timer, GDN, standard MoE wrapper, shared registry, or unrelated source file was edited. No commit or remote action was performed by this subtask.
- Applied skill: `diagnosing-bugs`, using deterministic CPU regressions through the real defective call paths. The already source-confirmed arithmetic and orchestration errors did not require a broad hypothesis search.

## Delivered behavior

### Multi-block expert sorting ownership

`make_moe_sorting_primitive` previously expected one sorted block owner per active expert. This disagreed with its own workload block count whenever an expert received more than 32 tokens. It also assigned rather than accumulated the number of observed routes for each expert.

The expected owner list now repeats each expert once per required block, validates all those blocks, and accumulates routes across blocks. No routing distribution, assignment reconstruction, AITER kernel call, quantized shape, or measured interval changed.

### Shared routed replay and real correctness/trace callbacks

`profile_routed_graph` previously owned a separate capture/replay loop. It never compared its returned tensors against the builder reference, did not reset mutable buffers, and emitted correctness and representative-trace flags without implementing their checks/probe.

The routed entrypoint now delegates to `profile_graph`. `make_primitive` uses the existing `MOE_ROUTED_PRIMITIVES` declaration to normalize the two routed builders through `_make_replay_call`. The established implementation now owns eager checks, warmup, reset before capture/replay, synchronized timing, replay checks, and the representative trace callback. Routed histogram and shape provenance remain attached to the row. `HIP_GRAPH_REPLAY` and experimental artifact boundaries remain unchanged.

The experimental routed API now returns `(row, trace_replay)`, matching the existing common replay API, instead of returning only a row. This permits the caller to execute the requested trace probe under its profiling context. Repository caller search found only invalid-input tests before this change, with no valid in-repository consumer requiring migration. Existing ordinary primitive callers keep their original tuple contract. External research callers of the routed API must unpack the tuple.

The shared comparison now uses exact equality for integral metadata while preserving the existing `atol=rtol=0.03` tolerance for floating/complex outputs. This is required for newly routed sorting metadata: the previous generic BF16-oriented tolerance incorrectly accepted a sorted-token count of 97 against an expected 96.

## Validation report

### Execution

- Working directory: `/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/feature-amd-sglang-gdn`.
- Interpreter: `/usr/bin/python`, Python 3.12.3; no active conda environment.
- pytest: 9.1.1; torch CPU tensors are used, with graph/event/AITER boundaries explicitly simulated.
- All raw logs are under `/data/ycfeng/tmp/pr33-w05-preflight-regression/`; no raw outputs or caches were placed in versioned source.

Initial sorting reproduction:

```bash
PYTHONPATH="$PWD" TMPDIR=/data/ycfeng/tmp /usr/bin/python -m pytest tests/unit/test_sglang_routed_sorting.py -q -p no:cacheprovider > /data/ycfeng/tmp/pr33-w05-preflight-regression/w07-sorting-red.log 2>&1
```

Observed **2 failed, 3 passed in 2.54s**. The counts `(33, 31)` and `(64, 33, 0)` failed in the actual production sorting oracle; the latter error was:

```text
AssertionError: The values for attribute 'shape' do not match: torch.Size([4]) != torch.Size([2]).
frontier/profiling/experimental/sglang/moe.py:204
```

Initial routed replay reproduction:

```bash
PYTHONPATH="$PWD" TMPDIR=/data/ycfeng/tmp /usr/bin/python -m pytest tests/unit/test_sglang_graph_replay_orchestration.py -q -p no:cacheprovider > /data/ycfeng/tmp/pr33-w05-preflight-regression/w07-replay-red.log 2>&1
```

Observed **4 failed, 8 passed in 2.41s**. Routed sorting and experts lacked the common trace callback return; injected eager/replay corruption also reached that unchecked return instead of failing the numerical comparison. The observed exception was `ValueError: too many values to unpack (expected 2)`. The absence of comparisons itself was source-confirmed; this reproduction should not be described as a native numerical failure.

Integral metadata reproduction:

```bash
PYTHONPATH="$PWD" TMPDIR=/data/ycfeng/tmp /usr/bin/python -m pytest tests/unit/test_sglang_graph_replay_orchestration.py::test_replay_requires_exact_integer_metadata -q -p no:cacheprovider > /data/ycfeng/tmp/pr33-w05-preflight-regression/w07-integer-red.log 2>&1
```

Observed **1 failed in 2.38s**, with `Failed: DID NOT RAISE AssertionError` for actual integer metadata `[97, 64]` versus expected `[96, 64]`.

Final integrated targeted command:

```bash
PYTHONPATH="$PWD" TMPDIR=/data/ycfeng/tmp /usr/bin/python -m pytest tests/unit/test_sglang_routed_sorting.py tests/unit/test_sglang_graph_replay_orchestration.py tests/unit/test_sglang_experimental_increment13.py -q -p no:cacheprovider > /data/ycfeng/tmp/pr33-w05-preflight-regression/w07-final.log 2>&1
```

Observed **25 passed in 3.06s**. `git diff --check` passed for the owned changed tracked source/test files.

### Acceptance evidence

1. Sorting checks loads below, exactly at, and above the 32-token block boundary, including multiple active multi-block experts and an inactive expert. The fake AITER boundary independently builds packed block owners and padding from the actual top-k assignment tensor; the production validation logic is executed unchanged except for the repair.
2. The existing CPU graph harness now also exercises routed sorting and experts. Stateful tensors increment on every invocation, so the expected result is achieved only if capture/replay and repeated trace invocations reset their buffers correctly.
3. Separate eager and replay-only corruption cases must raise AssertionError before returning a success row. Both pass after migration to common correctness checks.
4. Representative trace callbacks execute twice and preserve reset/correctness checks. Rows retain `measurement_type=HIP_GRAPH_REPLAY`, explicit histogram provenance, two trace-probe invocations, and the common callback contract.
5. Existing ordinary primitive, invalid-count, shape/provenance and trace-import tests remain green.

### Native evidence limits

**Native AMD/MI355X/AITER/HIP graph execution: NOT RUN in this subtask.** The retained task hardware boundary remains unavailable AMD hardware; no GPU inspection, worker command, package installation, or fabricated native pass was attempted. CPU simulated graphs establish orchestration and validation correctness, not kernel numerical correctness, timing fidelity, or actual native trace coverage. Native T12/T14 lanes remain separate acceptance work.

## Other W07 findings and handoff

- Standard MoE DEVICE_EVENT admission and backend-label truth were reported to the parent and remain in its ownership.
- Existing accelerator visibility, collective dtype/rank evidence, and native compatibility checks remain covered by the W00 profiling ledger, not implicitly closed by these 25 tests.
- Serialized `graph_replay` plan validation/rank boundary gaps, routed packed-shape divisibility checks, and top-k native-order versus reference-order concerns from P08 remain separate bounded follow-up findings. They were not silently repaired or claimed complete here.
- No additional source changes are pending for the scoped sorting/replay fixes. Next: parent review and commit the verified sub-step, record this evidence in the primary W07 test report, and preserve separate native hardware dispositions.

## W07 continuation: standard MoE and serialized boundary closure

The parent extended ownership after commit `c1e924db` to standard MoE, serialized replay admission, and the existing collectives precision boundary. The earlier scope description above describes the first increment only. This continuation edits eight production files and adds three focused CPU test modules; no existing test module was changed in this increment.

### Source corrections

- `moe/moe_impl.py`: remove the broad `AssertionError`/`RuntimeError` catch around `ReplicatedLinear`. Native constructor failures propagate. Explicitly disabled or unavailable native linear support retains the existing torch path. Record the selected linear implementation in gating output. Inspected local vLLM 0.10 source and the retained July source snapshot both resolve `disable_tp=True` to rank zero without querying the TP group; no speculative distributed-state fallback was introduced.
- `moe/moe_vllm_kernel.py`: pass actual `model_type` into the native config and include it in the cached-state key. The pinned online MXFP4 adapter admits Qwen3.5 MoE on ROCm with the functional API; unsupported identity/platform/API and malformed packed dimensions fail before allocating inputs/weights. The already unsupported functional FP8 path now fails before allocations. Validate packed weights/scales after quantization by their single-byte element storage and planned byte counts, permitting native shuffles/reshapes and packed dtype views. The selected backend must identify AITER. Return the actual selected backend with the timing result.
- `moe/moe_wrapper.py`: admit MXFP4 before output-directory and native-layer construction; extract the actual backend from the per-call kernel result and publish `moe_native_backend` outside numeric timing statistics. Keep the existing implementation-family label for compatibility. No hidden global provenance or cross-call mutable backend state was added.
- `experimental/sglang/graph_replay.py`: serialized plans re-enter `build_replay_plan`, including primitive/logical/context checks. Visibility no longer bypasses `rank < world_size` admission.
- `experimental/sglang/moe.py`: builder and serialized routed specs share hidden group-32 and local intermediate group-256 alignment validation, avoiding truncated packed shapes.
- `collectives/{collectives_input,main,benchmark_runner}.py`: one declarative precision-to-torch-dtype-name mapping in the existing input boundary owns choices and resolution. CLI/local/Ray callers reuse it; torch remains lazy at that boundary. Actual allocation dtype, bytes, rank selection, collective kernels, and timing aggregation remain unchanged.

Physical-layout source reference: retained local snapshot `.../task_memory/task_2026-07-26_step4_model_architecture_analysis/src_snapshots/sha256_b7791e32a027/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`, AITER branches near lines 980–1085. These branches shuffle scales/weights and view packed bytes as native FP4/FP8 types. The storage checks are executable validation, not evidence that a native online MXFP4 run has occurred.

### Exact CPU verification and observed results

Environment remains `/usr/bin/python` 3.12.3, pytest 9.1.1, no active conda environment. Commands were executed from the worktree root with `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp`; no GPU command or worker was launched.

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp python -m pytest tests/unit/test_sglang_replay_admission.py -q -p no:cacheprovider
```

RED before the replay/spec repair: **11 failed, 1 passed in 0.83s**. Invalid serialized primitive/logical/context values, visible rank outside world size, and truncated/misaligned packed dimensions were accepted. Log: `/data/ycfeng/tmp/pr33-w05-preflight-regression/w07-admission-red.log`.

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp python -m pytest tests/unit/test_sglang_replay_admission.py tests/unit/test_moe_gating_constructor_boundary.py tests/unit/test_moe_native_admission.py tests/unit/test_moe_fused_event_contract.py tests/unit/test_moe_mxfp4_increment10.py -q -p no:cacheprovider
```

GREEN: **52 passed in 3.41s**. Deterministic native-boundary simulation verifies constructor error propagation, forbidden allocation for invalid MXFP4/FP8 requests, actual per-call backend propagation to the output row, and missing/unpacked/truncated physical storage rejection. CPU packed tensors cover storage checks; simulated native state is explicitly not native validation. Log: `.../w07-boundary-green.log`.

An expanded verification command first named the nonexistent `tests/unit/test_profiling_moe_routed_width.py`; pytest exited 4 with `file or directory not found`, **no tests ran in 0.72s**. This was a command-path error, not a production failure. Preserved log: `.../w07-moe-sglang-regression.log`. After inspecting actual test paths, the corrected bounded command was:

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp python -m pytest tests/unit/test_sglang_replay_admission.py tests/unit/test_sglang_routed_sorting.py tests/unit/test_sglang_graph_replay_orchestration.py tests/unit/test_sglang_experimental_increment13.py tests/unit/test_moe_gating_constructor_boundary.py tests/unit/test_moe_native_admission.py tests/unit/test_moe_fused_event_contract.py tests/unit/test_moe_mxfp4_increment10.py tests/unit/test_moe_profiling_output_metadata.py tests/unit/test_moe_routing_runtime.py tests/unit/test_moe_shared_routing_helper.py tests/unit/test_moe_load_distribution_contract.py tests/unit/test_moe_routing_input_contract.py -q -p no:cacheprovider
```

Regression result: **110 passed in 5.28s**, log `.../w07-moe-sglang-regression-corrected.log`. These include the earlier sorting/replay corrections and existing routing/output contracts.

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp python -m pytest tests/unit/test_collectives_rocm_runner.py tests/unit/test_collectives_increment11.py -q -p no:cacheprovider
git diff --check
```

Collectives consolidation: **9 passed in 2.95s**; existing cases check precision, CLI choices, dtype resolution/rejection, element-size byte accounting, topology admission, and flat CSV output. Log `.../w07-collectives-green.log`. `git diff --check` emitted no errors.

### Full profiling hunk disposition at this checkpoint

The original W00 ledger remains the exact inventory of all **39 changed production files / 100 pinned-baseline hunks**. The following closure mapping covers every listed file; it supersedes the original proposed BUGFIX/REFACTOR action for the repaired hunks, without converting source review into native PASS.

| Original ledger files under `frontier/profiling/` | Current disposition |
| --- | --- |
| `attention/backends/{__init__,vllm_rocm_attention_wrapper}.py` | RETAIN explicit lazy ROCm adapter and existing metadata boundary; native attention ABI/output acceptance NOT RUN. |
| `collectives/{benchmark_runner,collectives_impl,collectives_input,collectives_wrapper,main}.py` | REPAIRED duplicated dtype/precision ownership; retain actual dtype/byte arithmetic and local finally teardown. Rank-zero reporting is the implemented contract, not rank-max evidence. Native NCCL/RCCL NOT RUN. |
| `common/accelerator.py` | RETAIN common visibility owner. Local collective workers correctly use runtime logical ranks under inherited visibility; discovery's physical-ID list is used only to establish available count in that path. Empty-variable behavior is an unresolved launcher-policy question without a supported failing scenario; no behavior change justified. |
| `common/{constants,cuda_timer,device_timer,timer_stats_store}.py` | Parent W05 owns and repaired timer/store admission; alias and operator declarations retained. This subtask does not duplicate that evidence. |
| `common/layers/{layernorm,rotary_embedding}.py`, `common/vllm_compat.py` | RETAIN bounded native compatibility/context boundary; no newly established source defect. Native version matrix NOT RUN. |
| `common/model_config.py` | Parent W01 owns normalized model/layer contracts; this subtask leaves its final disposition to the W01 evidence. |
| `experimental/__init__.py`, `experimental/sglang/__init__.py` | RETAIN explicit isolated experimental surface. |
| `experimental/sglang/{attention,dense,gdn,gdn_trace}.py` | RETAIN CPU shape/trace contracts and lazy native construction; no additional source-proven defect established. Native numerical/trace identity NOT RUN. |
| `experimental/sglang/{graph_replay,moe,routed_moe_replay}.py` | REPAIRED sorting ownership, common replay checks/reset/trace callback, exact integer checks, serialized plan/rank admission, and packed alignment. CPU regression PASS; native HIP graph/AITER NOT RUN. |
| `gdn/{__init__,inputs,main,vllm_wrapper}.py` | Parent W05 owns campaign preflight, ownership, timing samples, native acceptance entrypoint; original findings P01–P04 transferred and regression cases supplied. |
| `linear_op/{linear_op_impl,linear_op_wrapper,main,profiling_plan}.py` | RETAIN actual implementation selection, typed plan metadata, accelerator reuse, and explicit quantization boundary; native result/ABI evidence NOT RUN. |
| `moe/{main,moe_impl,moe_vllm_kernel,moe_wrapper}.py` | REPAIRED fused event admission by parent plus explicit constructor failures, quantization preflight, actual backend/config identity, packed storage checks, and existing shared routing use. CPU adapter evidence PASS; native optimized kernels NOT RUN. |
| `utils/{__init__,confirmation}.py` | RETAIN shared measurement choices and existing confirmation/discovery helper; no newly established source defect. |

### Remaining native and policy limitations

1. Native top-k ordering: the eager adapter accepts AITER expert ordering, while generic replay compares returned weights to the sorted torch reference. No installed supported native AITER/SGLang execution or exact ordering evidence established an actual divergence. This remains a specifically identified verification limit, not a claimed fixed defect. A future native case must exercise it before changing the measured boundary or introducing canonicalization.
2. Empty visibility variables: current common discovery treats them as absent; no supported launcher scenario demonstrated intended no-device semantics. Preserve existing behavior pending that evidence.
3. Native hardware lanes: repository inventory at this checkpoint contains collected `tests/integration/test_device_timer_native.py` and `test_gdn_native_acceptance.py` owned by W05. No corresponding collected attention/MXFP4/collective/SGLang native acceptance module was found. Those W07 native execution lanes remain an explicit delivery gap, separate from the passing CPU source/adapter tests. This subtask has not claimed hardware SKIPs for tests that do not exist.
4. Neither actual native packed layout correctness nor kernel numerical equivalence follows from successful CPU simulation. The new storage validator will enforce its contract on a real run; hardware execution remains NOT RUN.

## W07 native entrypoint delivery follow-up

The parent explicitly assigned one new module, `tests/integration/test_pr33_native_profiling_acceptance.py`, to close the missing collected-entrypoint gap recorded above. That delivery gap is now closed. **Native execution remains NOT RUN**: collecting tests and observing capability SKIPs on the CPU host does not establish GPU correctness.

The module imports only standard-library modules and pytest at collection time. GPU packages and Frontier native producers are imported behind capability fixtures. It provisions no worker, installs no dependencies, changes no backend-selection flags, and substitutes no native implementation. Requirements are precise: available CUDA/ROCm device; gfx950/MI355X for the AMD-specific lanes; required vLLM/AITER/SGLang modules; two visible devices and the NCCL/RCCL process-group backend for collectives; explicit existing AITER selection environment for online MXFP4; and a clean process-group context for isolated SGLang replay. Missing hardware/dependency capabilities SKIP; errors after admission remain failures.

### Real native cases and acceptance criteria

| Collected lane | Real entrypoint and acceptance |
| --- | --- |
| ROCm attention prefill and decode (2 cases) | `VllmRocmAttentionWrapper.init/begin_forward/forward/end_forward`; independent float32 causal attention reference, finite outputs, one measured active-phase sample. Decode primes the real paged KV cache with the prefix. |
| Online vLLM/AITER MXFP4 (1 case) | `profile_fused_moe_kernel` with a small 2-expert, 256-wide synthetic workload; actual selected AITER backend, finite nonnegative native timing summaries with positive mean, and packed native state storage validation. This is adapter/layout/timing acceptance; it does not claim full nonzero quantized numerical parity. |
| NCCL and RCCL collectives (4 cases) | Existing `_run_local_collective` spawns the real production workers and `CollectiveWrapper`, separately for BF16/FP32 and CUDA/ROCm. Requires two visible devices, rank-zero result, positive timing, and exact `128 * element_size` bytes. The existing production worker `finally` owns group teardown. This is native profiling/byte-accounting acceptance, not an all-reduce numerical oracle. |
| SGLang top-k, multi-block sorting, and MXFP4 experts (3 cases) | Real SGLang process groups plus existing `profile_graph` / `profile_routed_graph`, two invocations and five measured replays. Builders' native output/reference checks and mutable-buffer resets execute unchanged. The top-k lane uses 4 experts/top-2 to expose ordering differences; sorting uses counts 33/31. The returned trace callback runs under the real device profiler, exports Chrome trace, and `attach_kernel_evidence` must identify native kernels. Output stays `HIP_GRAPH_REPLAY`/experimental. The existing expert builder uses an exact-zero packed-weight oracle, which is explicitly weaker than general numerical parity. |

SGLang initialization/get-group/teardown signatures were checked against the [official SGLang parallel-state source](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/distributed/parallel_state.py). The fixture initializes its own one-rank group and destroys model-parallel and distributed state in `finally`; it does not borrow or destroy an existing owner. Runtime ABI execution is still unverified on this CPU host.

### CPU collection and capability evidence

Read `/data/ycfeng/stepfun-env-handbook/guidence.md` before the native capability check. No allocation job, `rlaunch`, Docker command, or native kernel was executed. Environment: `/usr/bin/python` 3.12.3, pytest 9.1.1, no active conda environment; temporary/cache root `/data/ycfeng/tmp`.

```bash
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp python -m pytest tests/integration/test_pr33_native_profiling_acceptance.py --collect-only -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp python -m pytest tests/integration/test_pr33_native_profiling_acceptance.py -q -rs -p no:cacheprovider
git diff --check
```

Observed results:

- Collection: **10 tests collected in 0.76s**. All four requested component lanes have concrete node IDs without importing native GPU packages during collection.
- Initial CPU execution: **10 skipped in 4.32s**.
- After moving SGLang capability checks before process-group initialization and making top-k ordering coverage meaningful: **10 skipped in 2.55s**.
- Exact reported reason for every CPU-host case: `Native profiling requires an available CUDA or ROCm GPU`.
- `git diff --check`: no errors.
- Logs: `/data/ycfeng/tmp/pr33-w05-preflight-regression/w07-native-collection.log`, `w07-native-cpu-skips.log`, and `w07-native-cpu-skips-final.log`.

On an already authorized worker, use the same test command in the existing native profiling environment. The online MXFP4 fixture requires the already documented `VLLM_ROCM_USE_AITER=1` and `VLLM_ROCM_USE_AITER_MOE=1` selections before Python starts; tests do not set these flags automatically. NCCL/RCCL cases require two visible devices. GPU-worker acquisition and environment setup remain governed by the handbook and parent authorization, outside this test module.
