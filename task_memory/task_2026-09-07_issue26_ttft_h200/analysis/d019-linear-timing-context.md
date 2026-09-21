## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Validated profiles-04: large repeatable event-only context effect, small timer-variant effect, unchanged kernels, and profiler-induced queue perturbation. |
| 2026-09-08 | Preserved profiles-03 failure and corrected per-label invocation-count validation from actual wrapper execution semantics. |
| 2026-09-08 | Prepared a bounded same-wrapper event-order and queue-context diagnostic; GPU execution pending. |

# D019 linear timer/context diagnostic

Status: **GPU diagnostic and independent artifact checks PASS; production correction and CUDA/TTFT gate remain pending**. Owner `/root/repair_review`; root controls H200 scheduling. Existing profiling data and production timers were not changed.

The question is why the same BF16 QKV-plus-QK-norm implementation measures approximately 0.146123 ms per layer in fresh Frontier profiling versus 3.870016/48 = 0.080625333 ms in the newer low-density vLLM scope. The direct F.linear and norm source match does not prove identical kernel execution time, launch gaps, or queue history. This experiment isolates two candidate influences without fitting either to the observed residual.

Script: `tests/performance/issue26_linear_timing_context.py`.

## Method

- Construct one existing `LinearOpWrapper` with its normal TP4 profiling plan, BF16 model, 4096 positions and input IDs. Preserve its QKV, Q/K normalization, RoPE, and synthetic output-projection implementation unchanged. Inputs, model weights and RNG seed are shared across all conditions.
- Run the original CudaTimer and a process-local variant that constructs both Event objects before recording start. The variant keeps deferred elapsed-time collection and adds no per-op synchronization. It is a diagnostic implementation of the construction-order control, not a production timer patch. CUDA Event objects can initialize their underlying events lazily on first record; moving Python object construction does not automatically remove every event initialization cost.
- Cross these timer modes with three explicitly named contexts: `synthetic` retains the ordinary queued model iterations; `cold_drain` synchronizes before each model iteration; `prefill_hot` directly calls the existing `MoEWrapper._run_prefill_hot_gating_prefix` before each model iteration. The hot prefix's initialization method and repeat constant are reused; its output is discarded so the tested wrapper inputs remain identical. It is the existing synthetic prefix, not a replayed real vLLM MoE layer.
- Use three warmups and twenty active samples per condition, two rounds with AB/BA timer order and reversed context order. Store raw per-op samples as well as mean/median/min/max. No artificial sleep, workload-dependent factor, or CPU residual is introduced.
- After event measurements finish, run one separate Torch profiler lifecycle covering all six conditions, two iterations each. Add source-timer annotations and record shapes; reuse the existing correlated-kernel analysis helper. Export raw Chrome trace, individual kernel/memcpy activities, families, and launch/device correlation coverage. These perturbed kernel observations diagnose implementation/queue behavior; they are not numerical replacements for unprofiled CUDA-event samples.

The cold/hot controls apply before the whole synthetic model iteration. They do not claim that each later operator starts on an empty or equally deep queue. The unmodified wrapper's intermediate work, including `randn_like(q)` before output projection, remains part of the context. The trace is needed to inspect the resulting queue behavior for each operator.

## Root-managed GPU command

Run serially with other diagnostics in the approved H200/step_main image and company proxy environment. Set `D019_LINEAR_CONTEXT_OUTPUT` to a new, nonexistent NFS evidence directory; the script refuses to overwrite an existing directory.

```bash
export PYTHONPATH=/data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907:/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908
export TMPDIR=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1
export VLLM_FRONTIER_INSTRUMENTATION=0
/local/ycfeng/anaconda3/envs/vllm-bs-0.10.2/bin/python \
  /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907/tests/performance/issue26_linear_timing_context.py \
  --output "$D019_LINEAR_CONTEXT_OUTPUT"
```

Single process, logical CUDA device 0; TP4 controls local operation shapes without constructing distributed allreduce groups. Do not run overlapping work on that GPU. The script saves interpreter, imported vLLM path, Torch/CUDA/GPU/model/TP identity, profiling plan, prefix repeat count and Frontier HEAD. Root's existing launch receipt should pin the actual image and source working tree.

## Acceptance and interpretation

1. Verify imported runtime and source identities; QKV/RoPE/output projection each supply 20 positive event samples for every round/context/timer pair.
2. Compare timer variants within the same context and inspect both orderings. A stable reduction supports an event-timing implementation contribution; it does not prove that every such gap is removable from real vLLM execution.
3. Compare contexts within the same timer implementation. A context effect with unchanged correlated kernel identities and similar active durations supports a queue/submission component; changed kernels or durations require a kernel/context explanation. The timer variant also changes Python branch execution slightly, so small differences must not be assigned entirely to Event allocation without further evidence.
4. Require complete launch/device correlations for kernel attribution. `PARTIAL_KERNEL_COVERAGE` preserves usable event samples but fails the kernel-coverage gate. A successful process alone does not settle coverage or causality.
5. Never add differences between event samples and profiler kernel sums as a CPU correction. The two measurement passes differ, and kernel-active time is a separate family. No production correction or cache/profile replacement follows automatically from this experiment.

## Preparation verification

Executed on CPU master using conda `dev-vidur-v03-hopper-e2e`, Python 3.13.13:

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PY'
import ast, pathlib, subprocess, sys
path = pathlib.Path('tests/performance/issue26_linear_timing_context.py')
ast.parse(path.read_text())
result = subprocess.run([sys.executable, str(path), '--help'], capture_output=True, text=True, check=True)
assert '--output OUTPUT' in result.stdout
assert '--samples SAMPLES' in result.stdout
print('PASS: AST and CPU-only CLI; GPU execution pending')
PY
```

Observed **PASS**. This detects Python syntax/CLI regressions without importing GPU dependencies; it does not verify runtime kernels or timer outputs. Source inspection also caught the model dtype property being read-only; the preparation now asserts the model's existing BF16 dtype rather than assigning that property. No GPU failure was masked. Pending: first GPU execution, interpretation of raw samples/correlations, and source-backed next correction if justified.

## Profiles-03 failure and correction

The first GPU attempt failed at the original assertion requiring exactly 20 records for every timer label: `AssertionError: ('emb', [40 positive values])`. Evidence: `h200-d019-profiles-03/runtime/timing-context.log`; the original script snapshot and RUNNING receipt are retained in that run. No complete event-sample file or kernel trace was produced, so this attempt does not establish a timing comparison.

The cause is a diagnostic count assumption. `GPTModel.forward` calls `embed_tokens` once before the repeat loop (`linear_op_impl.py:1089`) and once inside its single repeat (`:1092`). This gives two sequential embedding executions per iteration, rather than duplicated nested measurements. Other labels also have distinct semantics: `add` can appear under both input and post-attention norm, nested within the corresponding norm scope. Their samples must not be silently dropped or added to their inclusive parent timings.

The bounded script correction observes each label's call count during the first warmup and requires the same multiplicity in the next two warmups. Active samples must then equal `20 × calls_per_iteration`, remain positive, and retain exactly the same labels. QKV, RoPE and output projection still require exactly one invocation per iteration. Raw flat samples remain intact, and `samples_by_call_ms` separates repeated invocation positions. This repairs the failed assumption without weakening target-op coverage or changing the wrapper.

A CPU direct-source check executed the actual `GPTModel.forward` AST with tensor-free embedding/block doubles and observed `['emb', 'emb', 'block']`. A sample-partition check verified that 1-call and 2-call data each yield 20 samples per position and reconstruct the complete original observation sequence without loss. Python AST validation also passed. Corrected GPU run remains pending; root will run only the context phase in profiles-04, preserving the completed C sweep from profiles-03.

## Profiles-04 observed results

Corrected execution: `h200-d019-profiles-04/runtime/timing-context/`; status `COMPLETE_DIAGNOSTIC`. The receipt confirms NVIDIA H200, Python 3.10.16, Torch 2.8.0+cu128, diagnostic vLLM checkout, BF16, tokens4096, TP4, hidden2048, Q32/KV4, and QK norm enabled. The reused prefix runs its existing20 repetitions at expert width768. All72 event rows across6 labels and12 conditions have the required positive finite sample count and call-position partition. There are608 kernel launches and zero missing device correlations. Each target operation has12 independent profiler scope instances; each instance has the identical ordered kernel sequence across conditions.

All values in the next table are milliseconds per operation. Each cell is the mean of the two rounds' medians, not a kernel-active time or an E2E prediction.

| Operation | Synthetic original / precreated | Cold-drain original / precreated | Prefill-hot original / precreated | Original timer: hot versus synthetic |
| --- | ---: | ---: | ---: | ---: |
| QKV + Q/K norm | 0.133432 / 0.128680 | 0.132840 / 0.131256 | 0.078192 / 0.078192 | −0.055240 ms, −41.399% |
| RoPE | 0.031784 / 0.031720 | 0.031664 / 0.031720 | 0.019216 / 0.019176 | −0.012568 ms, −39.542% |
| Output projection, compute | 0.046752 / 0.046400 | 0.046792 / 0.046752 | 0.029824 / 0.029736 | −0.016928 ms, −36.208% |

The original/precreated QKV pair changes synthetic timings by−4.868% in round0 and−2.189% in reversed round1. Cold-drain differences are−1.086% and−1.299%. Hot differences reverse sign (−0.143%,+0.143%) and average to zero. RoPE and output-projection differences remain small and can change sign. Thus a dominant QKV/RoPE discrepancy caused solely by Python end-Event construction is not supported. A modest timer-variant contribution exists in some contexts, but this control also changes a few Python branches and does not isolate every underlying Event initialization operation.

The context effect is much larger and repeats in both orderings: original QKV synthetic medians0.136736/0.130128 versus hot0.078304/0.078080. The same implementation, weights, tested input sequence and timer have substantially different elapsed event timings under the existing hot prefix. This directly establishes sensitivity to surrounding execution context. Cold-drain and ordinary synthetic timings are similar; inserting a single synchronization is not equivalent to recreating the hot queue state.

### Comparison with new vLLM low-density scopes

V is the new first-real-forward DP0/TP0 attention event total divided by48, from `d019-compute-results/attention/first_batch_compute.csv`, local batch4666. The output projection counterpart is `row_parallel_gemm`, excluding TP allreduce. These are diagnostic runtime scopes, not a new clean E2E acceptance result.

| Frontier operation | Synthetic event S | Prefill-hot event S | vLLM event V | Synthetic−V / relative gap | Hot−V / relative gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| QKV + Q/K norm | 0.133432 | 0.078192 | 0.080625 | +0.052807 / +65.496% | −0.002433 / −3.018% |
| RoPE | 0.031784 | 0.019216 | 0.019576 | +0.012208 / +62.362% | −0.000360 / −1.839% |
| Output projection | 0.046752 | 0.029824 | 0.030309 | +0.016443 / +54.253% | −0.000485 / −1.599% |

These matches support further use of an explicit context-preserving measurement/selection path. They do not authorize silently replacing generic CSV rows by their shortest result. The prefix was reused with its existing repeat constant; no scale or prefix length was fitted against V. Converting the three original-timer context differences into a48-layer ledger sensitivity yields−4.067328ms. That arithmetic is not a new simulated TTFT and cannot settle other operator or communication defects.

### Kernel identity and the profiler limitation

QKV has five kernels per scope: one `nvjet_tst_320x128_64x3_1x2_h_bz_coopB_TNT` GEMM, a Q contiguous copy and RMSNorm, then a K contiguous copy and RMSNorm. RoPE uses one vLLM BF16 rotary kernel; output projection uses one `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_TNT` GEMM. All six conditions preserve these exact names and their ordering. The old complete vLLM K trace has the same corresponding compute identities; its old output-projection scope additionally includes NCCL, which is excluded from this compute comparison.

| Profiled operation | Active-time range across six conditions, mean of two scope instances | Kernel count per instance |
| --- | ---: | ---: |
| QKV + Q/K norm | 0.070561–0.071104 ms | 5 |
| RoPE | 0.016079–0.016672 ms | 1 |
| Output projection | 0.026160–0.026464 ms | 1 |

Kernel selection and observed active durations do not show a corresponding40% acceleration. However, **the profiler pass does not preserve the event-only hot timing behavior**. Its original-timer QKV scopes average:

| Context | Active union | Gap between owned device activities | Portion before next launch begins | Device envelope |
| --- | ---: | ---: | ---: | ---: |
| Synthetic | 0.071104 | 0.107233 | 0.081224 | 0.178337 |
| Cold-drain | 0.070720 | 0.075568 | 0.051974 | 0.146288 |
| Prefill-hot | 0.070561 | 0.079295 | 0.055579 | 0.149856 |

The profiler's hot device envelope0.149856ms is much longer than the separate event-only hot result0.078192ms. Profiling changes host submission/queue behavior sufficiently that its gap magnitudes cannot be assigned to the unprofiled control. Late launches directly prove submission gaps **in this profiler pass**; they do not establish CPU computation duration or a removable clean-runtime overhead. Single-kernel RoPE/output projection have zero gap *between their own kernels by definition*; this does not imply zero host/event-boundary overhead.

The supported causal conclusion is therefore: surrounding execution context strongly affects the event-based profile; large differences in selected CUDA kernels or active compute time are not supported by these captures; host submission/queue timing is a supported mechanism, while its exact clean per-op gap budget remains unmeasured. The profiler perturbation is preserved as contradictory evidence against a stronger numerical attribution.

## Decisions supported by this diagnostic

1. Do not apply a global CudaTimer production correction on the claim that end-Event construction explains the large gap. The observed effect is too small and context-dependent.
2. Do not retrain the same context-insensitive profile grid as the sole QKV/RoPE repair. The same op at the same shape changes materially with execution context even before any ML prediction.
3. A bounded explicit prefill context measurement/selection correction is now source- and measurement-supported for these three first-forward operators. Its integration must preserve context provenance and examine the reached runtime scopes; generic all-shape replacement and subtraction of a constant are unsupported.
4. Do not add these event/kernel differences to workflow CPU overhead. They concern work/gaps inside the modeled forward event interval, and the profiler data are independently perturbed.

Pending tasks: reviewed context-profile integration if selected by root/YC, fresh predictor consumption and first-forward validation; remaining op/communication errors remain independently owned. New production bugs established by this experiment: none. Newly established measurement issue: substantial context sensitivity and profiler perturbation. The profiles-03 diagnostic multiplicity failure is resolved and retained above.

Numerical evidence: `d019-linear-timing-context-results.json` contains all36 target profiler scope instances, their full kernel sequences, active unions/envelopes/gaps, six-condition event summaries and signed S/V comparisons. Raw event samples and trace/correlation evidence remain in the profiles-04 output. A dedicated reproducible verification report is `../test_report_2026-09-08_linear_timing_context.md`.

## Delivery record

Diagnostic script committed as `578a4b8d` (`test: diagnose linear CUDA timing and execution context`) after profiles-04 GPU and independent artifact checks passed. The commit contains only `tests/performance/issue26_linear_timing_context.py`. Reports and selected evidence remain persisted in the task directory under the existing Git ignore policy.
