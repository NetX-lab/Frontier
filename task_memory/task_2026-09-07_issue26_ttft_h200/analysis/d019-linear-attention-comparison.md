## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Compared three fresh4096 linear and attention repeats with exact P and qualified old vLLM event scopes; retained initial attention serialization failure and verified continuation. |

# D019 linear/attention P–S comparison

Status: seven linear/memory/attention operators COMPLETE for P–S attribution. This report is read-only and does not replace existing P or profile files. CUDA-span RCA/correction remains a separate gate.

The old linear anchors are broadly reproduced: the means of three new sample medians are1.11–6.53% higher, with repeat-median ranges0.41–3.17% of their means. Thus the large QKV/RoPE P-to-vLLM discrepancy is not explained by large instability of the old4096 CSV anchors in these new measurements. Reprofiling the same implementation makes their costs slightly higher, rather than closing their positive diagnostic gaps. This does not identify the remaining context/kernel/host-idle cause.

Replacing only these five per-layer values by their new S means changes their48-layer arithmetic sum by **+0.557568ms**. This is an operator-ledger sensitivity calculation, not a new simulator or E2E result; it shows the observed anchor refresh is much smaller than a20ms-scale discrepancy.

Fresh prefill attention reproduces its old anchor within−1.02%. KV save is the clear unstable small operator: its three fresh medians range0.012192–0.018400ms,43.40% of their mean0.014304ms. Relative to old P, that mean reduces the48-layer KV-save ledger by only0.120576ms. Across all seven operators the corresponding arithmetic refresh is **+0.365569ms**. No20ms-scale anchor error is established by these measurements.

## Values and timing boundaries

All table values are milliseconds per layer. P is the actual exact lookup value from `d019-predictor-query/query_receipt.json`. S is the mean of three repeat medians. Linear repeats run in separate processes with20 active measurements after3 warmups; attention repeats call the same wrapper three times with5 active measurements after3 warmups each. V_old is the first-forward48-layer event-scope total divided by48 from the prior all-op instrumented vLLM diagnostic. That run's full forward was116.210144ms, while batch-only timing was80.335617ms; V_old is a diagnostic comparator, not clean ground truth or kernel-active time. It must not be used as a clean CUDA-span acceptance gate.

| Operation | P | S repeat1 / repeat2 / repeat3 | S mean | (S−P)/P | V_old event | (S−V_old)/V_old |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| QKV + split + Q/K norm | 0.137168 | 0.146912 / 0.143552 / 0.147904 | 0.146123 | +6.53% | 0.081819 | +78.59% |
| RoPE | 0.032336 | 0.033088 / 0.032832 / 0.033888 | 0.033269 | +2.89% | 0.021438 | +55.19% |
| Attention output projection, compute | 0.046784 | 0.047856 / 0.047264 / 0.047712 | 0.047611 | +1.77% | missing pure scope | — |
| Input layernorm | 0.022608 | 0.023312 / 0.023216 / 0.023264 | 0.023264 | +2.90% | 0.026390 | −11.85% |
| Post-attention layernorm | 0.022112 | 0.022448 / 0.022208 / 0.022416 | 0.022357 | +1.11% | 0.025925 | −13.76% |
| Prefill attention | 0.146384 | 0.146176 / 0.145088 / 0.143424 | 0.144896 | −1.02% | 0.125984 | +15.01% |
| KV cache save | 0.016816 | 0.012320 / 0.018400 / 0.012192 | 0.014304 | −14.94% | 0.015027 | −4.81% |

Output projection V_old includes TP allreduce in its parent scope; no pure S/V gap is calculated by subtracting communication from another measurement. Layernorm V_old averages include actual first/subsequent-layer residual behavior; source/kernel coverage remains relevant to exact one-to-one interpretation.

## Identity and source validation

Fresh CSVs: `h200-d019-profiles-01/runtime/linear-{1,2,3}/compute/h200/qwen3-a3b-30b-moe/linear_op.csv`. Each contains two rows: physical line2 supplies replicatedTP1 layernorm/embedding; line3 suppliesTP4 QKV/RoPE/output projection. All selected rows were checked for tokens4096, BF16, CUDA_EVENT, no quantization, hidden2048, Q32/KV4, use_qk_norm=True, and the expected TP. The model reports generic architecture metadata consistently with the prior selected profile; no architecture mismatch is inferred from the label alone.

The saved worker/launch receipt pins H200/step_main and the approved image digest `b97a2fdc624953679562a4835ea66e085ec22563f2086fecfe6710a9e682dbbc`. GPU probe records8 H200 devices,143771MiB each, driver570.124.06, Python3.10.16 in `vllm-bs-0.10.2`, Torch2.8.0+cu128 and FlashInfer0.3.0. The probe itself imports `/data/ycfeng/tmp/vLLM-BS`; the worker then changes PYTHONPATH to the pinned diagnostic checkout4bc1bc026c91dff78bd7cf5ba6411f14d15e043d, and all three actual linear logs confirm that diagnostic import path. The run receipt records Frontier base388ae7a3 plus a MoE-only production diff; linear sources were not changed by that patch.

Old source profiling used the same BF16 CUDA-event wrapper over41 token sizes and8 independently assigned profiling GPUs. New profiling uses one GPU and only4096 per process, repeated three times. Their process placement and surrounding workloads therefore differ; these are measurements of anchor reproducibility under the new bounded context, not identical replayed GPU histories.

Fresh attention source is `h200-d019-profiles-02/runtime/attention/samples.json`, produced in a separate dedicated H200 allocation with the same pinned image/runtime. All three rows were checked for hidden2048,Q32/KV4,TP4,BS1,prefill4096,KV0,block16,max-model-len16384,FLASHINFER and5 active samples. The saved script explicitly passes `torch.bfloat16`; the JSON does not itself contain a dtype field, so precision attribution uses that source receipt. The wrapper receives310809 blocks, matching the frozen vLLM KV allocation. All per-sample timing statistics are retained in the comparison JSON.

## What the inspected source can establish

1. **Autograd is already disabled.** `LinearOpWrapper.profile` is decorated with `torch.inference_mode()` (`linear_op_wrapper.py:180`), and model construction applies the configured dtype, `.cuda()` and `.eval()` (`:88`). `AttentionWrapper.profile` is also decorated (`attention_wrapper.py:321`). Missing inference mode is not supported as the explanation for these profiles.
2. **CUDA unquantized GEMM dispatch agrees at the primitive level.** Frontier `_run_unquantized_gemm` calls `torch.nn.functional.linear` on CUDA (`common/parallel_utils/tensor_parallel_layers.py:212`). vLLM `UnquantizedLinearMethod.apply` calls its dispatcher, whose non-ROCm/non-CPU default also calls `torch.nn.functional.linear` (`vllm/model_executor/layers/utils.py:88,190`). This excludes a blanket claim that one side uses a different high-level GEMM primitive. Actual chosen kernel, memory placement and launch gaps still need activity/shape evidence.
3. **QKV scope intent agrees.** Frontier `CausalSelfAttention.forward` wraps GEMM, split and both Q/K norms (`linear_op_impl.py:500`). vLLM `Qwen3MoeAttention.forward` wraps the corresponding operations (`qwen3_moe.py:306`). Frontier's QKNorm delegates to the same vLLM RMSNorm kernel functions (`common/layers/layernorm.py:64`); this is not an unfused pure-PyTorch norm fallback.
4. **Event construction position differs.** Frontier creates its start event on entry and its end event on exit (`common/cuda_timer.py:55,94`). vLLM creates both Event objects before recording the start, then records the end on exit (`vllm/v1/utils.py:375`). Thus Frontier places Python end-Event construction between the two event records. This is a concrete candidate for host-induced idle inside a CUDA-event interval, but its magnitude has not been measured; it does not establish that this alone causes the QKV/RoPE gap. Both paths defer elapsed-time collection and do not synchronize after each normal CUDA-event scope.
5. **Synthetic surrounding work differs.** The linear profiler loops a synthetic layer and creates `torch.randn_like(q)` before its output projection (`linear_op_impl.py:545`), whereas vLLM executes attention and then output projection in a48-layer real forward. The tensor creation is outside the output-projection timer, but can affect queue depth/cache context. No numeric correction is inferred from this source difference.

Fresh repeat variation alone is small compared with the remaining QKV/RoPE diagnostic gap. It is not a complete host-idle or kernel attribution: no fresh linear kernel trace or precreated-event A/B has yet been collected, and three repeat medians do not provide a population confidence interval.

Within-run samples are less tight than their medians: fresh QKV sample minima are0.137824–0.141152ms and maxima0.183168–0.194816ms, with std0.010799–0.015337ms. The raw distributions are not available beyond the recorded summary statistics, so this spread cannot be classified into host launch stalls, kernel variation or hardware state from CSV alone. Even the smallest recorded QKV sample exceeds the old vLLM all-op scope mean0.081819ms substantially.

## Attention collection result

The initial `profiles-01` attention process completed its calls but failed while serializing `AttentionBackend` into JSON: `TypeError: Object of type AttentionBackend is not JSON serializable`. It produced no usable `samples.json`; elapsed profile execution cannot substitute for recorded timings. Root corrected enum serialization and ran the isolated `h200-d019-profiles-02` attention/communication continuation. The three fresh samples now pass the recorded shape/backend/count checks and are incorporated above. The original failure remains visible.

The two old exact attention rows68/82 have identical recorded non-timing metadata. Their KV-save medians differ substantially,0.023167999 versus0.010463999ms, while prefill-attention medians are0.146911994 versus0.145855993ms. Both rows have only5 active samples. Thus a different recorded batch/KV shape does not explain the KV-save spread. The new same-input repeats also show wide KV-save variation, with within-repeat maxima0.027744–0.029536ms and minima0.010624–0.012224ms. Existing metadata cannot separate launch timing, individual kernel variation or memory context; declaring a proven host-gap correction would require a corresponding trace or controlled timing-context comparison. Its absolute48-layer influence remains small in this experiment.

## Reproduction and evidence

The exact GPU commands, saved scripts, runtime environment and launch configuration are under `h200-d019-profiles-01/` and `h200-d019-profiles-02/`. Each selected CSV/JSON sample's full min/max/mean/median/std/count and physical row or JSON index are preserved in `d019-linear-attention-comparison.json`. CPU comparison uses `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python` (Python3.13.13), Python stdlib CSV/JSON/statistics, with no GPU calls or model fitting.

For each selected operation, read its nonempty `time_stats.<op>.median` from each fresh CSV, check its TP/token/precision/measurement tuple, compute the arithmetic mean of the three medians, and compare with the executed P receipt. Signed percentages use `(S−P)/P×100` and, only for supported scope pairs, `(S−V_old)/V_old×100`. The raw JSON retains all inputs and computed values; old P/V artifacts are unchanged.

Observed arithmetic check: **PASS** for seven distinct operators, three positive medians per operator, recomputed means/percentages and the48-layer sum. The exact check command, run from the worktree root, is:

```bash
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PY'
import json, math, statistics
from pathlib import Path
path = Path('task_memory/task_2026-09-07_issue26_ttft_h200/analysis/d019-linear-attention-comparison.json')
report = json.loads(path.read_text())
assert len(report['rows']) == len({row['op'] for row in report['rows']}) == 7
for row in report['rows']:
    assert len(row['samples']) == 3
    assert all(sample['statistics_ms']['median'] > 0 for sample in row['samples'])
    mean = statistics.mean(row['S_repeat_medians_ms'])
    assert math.isclose(mean, row['S_mean_of_medians_ms'], abs_tol=1e-14)
    assert math.isclose(100 * (mean / row['P_ms_per_layer'] - 1),
                        row['S_minus_P_percent'], abs_tol=1e-12)
assert math.isclose(sum(row['S_minus_P_ms'] for row in report['rows']) * 48,
                    report['seven_model_48_layer_arithmetic_delta_ms'], abs_tol=1e-12)
print('PASS: seven model comparisons and 48-layer arithmetic delta')
PY
```
