## Modification History

| Date       | Summary of Changes |
| ---------- | ------------------ |
| 2026-09-14 | Recorded Increment 6 GDN state/parameter memory, lifecycle, runtime-guard, and CPU regression evidence. |

# Increment 6 Verification Report — GDN State Memory, Parameter Memory, and Runtime Guards

## Execution

- Worktree: `feature-amd-sglang-gdn`.
- Python: `/usr/bin/python`, Python 3.12.3.
- Observed packages: NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, Plotly 6.8.0, PyTorch 2.5.1+cu124. `vllm`, `sglang`, and `aiter` are unavailable in this CPU environment.
- AMD/MI355X hardware is unavailable.

Focused Increment 6 and compatibility checks:

```bash
python -m pytest \
  tests/unit/test_gdn_increment6_memory.py \
  tests/unit/test_gdn_state_lifecycle.py \
  tests/unit/test_gdn_runtime_guards.py \
  tests/unit/test_gdn_semantic_core.py \
  tests/unit/test_pdaf_decode_attn_preemption.py \
  tests/unit/test_moe_ep_aggregate_admission.py \
  tests/unit/test_operator_registry.py \
  -q -p no:cacheprovider
```

Compilation and whitespace checks:

```bash
python -m compileall -q frontier tests
git diff --check
```

Full unit regression:

```bash
python -m pytest tests/unit -q -p no:cacheprovider
```

Real Qwen3.8 structural and capacity-path check:

```bash
python - <<'PY'
from types import SimpleNamespace
from frontier.config.model_config import BaseModelConfig
from frontier.scheduler.utils.memory_planner import MemoryPlanner
from frontier.types import ClusterType
from frontier.utils.param_counter import ParamCounter

name = "Qwen3.8-2.4T-A95B-Quark-MXFP4"
model = BaseModelConfig.create_from_name(name)
replica_config = SimpleNamespace(
    model_config=model, model_name=name, attn_tensor_parallel_size=1,
    moe_tensor_parallel_size=1, moe_expert_parallel_size=1, attn_dp=1,
    num_pipeline_stages=4, speculative_decoding_config=None,
)
replica = SimpleNamespace(
    num_layers=model.num_layers,
    num_layers_per_pipeline_stage=model.num_layers // 4,
    num_pipeline_stages=4,
    max_request_tokens=128,
    kv_heads_per_tensor_parallel_worker=model.num_kv_heads,
    attention_head_dim=model.get_head_dim(), total_memory_gb=1,
    memory_margin_fraction=0,
)
counter = ParamCounter(replica_config, ClusterType.MONOLITHIC)
print("layer_counts", counter.get_attention_stage_layer_counts())
print("attention_params_per_device", counter.get_num_attention_parameters_per_device())
print("attention_bytes_per_device", counter.get_attention_parameter_memory_per_device_bytes())
planner = MemoryPlanner(replica_config, replica, ClusterType.MONOLITHIC, max_num_seqs=1)
print("gdn_state_bytes_per_request", planner.get_gdn_state_memory_per_device_per_request_bytes())
try:
    planner.get_num_blocks(block_size=16, gpu_memory_utilization=1.0)
except Exception as exc:
    print("capacity_result", type(exc).__name__, str(exc))
PY
```

## Criteria and evidence

| Criterion | Expected result | Observed result |
| --- | --- | --- |
| GDN state layout | Fixed conv/recurrent bytes, TP-aware, no public speculative-token sizing | PASS; focused state-layout test passed, and speculative-token argument is rejected by the public API |
| Schedule-aware parameter memory | GDN/full-attention composition follows each PP stage; FP32 `A_log` is charged separately | PASS; focused tests passed. Synthetic PP2 counts were `((3, 1), (3, 1))`; real Qwen3.8 PP4 counts were `((18, 5), (17, 6), (17, 6), (17, 6))` |
| Fixed state reservation | Resident GDN state is reserved per request and multiplied by `max_num_seqs`; KV pages use only resident full-attention layers | PASS; memory-planner focused test passed; synthetic planner used one full-attention denominator and four request slots |
| D57 parameter policy | MLP/MoE/MTP remain at 2 bytes/parameter with a direct MXFP4 approximation comment and normal capacity checks | PASS; code comment is present; real Qwen3.8 path reached the approximation and produced `1204237977600` parameter bytes/device under the observed TP1/PP4 setup |
| State-slot lifecycle | Allocate, retain while waiting, resume same slot, release on completion/cancel; no tensor state stored | PASS; state lifecycle tests passed, including exhaustion, reuse, and idempotent release |
| Unsupported GDN runtime features | Prefix cache, P→D, speculative/MTP, PP>1, EP>1, attention-DP>1, cross-node, and state-dropping preemption fail before mutation | PASS; runtime guard and transfer/preemption tests passed. Waiting is accepted because it retains a slot |
| Non-GDN compatibility | Existing non-GDN memory planner and ordinary transfer path remain available | PASS; non-GDN planner/transfer tests passed |
| Predictor/mock compatibility | Existing admission-focused mocks continue to pass while production `ExecutionTime` values use `StageExecutionTime` | PASS; preemption suite 8/8 and MoE EP admission suite 31/31 passed |
| Full unit regression | Candidate must not add failures relative to the recorded baseline | PASS relative to baseline count: `3206 passed, 19 failed, 25 skipped, 576 warnings`; baseline was `3144 passed, 19 failed, 25 skipped`. The 19 failures are the same pre-existing missing `frontier.config_optimizer`, missing debug scripts, three analysis builder failures, and stale README/documentation contracts. The additional passing tests reflect Increment 6 and registry coverage; no new failure remains |
| Compilation and whitespace | Source and tests compile; no whitespace errors | PASS; both commands exited 0 |

The full unit command exits non-zero because the 19 baseline failures remain. They are recorded rather than relabeled as candidate failures or silently disabled.

## Regression fixes during verification

1. `VLLMv1EngineReplicaScheduler._preempt_request()` now obtains an optional `_replica_config` safely before invoking the GDN guard. Lightweight scheduler test doubles do not carry that field; fully initialized GDN schedulers still fail before request mutation.
2. MoE and disaggregation predictor dummy boundaries preserve the historical pass-through behavior for non-`ExecutionTime` test sentinels. Real production helper results continue through `StageExecutionTime.from_execution_time()`.
3. The attention-family registry test now includes the intentionally registered, execution-enabled GDN family.

## Hardware boundary

`SKIP: AMD/MI355X hardware unavailable` — no ROCm execution, real MI355X profiling, `DEVICE_EVENT` writer execution, or benchmark/groundtruth parity is claimed by this report. The real Qwen3.8 result is a CPU structural/capacity check; its modeled OOM is evidence that normal capacity checks remain active, not a serving-performance result.
