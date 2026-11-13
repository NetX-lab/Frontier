# Native balanced-routing boundary

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Inspected actual Step route and executed frozen CPU assignment bodies; reported unsupported built-in balanced assignment. |

The unchanged installed native source has no built-in deterministic balanced token assignment. Do not set `VLLM_MOE_ROUTING_SIMULATION_STRATEGY=uniform_random` and label it balanced. No routing setting was applied to capture03. The new pressure package is **not submission-ready for the balanced-aligned run** until the parent resolves this mechanism.

## Actual native route

Source root `/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm`; exact inspected source hashes are retained in `native_routing_receipt.json`.

- `models/step5/nvidia/model.py:1680–1822`: Step5MoE chooses FusedMoEFactory by default and passes correction-biased sigmoid, normalized weights and scaling factor3 from unchanged metadata;352 experts,top_k8. Explicit `deep_gemm_mega_moe` uses Step5MegaMoE instead.
- `model_executor/layers/fused_moe/layer.py:294–315`: when no router override exists, factory constructs `create_fused_moe_router`.
- `models/step5/nvidia/mega_moe.py:483–510`: mega branch also constructs that factory and calls select_experts.
- `model_executor/layers/fused_moe/router/router_factory.py:120–126`: nonempty simulation selector takes priority over the model's biased-sigmoid route. This is not EPLB; EPLB maps expert placement and replicas.
- `model_executor/layers/fused_moe/runner/moe_runner.py:620–642`: modular path calls select_experts, but monolithic quant kernels receive logits directly and bypass that call. Construction alone does not prove runtime routing is active. Original kernel/backend configuration is unchanged; no backend swap is authorized.
- `envs.py:1852–1854`: accepted strategy name is lowercased.
- `routing_simulator_router.py:226–235,279–298`: exactly `uniform_random` and `normal_routing` are registered. No balanced registration anywhere in this source. Unknown `balanced` raises ValueError.
- `routing_simulator_router.py:131–135,186–192`: uniform_random selects top-k random scores per token with no duplicates within one token, but no balancing across tokens/experts. It uses all-one weights, not model sigmoid normalization/scaling. The native warning explicitly says model outputs are invalid.

## CPU observations

Command from the owned repo:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B task_memory/task_2026-10-22_step5_d28_d24_pressure_stage/routing_probe.py
```

The read-only probe executes exact frozen native strategy class bodies with real CPU Torch tensors, not a GPU/model/factory construction. Seed0 only makes this counterexample reproducible; it is not a run-setting change.

| Tokens | Expert assignments | Min expert count | Max expert count | Exact balance |
| --- | --- | --- | --- | --- |
|8|64|0|2|No|
|44|352|0|4|No|
|128|1024|0|9|No|
|6144|49152|106|173|No|

Forty-four tokens give352 assignments, so exact one-per-expert balance would be possible, but the existing random implementation does not produce it. Statistical uniformity is not the requested deterministic balanced distribution.

## Parent decision needed

1. Provide a supported exact balanced mechanism in the locked source and demonstrate effective runtime assignment on the actual Step kernel route; or
2. Approve a separately scoped deterministic strategy via the existing registration seam, its per-call/DP/padding/TP and weight semantics, and any necessary native-kernel path changes. This is new implementation beyond this child's frozen-native pressure slice; do not silently add it.

The parent owns Frontier assignment semantics. Calibration ordering is workflow→GPU/operator gap→CPU overhead last. This investigation creates no CPU overhead measurement or suite. Capture02/original routing identities remain immutable; capture03 truthfully retains original routing pending the decision.
