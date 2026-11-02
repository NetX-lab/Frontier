# Native06 balanced routing identity RCA

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | W128-V4-NATIVE06-ROUTING-RCA: reproduced the frozen rejection, censused all 32 actual rank inventories, and proposed a reader-only contract correction without implementing it. |

## Result and limits

**Task ID:** `W128-V4-NATIVE06-ROUTING-RCA`.

The first failure is an **evidence-enumeration contract defect**, not an observed wrong balanced strategy activation. Frozen R3 derives all expected router names from `inventory.moe_backends`, an older field restricted to `Fp8MoEMethod`. The actual V4 run uses `Mxfp4MoEMethod`/MARLIN, so that field is `{}` on all 32 ranks. Independently collected routing and initialization inventories contain all 43 actual MoE runners and router owners per rank.

The retained run remains **UNVERIFIED**. The existing checker result remains seven predicates PASS and `rank_runtime_graph` FAIL. No checker, package, native/C26 source, raw run artifact or historical result was changed. No GPU, SSH, cloud, platform or Git operation was performed. No downstream graph predicate was evaluated with the failed routing identity bypassed.

## Exact first failed operands

Local case: `/data/ycfeng/tmp/glm53_dsv4/native06_case_20261023`.

Run: `runs/groundtruth_clean/dsv4-h800-p0-canonical-20261021-06`.

First file consumed in the frozen host/rank iteration:

`nodes/gpu-h800-0214.host.platform.shaipower.com/serving/rank0.json` (one-line JSON; prefill, DP owner 0, TP rank 0, global rank 0).

File SHA256: `037e9e5c108bcd18af558b7465a397ed949800fbb5e4327c4c36ce69df666a32`.

Frozen checker: `/data/ycfeng/Frontier/.worktrees/v4-balanced-native-model/task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py`.

| Operand | Actual | Frozen expectation / result |
| --- | --- | --- |
| `inventory.moe_backends` | `{}`, JSON/Python `dict` | R3:112 uses its keys as the complete routed-expert enumeration. |
| local `routers` | `set()` | R3:113 requires nonempty; **`not routers` is the first true failing condition**. |
| `inventory.moe_routing.routers` | 43 entries | Frozen expected map is `{}`; object inequality would also fail if reached. |
| `inventory.moe_routing.runners` | 43 entries | Frozen expected map is `{}`; same latent inequality. |
| sample router key | `model.layers.0.ffn.experts` | Missing from frozen expected map, not an incorrect class. Actual value is `vllm.model_executor.layers.fused_moe.router.routing_simulator_router.RoutingSimulatorRouter`. |
| sample runner key | `model.layers.0.ffn.experts` | Missing from frozen expected map. Actual value is `vllm.model_executor.layers.fused_moe.runner.moe_runner.MoERunner`. |

R3:113–121 uses a short-circuit `or`. The actual execution does **not** reach `actual != expected` because `not routers` already rejects it. Separate read-only operand comparison identifies only the `routers` and `runners` maps as unequal. The other six rejection conditions are false on all 32 ranks.

The complete scalar routing object has the expected serialized string types and values:

```json
{
  "environment": "balanced_topk",
  "runtime_path": "balanced_topk",
  "strategy": "balanced_routing.BalancedTopKRouting",
  "origin": "/data/ycfeng/Frontier/.worktrees/v4-balanced-native-model/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_06/tests/comparison/calibration/balanced_routing.py",
  "sha256": "6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea"
}
```

`inventory.moe_routing.native_sources` equals the expected four origin/hash objects on every rank. The node's `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` is `balanced_topk`; the engine selector matches it. The strategy file, manifest strategy hash and all four native source hashes match. Package freeze pins pass **43/43**; R3 and R2 hashes match the freeze.

## Census and independent retained facts

| Host | Role | DP owner | TP ranks | Rejected ranks |
| --- | --- | --- | --- | --- |
| `gpu-h800-0214.host.platform.shaipower.com` | prefill | 0 | 0–7 | 8 |
| `gpu-h800-0310.host.platform.shaipower.com` | prefill | 1 | 0–7 | 8 |
| `gpu-h800-0346.host.platform.shaipower.com` | decode | 0 | 0–7 | 8 |
| `gpu-h800-0399.host.platform.shaipower.com` | decode | 1 | 0–7 | 8 |

All 32 files are `native_serving_v1` / `pre_shutdown` with no recorded observation errors. Each contains 43 routing owners (layers 0–42), totaling **1,376 router entries and 1,376 runner entries**. All 32 fail the unchanged frozen verifier with the identical exception.

Independent raw initialization fields, not reconstructed from the expected policy:

- `inventory.initialization.modules["model.layers.0.ffn.experts"]` is the fully qualified native `MoERunner` type.
- `inventory.initialization.modules["model.layers.0.ffn.experts.routed_experts"]` is `vllm.model_executor.layers.fused_moe.routed_experts.RoutedExperts`.
- `inventory.initialization.backends["model.layers.0.ffn.experts.routed_experts"]` is `{"backend":"MARLIN","experts":"MarlinExperts","input_dtype":"None","prepare_finalize":"MoEPrepareAndFinalizeNaiveDPEPModular","quant_method":"Mxfp4MoEMethod"}`. `input_dtype` is a serialized string, not Python `None`.
- All 43 MXFP4 backend owner names, after removing exactly `.routed_experts`, equal the 43 independently recorded `MoERunner` module names and both routing map key sets, on every rank. The retained HF config reports 43 hidden layers.

These facts establish source-bound host object inventories. They are not a device expert histogram, route-content readback or proof that every routing call used the same strategy throughout serving. Server logs also contain native `Simulating MoE routing using a balanced_topk strategy` warnings; they corroborate native entry to the routing simulator but are not used to infer unobserved per-rank/per-call state.

## Evidence-bound cause and source references

Paths below use `PACKAGE` = the exact candidate path in the origin above without `/tests/comparison/calibration/balanced_routing.py`, and `NATIVE` = `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`.

1. **Enumeration mismatch:** `PACKAGE/tests/comparison/calibration/native_full_structure.py:280–285` collects `moe_backends` only where `type(method).__name__ == "Fp8MoEMethod"`. `Mxfp4MoEMethod` cannot enter this comprehension. Empty output is therefore expected for the observed actual backend, not evidence that no router exists.
2. **Independent routing producer:** the same file `:16–18,21–53` enumerates actual modules with a `router`, reads actual strategy registry objects, serializes type identity and checks native callable/module origins. It does not enumerate routers via `moe_backends`. `:279` stores routing separately. `:620–624` refreshes this routing object at emission after the post-warmup inventory, including pre-shutdown.
3. **Independent MXFP4 witness:** `PACKAGE/tests/comparison/calibration/native_dummy_initialization.py:268–303` separately handles actual `Mxfp4MoEMethod` and retains the backend/consumer facts; `:347–350` records every actual module's fully qualified type. These fields already exist in this run.
4. **Rejecting consumer:** frozen R3 `:112–117` assumes the narrower FP8 backend inventory is the complete router inventory. It rejects with the generic error at `:122`, before the retained routing object's own source/type values can qualify the object.
5. **Native seams:** `NATIVE/vllm/models/deepseek_v4/nvidia/model.py:618–621,691–701` constructs the non-MegaMoE fused path; `NATIVE/vllm/model_executor/layers/fused_moe/router/router_factory.py:114–120` selects the simulated router from the actual environment; `.../router/routing_simulator_router.py:295–301,332–339` obtains the registry object and calls its `route_tokens`; `.../runner/moe_runner.py:587–608` selects experts through the router for modular kernels. The observed actual backend is MARLIN, not an inferred FP8 selection.
6. **Why existing CPU identity control missed it:** `/data/ycfeng/Frontier/.worktrees/v4-balanced-native-model/tests/unit/test_native_reader_binding.py:463–475` uses a stub runner, calls only `balanced_routing_inventory`, then manually supplies `moe_backends={'experts.routed_experts': {}}` at `:467`. It never exercises the real `full_model_inventory` backend comprehension with the actual V4 MXFP4 method. This makes its key mapping self-consistent while missing the actual producer/consumer seam.

### Object identity versus class name

The registered strategy is a nested class in `PACKAGE/tests/comparison/calibration/balanced_routing.py:51–72`. The collector deliberately serializes `__module__ + "." + __name__`, not `__qualname__` (`native_full_structure.py:16–18`). A source-only CPU replay of the **unchanged registration AST**, with native imports stubbed and without executing `route_tokens`, produced:

- `__module__ = "balanced_routing"`
- `__name__ = "BalancedTopKRouting"`
- `__qualname__ = "register_balanced_topk.<locals>.BalancedTopKRouting"`
- collector identity = `"balanced_routing.BalancedTopKRouting"`
- bound method code filename = the exact retained strategy origin.

This explains the serialized representation; it does not claim to recover a live worker object. The actual raw string matches the frozen expected string exactly. **The nested class is not the cause of this failure.** Actual origin/hash/native source objects were compared, not only strategy names.

## Reproduction and verification

Local environment: `/usr/bin/python3`, Python `3.10.6`, GCC `11.3.0`. No conda activation or dependency installation. Bare `python` was unavailable (`zsh:1: command not found: python`, exit 127); using the existing `python3` interpreter resolved only the command name.

Minimal real verifier call (no `main()` call, no full-checker invocation, no Git subprocess, no result write):

```bash
/usr/bin/python3 -B /data/ycfeng/tmp/glm53_dsv4/native06_routing_rca_reproduce_20261023.py
```

The script loads frozen R3 via `runpy` with a non-main name, checks both frozen hashes, reads the original first rank and original node environment/config/engine/manifest, then calls:

```python
checker['verify_balanced_runtime'](
    payload['inventory'], result['server_env_set'], engine, config, manifest)
```

Two saved executions both exit **1** with identical traceback bytes (SHA256 `cb39fd4242e94ba44ca7d7a9f373305a156e0501fcd48da38d25c1b2b4ceccee`):

```text
Traceback (most recent call last):
  File "/data/ycfeng/tmp/glm53_dsv4/native06_routing_rca_reproduce_20261023.py", line 29, in <module>
    checker['verify_balanced_runtime'](
  File "/data/ycfeng/Frontier/.worktrees/v4-balanced-native-model/task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py", line 122, in verify_balanced_runtime
    raise ValueError('Actual balanced routing strategy/source/environment identity mismatch')
ValueError: Actual balanced routing strategy/source/environment identity mismatch
```

Census, package pin checks, Python identity inspection and original-artifact before/after hashes:

```bash
/usr/bin/python3 -B /data/ycfeng/tmp/glm53_dsv4/native06_routing_rca_census_20261023.py
```

| Check | Expectation | Observed | Result |
| --- | --- | --- | --- |
| Minimal reproduction | Exact reported exception twice | exit 1 twice, identical raw logs | PASS for RCA reproduction, not acceptance |
| Frozen identity | Exact R3/R2 freeze hashes | `ca8ddcea83c852c27295c8a73453746d755c51055e703bd94df48e8f1af7d917` / `29523700ac8492fa6e192ce272f8ab5a7a48d1528bd4064417e26e811533d10a` | PASS |
| Package pins | All 43 unchanged | 43/43 | PASS |
| Actual-rank rejection census | Inspect every original inventory | 32/32 same rejection; only router/runners maps differ | PASS for coverage |
| Independent module/backend/routing names | Compare original independent serialized objects | 43 names agree on 32/32 ranks | PASS for RCA witness |
| Original immutability | No evidence/checker/source/input byte changes during probes | 172 unique monitored paths, zero changed hashes | PASS |
| Downstream graph acceptance | Requires passing routing identity | Not run | UNVERIFIED |

Full temporary evidence: `/data/ycfeng/tmp/glm53_dsv4/native06_routing_rca_census_20261023.json`; raw logs: `/data/ycfeng/tmp/glm53_dsv4/native06_routing_rca_reproduction{1,2}_20261023.log`. Durable compact evidence is `routing_rca.json` beside this report. Probe scripts remain explicitly diagnostic in the approved temporary directory.

## Minimal repair proposal and collection decision

**Proposal only; root owns authorization and implementation.** Preserve R3/R2 and the run's historical FAIL. Create a versioned successor reader that derives router owner enumeration from independent retained actual module inventories, with an explicit supported native runner/child relationship, rather than the FP8-only `moe_backends` subset. For this run the existing `initialization.modules` provides all fully qualified `MoERunner` and `.routed_experts` child types; existing `initialization.backends` provides all 43 actual MXFP4 methods/consumers. Require exact set agreement with both routing maps, reject omissions/extras and unsupported consumers, and retain every current environment, actual source/origin/hash and strategy-object identity check. Do not merely allow `{}` or derive expected completeness from the same routing map being checked.

The regression seam must exercise the unchanged real collector's backend comprehension using an actual-shaped `Mxfp4MoEMethod`/`RoutedExperts`/`MoERunner` hierarchy; the earlier manually filled FP8 map is insufficient. Retain negative cases for missing routers, wrong strategy objects/origins, wrong native consumers and incomplete independent owner coverage. No native, C26, strategy, graph or metric change is indicated by this RCA.

**Fresh collection is not needed to resolve this specific enumeration rejection:** all independent owner/type/backend/source facts needed for a reader-only correction already exist on 32/32 ranks. This is not permission to retroactively admit the run. A reviewed versioned contract and full unchanged downstream acceptance checks would still be required, and root must decide whether historical evidence may be reconsidered under that successor. With frozen R3 kept authoritative, this run cannot PASS. If root instead requires new per-call strategy persistence, device expert-content/histogram evidence, stronger live-object facts or a changed collector/schema, those facts cannot be fabricated from these snapshots and would require fresh approved collection. The remaining graph, rank/phase/counter and runtime/reference qualification are intentionally unresolved here.
