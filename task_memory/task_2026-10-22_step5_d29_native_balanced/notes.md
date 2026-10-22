# Parent integration inputs

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record actual interface, sparse context and pending live inputs. |

## Native interface
Early commit e48cdc0 retains native_source.patch, four source snapshots in
native_source/ and interface_receipt.json.
Actual source is /data/ycfeng/tmp/d29-native-derived-20261023-candidate01/vllm.
Use native set_forward_context with is_padding, then
`with set_routing_positions(actual_device_positions): router.select_experts(...)`.
Factory uses existing `create_fused_moe_router(top_k, global_num_experts,
scoring_func='sigmoid', renormalize=..., routed_scaling_factor=...)`.
Environment selector is VLLM_MOE_ROUTING_SIMULATION_STRATEGY=position_cyclic.
No new expert backend or upstream runner API exists.

## Root image binding
Prepared final sparse context: /data/ycfeng/tmp/d29-native-build-context02.
The context manifest has the exact locked base and four reviewed derived hashes,
without a final image digest. Root must build/publish and inspect an actual changed
linux/amd64 digest, then set STEP5_POSITION_CYCLIC_IMAGE to repository@sha256:digest
when generating a new R2 definition. Tag-only and unchanged base digest are rejected.
The staged engine native_reference contains exact image/base_image/routing_strategy;
clients verify that engine.json, config engine and definition agree. Once staged,
the explicit engine identity remains consumable without the binding environment.

Root must export actual installed derived sources and53 installed source pins.
The base-build53 listing is50 existing sealed pins plus3 local base-source byte hashes;
it is NOT evidence of installed derived bytes/imports. Profiling's full installed
source receipt is separately root-emitted according to profile_source_design.md.

## T2 input and execution (NOT RUN)
Root supplies exact approved Step model metadata, neutral cwd, installed derived
native root, existing16-rank TP8/DP2/EP16 allocation, and a new writable output path.
No submitter/auth operation exists in T2. Pass --model-path, --installed-root,
--cases and --output to tests/integration/step5_environment/test_position_cyclic_graph.py
through the parent's existing torchrun launcher. PYTHONDONTWRITEBYTECODE=1,
OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 remain required. The normal derived
image selector must be position_cyclic and skip-padding enabled.

Cases JSON schema:
- graph: exactly2 objects, each {positions: integer array, mask: boolean array};
  both use the same physical-row length. Use actual tail6128..6143+dummy and
  padded q16144..6147 with stale masked positions, updating stable buffers.
- modules: objects {name, dp: [{positions, mask}, {positions, mask}]}, explicit
  per-DP local physical rows for source6128+dummy, tail16+dummy and padded q1.
  Existing frozen config determines experts/top_k/scoring/normalization/scale.

T2 uses actual BreakableCUDAGraphWrapper with a genuine in-place eager callback,
requires >0 eager breaks and >=2 graph segments, then checks post-break routing
and dependent output on2 replays. FULL captures outer Step under context mode NONE.
Thin opaque-op fixture is only graph binding evidence; separate actual FP8 layer3
and BF16 layer88 calls reuse dispatch probe components with positions/mask bound.
Diagnostic errors are soft and report UNVERIFIED; native execution/IO errors are
not converted into success. Actual expert classes/E8M0 are reported, not forced.

## Still UNVERIFIED
CUDA sigmoid arithmetic against locked biased-topk selected-score contract;
NONE/PW/FULL pointer/replay behavior; real modular FP8/BF16 auto backend/collectives;
all-rank imported model/router provenance; derived image/dependencies/library bytes;
G3 numerical state and bit-exact transfer; full serving and new operator/gap pricing.
No native CPU-overhead analysis was added.
