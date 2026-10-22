# Parent integration inputs

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record actual interface, sparse context and pending live inputs. |
| 2026-10-23 | Narrow thin-graph qualification and document checked backend-source staging. |
| 2026-10-23 | Bind accepted publication and prepared controlled T2 metadata/cases; engine7168 remains an explicit limit. |

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
Prepared sparse context: /data/ycfeng/tmp/d29-native-build-context02. Root accepted
native6827507 at9495cfb and published67d5b5e, summarized9090407. Use the verified
linux/amd64 platform reference, not its index digest or mutable tag:
hub.stepfun-inc.com/vllm-openai/vllm-openai@sha256:f037d9589cf30fe0fa30dfde9c64c9f21060adffc6c63726469f4db67e2451ee.
Set STEP5_POSITION_CYCLIC_IMAGE to that exact reference for new R2 staging.
Tag-only and unchanged base digest are rejected.
The staged engine native_reference contains exact image/base_image/routing_strategy;
clients verify that engine.json, config engine and definition agree. Once staged,
the explicit engine identity remains consumable without the binding environment.

Root exported actual installed derived sources and53 installed source pins at
/data/ycfeng/tmp/d29_published_sources01/source; publication receipts are in
root's task_2026-10-22_step5_position_routing/image_publication. This stream
consumes accepted evidence without repeating its source/CPU acceptance checks.
The base-build53 listing is50 existing sealed pins plus3 local base-source byte hashes;
it is NOT evidence of installed derived bytes/imports. Profiling's full installed
source receipt is separately root-emitted according to profile_source_design.md.
Strict derived R2 staging now additionally hashes the12 existing backend source
modules named by r2_runtime.BACKEND_MODULES in schema_source_hashes. Root's source
export must contain them; these are actual-source provenance inputs, not native
edits/new image pins/new family classification. D30 source_hashes remains13.
Root publication/source acceptance is complete; actual all-rank runtime/import
qualification remains pending. Native4 source hashes and
the retained interface patch are unchanged after the reviewer repairs.

## T2 input and execution (NOT RUN)
Controlled metadata/cases and exact execution/resource proposal are prepared in
t2_preparation/ at accepted9090407. Config.json is an unchanged78447-byte copy
of root's retained official revision364b3b03fed2477aa0953fe855a2c4c8c48ee3cc,
SHA256 e535c5c9e6385e5c39d55daf346fc0b42dd93496bf5e540ea442168b2524bb63.
See provenance.json/cases.json/execution_proposal.md/resource_proposal.json.
Inputs are controlled probes, NOT observed native/Frontier traces or a new run.
Root still supplies accepted/sealed source/input commit, actual16-rank TP8/DP2/EP16
resource/run approval, worker neutral cwd/import/mounts and new writable output.
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
It does not test real MRv2 serving inputs, attention or connector eager breaks,
or real FP8/BF16 backend graphs. The output explicitly marks both qualifications
false. Successful future execution reports PASS_THIN_GRAPH_AND_EAGER_MODULES,
not full T2/G3 acceptance. PW/FULL dummy replay precedes output checks and the
replay_baseline_counts receipt; the first PW call only captures graph segments.
Diagnostic errors are soft and report UNVERIFIED; native execution/IO errors are
not converted into success. Actual expert classes/E8M0 are reported, not forced.

Static preparation passes2 same-shape17-row graph cases,3 module frames with
physical/live DP counts[6129/6128,1/0],[1/0,17/16],[4/1,4/0]. All positions are
within[0,7168). Actual metadata retains max_position_embeddings1048576: both
existing EngineArgs calls omit max_model_len and no T2 CLI override exists.
An exact engine7168 requirement cannot be silently satisfied by rewriting config
or adding a wrapper. Root must resolve that dependency if needed; full-serving
G3's frozen7168/state criteria are unchanged. No state tolerance is selected.

## Still UNVERIFIED
CUDA sigmoid arithmetic against locked biased-topk selected-score contract;
NONE/PW/FULL pointer/replay behavior; real modular FP8/BF16 auto backend/collectives;
all-rank imported model/router provenance; derived image/dependencies/library bytes;
G3 numerical state and bit-exact transfer; full serving and new operator/gap pricing.
No native CPU-overhead analysis was added.
