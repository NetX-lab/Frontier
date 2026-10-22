# Balanced routing execution identity

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Completed source-level mismatch analysis; proposed explicit per-case runtime selection and existing native registration. |
| 2026-10-22 | W128 documented native109-line slice and exact unresolved next-generation binding. |
| 2026-10-22 | Updated actual capacity128/row-identity findings and complete22-file/843-line planning estimate. |
| 2026-10-22 | Superseded estimate with approved23/900, actual shared12/~428 basis and B final source/CPU binding. |

Analysis status: PASS (source question resolved). Correction state: approved under D-V4-BALANCED-RUNTIME-SELECTOR. Native/kernel/four-metric qualification: UNVERIFIED.

## Evidence and gain

- `frontier/moe_routing_runtime.py:48-62` derives `standard_fused_topk` from balanced load ratios. The model profile may override it globally, but no existing per-case runtime selector exists. Four production call sites resolve this identity: layer_contract_resolution, prediction_family_trainers, sklearn_moe_execution_time_predictor and moe_routing_workload.
- Pinned native router_factory selects RoutingSimulatorRouter when its existing simulation environment variable is set. The existing RoutingSimulator registry has uniform_random/normal_routing, not deterministic balanced routing. Its unchanged register_strategy API can add the required policy without changing native/C26 source.
- The native balanced tensor operation is a different operator implementation from standard_fused_topk and from the existing Frontier uniform_topk loop/clone. Relabeling either existing measurement as the new runtime would be invalid.
- A per-case selector delivers exact operator identity and protects old measurements/caches/default configurations. The existing distribution field cannot supply it: load distribution and executed routing kernel are distinct properties. A global V4 profile change would also change unrelated/default cases.

The latency/fidelity percentage of the new runtime has not been measured. The requested configuration exception is for selecting/identifying the actual operation, not fitting latency or adding a small tuning factor. Its measured cost remains missing until qualified evidence exists.

## Supported native strategy

Keep source2cf and the accepted single C26 overlay unchanged. Register balanced_topk through an explicitly called, standalone module, staged through the existing worker-extension and isolated-profile bootstrap paths. Do not mutate environment on module import or silently select another policy.

For TP/SP chunks in DP-major EP order, use the prefix sum of existing native CPU local_sizes as the local token base B. Route token t, choice k to `((B+t)*K+k) mod E`, with K/E from actual inputs. `T_ROWS=sum(TP*ceil(padded_width_dp/TP))`, `A=T_ROWS*K`; each expert receives floor(A/E), then the first A% E expert IDs receive one more assignment. This matches the existing Frontier integerization. Capture-time equal widths make FULL replay offsets fixed; eager unequal widths use the actual prefix, not rank*n. Padding participates. CPU controls must verify actual native context/rank conventions rather than assume them.

The simulation router overrides native hash routing as well as ordinary top-k. That is the requested temporary balanced policy and must be explicit in run identity. Actual selected backend/kernel and graph behavior remain fresh-run acceptance items.

## Proposed shared-interface extension

Add optional per-case `moe_gating_routing_runtime_path` to ReplicaConfig, defaulting to existing resolution. The central resolver handles explicit values through the existing runtime-path registry; register balanced_topk once and thread the selection through the four callers. Both producer and consumer retain the actual `routing_runtime_path` identity. Existing metadata/filter/hash ownership must observe the new selection; default resolution and unrelated cases must stay unchanged.

Use the same balanced tensor function when profiling that operation. A missing balanced_topk timing row must produce the existing typed coverage failure; never use standard_fused_topk/uniform_topk data or zero CPU costs as a replacement.

Expected maintained-code boundary: ReplicaConfig, central runtime registry/resolver, four call sites, the standalone strategy, profiling selection/implementation, existing bootstrap/binding logic and focused tests. Request an upper bound of14 source/test/binding files and500 changed lines for this routing feature, including the independently authorized strategy. Stop and report if the concrete diff exceeds it. Generated sealed snapshots remain byte-bound copies, not places for additional logic. No extra GPU allocation is requested; previously approved prefill/native budgets apply after readiness.

## Verification and dependency

1. Existing-registration strategy/CPU conservation and shared config/caller implementation are now approved under D-V4-BALANCED-RUNTIME-SELECTOR; the aggregate routing bound is14 maintained files/500 changed lines.
2. Verify TP1, equal and unequal DP/SP widths, padding, global expert counts/remainder, dtype and unique top-k; compare the real Frontier materializer.
3. Verify selected runtime reaches actual producer/consumer/cache paths; default behavior remains unchanged, and absent/wrong timing provenance rejects.
4. Review and bind the same module/configuration into fresh packages. Verify actual worker policy/backend/source origins and native/GPU execution in the already approved runs.
5. Align workflow and GPU/op gaps first. CPU-overhead measurement is deferred to the final calibration phase.

## W128 native binding delivery and remaining boundary

Implemented in `feat/v4-balanced-binding`: native environment producer uses the same optional `moe_gating_routing_runtime_path`; it removes inherited simulation selectors and only sets balanced_topk explicitly with FullModelInventory. The extension imports/registers the flat strategy before observer installation or Worker load_model. Existing inventory records actual vllm.envs value, actual registry class/code filename, loaded module path/SHA256 and every MoERunner.router. It refreshes that identity at pre_shutdown. R3 requires exact identity on each native rank and uses the unchanged R2 staged PathFinder guard for the new strategy import. No new schema or second hash registry was introduced.

Allocation used: five maintained files,107 added +2 deleted =109 lines. Source/binding62 lines, tests47 lines. Standalone strategy remains211 lines/two files; local combined feature320 lines/seven files. No new generation was selected or built. This is a partial runtime binding, not full package/GPU acceptance.

### Native next-generation changes (root-owned)

1. Existing builder is `/data/ycfeng/Frontier/.worktrees/v4-p0-native/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py`. Do not refresh candidate_h800_p0_05. A future builder must stage the committed strategy at `tests/comparison/calibration/balanced_routing.py`, using `harness_files['frontier/profiling/moe/balanced_routing.py']` and the existing committed/tool/input hashes. Other harness targets retain their current nested paths. R3 explicitly requires this destination; a same-name root module/package is rejected by its actual search-order resolver.
2. Set `engine['moe_gating_routing_runtime_path']='balanced_topk'` in all canonical engine copies. Use the modified vllm_replay.py; the R2 command/environment reproduction then expects the added strategy key rather than accepting a receipt label independently.
3. Dispatch source pins still must add dp_utils.py, states.py and nixl/pull_scheduler.py to67 existing pins; utils.py is already present, yielding70. This is the separate approved dispatch builder obligation, estimated five builder lines in its own488-line feature budget. It was not implemented here.
4. Routing source files are absent from old67 pins. Before admission, additionally pin unchanged native envs.py, router/router_factory.py, router/routing_simulator_router.py and runner/moe_runner.py from base2cf (four more source entries;74 total after dispatch). Observe actual imported router/env code origins against those installed files, not only class labels. Existing source/C26 bytes must remain unchanged.
5. Old native case has no `inputs/frontier.json`. Root must create the real Frontier comparison input with matching balanced distribution/runtime selector, include its input hash, and have R3 compare that actual input. Current R3 validates the engine and rank identities, but cannot establish Frontier comparison equality from an absent input.
6. Point fresh case checker provenance to the reviewed R3 and retain/bind its external hash-verified frozen R2 dependency. Do not modify the frozen R2 authority or accept old packages as balanced.

### Prefill next-generation changes (root-owned; design only)

The retained adapter is `/data/ycfeng/Frontier/.worktrees/v4-h800-prefill/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill_capacity128/profile_decode.py`; SHA25631c7c5c8f15680894967fdab45ef3d3e0d2d12359b17848c7602abfc020b7bdc. Do not edit it or its old package. Its sys.path/source_snapshot setup at lines25-26 precedes in-process LLM construction. Native2cf/C26 bootstrap stays byte-identical.

Use a maintained successor adapter copied by the future prefill builder, not string-injected generated code. After C26 bootstrap and sys.path setup, require config's explicit balanced runtime field; set the existing simulation environment selector explicitly; resolve/import the flat strategy from its tool-hash-bound destination; reject winning same-name packages/modules or mismatched bytes; register before importing the profiling CLI. Physical TP1/DP1/EP1 is in-process because dsa_vllm_runner.build_engine sets VLLM_ENABLE_V1_MULTIPROCESSING=0. No fallback/setdefault is allowed.

Actual prefill identity must come from the constructed model, not registration alone. Cheapest existing seam is dsv4_attention_runner.install_dsv4_hooks, which already returns the measured worker's inventory through collective_rpc. Add the same actual routing identity to that inventory and validate it against explicit config/module hashes. Root must coordinate this source edit with frontend ownership; it was not authorized to this agent. Preserve planned7168/8192/configured128/effective128 geometry, six layers, one measured request, warmup2/measured3 and both timing methods. Existing new GPU budgets do not authorize submission before all these checks.

### Allocation required before further edits

At least one additional maintained adapter file is necessary: the retained33-line adapter plus an explicit environment/origin/hash/registration block requires at least12 additional physical lines, so a45-line successor already exceeds the remaining zero-line/zero-file allocation. A source-backed prefill inventory producer also needs one existing frontend-owned file. Native Frontier equality requires at least four R3 lines to read the real input, enforce its input pin and compare the runtime selector. Native actual router/env origin facts require additional collector/checker lines; their precise final count depends on root's chosen existing receipt shape. Therefore the current14-file/500-line partition cannot be declared complete as delivered. Reallocate/reduce other slices or obtain additional explicit scope before implementation; no code has been hidden in snapshots to evade this limit.

## Complete binding estimate after root's integration finding

Historical estimate below is superseded by YC's explicit23-file/900-line approval. Actual aggregate planning basis is shared12/about428 plus nonshared11/about426, about23/854. Exclusive allocations are A12/450, B6/250, C5/200. No owner may silently use the aggregate ceiling; B currently253 and awaits root reallocation before expansion.

No source edits followed this finding. Shared final11 files/about417 lines plus the already implemented native five files/109 lines would be16 files/526 lines even before completing prefill. The following estimate is for the smallest ordinary readable complete wiring using current APIs, with no new strategy implementation or shared identity schema.

| Maintained files | Final routing lines estimated | Required gain / uncovered seam |
| --- | ---: | --- |
| Root shared11-file slice (including strategy/tests) |417 | Selector/materializer/operator producer/consumer/cache identity and role forwarding; root verifies registry correction |
| Existing native five files owned here |149 | Current109 plus40: known native router/env source origins and actual hashed Frontier input comparison; frozen R2 stays unchanged |
| New maintained prefill `profile_decode.py` successor outside all sealed packages |45 | Retained33 lines plus12 explicit flat-module origin/tool-hash/environment/registration lines before construction |
| `frontier/profiling/attention/dsv4_attention_runner.py` |28 | Constructed model registry/router/env/origin inventory, expected strategy validation and existing raw engine record propagation |
| `frontier/profiling/attention/dsv4_attention_rows.py` |10 | Preserve actual routing path/source identity in accepted CSV rows and reject mixed/missing balanced raw identity |
| `frontier/execution_time_predictor/deepseek_v4.py` |15 | Require matching per-row routing identity for explicitly balanced isolated layouts; preserve absent-selector legacy behavior |
| Existing native `build_candidate.py` successor delta |18 | Committed flat strategy target, explicit engine selector, four routing source pins, actual Frontier input binding and new checker dependency identity |
| Existing prefill `configured_capacity/prepare_package.py` successor delta |16 | Copy maintained adapter/flat strategy; explicit config, strategy and native source hashes, new distinct job/generation source identity |
| Additional controls in existing `tests/unit/test_native_reader_binding.py` |45 | Actual prefill pre-construction ordering/model identity, missing env/source, winning shadow and balanced/unbalanced row identity controls; no extra test file |
| **Projected total** | **843** | **22 distinct maintained files** |

These are planning estimates, not a fabricated diff. Existing five owned files' estimate149 includes their current two tests; the additional45 controls increase that row's test files without adding distinct files. Complete nonshared routing estimate is11 distinct files/426 lines. Compared with the original aggregate14/500, the projected necessary increase is eight files/343 lines. Recommended single amendment: maximum22 maintained files/900 changed lines for the complete routing feature, preserving all original GPU/source/old-package constraints. The57-line reserve covers evidence-driven origin/input/control details; stop again before exceeding it. Separate host-dispatch builder five lines remain in the previously approved488-line dispatch budget, not concealed in routing counts.

### Why current mechanisms alone are incomplete

- Native harness_files/tool_hashes can bind staged bytes and R2 can resolve actual import precedence, but neither has a strategy mapping before this slice; current native67 pins omit envs/factory/simulator/MoERunner. Class labels alone do not validate actual native code origins.
- Prefill bootstrap validates tools/native files before the adapter, but the retained adapter has no balanced selector/registration. Importing the module only in the parent would be insufficient for native32; its extension hook is essential. Prefill is legitimately in-process and its existing profiler inventory seam can observe the actual constructed worker.
- `dsa_vllm_runner.measure_engine:652-662` carries inventory into raw engine records automatically, so no second writer is needed. However `dsv4_attention_rows._model_columns:128-155` selects fixed attention columns and currently discards any routing identity; a raw label alone cannot keep accepted balanced/unbalanced rows distinguishable.
- RuntimeIdentity's existing source_revision/fingerprint protects different source generations, not two configurations of the same source revision. Keep routing identity in existing per-row metadata rather than replacing source identity or adding a second global fingerprint schema.
- `deepseek_v4.isolated_scope_time:630-647` validates fixed engine/layout metadata; decode additionally validates independent provenance, while prefill has no provenance columns. Merely emitting new routing columns would not make the current consumer enforce them. Explicit balanced selection must require matching row identity; legacy unselected rows keep their existing behavior. This existing consumer adds one necessary maintained file/15 estimated lines.
- Exact configured-capacity successor builder is `/data/ycfeng/Frontier/.worktrees/v4-h800-prefill/task_memory/task_2026-10-22_v4_h800_prefill/configured_capacity/prepare_package.py`. Its engine assertion is7168/8192/planner128/effective128, not the failed prior effective1 generation. Both old generations remain immutable.

### Root decisions and dependencies

1. Confirm actual shared file/line count after mandatory forwarding and demonstrated registry reuse correction. If its final count differs from11/417, adjust the843-line estimate by that exact difference.
2. Approve/reallocate the complete22/900 routing scope before further implementation. Coordinate the two profiler-owned source files and existing deepseek_v4 consumer with frontend ownership. A review-only strategy freeze need not change; same strategy bytes remain6562ec1c... on both paths.
3. Root selects actual Frontier comparison input and fresh generation destinations; builders use those existing inputs, not fabricated miniature substitutes. Source implementation/review -> complete CPU identity/negative controls -> fresh-generation choice/pins -> root sealing/runtime validation. No live GPU proof can replace missing source wiring.

## Actual native-final wiring and remaining generation evidence

- Actual native source receipt reuses moe_routing.native_sources, keyed by the four existing native source paths. Env cached getter is unwrapped; strategy/factory/simulator/router/MoERunner callables must have co_filename equal to actual module origin. R3 compares every origin to installed_package.parent/path and every hash to the unchanged pinned native checkout. No extra device read/fence/collective is added.
- Actual input is root's complete flat SimulationConfig,775 fields/SHA747a08e5da6759f8189d29b2c8419ab48eb215ac2db4a2596af4f1d15ef34897. The native builder copies those bytes to case/inputs/frontier.json; retained_files automatically adds its actual absolute path to existing prelaunch inputs_sha256. R3's harness predicate verifies all three existing hash records, complete current flat schema and canonical P/D/model/parallel/device/balanced keys. Frozen R2 inputs predicate remains unchanged.
- Maintained builder path in this worktree: task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py. Original bytes/hash21b5a723... are in evidence/build_candidate.baseline.py, not a parallel implementation. New strategy target and R3/R2 tool hashes are explicit. Dispatch adds three lines separately; routing adds four pins to74. Existing source guards require a clean committed checkout and absent selected06 destination; all old packages remain unchanged.
- Source wiring and130 focused CPU controls are implemented; local253/250 scope correction and forthcoming root-supplied prefill negatives remain open. Root owns effective role check, common review/freeze, committed generation choice/creation/seal and runtime evidence. Existing builder timing defaults10800/7200 remain unqualified historical values; root must align the selected formal launch envelope to its approved60min budget before submission, without inferring increased GPU authorization.
