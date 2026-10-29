# Balanced routing execution identity

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | W128-V4-NATIVE06-ADMISSION-RECORDS: final R4 reference accepted and D-Q10 integrated; native graph execution proof remains distinct from predictor/operator-cost/calibration qualification. |
| 2026-10-23 | W128-V4-NATIVE06-RUNNING-RECORDS final delta: native06 terminal successful but runtime identity FAIL; D-Q10 matrix/isolated acceptance complete, no design/source or full-admission change. |
| 2026-10-23 | W128-V4-NATIVE06-RUNNING-RECORDS: updated execution boundary to frozen/sealed Running native06 and independent D-Q10 delivered source/pending matrix, without changing design or historical analysis. |
| 2026-10-23 | W128-V4-CHECKPOINT-CORRECTIONS-20261023: corrected current953/source/package/fidelity boundaries and historical952 governance; independent D-Q10 consumer work does not imply serving or new GPU approval. |
| 2026-10-23 | Resolved primary/B design archive; current accepted c491e6a/0832200 and23/952 supersede historical limits and pending wiring. |
| 2026-10-22 | Completed source-level mismatch analysis; proposed explicit per-case runtime selection and existing native registration. |
| 2026-10-22 | W128 documented native109-line slice and exact unresolved next-generation binding. |
| 2026-10-22 | Updated actual capacity128/row-identity findings and complete22-file/843-line planning estimate. |
| 2026-10-22 | Superseded estimate with approved23/900, actual shared12/~428 basis and B final source/CPU binding. |
| 2026-10-23 | Native R1 P2s invalidate complete input-source/DP admission; retained isolated-harness/reconstruction proposal requires net46 lines. |
| 2026-10-23 | Approved R1 implemented with observed-role/77-source receipt and immutable revision at B312; builder invocation root-owned. |

Analysis status: PASS (source question resolved). Native06 P0 reference: ACCEPTED_NATIVE_REFERENCE_R4. Measured kernel/operator costs, Frontier prediction and four-metric qualification remain unfinished.

Current native06 P0 reference ACCEPTED_NATIVE_REFERENCE_R4, frozen8/8PASS under reader d278659/SHA67f31923 against unchanged b62c155 run/12.002222GPUh. D-Q10 integrated e89b45d; [current root state](progress.md#current-root-state) links final JSON. Historical routing953/dispatch491 and D-Q103/197/core55 unchanged; R4 additional2files223lines/zeroCore under3/300 separately accounted. Old R3 FAIL/native05 UNVERIFIED remain historical; no full05/09/18/19, Frontier prediction/four-metric calibration/operator or graph costs/physicalTP8 profiling/independent expert histogram or new GPU budget.

## Original source analysis — rationale retained

This source analysis describes the pre-selector code inspected on2026-10-22. Its interface and allocation proposals are historical; implemented contracts and current package boundaries follow below. The rationale for distinguishing routing distribution from executed operator identity remains applicable.

### Evidence and gain

- `frontier/moe_routing_runtime.py:48-62` derives `standard_fused_topk` from balanced load ratios. The model profile may override it globally, but no existing per-case runtime selector exists. Four production call sites resolve this identity: layer_contract_resolution, prediction_family_trainers, sklearn_moe_execution_time_predictor and moe_routing_workload.
- Pinned native router_factory selects RoutingSimulatorRouter when its existing simulation environment variable is set. The existing RoutingSimulator registry has uniform_random/normal_routing, not deterministic balanced routing. Its unchanged register_strategy API can add the required policy without changing native/C26 source.
- The native balanced tensor operation is a different operator implementation from standard_fused_topk and from the existing Frontier uniform_topk loop/clone. Relabeling either existing measurement as the new runtime would be invalid.
- A per-case selector delivers exact operator identity and protects old measurements/caches/default configurations. The existing distribution field cannot supply it: load distribution and executed routing kernel are distinct properties. A global V4 profile change would also change unrelated/default cases.

The latency/fidelity percentage of the new runtime has not been measured. The requested configuration exception is for selecting/identifying the actual operation, not fitting latency or adding a small tuning factor. Its measured cost remains missing until qualified evidence exists.

## Supported native strategy

Keep source2cf and the accepted single C26 overlay unchanged. Register balanced_topk through an explicitly called, standalone module, staged through the existing worker-extension and isolated-profile bootstrap paths. Do not mutate environment on module import or silently select another policy.

For TP/SP chunks in DP-major EP order, use the prefix sum of existing native CPU local_sizes as the local token base B. Route token t, choice k to `((B+t)*K+k) mod E`, with K/E from actual inputs. `T_ROWS=sum(TP*ceil(padded_width_dp/TP))`, `A=T_ROWS*K`; each expert receives floor(A/E), then the first A% E expert IDs receive one more assignment. This matches the existing Frontier integerization. Capture-time equal widths make FULL replay offsets fixed; eager unequal widths use the actual prefix, not rank*n. Padding participates. CPU controls must verify actual native context/rank conventions rather than assume them.

The simulation router overrides native hash routing as well as ordinary top-k. That is the requested temporary balanced policy and must be explicit in run identity. Actual selected backend/kernel and graph behavior remain fresh-run acceptance items.

## Historical shared-interface proposal

Add optional per-case `moe_gating_routing_runtime_path` to ReplicaConfig, defaulting to existing resolution. The central resolver handles explicit values through the existing runtime-path registry; register balanced_topk once and thread the selection through the four callers. Both producer and consumer retain the actual `routing_runtime_path` identity. Existing metadata/filter/hash ownership must observe the new selection; default resolution and unrelated cases must stay unchanged.

Use the same balanced tensor function when profiling that operation. A missing balanced_topk timing row must produce the existing typed coverage failure; never use standard_fused_topk/uniform_topk data or zero CPU costs as a replacement.

Expected maintained-code boundary: ReplicaConfig, central runtime registry/resolver, four call sites, the standalone strategy, profiling selection/implementation, existing bootstrap/binding logic and focused tests. Request an upper bound of14 source/test/binding files and500 changed lines for this routing feature, including the independently authorized strategy. Stop and report if the concrete diff exceeds it. Generated sealed snapshots remain byte-bound copies, not places for additional logic. No extra GPU allocation is requested; previously approved prefill/native budgets apply after readiness.

## Original verification strategy

1. Existing-registration strategy/CPU conservation and shared config/caller implementation are now approved under D-V4-BALANCED-RUNTIME-SELECTOR; the aggregate routing bound is14 maintained files/500 changed lines.
2. Verify TP1, equal and unequal DP/SP widths, padding, global expert counts/remainder, dtype and unique top-k; compare the real Frontier materializer.
3. Verify selected runtime reaches actual producer/consumer/cache paths; default behavior remains unchanged, and absent/wrong timing provenance rejects.
4. Review and bind the same module/configuration into fresh packages. Verify actual worker policy/backend/source origins and native/GPU execution in the already approved runs.
5. Align workflow and GPU/op gaps first. CPU-overhead measurement is deferred to the final calibration phase.
## Historical native design archive — superseded by accepted source

All delivery counts, incomplete seams, proposed allocations and pending actions in this archive record the original B handoff. They preserve investigation/trade-offs; requirements.md's root model-identity correction and plan.md govern current23/953 scope. c491e6a closes both R1 findings, C implements prefill seams,0832200 completes deadlines, e9e79d4 removes redundant worker hashes and b62c155 completes existing model input. Archived pending source tasks are not instructions to reopen completed work.

### W128 native binding delivery and remaining boundary

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

### Complete binding estimate after root's integration finding

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

### Native-final wiring handoff before R1 review

- Actual native source receipt reuses moe_routing.native_sources, keyed by the four existing native source paths. Env cached getter is unwrapped; strategy/factory/simulator/router/MoERunner callables must have co_filename equal to actual module origin. R3 compares every origin to installed_package.parent/path and every hash to the unchanged pinned native checkout. No extra device read/fence/collective is added.
- Actual input is root's complete flat SimulationConfig,775 fields/SHA747a08e5da6759f8189d29b2c8419ab48eb215ac2db4a2596af4f1d15ef34897. The native builder copies those bytes to case/inputs/frontier.json; retained_files automatically adds its actual absolute path to existing prelaunch inputs_sha256. R3's harness predicate verifies all three existing hash records, complete current flat schema and canonical P/D/model/parallel/device/balanced keys. Frozen R2 inputs predicate remains unchanged.
- Maintained builder path in this worktree: task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py. Original bytes/hash21b5a723... are in evidence/build_candidate.baseline.py, not a parallel implementation. New strategy target and R3/R2 tool hashes are explicit. Dispatch adds three lines separately; routing adds four pins to74. Existing source guards require a clean committed checkout and absent selected06 destination; all old packages remain unchanged.
- Source wiring and130 focused CPU controls are implemented; local253/250 scope correction and forthcoming root-supplied prefill negatives remain open. Root owns effective role check, common review/freeze, committed generation choice/creation/seal and runtime evidence. Existing builder timing defaults10800/7200 remain unqualified historical values; root must align the selected formal launch envelope to its approved60min budget before submission, without inferring increased GPU authorization.

### R1 findings and exact scope dependency before approval

The prior130CPU PASS did not cover ambient schema identity or attn_dp mismatch. Root now permits retainedB253, but actual native input admission remains incomplete: current schema acceptance depends on ambient B774/A775 imports, and a rehashed complete DP1 input accepts against nativeDP2. Both are directly reproduced; this section supersedes any source-complete claim for that input boundary.

Existing frozen R2 harness_readers already validates recorded clean HEAD/branch and launches isolated Python with selected harness. Reuse that mechanism in R3: do not import SimulationConfig in the checker process, and do not infer the schema from candidate JSON keys. A fresh -I/-B interpreter, cwd at recorded harness (model config JSON lookup is relative), builds the current flat schema, reconstructs real bytes, materializes existing P/D roles and compares actual replica properties. Every imported frontier.* source must resolve within the recorded harness and have bytes equal to git-show of recorded commit; file labels or selected PYTHONPATH alone are insufficient. Preserve R2 predicates and all existing input hashes.

Reconstruction writes a config archive in its native post-init. The one-off substrate experiment suppressed only that write method to preserve root's existing archive; all actual model/cluster constructors and role builder ran. The proposed checker needs the same explicitly scoped observational-write suppression (not a runtime/model/coverage bypass), or a separately approved alternative output isolation; no silent overwrites of the original root archive. No prediction fitting or Simulator run is needed.

The ordinary complete proposal adds41/deletes6 R3 lines relative65743ea, and adds21/deletes10 existing test lines. Aggregate source accounting remains from9d44675: R3 routing70->105, tests99->110, net46. Existing R2 provenance predicates supply clean source checks; this proposal adds actual imported source-byte binding, strict full-schema check, reconstructed effective roles and a real top-levelDP condition. Targeted tests reuse the existing full-input fixture and add rehashedDP/ambient-preload controls plus existing manifest.harness_worktree/commit context.

Exact current planning basis: A432+B253+C189=874; reserved C12 generation+C1 runner+B4 timeout=17; repair46 yields937, still23 maintained files, B303 vs266. Root must choose at least37 aggregate extra lines or legitimate reductions elsewhere before implementation. The optional10-line C consumer persisted regression was never applied and is not in these counts. No new helper/schema/source files, package generation or native/C26 edits are proposed.

### R1 source-completion handoff before root builder finalization

Historical estimate/blocked status above is superseded by YC23/950. Actual implementation uses six B maintained files/312 routing lines before root timeout4. Expected/source labels are not receipts: reconstruction validates actual role attributes and emits those observed attributes, plus actual imported module origins and recorded-revision byte hashes. Full commit object ID is required; manifest supplies harness root/revision in the formal checker, explicit config supplies them in standalone pre-build verification. No ambient schema import and no selected-current-HEAD substitute exists in the verifier.

Existing stdout warnings precede final JSON; verifier consumes the child final structured line and converts failed child stderr to ValueError with original rejection text. Actual architecture guards run. Only archive-output observer is suppressed to preserve original bytes; constructors/role builder are not replaced. Returned77-source receipt is retained locally for CPU checks, not a new runtime receipt/schema requiring worker GPU work.

Fresh-generation builder must pass its existing verified clean `commit` through a temporary validation config: `checker.verify_balanced_frontier(config | {'harness_commit': commit}, engine)`. This preserves the existing config/retained/tool hashes and uses the immutable snapshot already written to source_lock. It replaces an existing routing-added line with no net cumulative routing increase. Root owns this and the four reserved timeout replacement lines. Current builder intentionally cannot validate without a declared revision; no package was generated. Final effective role proof on the merged harness and actual native source/FP8/backend/runtime proof remain root gates.

## Current native source and input contracts

- Explicit policy: `engine.moe_gating_routing_runtime_path=balanced_topk` drives the existing native clean environment producer. Inherited simulation selection is removed, and the selector is set explicitly only with `native_full_structure.FullModelInventory`. The standalone strategy registers before Worker construction/load_model; absent-selector behavior remains unchanged.
- Actual runtime identity: existing `moe_routing` inventory records the environment value, registry strategy class, actual strategy module bytes/co_filename, every router/MoERunner and four native source origins. The collector unwraps the native cached env getter and follows the factory callable imported by layer.py; factory/simulator must reference the same env module. R3 compares exact installed origins and unchanged source pins, rather than accepting labels. Existing observer errors remain soft gates for32GPU workloads; no extra GPU read/fence/collective was added.
- Import precedence: the strategy is staged flat at `tests/comparison/calibration/balanced_routing.py`, mapped from `frontier/profiling/moe/balanced_routing.py` through existing harness/tool hashes. R3 extends the existing R2 actual-search-order resolver; same-name stage-root modules/packages cannot shadow the validated nested bytes unnoticed. Strategy SHA256 remains `6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea`.
- Real input: root's complete775-field flat SimulationConfig at `balanced_frontier_input/frontier.json`, SHA256 `747a08e5da6759f8189d29b2c8419ab48eb215ac2db4a2596af4f1d15ef34897`, is copied byte-for-byte to `case/inputs/frontier.json`. The existing retained_files/tool_hashes/prelaunch inputs_sha256 all bind its bytes. R3 builds the actual flat schema in a fresh `-I/-B` interpreter with cwd at the declared harness, reconstructs the complete input and materializes existing P/D roles. It checks actual TP8/DP2/EP16/PP1/world16/model/routing, including replica_config_attn_dp, without a miniature schema or Simulator/predictor run.
- Immutable schema provenance: manifest.harness_worktree/harness_commit, or explicit standalone config, declares the full immutable revision. Every actual imported frontier.* module origin is within that checkout and loaded bytes equal git-show of the declared revision, not current HEAD. A/B ambient imports therefore cannot change admission. Actual77-source and observed-role receipts, specific rejection errors and RED/GREEN controls are retained in `evidence/native_r1_{red,green}.json`.
- Reconstruction preserves root's original config archive by suppressing only its archive-write observer; model/cluster constructors and role materialization still execute. Existing architecture warnings precede the final stdout JSON; R3 parses that final structured receipt and retains failed child stderr. This is not a graph/runtime/coverage bypass. Archive SHA256 remains `3808af195b09e32074922b2f068e4b670ba1a8ed80fe916a2527967ab3997223`.

## Current builder and qualification boundary

The maintained builder is `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py`; original bytes remain `evidence/build_candidate.baseline.py`, SHA256 `21b5a7232e31df74052856d933c003b5af9caca7ce2fc6362fc1dbca36593560`. It refuses an existing selected destination and preserves old packages. Complete candidate06 now exists only in the new native-model worktree; old e9e79d4 partial13-file06 remains intact. Host R3 manifest path/hash and fixed external R2 dependency remain bound; they are not redundant worker tool entries. Dispatch67->70 is separately488+3=491; four unchanged routing-source pins bring total74.

Root0832200 supplies `checker.verify_balanced_frontier(config | {'harness_commit': commit}, engine)` with the immutable snapshot, net0 cumulative routing while retaining input hashes. Payload3600/startup1800/replay1800/readiness32GPUh are unchanged. Historical B312+6deadline=318/aggregate952, then e9e79d4 removes2 redundant host-only worker hash entries, yielding B316/aggregate950. b62c155 independently uses the retained model case input (+2/-1), yielding current B319/aggregate953; no native/C26, server argv, R3/R2, interface or GPU-resource change.

R4 uses independently retained owner/child/backend metadata to correct only the FP8-only enumeration seam; frozenR3ca8ddcea/R229523700 and all downstream criteria remain unchanged. Parent final acceptance/result JSON admits native06 P0 reference after R4 freeze and separate8/8PASS evaluation (runs symlink original; inputs byte/mtime preserved). Physical graph P NONE/D PIECEWISE+FULL verified across32ranks/six phases; this is native execution proof, not measured graph/physicalTP8 operator costs or Frontier qualification. Historical R3 FAIL/43sealed inputs/manifest8eb3f794/caps3600/1800/1800/32GPUh preserved; no new GPU. Exact pointers: [current root state](progress.md#current-root-state).

D-Q10 source2f87828 is INTEGRATED_ISOLATED_PREREQUISITE e89b45d, merged144PASS/94exact/legacy102positive72negative,3files197lines/core55 separately accounted. Contract-task `integration_receipt.json` preserves H200 defaults/isolated-only limits. R4 `matrix_accounting.json` reuses actual576d27877/77/451hashes/519cache names/777runtime-input bytes with same environment/gitlink and unreached unit-file change, NOT new R4 capture. Full predictor/operator-cost/family/PD/graph/serving remains unqualified; no historical-q1 relabeling, CPU-overhead remains last/GLM paused.
