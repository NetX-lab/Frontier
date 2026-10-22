# D29 profiling runtime, image and source binding

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Trace the entire existing P0 profile package path and specify the missing derived-runtime/source propagation, immutable artifacts, reuse checks and gross implementation ceilings. Design only. |

TaskID: **D29-PROFILE-SOURCE-DESIGN**.

## 1. Outcome and ownership

The profiling path needs **13 existing code/test files, +633/-40 = 673 gross
lines** at most. Three files overlap the Frontier/profile design; **10 files
are genuinely additional**. Replacing that design's incomplete `job.py`
allowance, rather than counting it twice, adds **662 gross lines** to its
published total. No additional native source, image builder, Dockerfile,
selector registry, submitter or maintained campaign JSON is proposed.

The smallest complete propagation is:

```
new explicit profile plan
  -> existing registered step5_position_cyclic adapter
  -> existing describe_job / stage_package
  -> explicit --image in the existing submission command
  -> ResourceConfig(image=args.image)
  -> existing run_on_worker.sh, neutral-cwd installed-source preflight
  -> same-process native import checks in existing producers
  -> existing RuntimeIdentity and attention evidence columns
  -> sealed stage / native-source receipts / new-generation output
```

Changing only `frontier/profiling/step5/job.py` is incomplete: the descriptor
and actual launcher both select the base image, and attention rows independently
declare that image. Package version equality cannot detect any of these errors.

This document owns source binding, not routing input schema/count mathematics,
source-local operator pricing or collective pricing. Those remain with
`design.md` and the separately assigned pricing owner. The native policy,
factory/context interface, four derived native files and derived image are
exactly the reviewed `native_design.md` at **c9e9274**, under
`task_memory/task_2026-10-22_step5_d29_native_balanced/`. Its **19 files /
1003–1582 gross lines** include D24 native packaging, not profiling launchers.

Inspection used the current source tree, with HEAD observed at
`1cf1d5a5c8080b5e31e5efe4afeb973d69d15297`. Parent-owned changes were present.
Only this new document is written; no source/test edit, test execution, commit,
GPU, Docker, platform, authentication, cloud or additional agent action occurs.

## 2. Complete current-path trace and hard locks

Paths are repository-relative. These are source observations, not worker facts.

| Path / lines | Actual responsibility and current limitation | Required disposition |
| --- | --- | --- |
| `frontier/profiling/step5/job.py:160–179,238–241,244–273` | Existing P0-role describe/stage/CLI reads only the maintained `p0_profile_plan.json`; its additional payload is only WORKLOAD. | Accept an explicitly supplied new generated plan through this same entrypoint; keep no-option historical behavior. |
| `frontier/profiling/step5/p0_profile_plan.json:4–18,20–35` | Locks H800/codesign/non-preemptible/one-hour and serving topology; P0 attention files and scalar MoE points explicitly do not establish complete native coverage. | Leave this file and both original P0 input JSONs byte-identical. New generation has its own plan/input artifacts and honest coverage limits. |
| `frontier/profiling/step5/job.py:180–204,206–235` | Builds attention + two linear + two MoE commands; MoE explicitly uses `standard_fused_topk`; neither linear nor MoE command enables existing source-revision metadata. | Derive new routing path from the selected adapter, pass the core-owned MoE input option, and enable existing revision arguments for new commands. |
| `frontier/profiling/common/step5_job.py:12–13,19–42` | `describe_job` always returns base IMAGE. Generated submission command has no image argument. Declared manifest image is not consumed by the launcher. | Add an explicit image parameter, default unchanged; serialize that exact immutable image in `--image`. |
| `frontier/profiling/common/step5_job.py:46–78` | Copies Frontier/config/offline_site and a small script list, then hashes payloads. `additional_paths` uses only `copytree`; registry files, external new input files and native receipts are not staged. No source commit is recorded. | Reuse this copier/seal, support needed file payloads, copy explicit generated inputs/receipts, and record/verify committed source for the new generation. |
| `tests/integration/step5_profiling/submit_profiling_job.py:23–26,32–46,59–86` | Second independent base-image constant goes directly into `ResourceConfig.image` at71. Existing launcher already owns personal RJob submission and source mount. | Preserve direct-CLI base default; add `--image` and consume it at71. No alternative launcher or platform path. |
| `tests/integration/step5_profiling/run_on_worker.sh:16–29,31–61,63–73` | Source root, worker dependency site and inherited PYTHONPATH precede installed packages; current cwd is source root. Only CUDA/device checks run. Dependency installer intentionally does not replace torch/vLLM/Triton. | Keep driver/dependency/device behavior. Add a selected-generation neutral-cwd/source-origin check and verify actual producer-process imports too. Do not solve this by putting an exported `vllm/` on PYTHONPATH. |
| `frontier/profiling/step5/vllm_wrapper.py:17,38–46,86–98,125–128,586–620` | Attention imports the model from Python search paths but records `declared_image={IMAGE}` from the base job module. Its row writer already accepts arbitrary evidence. | Remove that declaration lock for the selected generation; derive signature/evidence from the bound profile manifest and actual module origins. No attention geometry or timing change. |
| `frontier/profiling/moe/moe_impl.py:267–314` | Existing Step gating directly calls native `fused_topk_bias`; this is not a router-factory invocation and cannot select the new native strategy just because an env variable is set. | Already owned by core `design.md` §6: use the actual native factory/context/router and stamp its actual class/config/source. This document adds no substitute router. |
| `frontier/profiling/moe/main.py:340–376,1026–1043,1205–1209` | Runtime paths are an old two-value choice list; paired `--source_revision` / `--model_config_revision` already exist; RuntimeIdentity attaches only when requested. | Choice/input/native adapter changes are core-owned. Add mandatory source binding for position mode and reuse the existing attachment path. |
| `frontier/profiling/linear_op/main.py:369–385,662–679,1096–1097` | Same existing paired revision interface and optional identity attachment; no image/source-pin proof. | Pass existing arguments; small same-process check, no new linear CLI. |
| `frontier/profiling/common/runtime_identity.py:41–54,95–157,174–216` | Records measured versions, Frontier source_revision, device and graph facts. Fingerprint has no image/native-source identity; both images can have the same versions and Frontier revision. | Add opt-in bound native facts to this existing module/fingerprint, not a separate identity system. Keep legacy output/fingerprint behavior unchanged when no binding exists. |
| `frontier/profiling/moe/moe_vllm_kernel.py:132–160` | Chooses low-level versus modular API by import availability and prints installed version; it does not pin the package. | No edit here: preserve actual API selection; source checks prove imported current modules, not a version-label comparison. |
| `tests/integration/step5_profiling/verify_fixed_probe.py:23–49,50–93` | Existing sealed inventory/hash, shell syntax, resource and producer dry-run checks. P0 MoE verification assumes scalar `moe_tokens`; there is no image/adapter/source-binding comparison. | Extend this existing verifier for explicitly bound plans; keep legacy branch. Core-owned dry-run is authoritative for input shape validation. |
| `tests/comparison/dp_placement_pp/vllm_replay.py:127–190,194–221,224–239,312–335` | Existing RUNTIME_ADAPTERS, adapter_record/check_native_reference/runtime_version/server_env own runtime identity and selector sanitization. Derived member is planned, not currently implemented. | Reuse native-design P3. No profiling adapter registry or derived-image literal elsewhere. |
| `tests/integration/step5_pd/build_stage.py:51–76,192–203,226–231` | Already supplies sha256, installed_source_hashes, source commit and exact adapter native_reference. | Reuse these host-side functions without editing this file or staging its whole serving dependency tree. |

These exhaust the executable descriptor/launcher/worker/producer/verification
chain used by the actual P0 profile path. Additional inspected launchers are
**not** on that path: `frontier/profiling/linear_op/step5_job.py:108–149` is a
standalone old smoke (P0 only reuses its argument/plan functions), and
`tests/integration/step5_profiling/boundary_probe_job.py:41–83,96–99` is an
explicit old-image boundary experiment. Historical native-environment source
inspection launchers likewise are not profile staging/export machinery. Do not
change their plans, frozen hash constants or defaults to make D29 appear ready.

## 3. Explicit interface: selection once, immutable inputs

### 3.1 Host-facing existing entrypoints

Extend only the P0-role path:

```
describe_p0_profiles(stage, run_id, *, role, gpus=1, profile_plan=None)
stage_p0_profiles(stage, run_id, *, role, dependency_site, gpus=1, profile_plan=None)
python -m frontier.profiling.step5.job --p0-role PREFILL|DECODE
    --profile-plan /absolute/new-generation/profile_plan.json
    --run-id <new-unused-id> --stage <new-unused-stage> ...
```

`--profile-plan` is an artifact selector, not another routing/image selector.
Its omission follows the unchanged original P0 plan. It is not admitted for
the fixed probe/smoke branches. An explicit new plan requires its declared
generation to equal `run_id`; do not inherit the old default probe run ID.

The generated plan retains the existing resource/model/role/attention/linear
fields and adds the following source-binding fields. They reuse native naming
where one exists:

| Field | Meaning / invariant |
| --- | --- |
| `generation` | New unused profile generation; equals requested run_id. |
| `vllm_version` | Explicit existing adapter label `step5_position_cyclic`; not installed package version. |
| `native_reference` | Exact adapter-owned object with derived `image`, `base_image`, `routing_strategy`; use `check_native_reference`, no image override. |
| `source` | Existing committed_source result `{repository, commit}` from final integrated Frontier code; staged files must be those committed bytes. |
| `model_config_revision` | Actual retained upstream model-config revision, not a guessed date/version. Existing config-file SHA is separately retained. |
| `image_paths.installed_package` | Exact native installed vLLM root, as used by native packaging; not local source-export root. |
| `image_source_hashes` | Actual fresh derived-image installed-source listing; host reads it with existing installed_source_hashes. |
| `native_derivation_manifest` | Native P1/generated image/source receipt tying base/derived identities and exact approved source changes together. |
| `roles[role].moe_input_file` | Core-owned census-derived routing input artifact; this design does not define or infer its schema, counts or coverage. |

The native source receipt supplies the approved source-path inventory and hashes;
do not create another maintained profile pin inventory. The installed pin set
is **53**, as established by native design: the already-pinned model is updated,
and forward_context/router_factory/routing_simulator_router are the three new
pins. Reject missing, extra, inconsistent or base-image source bindings using
the native generated inventory; do not silently accept50 or54.

Call adapter_record once when describing the package; compare exact
native_reference with that member. New MoE `routing_runtime_path` is derived
from its `routing_strategy` through the existing core routing-runtime interface.
No separate `--image`, `--routing-strategy` or duplicated policy choice is
exposed at profile-plan CLI level. The MoE executable keeps its existing public
`--routing_runtime_path`; generated commands set it, rather than asking users
to select the same policy twice. The additional MoE input option is the
core-owner's §6.2 option (proposed spelling `--routing_input_file`); payload
semantics remain wholly there.

### 3.2 Existing package and launcher seam

`describe_job(..., image=IMAGE)` keeps its base default. The selected P0 plan
passes only `adapter['native_reference']['image']`. Generated submission command
always contains `--image <that-exact-digest-reference>`. Existing launcher adds
`--image`, retains STEP_IMAGE as the legacy direct-CLI default, and uses
`ResourceConfig(image=args.image)`. The launcher performs no independent
policy selection or image lookup.

Reuse `stage_package` and its exclusive new-directory creation/hash seal.
Allow files as well as directories in existing `additional_paths`, and one
explicit `additional_files=()` sequence of `(source, stage_relative_path)` for
the externally generated plan, core inputs, pins and derivation receipt.
Require relative destinations within the new stage. No new tree copier or
submission wrapper is introduced.

Stage the already-existing registry implementation and its actual import
dependency in their repository locations:

- `tests/comparison/dp_placement_pp/vllm_replay.py`;
- `tests/comparison/stage_admission_pp/vllm_burst_driver.py`.

`vllm_replay.py:100–102` imports the latter. Staging only the registry file is
therefore insufficient even for offline/neutral-cwd identity checks. Use the
existing direct-file import pattern to load this staged registry explicitly;
do not rely on an installed third-party `tests` package or copy all tests.
Host-side installed_source_hashes/committed_source remain reused from
`step5_pd/build_stage.py`; no worker import of its serving dependency tree is
needed. The parsed pin mapping is frozen in the profile manifest.

For the new generation, stage verifies committed Frontier/config/registry/
launcher/verifier payloads against the recorded commit, then seals every
actual payload with the existing manifest sha256 inventory. Offline_site is a
dependency artifact, not Git source. Sealed old stages are not regenerated or
edited. Stage creation still makes no submit/job marker.

### 3.3 Worker binding: one artifact locator, not a second selector

The new worker command sets
`FRONTIER_STEP5_PROFILE_MANIFEST=<absolute-stage>/manifest.json`. This contains
the sole selected adapter/reference/generation/pin mapping. It is an artifact
locator, never a free-form image or strategy setting.

The worker command clears inherited `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` and
sets the existing native variable from the validated adapter routing_strategy.
Also require the frozen skip-padding setting; do not introduce another mask
switch. Set the existing breakable graph environment before model imports.
Legacy unbound worker commands/defaults remain unchanged.

The existing run_on_worker.sh performs the neutral check only when this binding
is present. It starts a fresh image interpreter with **neutral cwd under the
worker TMPDIR and `python3 -I -B`**, explicitly exposing only the staged Frontier
and registry code needed for verification. No exported `image_src/vllm`,
offline_site, worker dependency site, inherited PYTHONPATH or current source
directory may supply native modules during this check. It returns to the
existing source-root execution for producers, whose model-config paths are
currently relative. Neutral cwd is a verification scenario, not a speculative
relocation of all producer paths.

The same-process checks below are essential: a successful clean preflight
process alone cannot establish what a later Python process with different
PYTHONPATH imported.

## 4. Current-source checks and truthful row identity

Add the bound-native validation/row facts to the existing
`frontier/profiling/common/runtime_identity.py`. It owns source identity already;
no new global infrastructure module is needed. The selected-generation
interface is `verify_step5_profile_runtime(manifest_path) -> native facts` plus
an optional native-facts field on RuntimeIdentity. Read the manifest from the
explicit locator when collecting measured identity. Absent binding retains
legacy identity behavior; **position mode without binding is an error**, not
a fallback to legacy rows.

Required checks, in order:

1. Read the sealed binding and staged registry; compare adapter/native_reference,
   generation, staged receipt hashes and selected effective native env. Verify
   pins against the exact installed root. Existing `file_sha256`, native
   installed_source_hashes and stage inventory/hash checks are reused.
2. Check all53 listed installed files against fresh derived pins. Verify the
   four derived paths match the derivation receipt; other listed source bytes
   must retain the base identity. Do not accept a base source export with only
   new policy labels or an image tag whose installed version happens to match.
3. In a fresh neutral interpreter, import the actual native package and four
   modules: `vllm.models.step5.nvidia.model`, `vllm.forward_context`,
   `vllm.model_executor.layers.fused_moe.router.router_factory`, and
   `vllm.model_executor.layers.fused_moe.router.routing_simulator_router`.
   Require resolved `__file__` **and** `__spec__.origin` to be each exact installed
   path, and verify hashes. Reject a byte-identical foreign-root copy too.
4. In the **actual producer process**, check those modules and every pinned
   native module already imported for its operation through sys.modules, with
   the same root/origin/hash rules. Do not force-import every pinned backend
   merely to fill an inventory. Core native adapter construction must also bind
   the actual router class/method source to the checked simulator module; the
   attention wrapper binds its actual StepAttention class/method to the checked
   model module. This avoids treating unrelated imported modules as proof that
   the timed callable came from them.
5. Record measured package/runtime versions and actual import facts. Require
   source commit/config SHA and native fields to agree across outputs; let
   existing identity attachment reject conflicting columns. A source-origin
   failure leaves new output unverified; it never becomes accepted timing by
   adding labels afterward. No retry, image-version shortcut or source-shadow
   fallback.

The new generation passes existing `--source_revision <source.commit>` and
`--model_config_revision <actual-revision>` to **both linear and MoE** commands.
MoE source validation occurs before timing and again when its table-wide
identity is collected; the actual routing callable check is folded into the
already core-owned native adapter seam. Linear adds the same small pre-timing
check and uses its unchanged existing identity writer. Active path remains
`--disable_ray --num_gpus 1`; no parallel profiling launcher or Ray source
distribution system is added.

Keep these opt-in row facts in existing RuntimeIdentity/attention evidence:

- `native_runtime_adapter`, `profile_generation`;
- exact canonical `native_reference` JSON (derived image/base/policy);
- `native_source_hashes` from the sealed fresh installed pins;
- source-pin and derivation-receipt artifact hashes;
- actual native imported path/spec-origin/hash facts and verification status;
- the existing measured runtime_versions/source_revision/config SHA/device/
  graph/quant/scope metadata.

RuntimeIdentity's existing fingerprint includes bound reference/source facts
only when present. Thus a derived image/source with the same package version
cannot alias a legacy source fingerprint. Do not add another fingerprint,
cache, dataset registry or consumer schema migration. Attention uses its
existing runtime_stack_signature and evidence spread at wrapper586–620, rather
than declaring base IMAGE; no `rows.py` edit is necessary.

Distinguish **declared image** from observed installed-source proof. Python
module origins and hashes prove code identity, not the platform-selected OCI
digest. Parent's later launch/platform receipt must show the actual digest
equals the sealed manifest/ResourceConfig image. Do not set
`launched_image_verified=True` merely because imports passed. Native image
receipts must retain unchanged compiled-library/dependency identity as required
by native design; these are generated receipts, not additional maintained
profiling code or a GPU-only pin mechanism.

Source proof also is not numerical/kernel equivalence. For example,
`fp8_block_experts.py:216–225` currently constructs an isolated expert config
with RoutingMethodType.Unspecified, whereas the native design keeps the
selected router classified Simulated. Its actual oracle/kernel must be compared
with native T2's executed backend evidence before timing reuse. This document
does not change that classification or price its operators; core/pricing owners
must not treat matching source/version strings as kernel-scope parity.

## 5. Exact gross ceilings and union accounting

These are **maximum added/deleted diff lines**, including tests/imports/comments,
not net growth, implementation evidence or authorization. Files may be edited
only after the parent's combined numerical approval. No new source module is
listed here.

| ID | Exact file | + ceiling | - ceiling | Scope accounting / responsibility |
| --- | --- | ---: | ---: | --- |
| S1 | `frontier/profiling/common/step5_job.py` | 100 | 10 | ADDITIONAL: explicit image/submission, existing seal supports required files/receipts and committed payloads. |
| S2 | `frontier/profiling/step5/job.py` | 70 | 12 | OVERLAP: replace core design B6 allowance of +8/-3; complete generated plan/input/adapter/source/revision/worker propagation. |
| S3 | `tests/integration/step5_profiling/submit_profiling_job.py` | 8 | 1 | ADDITIONAL: --image with unchanged legacy default, ResourceConfig consumes it. |
| S4 | `tests/integration/step5_profiling/run_on_worker.sh` | 24 | 2 | ADDITIONAL: opt-in neutral-cwd interpreter check, existing final execution unchanged. |
| S5 | `frontier/profiling/common/runtime_identity.py` | 110 | 4 | ADDITIONAL: bound native root/spec/hash validation and opt-in existing row/fingerprint facts. |
| S6 | `frontier/profiling/step5/vllm_wrapper.py` | 20 | 4 | ADDITIONAL: actual same-process model binding and truthful attention signature/evidence. |
| S7 | `frontier/profiling/linear_op/main.py` | 18 | 1 | ADDITIONAL: pre-timing binding check; existing runtime identity attachment remains. |
| S8 | `frontier/profiling/moe/main.py` | 18 | 1 | OVERLAP: additive source checks on top of core +65/-12 input/selection allowance; merged ceiling +83/-13. |
| S9 | `tests/integration/step5_profiling/verify_fixed_probe.py` | 55 | 5 | ADDITIONAL: exact source/adapter/image/command/artifact checks; dispatch input verification to core dry-run, preserve legacy scalar branch. |
| S10 | `tests/unit/test_step5_fixed_probe.py` | 120 | 0 | ADDITIONAL: extend real staged launcher stub, sealed payload and worker shell fixtures; no new launcher/test harness. |
| S11 | `tests/unit/test_step5_attention_profiling.py` | 25 | 0 | ADDITIONAL: same-version different-source identity and actual callable/signature evidence. |
| S12 | `tests/unit/test_step5_moe_profiling.py` | 35 | 0 | OVERLAP: additive source-origin/router-callable checks; core +55/-0 becomes merged +90/-0. |
| S13 | `tests/unit/test_step5_request_free_profiling.py` | 30 | 0 | ADDITIONAL: old P0 plan/files/defaults unchanged; new explicit plan/generation passes the existing role packaging path. |

Subtotal: **13 unique files, +633/-40 = 673 gross lines**.
Core overlaps are S2/S8/S12, so **10 additional unique files**. Deduct the
already counted S2 +8/-3 once: **+625/-37 = 662 additional gross lines**.

Native19 scope is reused without any new file counted there:

- P1/P2/P10 already own sparse four-file image construction, image/source
  receipts, allowlist/source validation and its tests;
- P3 already owns the one derived adapter, exact native_reference and existing
  native selector setting/sanitization;
- P5/P6/P7/P9 already own D24 model/context/router schema pins and actual serving
  import-root/callable evidence; their serving observer is **not** installed as a
  new profiling observer. Profile S5/S6/core adapter must check their own process;
- T1/T2 already own native router/context and actual graph/module tests. Do not
  duplicate PW/FULL graph suites in profile source binding;
- P4/P8/P11/P12/P13 already own D24 runtime-label/client/HTTP/reference tests.
  They do not propagate profiling ResourceConfig or attention metadata.

For arithmetic only, the currently published core32/1699-gross snapshot
(including required actual-main fixture companion), native19/1003–1582 and
this source addition union to **61 unique files /3364–3943 gross lines**, before
the separately owned pricing correction and task documents. If the core owner
revises its table, parent must recompute that union from the actual paths. This
document's fixed deliverable is the **13-path table and its10-file/662-line
increment**, not permission to treat the older combined subtotal as final.

Unchanged maintained files include original P0 JSONs, routed_layer_groups.py,
profile shell examples, linear_op/step5_job.py, boundary_probe_job.py,
moe_vllm_kernel.py, step5_pd/build_stage.py, step5 rows/timing/graph files,
historical source-inspection launchers, README and package/dependency files.

## 6. Reusable checks, acceptance and evidence tiers

Required future checks; none are executed in this design:

| Check / reused evidence | Acceptance |
| --- | --- |
| `test_step5_fixed_probe.py:135–151,154–185` | Existing exclusive-stage/hash fixture and mocked actual launcher consume the new --image; captured ResourceConfig image equals sealed manifest/adapter. No StepMind backend is instantiated. |
| `test_step5_fixed_probe.py:113–132` | Existing shell/Python/nvidia-smi stubs retain driver compatibility behavior; new stub checks neutral cwd/interpreter invocation. This is shell-interface proof, not an installed-native import observation. |
| `test_step5_request_free_profiling.py:233–255` | Existing old-P0 five-command/resources/input expectations remain; original three P0 files retain exact bytes. New plan gives a new generation and core input artifact without modifying those expectations. |
| `verify_fixed_probe.py:23–49` | Exact sealed file inventory/hashes, syntax, path confinement, resources, and final worker-command argument. Add equality across plan native_reference, manifest image, parsed --image and recorded generation/source artifacts. |
| `verify_fixed_probe.py:50–93` + current producer --dry-run | All attention/linear/MoE commands consume staged paths. New MoE input coverage/count validation is the core-owned dry-run's result, not a scalar N comparison or a second validator. |
| Existing RuntimeIdentity tests in `test_glm_moe_profiling.py:61–79`, `test_glm_linear_op_profiling.py:171...`, `test_dsa_attention_profiling.py:343–366` | Reuse unchanged legacy identity behavior; add targeted source-binding cases only in the S10/S11/S12 existing files, not a broad schema rewrite. |
| Native P9 wrong-root/spec/hash fixture pattern and P10 allowlist/base-source checks | Reuse their adversarial scenarios: correct root/hash succeeds; same-byte foreign root, wrong spec.origin, wrong hash, missing module/pin, changed receipt or base image with new label fails. No need to rerun unrelated historical investigations. |
| Parent later derived-image neutral-cwd preflight + actual profiler-process records | Image Python imports only the expected installed modules; exact53 pins and all used callable sources agree; measured CSVs carry derived/native/source/generation identity. This observation is still not full serving/G3/kernel parity. |
| Parent launch/platform/image receipt | Actual launched immutable image digest agrees with adapter/stage/ResourceConfig; source-only proof is not substituted for it. No new resource/GPU budget is requested here. |

A real current-source negative test must hold package versions constant while
changing native source/reference; require a distinct bound fingerprint or
failure. A neutral-cwd success followed by source-shadowed actual producer
imports must fail the second check. The new selected policy without a profile
binding must fail before timing. Legacy no-binding rows keep their exact old
identity contract.

Existing verification commands are reused and narrowly extended after approval:

```
python -m pytest -q tests/unit/test_step5_fixed_probe.py tests/unit/test_step5_attention_profiling.py tests/unit/test_step5_moe_profiling.py tests/unit/test_step5_request_free_profiling.py
python -B <new-stage>/tests/integration/step5_profiling/verify_fixed_probe.py --stage <new-stage>
```

These commands are future CPU/source/package checks, not new profiling/GPU
execution authority. Before closing an implemented step, parent records exact
commands/environment/results in its existing test report/progress. On any
later32+-GPU run, auxiliary acceptance/harness failures remain non-blocking soft
gates under the standing rules; this design adds no cancellation path.

## 7. Generated artifacts, prerequisites and remaining decisions

**Generated, not maintained code:**

1. Native candidate snapshot/unified diff, before/after four-file source maps,
   sparse image context, derivation manifest, base/derived installed-source
   receipts,53-pin listing, compiled-library/dependency receipts and actual
   immutable published image digest: native P1/P2 and parent image operations.
2. A new profile-generation plan and core-owner census-derived MoE/attention
   inputs; output paths and coverage records name that generation. Do not edit
   `p0_profile_plan.json`, `p0_prefill_inputs.json`, `p0_decode_inputs.json` or
   pretend their representative points are complete D29 coverage.
3. Sealed existing stage manifest, copied plan/inputs/pins/derivation receipts,
   registry payload/dependency, submit.sh/submit_with_env.sh and source hashes.
4. Later launch receipt, neutral-cwd import report, per-command actual import
   evidence, producer config YAML/summary/CSV/trace/kernel-gap artifacts and
   parent admission/coverage report, all with the new generation. Durable cloud
   paths remain confined to the parent's personal root; no cloud action here.

**No source-binding semantic decision remains open.** The recommended interfaces
are concrete and reuse the existing adapter/native variable/launcher/identity
mechanisms. Missing actual derived digest/pins, final committed source and
census inputs are implementation/measurement prerequisites, not reasons to
invent a digest, fallback route or new budget. Core input-option spelling is a
routine owner integration detail; this design intentionally does not redesign
its schema.

Parent must union this scope with the revised core/fixture and pricing scopes
before one numerical implementation approval. Source binding alone does not
resolve local-router versus gathered-expert dimensions, collective identity,
empty-scope timing or independent G3 numeric agreement. Workflow -> GPU/operator
gap -> CPU gap last remains unchanged.

## 8. Local design validation

Only document/path/accounting checks are allowed here. The source trace is
recorded in §2; native-design source is the exact reviewed commit c9e9274, not a
guessed live file. Implementation checks in §6 are NOT_RUN. The final parent
handoff must report actual document accounting/format checks separately from
any production correctness or GPU evidence.

Observed local checks, Python3.10.6, current repository cwd:

- Inline `python3 -B` document parser checked S1–S13 uniqueness, all listed
  existing paths, sums, three overlaps and10 genuinely additional files:
  **PASS +633/-40=673; union increment662; provisional61/3364–3943**. The first
  arithmetic check failed because the prose total overstated additions by100;
  corrected the prose totals against the unchanged table and reran PASS.
- `git diff --no-index --check /dev/null task_memory/task_2026-10-22_step5_position_routing/profile_source_design.md`
  emitted no whitespace diagnostics (exit1 is the expected new-file difference);
  the independent parser also found no trailing whitespace and a final newline.
- Reread requirements.md before delivery. Parent source/status changed during
  this shared-tree session; those unrelated changes were not edited or reverted.
  This artifact is ignored by the existing task_memory rule, so the parent must
  explicitly add it if recording it in Git. No commit was created here.

Reproducible document accounting check, not a production test:

```bash
python3 -B - <<'PY'
from pathlib import Path
import re
path = Path('task_memory/task_2026-10-22_step5_position_routing/profile_source_design.md')
text = path.read_text()
pattern = r'^\| (S\d+) \| `([^`]+)` \| (\d+) \| (\d+) \| ([A-Z]+):'
rows = [(key, name, int(add), int(delete), kind)
        for key, name, add, delete, kind in re.findall(pattern, text, re.M)]
assert len(rows) == len({row[1] for row in rows}) == 13
assert (sum(row[2] for row in rows), sum(row[3] for row in rows)) == (633, 40)
assert sum(row[4] == 'ADDITIONAL' for row in rows) == 10
assert {row[0] for row in rows if row[4] == 'OVERLAP'} == {'S2', 'S8', 'S12'}
assert all(Path(row[1]).is_file() for row in rows)
assert all(line.rstrip() == line for line in text.splitlines())
assert text.endswith('\n')
assert 633 + 40 - 11 == 662
assert 32 + 19 + 10 == 61
assert (1699 + 662 + 1003, 1699 + 662 + 1582) == (3364, 3943)
print('Document accounting/path/format PASS; no production test or platform action')
PY
```

## 1. 规划完成情况

**汇总：** PASS 3｜BLOCKED/FAILED/SKIPPED 1｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 实际 profile staging/launcher/import/metadata 全链路 | PASS | 本文2覆盖两个 image锁、attention声明、source writer和stage verifier。 |
| 最小完整 binding interface 与去重 scope | PASS | 本文3–5：13文件、10新增、662额外gross行，无新registry/submitter。 |
| 复用checks及生成artifact清单 | PASS | 本文6–7区分CPU/source/image/实际import证据。 |
| source/tests/GPU/platform/commit执行 | SKIPPED | 本child仅有本文新增写权限。 |
| 联合numerical approval与实现 | TODO | parent合并core/pricing/native/source范围后推进。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** 本child design任务开始至今｜**当前前沿：** 完成binding设计交付，parent联合范围确认。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 只设计真实路径必需传播；不新增profiling基础设施。 |
| D2 细节反复重试 | CLEAR | 路径定位失配改读实际文件，无production修复循环。 |
| D3 支线问题修补 | CLEAR | 所有检查对应source/image/import/metadata acceptance。 |
| D4 可并行任务停滞 | CLEAR | 用户禁止额外agent；core/pricing由parent独立协调。 |
| D5 超范围镀金 | CLEAR | 唯一新增本文，所有代码行是未来gross上限。 |
| D6 分析空转 | CLEAR | design是交付，形成具体接口/路径/预算。 |
| D7 路线违背 | CLEAR | 无source/test/GPU/Docker/platform/cloud/auth/commit操作。 |

**文档检查：** S2/S5：`design.md:538–544,645–649`的job-only传播和预算未覆盖真实common image、ResourceConfig、attention声明、stage/verifier source链路；本文已接管完整binding设计。S1/S3/S4/S6未发现本slice明显缺陷。
**归因：** 无执行偏离。

## 3. 纠正建议

**分支：** B：spec/plan遗漏必要binding范围，先更新parent联合预算。

**文档更新（先执行）：**
1. 本文：写入完整source-binding补充及overlap算术（已执行）。
2. parent `plan.md`/`progress.md`：将本文13-path表与core/pricing/native表union，替换job-only估计（parent可直接执行）。
3. parent数值范围批准：明确完整文件/gross行上限后实施（需用户确认）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. parent联合scope：精确去重全部路径 → 完成标志：含profile launcher/import/metadata且pricing scope明确。
2. 批准后实施existing adapter/image/source传播与core输入 → 完成标志：现有mock launcher、sealed-stage、source-negative checks通过。
3. parent新image/generation证据 → 完成标志：真实derived digest、53 pins、neutral和实际producer imports、数值/coverage逐项验证，不冒充G3通过。

## 4. 最值得关注的 3 点

1. 两个独立image选择和attention声明都必须传播，包版本相同不能证明source一致。
2. native19已有image/adapter/D24 provenance；profile独立执行仍需10新增文件/662gross行，不能重复计数或省略。
3. 原P0 JSON/包保持不变；新digest/pins/输入/generation是生成artifact，未产生任何GPU或数值验收证据。
