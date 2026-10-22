# D29 profiling runtime, image and source binding

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Parent specifies the missing post-build receipt command (51 Python lines including its container payload); syntax checked without executing Docker. No maintained exporter change. |
| 2026-10-22 | Close source-review P2: verify actual gate/norm/oracle and constructed callables from the generated full installed-source receipt; separate shared facts from import evidence, preserve omitted legacy fields, bind the real local config revision, and replace obsolete overlap arithmetic. |
| 2026-10-22 | Trace the entire existing P0 profile package path and specify the missing derived-runtime/source propagation, immutable artifacts, reuse checks and gross implementation ceilings. Design only. |

TaskID: **D29-PROFILE-SOURCE-CORRECTION** (continues D29-PROFILE-SOURCE-DESIGN).

## 1. Outcome and ownership

The canonical source-binding table is **16 existing code/test files,
+835/-44 = 879 gross lines** at most. It includes the former canonical13 table
plus three required producer provenance hunks: `moe_impl.py`, native norm's
Frontier wrapper and `fp8_block_experts.py`. Core no longer budgets `job.py`.
The prior **10-file/662-line increment is obsolete**: union current paths and
owned hunks, rather than subtracting that old job allowance. Four current core
paths overlap (main/test plus the added gate/oracle seams); **12 paths are
additional to core**. Native19 is unchanged. No native code, maintained pin
registry, image builder, Dockerfile, selector registry, submitter or maintained
campaign JSON is added.

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
The correction is restricted to reviewer-specified provenance/config/identity
seams; it does not examine or redesign the separately owned pricing proposal.

### Review finding closure (design, not implementation acceptance)

| Finding | Corrected contract / owned hunks |
| --- | --- |
| Original launcher P1 | Retained S1/S2/S3 explicit adapter-derived --image -> ResourceConfig propagation and S6 attention declaration correction. No new launcher. |
| P2 timed source omitted by53 pins/four modules | Full actual-image native_installed_sources.json supplies hashes; S5 validates actual module plus bound callable, S14 gate/core-created routing objects, S15 native norm, S16 actual config/FP8/BF16 oracle/constructed kernel; S10–S12 cover negative source and callable cases. Serving53 remains a subset, not profiling completeness. |
| Producer maps incorrectly forced equal | Shared identity agrees; actual imports/callables are producer-specific evidence. S5/S7/S8 and S10 specify that comparison and lazy-import collection. |
| Legacy identity gains null/empty native_facts | S5 omits field entirely from old fingerprint payload/columns/to_json; S10 compares exact old keys/hash. |
| Unavailable upstream config revision | Use real committed local config revision/hash, byte/copy-validated by staging and S6/S9/S10; no upstream identity invented. |
| Obsolete job/union accounting | Canonical16 source table below includes all required provenance hunks. Old10/662 is removed from current union arithmetic; four inspected core paths overlap but their behavior and provenance hunks have separate budgets. |

Closure is source-inspected design specification and document validation only;
actual source/tests/native-image imports remain unimplemented/unverified.

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
| `frontier/profiling/moe/moe_impl.py:270–282,301–303` | The actually timed gate is the constructed native GateLinear, not one of the four changed modules. | S14 verifies the exact gate object's class and bound forward using the full generated receipt, outside its timer. Core still owns factory/context/select_experts behavior. |
| `frontier/profiling/common/layers/layernorm.py:156–181` | Lazy make_step5_rms_norm returns the native norm implementation later timed by impl(x,...). | S15 verifies the actual factory and returned object's bound forward when constructed, before the norm timer; not just the model module. |
| `frontier/profiling/moe/fp8_block_experts.py:170–177,232–304,308–317` | Native config classes plus FP8 or unquantized/BF16 oracle choose and construct the kernel/experts; the returned kernel.apply is used at97–116. | S16 verifies actual selection/conversion/construction functions, returned class and bound kernel/expert callable sources, without changing backend/scopes. |
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
| `model_config_revision` | Authentic Git revision of the retained locally generated effective config file, not an upstream revision. Current observed config revision is b60d64d6073b126fd6439bdba5a13e3951ab758d, byte-verified below. |
| `model_config_sha256` | Existing file identity of staged data/config/models/stepfun-ai__Step-5-Preview.json and its identical worker model/config.json copy. |
| `image_paths.installed_package` | Exact native installed vLLM root, as used by native packaging; not local source-export root. |
| `image_source_hashes` | Actual fresh derived-image installed-source listing; host reads it with existing installed_source_hashes. |
| `native_derivation_manifest` | Native P1/generated image/source receipt tying base/derived identities and exact approved source changes together. |
| `native_source_receipt` | Generated native_installed_sources.json from the actual finalized derived image; full installed vllm Python-file inventory, including gate/norm/oracle and dynamically selected Python backend code. Not the four-file derivation map or53 serving pins. |
| `roles[role].moe_input_file` | Core-owned census-derived routing input artifact; this design does not define or infer its schema, counts or coverage. |

The serving installed pin set remains **53**, as established by native design:
the already-pinned model is updated and the three context/router paths are
added. This is the serving admission inventory, **not a complete profiling
dependency inventory**. Its exact53 membership agrees with the native serving
receipt; profiling's required source hashes come from the separate full
installed-source receipt defined below. Do not enlarge a maintained pin list
or infer a required dependency's hash from the imported worker copy itself.

#### Generated native-source receipt and emitter

The **parent native-image/source owner** emits `native_installed_sources.json`
after final image publication/digest verification, alongside the installed-source
export. The earlier claim of an existing retained enumeration command was not
supported; the explicit case-local command below supplies that missing interface.
It is a newly specified operational command, not a reused exporter and not an
additional maintained repository module. Generation reads the finalized image,
not the sparse COPY context, candidate tree or profiling worker. It runs only
after the approved image phase; it has not been executed for this design.

Exact receipt interface:

```
native_installed_sources.json:
  schema_version: native_installed_sources.v1
  image: <actual finalized immutable derived image reference>
  base_image: <locked base reference>
  installed_package: <exact installed vllm root>
  python_version: <actual image interpreter version>
  derivation_manifest_sha256: <sealed finalized native derivation receipt hash>
  sources: {vllm/<installed-relative-file.py>: <sha256 of actual image bytes>, ...}
```

Exact post-build command, run from the repository root. The image/source owner
supplies five inputs: `FINAL_IMAGE` and `BASE_IMAGE` are externally verified
immutable digest references; `INSTALLED_ROOT` is the actual installed package
root from image inspection; `DERIVATION_MANIFEST` is the sealed native derivation
artifact; `RECEIPT_PATH` is a new unused local-workspace path ending in
`native_installed_sources.json`. Its parent directory already exists. There are
no source mounts and no GPU request. The finalized image must already be local.

```bash
/data/ycfeng/envs/frontier-py310/bin/python -B - \
  "${FINAL_IMAGE:?}" "${BASE_IMAGE:?}" "${INSTALLED_ROOT:?}" \
  "${DERIVATION_MANIFEST:?}" "${RECEIPT_PATH:?}" <<'PY'
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

image, base, root, manifest, destination = sys.argv[1:]
for reference in (image, base):
    if re.fullmatch(r".+@sha256:[0-9a-f]{64}", reference) is None:
        raise ValueError("An immutable image digest reference is required")
manifest_hash = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
container_code = r'''
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

image, base, expected_root, manifest_hash = sys.argv[1:]
root = Path(expected_root).resolve(strict=True)
spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise RuntimeError("Installed vllm package was not found")
if Path(spec.origin).resolve(strict=True).parent != root:
    raise RuntimeError("Installed vllm root differs from the inspected root")
sources = {}
for path in sorted(root.rglob("*.py")):
    path.resolve(strict=True).relative_to(root)
    key = "vllm/" + path.relative_to(root).as_posix()
    sources[key] = hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {
    "schema_version": "native_installed_sources.v1",
    "image": image,
    "base_image": base,
    "installed_package": str(root),
    "python_version": sys.version.split()[0],
    "derivation_manifest_sha256": manifest_hash,
    "sources": sources,
}
print(json.dumps(receipt, sort_keys=True))
'''
result = subprocess.run(
    ["docker", "run", "--rm", "--pull=never", "--network=none",
     "--workdir", "/", "--entrypoint", "python", "-i", image,
     "-B", "-", image, base, root, manifest_hash],
    input=container_code, text=True, capture_output=True, check=True, timeout=60,
)
receipt = json.loads(result.stdout)
with Path(destination).open("x", encoding="utf-8") as output:
    output.write(json.dumps(receipt, sort_keys=True) + "\n")
PY
```

The derivation-manifest hash is computed outside the container from its exact
sealed bytes. The final image reference is passed after publication; this receipt
is not baked back into that image, avoiding a self-digest cycle. Exclusive `x`
creation preserves earlier artifacts. A failed command produces no admitted
receipt; existing staging checks still require serving pins, approved changed
files and consumed callable hashes to match. At execution, retain the expanded
command, its payload hash, UTC times, status and receipt hash in the task record.
This documentation adds no builder/exporter hunk; if execution later requires a
maintained exporter change, include that change in the scope before proceeding.

`sources` is generated by enumerating **every installed vllm/**/*.py file**;
membership is determined by the actual package, not a maintained module-name
registry. Source-backed paths must resolve inside the declared installed root.
The53 serving pins and four changed paths are required matching subsets. For
the gate/norm/oracle seam this necessarily includes:

- `vllm/model_executor/layers/fused_moe/router/gate_linear.py`;
- `vllm/models/step5/nvidia/layernorm.py`, its used `ops/rms_norm.py` path;
- `vllm/model_executor/layers/fused_moe/config.py`, `activation.py`;
- `vllm/model_executor/layers/fused_moe/oracle/fp8.py` and `unquantized.py`;
- whichever actual selected kernel/expert/linear Python files define bound
  callable code; their hashes are looked up by resolved module-relative path.

These names identify the required current call seam, not a new pin inventory.
Existing compiled-library/dependency receipts still attest native extensions;
the Python inventory does not claim bytecode/DSO/kernel execution attestation.
Stage seals this full receipt and its file hash. A used Python module or bound
callable with no matching receipt entry fails provenance; there is no
worker-self-hash trust fallback. The sparse builder need not learn new profile
dependencies or modify the native four-file allowlist.

#### Authentic effective-model configuration identity

Current APIs do not demand an upstream Git/HF revision:
`RuntimeIdentity.__post_init__:110–123` requires a nonempty string and
`measure_runtime_identity:174–198` hashes `model_config_json_path`; profiling
`ModelConfig.from_model_name:458–497` reads the local checked-in JSON. The
attention worker copies that file to model/config.json (`job.py:210`), then
EngineArgs loads it and selects its text config (`vllm_wrapper.py:72–85`).

Observed current file SHA256 is
`e535c5c9e6385e5c39d55daf346fc0b42dd93496bf5e540ea442168b2524bb63`,
exactly equal to the file bytes at its last config commit
`b60d64d6073b126fd6439bdba5a13e3951ab758d` (commit message: Represent Step-5
text layers, precision and cache components). This is an authentic revision
of the **local effective configuration**, not the upstream reference metadata.
Use this real commit in the existing model_config_revision argument while
those bytes remain selected. If final integrated code changes this file, stage
requires its actual new committed revision and actual hash, verified against
the selected commit; never substitute an unavailable upstream revision.
The staging host verifies revision/file bytes and seals the result; workers
verify the staged file and identical attention model/config.json hash without
needing repository Git. EngineArgs' existing text normalization is unchanged;
record its existing effective config evidence, do not invent an upstream
identity for it or hash an unrelated reference_inputs/config.json.

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
the externally generated plan, core inputs, serving pins, full installed-source
receipt and derivation receipt.
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
needed. The parsed serving-pin mapping and sealed full-source receipt reference
are frozen in the profile manifest; neither is an actual-import map.

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
interface is `verify_step5_profile_runtime(manifest_path, *, callables=()) ->
native facts` plus an optional `native_facts` field on RuntimeIdentity. Actual
object/function references, not configured names, are supplied at producer
construction/lazy initialization seams. Read the sealed receipt through the
explicit manifest locator. Absent binding retains legacy behavior; **position
mode without binding is an error**, not a fallback to legacy rows.

Required checks, in order:

1. Read the sealed binding and staged registry; compare adapter/native_reference,
   generation, staged receipt hashes and selected effective native env. Verify
   pins against the exact installed root. Existing `file_sha256`, native
   installed_source_hashes and stage inventory/hash checks are reused.
2. Check all53 serving files against fresh derived serving pins and require
   agreement with corresponding entries in native_installed_sources.json.
   Verify the four derived paths match the derivation receipt; unchanged files
   retain the native base identity. Profiling dependencies outside53 are checked
   against the full generated receipt, not ignored. Do not accept a base source
   export with only new policy labels or a same-version image label.
3. In a fresh neutral interpreter, import the actual native package and four
   modules: `vllm.models.step5.nvidia.model`, `vllm.forward_context`,
   `vllm.model_executor.layers.fused_moe.router.router_factory`, and
   `vllm.model_executor.layers.fused_moe.router.routing_simulator_router`.
   Require resolved `__file__` **and** `__spec__.origin` to be each exact installed
   path, and verify hashes. Reject a byte-identical foreign-root copy too.
4. In the **actual producer process**, validate all imported installed vllm
   Python modules through sys.modules against the full receipt, not only
   PINNED modules. Require __file__/__spec__.origin/receipt hash agreement at
   each exact installed path. Packages and lazily imported Python modules are
   included once present; do not import unused backends to equalize maps.
   Native extension objects use the already generated compiled-library receipt
   and are not represented as Python-source proof.
5. At the exact producer seams in the table below, verify actual supplied
   functions/classes **and bound callable code**: bound __func__, defining
   module, resolved source/code filename, qualified name and same receipt hash
   must agree with the module's installed root/spec/origin. Do not trust only
   callable.__module__ or an unrelated sys.modules entry. Inspect the ordinary
   Python function/JIT-wrapper's exposed underlying source function; do not
   claim an uninspectable compiled entry has Python-source proof. No new native
   observer/import hook or maintained callable registry is introduced.
6. Recheck the producer's actual imported map after execution and before row
   identity attachment to include lazy gate/norm/oracle/selected backend imports.
   Failed provenance cannot yield accepted timing by attaching labels afterward.
   Keep actual callable evidence through existing gate/kernel/norm producer
   metadata and the bound native_facts field; S7/S8 use the same S5 verification
   result for table identity. No retry, version shortcut or source-shadow fallback.

| Actual used object/function | Required same-process source seam; no routing/compute change |
| --- | --- |
| Native GateLinear | S14 `moe_impl.py:270–282`: after gate construction and before gating timer, check the supplied GateLinear constructor, actual `type(self.gate)` and `self.gate.forward.__func__`. Current GateLinear.forward is gate_linear.py:229, not a proof about only model.py or the four edits. Validate actual used Python dispatch module imports after profiling too; no backend selection is altered. |
| Native RMSNorm | S15 `common/layers/layernorm.py:156–172`: validate actual make_step5_rms_norm and the returned `_impl` class/bound forward before `_norm_timer` opens at179. The retained native class/factory are layernorm.py:110–166; its actual RMS/add-RMS Python/JIT source imports are included through the full receipt. Retain facts on the existing norm instance for S7 row-identity collection; do not hash files inside the timed forward or repeatedly verify each replay. |
| Native FP8/BF16 oracle/config | S16 `fp8_block_experts.py:170–177,232–304`: check actual FusedMoEConfig/ParallelConfig/QuantConfig classes; FP8 select/convert/make_quant/make_kernel functions or BF16 select/convert/make_kernel functions against their real defining modules. On constructed kernel, verify `kernel.apply.__func__`, selected experts class and bound source methods exposed by the actual kernel/experts object. Extend its existing returned kernel identity with provenance, not a label that assumes selected class was executed. |
| Native routing | Core still owns actual factory/context/set_routing_positions/select_experts in moe_impl.py. S14 only passes the **actual core-created** factory/router/context callables into S5 root/spec/source validation and records the returned facts; it does not replace or rebuild the adapter. |
| Attention | S6 checks the actual StepAttention class/bound forward and existing native model/loaded operation modules; no geometry/timing change. |

S7's active single-GPU path already has `wrapper.model`
(`linear_op/main.py:923–934`, `linear_op_wrapper.py:79–98`); it walks those
actual named modules after profiling to collect S15's norm-instance facts.
This is a bounded producer-object walk, not a new maintained pin/callable
registry or new linear wrapper file. S14 retains gate/core-router provenance
in existing gate metadata; S16 adds source facts to the returned
NativeRoutedExperts/kernel identity already consumed by the core. S8 retains
that actual per-result evidence while attaching common shared source facts.
The active one-process producer needs no extra process import transport.

The callable checks run at ordinary construction/lazy-creation seams outside
timers, graph capture and repeated profiled loops. The post-run map is evidence
of what the process imported, not a list of executed kernels. Missing full
receipt entries, same-byte foreign copies or a monkeypatched bound method from
another root fail. An unused module need not appear in a producer's import map.

The new generation passes existing `--source_revision <source.commit>` and
`--model_config_revision <authentic-local-config-commit>` to **both linear and
MoE** commands, with the file/copy hash contract in3.1.
MoE source validation occurs before timing and again when its table-wide
identity is collected; S14 provenance checks consume the actual routing objects
at the core-owned adapter seam. Linear's S7 collects the lazy norm instance
facts and final actual imported map using its existing identity writer. Active path remains
`--disable_ray --num_gpus 1`; no parallel profiling launcher or Ray source
distribution system is added.

Keep these opt-in row facts in existing RuntimeIdentity/attention evidence:

- `native_runtime_adapter`, `profile_generation`;
- exact canonical `native_reference` JSON (derived image/base/policy);
- serving pin hashes plus the full installed-source receipt artifact hash;
- source-pin and derivation-receipt artifact hashes;
- actual native imported path/spec-origin/hash facts and verification status;
- the existing measured runtime_versions/source_revision/config SHA/device/
  graph/quant/scope metadata.

Represent bound native_facts as **shared source identity plus actual evidence**:
`shared` contains adapter/reference/generation, serving-pin/full-source receipt
hashes, derivation hash, installed root and authentic config commit/hash;
`imports` and `callables` contain that producer's verified actual observations.
Shared image/source/config facts must match across producers of this generation.
Actual import/callable maps differ legitimately for attention, linear, FP8 and
BF16 and can grow through lazy imports. Never require their maps to be equal or
fill an unused oracle by forcing an import. `attach_identity_columns` still
checks conflicts within one table's identity; cross-producer comparisons compare
shared facts, not whole native_facts JSON or whole per-graph fingerprint.

RuntimeIdentity's existing fingerprint appends **shared native facts only when
bound**; actual import-map differences do not change source identity. Different
graph/device facts can still give distinct existing fingerprints. For unbound
legacy identity, the fingerprint payload, `columns()` and `to_json()` **omit
native_facts entirely**: no `native_facts:null`, `{}` or extra native columns,
and no extra IDENTITY_COLUMNS requirement. Keep exactly the old payload/key
set and hash; an optional dataclass field is not serialized via asdict.
Thus a derived image/source with the same package version cannot alias a legacy
source fingerprint. Do not add another fingerprint, cache, dataset registry or
consumer schema migration. Attention uses its
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
| S2 | `frontier/profiling/step5/job.py` | 70 | 12 | ADDITIONAL: canonical job allowance; core's former +8/-3 is removed, not subtracted again. Complete generated plan/input/adapter/source/revision/worker propagation. |
| S3 | `tests/integration/step5_profiling/submit_profiling_job.py` | 8 | 1 | ADDITIONAL: --image with unchanged legacy default, ResourceConfig consumes it. |
| S4 | `tests/integration/step5_profiling/run_on_worker.sh` | 24 | 2 | ADDITIONAL: opt-in neutral-cwd interpreter check, existing final execution unchanged. |
| S5 | `frontier/profiling/common/runtime_identity.py` | 170 | 6 | ADDITIONAL: full generated receipt, actual module/bound-callable root/spec/source/hash checks; shared/evidence split; exact omission of legacy native_facts payload/columns/to_json. |
| S6 | `frontier/profiling/step5/vllm_wrapper.py` | 25 | 4 | ADDITIONAL: actual same-process model/callable binding, authentic config-copy SHA and truthful attention signature/evidence. |
| S7 | `frontier/profiling/linear_op/main.py` | 28 | 1 | ADDITIONAL: pre-timing binding plus final imported/lazy norm evidence collection through existing runtime identity attachment. |
| S8 | `frontier/profiling/moe/main.py` | 25 | 1 | OVERLAP: provenance only, additive to current core +110/-30; merged +135/-31. No duplicate adapter implementation. |
| S9 | `tests/integration/step5_profiling/verify_fixed_probe.py` | 60 | 5 | ADDITIONAL: exact source/adapter/image/artifact/config-revision/full-receipt subset checks; dispatch input verification to core dry-run, preserve legacy scalar branch. |
| S10 | `tests/unit/test_step5_fixed_probe.py` | 155 | 0 | ADDITIONAL: existing launcher/stage/worker fixtures plus full-receipt, shared-versus-import maps, exact legacy omission/hash and local-config revision/copy checks. |
| S11 | `tests/unit/test_step5_attention_profiling.py` | 30 | 0 | ADDITIONAL: same-version different-source identity, actual bound callable/signature/config-copy evidence. |
| S12 | `tests/unit/test_step5_moe_profiling.py` | 50 | 0 | OVERLAP: additive provenance tests for gate/norm/FP8/BF16 oracle/kernel callable wrong-root/spec/hash/missing-receipt and monkeypatched method; current core +135/-0 becomes merged +185/-0. |
| S13 | `tests/unit/test_step5_request_free_profiling.py` | 30 | 0 | ADDITIONAL: old P0 plan/files/defaults unchanged; new explicit plan/generation passes the existing role packaging path. |
| S14 | `frontier/profiling/moe/moe_impl.py` | 18 | 0 | OVERLAP: after actual core-created gate/router construction, pass used constructor/class/bound forward/router callables to S5 and retain provenance outside timers. Core +95/-18 becomes merged +113/-18; factory/context/select_experts remain entirely core-owned. |
| S15 | `frontier/profiling/common/layers/layernorm.py` | 20 | 1 | ADDITIONAL: lazy native factory/actual returned norm bound forward receipt check and instance evidence before _norm_timer; no native implementation or replay-time check. |
| S16 | `frontier/profiling/moe/fp8_block_experts.py` | 22 | 1 | OVERLAP: actual config/oracle selection/conversion/construction function plus returned kernel/expert bound source verification and existing identity evidence. Core +20/-2 becomes merged +42/-3; no scope/backend change. |

Subtotal: **16 unique files, +835/-44 = 879 gross lines**. The corrected
canonical13 rows S1–S13 total **+775/-42 = 817**; S14–S16 add **+60/-2 = 62**.
The canonical13 now overlaps core only at S8/S12. The newly required S14/S16
producer hunks also share current core paths, giving **four shared paths /
12 additional unique files** overall. Source-owned hunks are additive to
core-owned behavior hunks; no job.py allowance is subtracted. Current source
amount to add to the revised core line total is **879 gross lines**, while
unique paths must be unioned (12 additional to the inspected core).

Correction versus the first source-binding table: **+202/-4 = 206 additional
gross lines**, comprising expanded S5/S6/S7/S8/S9/S10/S11/S12 and the62-line
S14–S16 producer hunks. These required hunks cannot be described as zero-cost
because the core already changes the same filenames. Parent must apply actual
diff accounting once implemented.

Native19 scope is reused without any new file counted there:

- P1/P2/P10 already own sparse four-file image construction, image/source
  receipts, allowlist/source validation and its tests;
- P3 already owns the one derived adapter, exact native_reference and existing
  native selector setting/sanitization;
- P5/P6/P7/P9 already own D24 model/context/router schema pins and actual serving
  import-root/callable evidence; their serving observer is **not** installed as a
  new profiling observer. Profile S5/S6/S14–S16 must check their own process;
- T1/T2 already own native router/context and actual graph/module tests. Do not
  duplicate PW/FULL graph suites in profile source binding;
- P4/P8/P11/P12/P13 already own D24 runtime-label/client/HTTP/reference tests.
  They do not propagate profiling ResourceConfig or attention metadata.

The old core32/native19/source10 combined estimate and10/662 increment are
obsolete. The parent now owns the revised core/fixture/pricing union; this child
does not inspect pricing architecture or publish a new combined total. Its
input is this canonical **16-path/+835/-44 source table**, the four explicit
current core overlaps and the unchanged native19 contribution. Any pricing
path overlap is handled by parent using assigned hunk ownership, not assumed
here. This source design contains no new semantic compute/collective proposal.

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
| S12 existing GateLinear/native FP8/BF16 builder fixtures (`test_step5_moe_profiling.py:86–127,149–225`) plus bounded fake norm factory | Positive correct receipt/module/bound method; negative gate, norm, oracle, actual kernel/expert method wrong root/spec/hash or absent full-receipt hash; an unchanged sys.modules entry with a monkeypatched foreign bound callable fails. Tests exercise S14/S15/S16 seams, not only generic S5. Native oracle monkeypatch controls remain legacy-unbound, or are explicitly used as negative bound-provenance cases. |
| S10 existing identity/stage fixtures | Unbound fingerprint equals the exact old five-key payload/hash; columns/to_json have exactly old keys with native_facts absent, not null/{}. Bound shared facts agree across linear/attention/FP8/BF16 while distinct imports/callables are retained; shared mismatches fail. A locally generated config with no upstream revision succeeds using its authentic committed file revision/hash; altered staged config or attention copy fails. |
| Parent later derived-image neutral-cwd preflight + actual profiler-process records | Serving53 is verified as a subset; actual gate/norm/oracle/config/kernel callable Python sources use the full generated installed receipt. Measured rows carry shared derived/source/generation identity and producer-specific actual evidence. This is not serving/G3/kernel execution parity. |
| Parent launch/platform/image receipt | Actual launched immutable image digest agrees with adapter/stage/ResourceConfig; source-only proof is not substituted for it. No new resource/GPU budget is requested here. |

A real current-source negative test must hold package versions constant while
changing native source/reference; require a distinct bound fingerprint or
failure. A neutral-cwd success followed by source-shadowed actual producer
imports must fail the second check. The new selected policy without a profile
binding must fail before timing. Legacy no-binding rows keep their exact old
identity contract, including complete omission of native_facts from fingerprint
payload/columns/to_json. Gate/norm/oracle and bound-callable wrong-source cases
must fail even when package versions and all53 serving hashes are unchanged.

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
   receipts,53 serving-pin listing, **native_installed_sources.json full Python
   inventory**, compiled-library/dependency receipts and actual immutable
   published image digest: native P1/P2 and parent image/source owner operations.
   Full source enumeration is a generated inline image-export artifact, not a
   maintained native pin registry or an additional source-image-builder edit.
2. A new profile-generation plan and core-owner census-derived MoE/attention
   inputs; output paths and coverage records name that generation. Do not edit
   `p0_profile_plan.json`, `p0_prefill_inputs.json`, `p0_decode_inputs.json` or
   pretend their representative points are complete D29 coverage.
3. Sealed existing stage manifest, copied plan/inputs/serving-pins/full-source/derivation receipts,
   registry payload/dependency, submit.sh/submit_with_env.sh and source hashes.
4. Later launch receipt, neutral-cwd import report, per-command actual import
   evidence, producer config YAML/summary/CSV/trace/kernel-gap artifacts and
   parent admission/coverage report, all with the new generation. Durable cloud
   paths remain confined to the parent's personal root; no cloud action here.

**No source-binding semantic decision remains open.** The recommended interfaces
are concrete and reuse the existing adapter/native variable/launcher/identity
mechanisms. Routine remaining owner dependencies are: parent native image/source
owner emits actual derived digest/53 serving pins/full native_installed_sources
receipt; core supplies its actual gate/router/context/kernel objects and census
input artifact; staging owner binds final committed Frontier and authentic local
config revision/hash. No unavailable upstream-config revision is a prerequisite.
These are artifact/interface delivery dependencies, not new semantic decisions,
fallback routes or GPU-budget requests. Parent alone combines pricing scope;
this design does not revise its architecture.

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

First-design historical accounting PASS was13/+633/-40; its10/662 overlap
increment is now superseded, not an input to the final union. This correction
changes only this document, including the canonical16-path table below. Actual
production tests/GPU/image checks remain NOT_RUN.

Observed targeted config check (current repository cwd, Python3.10.6):
`git log -1 --format='%H %s' -- data/config/models/stepfun-ai__Step-5-Preview.json`
returned b60d64d6073b126fd6439bdba5a13e3951ab758d. A read-only inline Python
SHA/`git show <commit>:<path>` comparison returned
`e535c5c9e6385e5c39d55daf346fc0b42dd93496bf5e540ea442168b2524bb63`,
same_as_last_config_commit=True. This settles the current authentic local
revision/hash prerequisite; it is not native numerical or profiling execution.
The retained helper/producer/source seams were read only; no tests ran.

The artifact remains ignored by the existing task_memory rule. Parent may
explicitly add it in its record commit; this child makes no commit and does not
modify/revert shared parent work. The reproducible current document check
verifies16 paths, +835/-44 totals, four current core overlaps, legacy omission
language and exact receipt/config interface; its PASS is document evidence only.
Executed that check with Python3.10.6: **PASS**. Independent
`git diff --no-index --check /dev/null <this-document>` produced no whitespace
diagnostics (new-file diff exit1 expected): **PASS**. Reread active
requirements.md before handoff. No production test, source edit, image/platform
operation, agent delegation or commit ran in this correction.

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
assert len(rows) == len({row[1] for row in rows}) == 16
assert (sum(row[2] for row in rows), sum(row[3] for row in rows)) == (835, 44)
assert sum(row[4] == 'ADDITIONAL' for row in rows) == 12
assert {row[0] for row in rows if row[4] == 'OVERLAP'} == {'S8', 'S12', 'S14', 'S16'}
first13 = [row for row in rows if int(row[0][1:]) <= 13]
assert (sum(row[2] for row in first13), sum(row[3] for row in first13)) == (775, 42)
assert all(Path(row[1]).is_file() for row in rows)
assert all(line.rstrip() == line for line in text.splitlines())
assert text.endswith('\n')
assert 835 + 44 == 879
assert 879 - 673 == 206
assert 'native_installed_sources.json' in text
assert 'b60d64d6073b126fd6439bdba5a13e3951ab758d' in text
assert 'native_facts entirely' in text
print('Document accounting/path/format PASS; no production test or platform action')
PY
```

## 1. 规划完成情况

**汇总：** PASS 3｜BLOCKED/FAILED/SKIPPED 1｜TODO 1

| 子任务 | 状态 | 简要依据 |
| --- | --- | --- |
| 实际 profile staging/launcher/import/metadata 全链路 | PASS | 本文2覆盖两个 image锁、attention声明、source writer和stage verifier。 |
| source P2、legacy/config接口与去重scope纠正 | PASS | 本文3–5：16文件/+835/-44=879，实际gate/norm/oracle callable来源；无新registry/submitter。 |
| 复用checks及生成artifact清单 | PASS | 本文6–7区分CPU/source/image/实际import证据。 |
| source/tests/GPU/platform/commit执行 | SKIPPED | 本child仅有本文新增写权限。 |
| 联合numerical approval与实现 | TODO | parent合并core/pricing/native/source范围后推进。 |

## 2. 偏离检测

**结论：** 未偏移，正常推进。
**检测窗口：** source-review correction至今｜**当前前沿：** 交付纠正设计，parent联合范围确认。

| 模式 | 判定 | 依据 |
| --- | --- | --- |
| D1 工具打磨 | CLEAR | 只设计真实路径必需传播；不新增profiling基础设施。 |
| D2 细节反复重试 | CLEAR | 路径定位失配改读实际文件，无production修复循环。 |
| D3 支线问题修补 | CLEAR | 所有检查对应source/image/import/metadata acceptance。 |
| D4 可并行任务停滞 | CLEAR | 用户禁止额外agent；core/pricing由parent独立协调。 |
| D5 超范围镀金 | CLEAR | 唯一新增本文，所有代码行是未来gross上限。 |
| D6 分析空转 | CLEAR | design是交付，形成具体接口/路径/预算。 |
| D7 路线违背 | CLEAR | 无source/test/GPU/Docker/platform/cloud/auth/commit操作。 |

**文档检查：** S5：本文旧53-pin/4-module检查漏实际gate/norm/oracle；旧upstream-config prerequisite和10/662去重已被后续决定替代。本次已改全source receipt+实际callable seams、authentic local config、16-path table；S1/S2/S3/S4/S6未发现当前slice明显缺陷。
**归因：** 无执行偏离。

## 3. 纠正建议

**分支：** B：spec/plan遗漏必要binding范围，先更新parent联合预算。

**文档更新（先执行）：**
1. 本文：纠正review P2、shared/evidence、legacy遗漏字段、local config、current overlap表（已执行）。
2. parent `plan.md`/`progress.md`：使用本文16-path表与core/pricing/native表union，删除旧10/662增量（parent可直接执行）。
3. parent数值范围批准：明确完整文件/gross行上限后实施（需用户确认）。

**需停止的偏离行为：** 无。

**接下去的路线：**
1. parent联合scope：精确去重全部路径 → 完成标志：含profile launcher/import/metadata且pricing scope明确。
2. 批准后实施existing adapter/image/source传播与core输入 → 完成标志：现有mock launcher、sealed-stage、source-negative checks通过。
3. parent新image/generation证据 → 完成标志：真实derived digest、53 serving pins、full generated source receipt、neutral和实际producer callable/import事实，数值/coverage由各owner验证，不冒充G3通过。

## 4. 最值得关注的 3 点

1. Serving53不覆盖所有profiling依赖；actual gate/norm/oracle+bound callable均以parent生成的full installed receipt校验。
2. 当前canonical16文件/+835/-44=879，四core路径共享但source hunks独立计数；旧10/662增量失效。
3. Legacy完全省略native_facts；local effective config已有真实commit/hash，无需虚构upstream revision或改变pricing架构。
