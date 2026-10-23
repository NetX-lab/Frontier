# Diagnostic DP sequence source/CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded baseline, RED/GREEN, bounded context correction, final CPU results and unchanged-domain bindings. |

## Outcome and environment

Task W128-DP-SEQUENCE-IMPLEMENTATION: **PASS_SOURCE_CPU_WITH_SKIPS**, not native pairing or final root acceptance. Final source has18/40 production and138/140 test gross changed lines, total156/180. Source set SHA256 `52c2596a8705b2002ad3a2de45360de45592be5fe806fde992f5d4b735ae838a`; exact per-file base/candidate hashes and +/- counts are in `source_binding.json`. `source.diff` SHA256 `71ca353c76af4f5b392291f7226328c0d6547a7aefca7eff5e3a970656f04a78`.

Working directory `/data/ycfeng/Frontier/.worktrees/v4-dp-sequence-record`; base `44f2ddb9736f0abaaf6a52a5fb768519996b7da3`. CPU test Python `/data/ycfeng/envs/asyncopd-py312/bin/python`,3.12.13/Clang22.1.3; wrapper/binding Python `python3`,3.10.6/GCC11.3.0. No conda activation, installation, GPU/platform/network, package generation or further delegation. The native checkout is read-only at `2cf0a6915ce544dc493a0990f2ea38d81601128a`; CPU native bodies are extracted without importing their CUDA dependencies. Real Torch used by the dispatch oracle is the existing local2.9.1+cu128 environment, not locked-image2.13.0+cu130. Locked-image WS2 symmetry remains inherited evidence only.

Each receipt preserves exact argv, cwd, environment, UTC start, return code, test counts, raw log/XML paths and SHA256. Temporary scripts and logs: `/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/`. No temporary log/script is written outside that approved root.

## Commands and results

Executed wrapper commands:

```bash
python3 -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/run_suites.py baseline
python3 -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/run_suites.py red native_v2_serving_inventory native_serve_profile
python3 -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/run_suites.py green
python3 -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/run_suites.py green2 native_serve_profile
python3 -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/run_suites.py final native_v2_serving_inventory native_serve_profile native_serving_inventory native_groundtruth_check native_full_structure
python3 -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/run_suites.py final2 native_serve_profile
python3 -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/run_suites.py context_red native_serve_profile
python3 -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/run_suites.py final_context native_v2_serving_inventory native_serve_profile native_serving_inventory native_groundtruth_check native_full_structure
python3 -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/verify_bindings.py
git diff --check
```

Final expanded test commands, with common environment:

```bash
export PYTHONPATH=.:/data/ycfeng/tmp/d29_py312_cpu_extras01/site
export FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69
export TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q tests/unit/test_native_v2_serving_inventory.py -p no:cacheprovider --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/final_context_native_v2_serving_inventory.xml
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q tests/unit/test_native_serve_profile.py -p no:cacheprovider --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/final_context_native_serve_profile.xml
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q tests/unit/test_native_serving_inventory.py -p no:cacheprovider --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/final_context_native_serving_inventory.xml
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q tests/unit/test_native_groundtruth_check.py -p no:cacheprovider --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/final_context_native_groundtruth_check.xml
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest -q tests/unit/test_native_full_structure.py -p no:cacheprovider --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128-dp-sequence-implementation/final_context_native_full_structure.xml
```

| Phase/suite | PASS | FAIL | ERROR | SKIP | Expected/actual |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline V2/profile/inventory/groundtruth | 184 | 0 | 0 | 21 | No pre-existing failures; PASS with recorded unavailable fixtures |
| RED V2 | 45 | 1 | 0 | 0 | Raw side state absent on unchanged production; expected RED |
| RED profile | 22 | 14 | 0 | 0 | 13 missing-field failures plus1 new stand-in pooling AttributeError; not all14 are valid RED controls |
| Initial GREEN profile | 33 | 3 | 0 | 0 | New stand-in setup defects only; corrected below |
| Corrected GREEN2 profile | 36 | 0 | 0 | 0 | PASS |
| Context correction RED profile | 26 | 10 | 0 | 0 | Annotation counter absent; expected RED |
| Final V2 | 46 | 0 | 0 | 0 | PASS |
| Final profile | 36 | 0 | 0 | 0 | PASS |
| Final inventory | 28 | 0 | 0 | 0 | PASS |
| Final groundtruth | 78 | 0 | 0 | 21 | PASS with unchanged skips |
| Adjacent native_full_structure | 6 | 0 | 0 | 0 | PASS; covers the directly composed FullModelInventory/Cutlass source module |
| Final total | 194 | 0 | 0 | 21 | All executed CPU checks PASS; not a skip-free groundtruth gate |

Groundtruth skips are identical before/after:19 `Approved C26 source fixture is unavailable`;2 `External legacy checker baseline is unavailable: .../h200_glm_canonical_integration_20261021/check_groundtruth_run.before.py`. Do not interpret these as passed native/C26 controls. No substitute fixture, installation or external source edit was attempted.

## Requirement evidence

| Acceptance | Observed result |
| --- | --- |
| Actual explicit group; raw nonzero startup and ordering | Local group starts37; native CPU reduce completes/increments before getter records38. Wrong/default group is not substituted. Missing `group` yields exactly a soft `KeyError: 'group'`, while return/vector remain intact. |
| Repeated and entirely zero vectors | Source-backed dispatch test retains38 then39/40/41 for repeated zero4×2 vectors. Inactive dispatch completes the reduction but does not read the getter. No offset is subtracted. |
| Native return/failure identity and observation isolation | Native all-reduce marker returns unchanged; native exception object is unchanged and getter is not called after failure. Missing/throwing getter leaves step/annotation counter absent; vector and exact dispatch keys survive. |
| Per-execution reset | Unedited pinned Worker.execute_model test raises during pending-send wait before runner entry; next retained step has no stale counter. Zero/no-forward and subsequent successful execution reset correctly. Native forward failure after successful reduction retains that execution's raw counter, and the next zero execution has none. |
| Direct dummy peer coverage | Across10 existing FULL/PIECEWISE descriptor cases, the unedited pinned Worker.execute_dummy_batch body calls the CPU-carried runner `_dummy_run` directly. Ordinary annotation/step counter38; direct dummy annotation39 with `step=None`; no extra Worker step is invented. Native V2 `_dummy_run:695–702` was inspected to confirm direct runner execution with `dummy_run=True`. |
| Saved-step context after later dummy execution | Existing sample_tokens hook executes its original CPU carrier and returns the marker unchanged. Its context uses detached step counter38, not the later dummy counter39. With no active dispatch/step, the context omits the field. |
| Detachment and byte admission | Final step enrichment remains before JSON detachment/append; enriched row exceeding configured bytes is dropped. Mutation of source dispatch/counter does not change retained row. Final context tests reconcile retained_bytes exactly with all encoded step/annotation/synchronization rows including the optional counter. |
| Frozen schemas, base inventory and hooks | Existing R3 positive controls pass and the additional dispatch-dictionary sequence key is rejected. Existing ambiguity and negative controls remain. Base `count`/`emit` AST and all inventory hook statements are byte-equivalent/AST-equivalent to base; no base row/dedup/schema change. |

## Original errors and scoped corrections

Initial RED/GREEN errors remain in `red.json`, `green.json` and their raw logs. RED profile has13 valid missing-field controls and1 stand-in `AttributeError: 'GPUModelRunner' object has no attribute 'is_pooling_model'`; that fixture error is not counted as proof of missing-counter behavior. In initial GREEN the two new NONE-mode profile tests had `KeyError: 'dispatch_evidence'` because that fake execution made no profile range/context call, and the same Worker pooling AttributeError remained. Only local fixtures were corrected to FULL mode/pooling=False; production was not altered to suppress these errors.

The first binding-script attempt had `FileNotFoundError` for a worktree-local R2 checker. Inspection confirmed R2 is an external fixed-path dependency, not a tracked file in this worktree/base. The temporary verifier now checks the real configured external R2 bytes against its existing frozen SHA256, and compares tracked R3 bytes to base. No R2 copy or checker edit was created.

## Frozen/native and matrix binding

`frozen_native_binding.json`: R2 SHA256 `29523700ac8492fa6e192ce272f8ab5a7a48d1528bd4064417e26e811533d10a` matches R3's existing pin; tracked R3 SHA256 `ca8ddcea83c852c27295c8a73453746d755c51055e703bd94df48e8f1af7d917` equals base bytes. Pinned Worker, V2 runner and DP source equal their native Git objects; DP SHA256 `3c882f85109ba47e473953351d166c4377ceb995205e91cbf128ca5075775a5d` matches retained exact-image proof.

`matrix_binding.json`: explicitly inherits root stream-probe binding SHA256 `39e2435a25ef718dadafa0a8f0a67c4b91fe559f409aa929e5014338c901a57b`, source `5f7d0950b78276a77a66b7330cf0041a9f380f12`; original comparison `4a1b53e6033134c59b4561306eb2d586bd3d8300`, SHA256 `7a8728ecc0851373827d9cdc7dfb68c552299e6c54b5fae75bd8cc5429d36ad6`,77 identical cases. The five domains' Git tree SHA256 is identical at comparison source, inherited source and owner base (`e9be15c60dffc0083b887460ddfbad61718f40f16246be201aa22571f6d12d62`), and none changes in the candidate. **No new matrix capture.**

## Limits

Optional diagnostics only. No actual native Worker distributed run, native startup-offset measurement, cross-DP pairing, native-reference admission, serving calibration, GPU timing or historical profile02 reconstruction is established. Future pairing uses run generation/role/ordered DP group ranks with lane/rank as members, never local Worker steps or corrected offsets. Root owns independent review, final integration and fresh native collection.
