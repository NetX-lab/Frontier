# Routing reader verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded actual-comprehension RED/GREEN, final regression, diagnostic and immutable evidence checks. |

Task ID: W128-V4-NATIVE06-ROUTING-READER. Decision: D-V4-NATIVE06-ROUTER-OWNER-READER.

## Environment and scope

Workdir: `/data/ycfeng/Frontier/.worktrees/v4-native-routing-reader`, baseline b62c155.
CPU-only Linux, `/data/ycfeng/envs/asyncopd-py312/bin/python`, Python 3.12.13, Clang 22.1.3, PyTorch 2.9.1+cu128. No conda activation, dependency install, GPU/platform/SSH/cloud/Docker action. Initial read-only hash inspection used `/usr/bin/python3` 3.10.6. Task filenames follow the requested local 2026-10-23 date; diagnostic UTC timestamps are retained in JSON.

Maintained correction: new 52-line R4 and one existing native unit-test file. Exact delta 216 additions/seven deletions = 223 changed lines, two files, zero `frontier/` edits. Task records are separate.

## Commands

```bash
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_reader_binding.py -k 'balanced_strategy_actual_import_and_missing_identity or test_r4' -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-routing-reader-focused --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-routing-reader-focused.xml
env PYTHONDONTWRITEBYTECODE=1 FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 NATIVE_READER_BINDING_CHECKER=/data/ycfeng/Frontier/.worktrees/v4-native-routing-reader/task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r4.py /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_reader_binding.py tests/unit/test_native_v2_serving_inventory.py tests/unit/test_native_serving_inventory.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-routing-reader-final --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-routing-reader-final.xml
git diff --check
```

Full diagnostic used `env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B -` with this operative body (no `main()`):

```python
spec = importlib.util.spec_from_file_location('routing_reader_diagnostic', checker_path)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
result = checker.check(
    Path('/data/ycfeng/tmp/glm53_dsv4/native06_case_20261023'),
    'dsv4-h800-p0-canonical-20261021-06', 'clean')
```

The result was written only to new `native06_r4_diagnostic.json` in this task. The original case analysis was read/hash-checked only. Integrity checks used Python SHA256 reads of 199 original files and byte comparison to `git show b62c155:<path>` for 744 regular production/matrix-input files; git mode 160000 was checked separately with `git ls-tree` and `git submodule status`. See `integrity_and_matrix_scope.json` and the before/after snapshot `/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-native-routing-reader-original-hashes.json`.

## Results

| Check and passing condition | Actual | Result |
| --- | --- | --- |
| Real unchanged collector comprehension excludes MXFP4; independent recorded identities remain complete | Native source hash assertions pass; actual-shaped MoERunner/RoutedExperts/Mxfp4MoEMethod produces `{}` in moe_backends, frozen R3 raises the original identity error, R4 passes | PASS |
| Legacy FP8/TRITON collector remains accepted | Real comprehension emits its backend; frozen R3 and R4 both accept | PASS |
| Focused owner/source/dependency controls | 75 PASS, 53 deselected, 5.64s, exit 0 before final seven FP8 controls | PASS |
| Existing reader/dispatch regression after final test edits | 208 PASS, 51.23s, exit 0 | PASS |
| Actual native06 positive identity census | 32/32 ranks, 43 independent owners each; frozen R3 rejects all 32, R4 passes all 32 | PASS |
| Missing/extra/wrong owner/module/child/backend/router/runner | All 15 map controls plus renamed module/child controls reject | PASS |
| Empty coverage, wrong consumer tuple, strategy/source/environment/hash identity | All controls reject; serialized `input_dtype="None"` cannot be null | PASS |
| Legacy absent selector and frozen dependency tamper | Absent selector remains unchanged; unsolicited selector rejects; R3 and R2 altered bytes reject | PASS |
| Preserve every non-owner seam | Reverse the one-line R4 replacement reproduces byte-exact R3; generated R2 successor text equals frozen R3 output | PASS |
| Read-only full R4 diagnostic, no new downstream failure | 8/8 predicates PASS; rank_runtime_graph reports 32 physical ranks, six phases/rank, original collector SHA256 | PASS, diagnostic only |
| Frozen R3/R2, 43 package inputs and original evidence | 199 unique original files unchanged; 43/43 pins; original R3 analysis remains FAIL | PASS |
| Production and comprehensive-matrix inputs unchanged | 744 regular files byte-identical to b62c155; collective-sim gitlink unchanged/uninitialized; no matrix path imports the changed checker or unit test | PASS, root capture accounting pending |
| Diff whitespace | No errors | PASS |

An auxiliary integrity probe initially failed with `AssertionError` after misclassifying the uninitialized collective-sim gitlink as a missing regular file. Mode 160000 inspection confirmed unchanged d28fe917370c2d55230c3ea226e5660785b44aca; corrected read-only inspection passed. This was not a downstream R4 failure and caused no code change.

## Limits

No native/reference admission, device completion/timing claim, new collection or metric change. Root must independently review, rerun and freeze R4, then decide re-admission. Root also maps the recorded unchanged-byte/reachability evidence to the prior 77/77 comprehensive capture or runs the affected default matrix; this task does not waive or independently close that gate.
