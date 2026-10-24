# PROFILE04 package verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded pre-generation CPU results, the preserved environment mistake and bound native controls. |

## Environment and commands

Task W128-PROFILE04-PACKAGE; cwd `/data/ycfeng/Frontier/.worktrees/v4-native-profile04-run`, branch `v4-native-profile04-run`, fixed base53398d06. No dependency install or conda activation. Builder/check driver Python `/data/ycfeng/envs/frontier-py310/bin/python`,3.10.6/GCC11.3.0; worker/manifest/native-config Python `/data/ycfeng/envs/asyncopd-py312/bin/python`,3.12.13/Clang22.1.3. Scratch `/data/ycfeng/tmp/glm53_dsv4/profile04_package_W128`; durable receipts in cpu_evidence/.

Exact commands, cwd, sanitized environment and versions are retained in cpu_evidence/*.command.json. Source and provenance preparation used inline Python3.10 `-B -` with Git/path/JSON reads and new local record writes. No platform/image/GPU/auth operations. Initial overly broad tests used local HTTP stand-ins; this was stopped and is not part of the accepted verification.

```bash
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_pd_worker.py tests/unit/test_native_pd_manifest.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/glm53_dsv4/profile04_package_W128/pytest_py312 --junitxml /data/ycfeng/Frontier/.worktrees/v4-native-profile04-run/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/cpu_evidence/tests_py312.xml
```

The accepted test environment sets PYTHONPATH to this worktree plus the previously verified `/data/ycfeng/tmp/d29_py312_cpu_extras01/site`, CUDA_VISIBLE_DEVICES empty, PYTHONDONTWRITEBYTECODE=1, TMPDIR to the scratch root, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 and OPENBLAS_NUM_THREADS/OMP_NUM_THREADS=1. PATH=/usr/bin:/bin, HOME=/home/brainpp, LANG=C.UTF-8. It does not inherit platform credentials.

## Observed results

| Check | Passing condition / observation | Verdict |
| --- | --- | --- |
| Worker and manifest | 55PASS/0FAIL/0ERROR/0SKIP; original36worker cases become41 with5 additional parameter combinations, manifest14 unchanged | PASS |
| Four-lane regression | true and false both reach P0/P1/D0/D1 native profiler argv with exact dictionary equality; role512/64 and distinct paths; input config unchanged | PASS |
| Maintained budget | builder1add/1del=2; fresh plan50add=50; test17add/4del=21; source+plan52/100, tests21/60, total73 | PASS |
| Native06 AST | Existing profile-elision comparison equals both53398d06 and accepted1437ace6 | PASS |
| Fresh plan behavior | Only identities/provenance paths plus existing optional dump=false differ from03; caps/limits/window/pins unchanged | PASS |
| Historical77 matrix | Copy is byte-identical to03;77identical/0mismatch/new_capture=false, domain hashes unchanged; final rehash pending generation | BOUND_EXISTING_NOT_NEW_CAPTURE |
| Root actual-image CPU control | Existing native constructor CPU+CUDA activities, CUDA uninitialized; both exports1, annotations8; true statistics/lazy1/1, false0/0; table only true | BOUND_EXISTING_NOT_RERUN |
| Native virtual-duration control | Existing true RED/false GREEN; export-too-slow false RED; CPU-only false RED; false FIFO GREEN; native300-second budget unchanged | BOUND_EXISTING_NOT_RERUN |
| Package generation/entry/closure | Not yet performed; remains next after clean source commit | PENDING |

## Preserved initial failure

cpu_evidence/tests.command.json/logs/XML preserve the first Python3.10 command:101PASS/22FAIL. Four existing child fixtures fail `Unknown option: -P`; the extra calibration suite lacks aiohttp, causing sixteen direct dependency failures and two failed local proxy starts. No package/runtime defect was established. Reusing the original Python3.12 environment and narrowing to required worker/manifest suites gave55PASS without source/fixture/default/timeout changes. No dependency installation or repeated broad suite; original known-positive fixture failure #9 remains deferred and untouched.

## Limits

This owner provides prepare-only CPU/config/source evidence. Inherited source acceptance465PASS/1known-baselineFAIL/0SKIP and77-case matrix are bound, not new captures. Root retains independent reviews, actual image/SDK/source/binary controls, virtual-duration final acceptance, quota, seal, one run and actual data acceptance. Profile03 remains FAILED. Captured ownership, raw cross-DP pairing, useful kernel coverage, stream correlation, post-stop evidence and serving/operator costs remain unverified.
