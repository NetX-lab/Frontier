# PROFILE04 package verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Added the single generation, successful staged-entry/config/closure controls and final immutable inventories. |
| 2026-10-24 | Recorded pre-generation CPU results, the preserved environment mistake and bound native controls. |

## Environment and commands

Task W128-PROFILE04-PACKAGE; cwd `/data/ycfeng/Frontier/.worktrees/v4-native-profile04-run`, branch `v4-native-profile04-run`, fixed base53398d06. No dependency install or conda activation. Builder/check driver Python `/data/ycfeng/envs/frontier-py310/bin/python`,3.10.6/GCC11.3.0; worker/manifest/native-config Python `/data/ycfeng/envs/asyncopd-py312/bin/python`,3.12.13/Clang22.1.3. Scratch `/data/ycfeng/tmp/glm53_dsv4/profile04_package_W128`; durable receipts in cpu_evidence/.

Exact commands, cwd, sanitized environment and versions are retained in cpu_evidence/*.command.json. Source and provenance preparation used inline Python3.10 `-B -` with Git/path/JSON reads and new local record writes. No platform/image/GPU/auth operations. Initial overly broad tests used local HTTP stand-ins; this was stopped and is not part of the accepted verification.

```bash
/data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_pd_worker.py tests/unit/test_native_pd_manifest.py -q -p no:cacheprovider --basetemp /data/ycfeng/tmp/glm53_dsv4/profile04_package_W128/pytest_py312 --junitxml /data/ycfeng/Frontier/.worktrees/v4-native-profile04-run/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/cpu_evidence/tests_py312.xml
/data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py --generation 4 --profile-plan task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/profile_plan.json
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/glm53_dsv4/profile04_package_W128/run_check.py entry /data/ycfeng/envs/frontier-py310/bin/python entry_controls.py
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/glm53_dsv4/profile04_package_W128/run_check.py native_config /data/ycfeng/envs/asyncopd-py312/bin/python native_profiler_controls.py
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/glm53_dsv4/profile04_package_W128/run_check.py integrity /data/ycfeng/envs/frontier-py310/bin/python integrity_controls.py
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/glm53_dsv4/profile04_package_W128/run_check.py balanced /data/ycfeng/envs/frontier-py310/bin/python balanced_frontier_controls.py
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/glm53_dsv4/profile04_package_W128/entry_r1/run_check.py entry /data/ycfeng/envs/frontier-py310/bin/python entry_controls.py
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/glm53_dsv4/profile04_package_W128/run_check.py package /data/ycfeng/envs/frontier-py310/bin/python package_controls.py
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/glm53_dsv4/profile04_package_W128/run_check.py final /data/ycfeng/envs/frontier-py310/bin/python final_controls.py
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
| Historical77 matrix | Copy byte-identical to03;77identical/0mismatch/new_capture=false; comparison SHA and5domain Git trees rehashed unchanged | PASS_BINDING_NOT_NEW_CAPTURE |
| Root actual-image CPU control | Existing native constructor CPU+CUDA activities, CUDA uninitialized; both exports1, annotations8; true statistics/lazy1/1, false0/0; table only true | BOUND_EXISTING_NOT_RERUN |
| Native virtual-duration control | Existing true RED/false GREEN; export-too-slow false RED; CPU-only false RED; false FIFO GREEN; native300-second budget unchanged | BOUND_EXISTING_NOT_RERUN |
| Single generation | Clean committed8cbf7ab9; one actual builder invocation;45files/6,983,749bytes/280,069lines; no seal/manifest/runs/pycache | PASS |
| Actual staged entry | Fresh launcher before write_run_manifest body, no pd_replay preload/path supplement; actual worker own imports complete before output path; disposable launch --check-only exit0 | PASS_CPU_WIRING |
| Bootstrap child | Actual generated bootstrap with explicit fake installed-source/binary/PathFinder stand-ins; native worker imports through generated PYTHONPATH | PASS_CPU_WIRING_NOT_IMAGE |
| Native profiler config | Actual pinned decorator/class/validators via AST; original default true, package false; positive512/64 and3negative controls | PASS_CPU_CONFIG |
| Hash closure |13runtime identities exact1437/03;87native local source hashes/20unchanged retained binary pins;23tool/10retained/4model/5workload identities;45-file prospective manifest SHA25bc61beca71021954aea4b875bd42ad4477e4482871d3422d0f35881ed3d58f | PASS_SOURCE_METADATA_NOT_INSTALLED_IMAGE |
| Workload/resources/soft gates |128formal+16warmup templates×3=432; P0/P1=512,D0/D1=64; limits600000/268435456;4×8H800/codesign,64CPU/512mem_gb,3600/1800/1800/1200/90, localDP1, unchanged cloud/local-source/RDMA mechanism, non-preemptible/soft policy | PASS_DECLARED_CONFIG |
| Balanced source | Existing frozen R3 reconstruction84sources; both rolesTP8/DP2/EP16/PP1/world_size16, balanced/balanced_topk | PASS |
| History/syntax | Exact03 full49files/45sealed inputs unchanged;123historicalfiles/3checkers unchanged;36new-package files byte-identical to03, nine expected fresh config/path/identity records;20Python files compile | PASS |

## Preserved initial failure

cpu_evidence/tests.command.json/logs/XML preserve the first Python3.10 command:101PASS/22FAIL. Four existing child fixtures fail `Unknown option: -P`; the extra calibration suite lacks aiohttp, causing sixteen direct dependency failures and two failed local proxy starts. No package/runtime defect was established. Reusing the original Python3.12 environment and narrowing to required worker/manifest suites gave55PASS without source/fixture/default/timeout changes. No dependency installation or repeated broad suite; original known-positive fixture failure #9 remains deferred and untouched.

The initial adapted entry check failed only its old-package absent-runs assertion after all fresh entry controls passed: sealed/launched03 already has retained case/runs metadata. Original failed script/receipt/logs are preserved; entry_r1 changes only that historical assertion and passes. The first inline final check incorrectly guessed35unchanged files; the actual36/9inventory now uses the explicit nine-record change set in final_controls.py and passes. No package bytes changed or regenerated. The whole-index git --check reports whitespace from raw pytest/XML evidence and unified-diff context; precommit_git_check.json records a passing maintained-source/Markdown check without altering raw evidence.

## Artifact and commit evidence

Source8cbf7ab9dc0690b537bc95f07e30ec545e40779c was clean before generation. source_changes.diff/.numstat and source_budget.receipt.json are preserved unchanged; source_budget_commit.receipt.json verifies exact equality against actual `git diff 53398d06 8cbf7ab9 -- <three maintained paths>`. Generated immutable files are separate from73gross maintained lines. hash_inventory.json records each generated file/runtime/native/binary identity; final_verification.json records all accepted outcomes. Evidence scripts/receipts copied byte-identically are indexed by evidence_inventory.json; disposable copies and old GPU logs are not vendored.

## Limits

This owner provides prepare-only CPU/config/source evidence. Inherited source acceptance465PASS/1known-baselineFAIL/0SKIP and77-case matrix are bound, not new captures. Root retains independent reviews, actual image/SDK/source/binary controls, virtual-duration final acceptance, quota, seal, one run and actual data acceptance. Profile03 remains FAILED. Captured ownership, raw cross-DP pairing, useful kernel coverage, stream correlation, post-stop evidence and serving/operator costs remain unverified.
