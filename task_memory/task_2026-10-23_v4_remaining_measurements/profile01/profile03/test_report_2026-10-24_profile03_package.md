# PROFILE03 package CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded immutable generation, five accepted CPU checks, one corrected checker assertion and remaining root gates. |

## Source and environment

Task W128-PROFILE03-PACKAGE; worktree `/data/ycfeng/Frontier/.worktrees/v4-native-profile03-run`, branch `v4-native-profile03-run`, base/runtime1437ace6f4a438c1bb879d904e2bede0c777303c. Source9946771815903f6cd6de8147765fd41395deee21 was committed clean before the single generation. Root integration465PASS/1known-baselineFAIL/0SKIP and four No-findings reviews are bound byte-identically by primary_integration_acceptance.json; the twelve suites were not rerun by this package owner.

Builder and most CPU checks: `/data/ycfeng/envs/frontier-py310/bin/python`, Python3.10.6/GCC11.3.0. Native config AST checks: `/data/ycfeng/envs/asyncopd-py312/bin/python`, Python3.12.13/Clang22.1.3. No conda activation, dependency changes or GPU initialization. Raw logs/scripts/disposable copies under `/data/ycfeng/tmp/glm53_dsv4/profile03_package_W128`;42 selected evidence files copied byte-identically to cpu_evidence/. No local `/mnt` write, Docker, network, StepMind, auth, cloud, seal or launch operation.

All accepted check subprocesses used `-I -B` with only PATH=/usr/bin:/bin, HOME=/home/brainpp, LANG=C.UTF-8, TMPDIR=<raw directory>, CUDA_VISIBLE_DEVICES empty and PYTHONDONTWRITEBYTECODE=1. The entry-only child adds literal CPU_ONLY_SENTINEL_NEVER_SUBMIT authorization and STEPMIND_BACKEND=rjob solely to reach the traced pre-write boundary; no credential file or SDK import is reached. These are safe stand-ins, not authorization for an actual launch.

## Exact commands

Working directory is the worktree above unless a receipt says otherwise. Full actual argv, cwd, sanitized environment and timestamps are in cpu_evidence/generation.receipt.json and each *.command.json. Executed scripts are retained byte-identically in cpu_evidence/; check_adaptations.json binds the prior existing scripts and exact adapted bytes.

```bash
git worktree add -b v4-native-profile03-run /data/ycfeng/Frontier/.worktrees/v4-native-profile03-run 1437ace6f4a438c1bb879d904e2bede0c777303c
git add -f task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile03
git diff --cached --check
git diff --cached --numstat
git commit -m "Prepare PROFILE03 limits and bind integrated diagnostic runtime"
/data/ycfeng/envs/frontier-py310/bin/python -B task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/build_candidate.py --generation 3 --profile-plan task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile03/profile_plan.json
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/glm53_dsv4/profile03_package_W128/run_check.py entry /data/ycfeng/envs/frontier-py310/bin/python entry_controls.py
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/glm53_dsv4/profile03_package_W128/run_check.py native_config /data/ycfeng/envs/asyncopd-py312/bin/python native_profiler_controls.py
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/glm53_dsv4/profile03_package_W128/run_check.py integrity /data/ycfeng/envs/frontier-py310/bin/python integrity_controls.py
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/glm53_dsv4/profile03_package_W128/run_check.py package /data/ycfeng/envs/frontier-py310/bin/python package_controls.py
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/glm53_dsv4/profile03_package_W128/run_check.py integrity_r1 /data/ycfeng/envs/frontier-py310/bin/python integrity_controls.py
/data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/glm53_dsv4/profile03_package_W128/run_check.py balanced /data/ycfeng/envs/frontier-py310/bin/python balanced_frontier_controls.py
```

Generation ran through an inline CPU-only subprocess receipt wrapper, exactly the builder argv displayed above; the receipt includes stdout/stderr and clean source SHA. Preparation used inline Python3.10 `-B -` to derive bindings from Git and existing immutable inputs. Native06-elided AST and incremental gross-budget verification are repeated in retained integrity_controls.py, not inferred from a successful build.

## Expectations and observations

| Check | Expected and observed | Verdict |
| --- | --- | --- |
| Generation | One clean-source build, no overwrite;45 files/6,983,474bytes, no actual manifest/runs/pycache | PASS |
| Current source/acceptance |13 modules equal1437 Git bytes;3 diagnostic modules changed versus02,10unchanged; actual status/source_revision, acceptance SHA96d6e17cac8ca4ff1a8503ec525505c24cbc30690f7ad1d60d743fd5d6b67bc4 | PASS |
| Native/source/binary pins |87 local native hashes including accepted C26 leaf exact;20 binary pins byte-identical to prior overlay map | PASS metadata/source; installed image/binaries ROOT PENDING |
| Manifest closure | Existing seal's read-only file-list expression covers exactly all45 package files; prospective manifest SHA4e827ca74e26abfba5816b61f6bac095ca1da326957b80d0f5e4d5c0c59423b7; no seal invoked | PASS |
| Tool/model/workload |23tool/10retained/4model/5workload hashes; both workload copies exact;128formal/16warmup templates x3=384formal/48warmup | PASS |
| Actual launcher entry | Disposable test-only manifest; fresh generated launcher imports worker/vllm_replay without pd_replay preload or test-added dependency path; trace exits before write_run_manifest body | PASS |
| Actual bootstrap child | Actual generated bootstrap with explicit fake installed-source/binary/PathFinder stand-ins; generated PYTHONPATH is stage plus staged dp_placement_pp; fresh child imports worker unassisted | PASS wiring only |
| Actual worker pre-output entry | Real run() own imports complete; trace stops before root=Path(output_dir), no output-path access | PASS |
| Four node plans/commands | P0/P1=512,D0/D1=64; four distinct role/lane trace paths, localDP1 and correct headless/start-rank behavior;600000/268435456 parsed by actual collector | PASS |
| Native profiler validators | Actual pinned class/decorator/validators via AST; global64 and positive512/64, fixed flags retained; negative count/key/kind rejected3/3 | PASS CPU config |
| Balanced Frontier | Existing frozen R3 reconstruction:84sources, each role TP8/DP2/EP16/PP1/world_size16, balanced/balanced_topk | PASS |
| Resources/soft gates |4x8H800/codesign,64CPU/512mem_gb,3600/1800/1800/1200/90, localDP1, cloud mount unchanged; existing launcher bytes unchanged; policy non-preemptible/soft validators retained | PASS declared configuration; actual SDK RDMA/NFS/defaults ROOT PENDING |
| Native06/history/matrix | Native06 AST identical after profile-branch elision;123old files/3checkers, prior01/02 inputs and sealed02/native06 manifests rehashed; unchanged77-case comparison and5domain trees | PASS by exact binding; no new capture |

## Preserved checker failure

Initial integrity command exited1 at line82 because the disposable checker incorrectly expected `enable_rdma=True` and `host_path=str(workspace)` literals that the existing launcher does not use. This is not a package failure. Original script is integrity_initial_failed.py, SHA d4f1568797b2a64332af1bfd92da87e7f9db6adcd797fa168ddb0d84bd40a81f; exact failure and command are integrity.stderr.log and integrity.command.json. Inspection found ResourceConfig arguments at submit_native_diagnostic.py:36–48 and spawn_tasks code_mount_point at49–51. Corrected only that checker to require identical prior launcher bytes and actual declared mounts/memory/code_mount_point; integrity_r1 returned0 and explicit PASS. No package regeneration, suppression, runtime flag or SDK substitution.

## Budget, inventory and limits

Incremental maintained change against1437: builder14add/5del=19gross plus new plan49add=68/140gross. Prior profile02 historical152gross remains separate. Immutable generated candidate:45files/6,983,474bytes/280,064 artifact lines including inherited model/tokenizer and runtime bytes, not charged to the maintained-change allocation. Detailed module/native/binary/tool/model/package/evidence inventories and map digests are in hash_inventory.json and delivery.json.

No runtime/native-reference/operator-cost/E2E qualification. Root still owns independent source/package reviews, exact-image87source/23tool/20binary and fresh bootstrap checks, actual SDK/RDMA/NFS/non-preemptible checks, personal auth/quota/destination, seal/submission and actual retained-data acceptance. Actual >=3 useful steps per role/lane, zero unintended drops, raw cross-DP pairing, stream joins and post-stop evidence remain unverified; captured per-op ownership remains UNKNOWN. Existing soft diagnostics remain unchanged.
