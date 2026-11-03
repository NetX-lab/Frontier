# Formal P2-64 generation03 preparation and execution

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-03 | Independently accepted the third-generation package after generation02 preemption, under the existing three-job allowance. |

## Scope and result

The existing D-P2-64-FORMAL-B preemption rule permits one remaining GPU submission after exp-1102-164536-160317 and exp-1102-233941-576872. The builder adds only the `p2_64_formal_retry03` CASES entry. It keeps native `a06a72a4326cba02234df6cfbdc21b78d3a15d26`, harness `f33306d40443cfc0c44842d6ee044f6703c736e7`, model, engine,3repetitions of128formal6144/1024requests,16pair warmups per repeat, promotion receipt,64H200/all-preemptible/step_main/3600s/64GPUh. Run tag: `dsv4-h200-p2-64-formal-20261103-03`.

Package: `h200_mainline/p2_64_formal_retry03/package_01`. Manifest SHA256 `9f9ec3c1f9d2b72b957d0678a0e8f06e05de8bd1fbd68b8addc3c092043bb474`. It binds promotion receipt SHA256 `d505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596`. The old generation02 package remains unchanged at manifest `9f9a1ab954ef3b7d61ff67f2f62c891e9da8547590a21b19db020c8557e8ed7d`.

## Verification

Repository cwd: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Environment `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp PYTHONPATH=$PWD`; no conda activation. Build/check-only Python3.12.13 `/data/ycfeng/envs/openmopd-py312/bin/python`; lead audit system Python3.10.6.

```bash
/data/ycfeng/envs/openmopd-py312/bin/python -B task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_mainline/build_package.py p2_64_formal_retry03 --package package_01 --harness /data/ycfeng/Frontier/.worktrees/p2-64-startup-harness
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/tmp/glm53_dsv4/h200/audit_formal03_package.py
```

The audit invokes the new package's `launch.py --check-only` with the3.12interpreter and independently rehashes both generations.

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Existing builder seal/preflight/R3 reconstruction/offline launch path | Accepted input identity and8TP/4DP/32EP/PP1 per role |47sealed files; all stages PASS | PASS |
| Lead seal hashes | Every input equals its manifest |47/47new and47/47old match | PASS |
| Generation comparison | Only identity/path/provenance changes |41/47 byte-identical;6files differ only by exact package-root/run-tag/cloud-path/build-time/purpose/overlay-digest replacements | PASS |
| Source/workload/resource preservation | Clean native/harness pins, unchanged code and workload, approved resource shape | Both commits clean; every code/trace/engine/Frontier input byte-identical;8x8H200/preemptible/3600s | PASS |
| Lead launch check-only | No submission and explicit hash success |exit0,INPUT_HASHES_PASS_NO_SUBMISSION | PASS |
| Fresh generation | No existing launch markers/case runs | No markers or case/runs;submitted=false | PASS, local scope |

The first lead comparison failed because its replacement order changed the run tag before replacing the containing cloud path. The second failed because the YAML generation omits the `dsv4-` prefix. The audit now applies longest identity substitutions first and includes the exact YAML generation string; all remaining bytes compare equal. Package files were never changed to satisfy the audit. No runtime tests were repeated for this registry-only addition; actual build/check-only settle its acceptance. Evidence: `package_01_build.json`, `lead_package_acceptance.json`, and `executor_build.log` in the new generation directory. Pi label p64-formal3-build-1103a, session68a759cf-88df-4d7e-8532-fa8c9266a23c, settled/exit0; lead reviewed full7-line diff.

## Execution and limits

Submission is pending after this verified code/package commit. Launch through `/ds4-exec`, retain the launcher, inspect the exact job, and collect terminal/copy evidence. It is the third GPU job; no fourth or automatic restart is authorized. The24.3GPUh historical estimate is not measured usage of this pair-warmed generation; the unchanged ceiling is64GPUh. Post-run coverage/R5/all-request/warm checks require a complete reference. Prior formal attempts remain incomplete, and missing end clocks keep their exact usage open. Do not repeat the completed Frontier replay.

Commit whitespace check: three inherited trailing-space lines in frozen native_config_parallel.py at228,271,274 remain byte-identical to the approved native source and generation02. The authored-file check passes; immutable source bytes are preserved.
