# Profile04 package review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Closed independent Standards/Spec reviews and primary source/test/image/SDK checks. |

## Standards

Task `W128-PROFILE04-PACKAGE-STANDARDS`, reviewer `ses_e92f6b619ffeSL3jzMyWKcBdeN`; fixed comparison `53398d06…6ec96b4c`. Inspected instructions, main/local task records, maintained source diff, generated config/launch/source lock and closure records. **No findings.** Builder generation guard, existing optional bool and four-node true/false argv regression conform to documented standards. Maintained source+plan52/100 and tests21/60 gross lines; no new abstraction, runtime rewrite or admission relaxation.

## Spec

Task `W128-PROFILE04-PACKAGE-SPEC`, reviewer `ses_e92f6b22fffeDtfv61hSk77QwL`; same fixed comparison and task records, plus input freeze, prospective manifest and native stop ordering. **No findings.** Fresh04 identity and `torch_profiler_dump_cuda_time_total=false` preserve512/64, record limits, native/runtime/C26 bytes, workload and300-second RPC. Raw export remains synchronous; CPU-only automatic statistics remain separate.

Both reviews are static: supplied55PASS/45-input/source-pin receipts were not executed or rehashed by reviewers. Preserved101PASS/22environment failures, local HTTP stand-ins and checker mistakes are not silently relabeled. Neither review grants GPU, serving or cost admission.

## Primary verification

- Independent required tests:55PASS/0FAIL/0ERROR/0SKIP; `/data/ycfeng/tmp/glm53_dsv4/profile04_root_tests_g2vtrw1t/receipt.json` preserves exact command and sources. The first environment's regular editable `verl.tests` package shadowed the Frontier namespace; the owner-verified asyncopd environment resolves it without a source edit.
- Five actual locked-image checks: `/data/ycfeng/tmp/glm53_dsv4/profile04_root_image_egwpqxbz/receipt.json`, image entry plus positions0–3 PASS,45 hashes unchanged. Reused prior image script unchanged; hook changes only the container-local diagnostic path and existing false-field assertion. The first import exhausted128PIDs; direct cgroup evidence and failed inputs are preserved. BLAS/OMP/MKL threads are limited only in the CPU sandbox, not the GPU package.
- Actual SDK ResourceConfig expression: `/data/ycfeng/tmp/glm53_dsv4/profile04_root_sdk_09unc5v7/receipt.json`,4×8H800/codesign/non-preemptible,64CPU/512mem_gb, required cloud/RDMA/anti-affinity configuration; no platform call.
- Independent153 source/Git/native/package checks: `/data/ycfeng/tmp/glm53_dsv4/profile04_root_final_0esz0v_h/receipt.json`,45 owner/merged inputs,13 runtime modules identical to accepted1437, native/profile-input hashes and stopped source writer PASS.
- Final metadata/native-duration selection is accepted only as [bounded offline/CPU evidence](../profile03/failure/primary_acceptance.json). Raw export may still exceed300seconds. Seal/submission and actual432-request/32-GPU/coverage/DP/stream/post-stop acceptance are separate primary responsibilities.
