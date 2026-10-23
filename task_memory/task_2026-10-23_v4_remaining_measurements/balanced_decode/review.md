# Balanced decode package review

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded independent source reviews and root package/image/SDK acceptance before the isolated GPU run. |

## Source preparation

Component: `prepare_package.py` and `package_plan.json`, base97803f07 to64133239; records9fb4bd7e; integrated56e9405d. Root's `git diff --numstat` gives39+175 additions, zero deletions, exactly214/300 maintained lines. `source.diff` is the complete two-file diff; its digest is in `root_seal.json`.

| Axis | Reviewer | Finding |
| --- | --- | --- |
| Specification | `ses_e97145e90ffeDHy4ZobE6HQCoL`, W128-BALANCED-DECODE-SPEC | No findings in the complete changed files, actual generated entries/configuration, producer call chain and supplied records. |
| Standards | `ses_e97145e7affe5inF1CF3n14ILz`, W128-BALANCED-DECODE-STANDARDS | No findings in reuse, origin preservation, scope, refusal behavior or recorded limitations. |

Both reviewers inspected full source and supplied evidence. They did not independently run Git, rehash all inputs, rerun tests or operate the platform. Root subsequently checked the complete Git change list, hashes and actual image/SDK entry. Neither review grants measured-data or serving admission.

## Root verification

`root_seal.json` binds530 package inputs,512 snapshot files byte-identical to Git97803f07, originals527/529 and historical candidate01's530 inputs unchanged. Actual locked-image bootstrap checks60 native sources,520 tools and20 binaries. Its fresh child executes the real balanced wrapper through origin/hash validation and stops at entry to routing registration, before the function body. Actual SDK launcher execution reaches `spawn_tasks` with1H800/codesign/16CPU/mem_gb128/non-preemptible and the correct local mount, then stops before submission. Existing77-case evidence is rebound with unchanged source domains; no fresh capture.

The subsequent real submission is `exp-1023-213905-916183`; `platform_launch.json` and `platform_launch_check.json` establish creator, non-preemptible placement, resources and local NFS source. The first verifier incorrectly compared memory strings128000Mi and125Gi; Kubernetes quantity parsing proves exact byte equality. This was an auxiliary assertion failure; the workload continued unchanged.

## Open acceptance

The platform showed Running at13:40:46Z. The13:44:39Z worker receipt confirms one NVIDIA H800, CUDA available, native vLLM0.28.0 and the measurement phase running. Final outputs, constructor-effective values, routing evidence, trace integrity and terminal state remain unverified. Physical TP1/eager/six-layer attention data will not establish serving, graph or physical TP8/DP2/EP16 cost equivalence.
