# Balanced q1 decode package

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded the bounded Q2a package preparation assignment. |

[Original Request] Task W128-V4-BALANCED-DECODE-PACKAGE: work only in the new `v4-balanced-decode-run` worktree at `97803f07`; deliver an executable fresh 1-H800 balanced q1 decode package and CPU verification, not a launch. Keep profile02 queued. Use existing producer/bootstrap/aggregate mechanisms, accepted q1 and balanced-prefill inputs, native 0.28 image and C26/model/strategy pins. Preserve original bytes.

[Original Request] Own only task-local `balanced_decode/` records and at most two maintained non-core code/config files / 300 gross changed lines; generated immutable package contents and records are excluded. No core/native/C26/source-collector changes, external publication, cloud/StepMind/Docker operations or further delegation. Root owns actual image checks, final prelaunch, seal and platform operations.

[Original Request] Measure the existing q1 case at cached contexts 6143–7166, physical TP1 with target TP8-local dimensions, six layers, eager execution, two unmeasured decode warmups and 1024 measured steps per CUDA_EVENT/KERNEL_ONLY; keep traces. Resources: 1 H800/codesign, non-preemptible, 16 CPU/128 GiB SDK setting, 1800 seconds / 0.5 GPUh ceiling. Historical 664 seconds / 0.184444 GPUh is a reference only.

[Original Request] Inspect actual builders, derive expected artifacts from the current plan, verify routing/dependencies/model/provenance pins, run fresh launcher/bootstrap imports without dependency preload, prove original input hashes unchanged, run affected CPU checks and explicitly bind existing matrix evidence when source domains remain unchanged. Commit builder/config and records separately; stop writing after delivery.

[Decision] Preserve the existing q1 producer's planner `max_num_seqs=128` and effective isolated-engine `max_num_seqs=1`. Record CLI chunk budget, planned LLM constructor argument and actual native value separately; actual native configuration remains unverified until root's GPU run. No admission or decoder sidecar is added. Existing authority is D-V4-H800-RJOB-AUTHORITY; no additional resource approval is needed for the root's task-required H800 submission.
