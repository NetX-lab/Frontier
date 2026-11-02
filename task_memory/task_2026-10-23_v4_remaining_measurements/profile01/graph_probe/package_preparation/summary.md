# Graph payload package preparation archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Bound maintenance source/records to6f84a88b; local preparation complete. |
| 2026-10-24 | Archived the prepared immutable package, scoped CPU validation and remaining root gates. |

Task ID: `W128-GRAPH-PAYLOAD-PACKAGE`.
Branch/worktree: `v4-graph-payload-package`, `/data/ycfeng/Frontier/.worktrees/v4-graph-payload-package`.
Base: `a2c4ad87f231a113f0d4d543050e3697a208dcba`.
Verified maintenance/records commit: `6f84a88bd17ef3cdd1626a6beb193275dab5df62`; final archive commit returned separately.

## Outcome and deliverables

Prepared, not sealed or launched, package:
`/data/ycfeng/Frontier/.worktrees/v4-graph-payload-package/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/graph_probe/h800_graph_probe_01`.

Its8 inputs are `launch.py`, `config.json`, unchanged `probe_diagnostic.py`, `source_lock.json`, and `inputs/{build_package.py,profile_plan.json,launcher_template.py,template_config.json}`. `manifest.json` is the ninth file; SHA256 `ec816706edeb0930756ad7077684a2872b93c7728ae8851264e3e2592409c146`, PREPARED_NOT_SEALED/submitted=false. The package remains untracked and has no launcher receipts.

The exact frozen probe revision is `1aa1e4a5ce24bba8aafc8dd9f2fb41236d016077`,669 lines/34078B, SHA256 `a9d6a24e0aff13ef61088d75b5f693b599f42e8c50c2702342fbb84d3754d1a0`. No reviewed worktree/source/records, old package, subscriber/stage/strict9-node5-kernel4-event logic or native/C26/simulator source changed.

Committed maintenance/records reside only in sibling `package_preparation/`: builder/plan, requirements/progress/summary/test report, original pins, generated-entry/stand-in/selected-output receipts, package inventory, launcher diff, source provenance and existing matrix binding. Builder113+plan20=133/220 maintenance lines (162 including generated config); generated launcher73/100, old template65, delta12 added/4 deleted. `package_inventory.json` retains every exact source byte/line/hash; final `artifact_hashes.json` binds all preparation and generated artifacts. The old generated launcher is content-hash-bound rather than assigned an invented Git revision; source provenance distinguishes its old profile-source revision from launcher provenance.

## Configuration and validation

Fixed oneH800/codesign/non-preemptible, replica1,CPU16/mem64/600s; exact image2286e853, CUDA13.0 compat, standard cloud mount and local package workspace. Cloud root `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/graph_payload_probe/1aa1e4a5_h800_20261024_01` has sibling `diagnostics` created before CLI; `probe` remains absent until the probe creates it. Worker tmp/cache are local. Initial nvidia-smi, timeout/exit preservation, host/worker manifest+source checks and existing ResourceConfig/spawn/launch receipts are retained without a new command wrapper or retry.

Real generated launcher --help/dry-run/bash syntax passed;4 Python sources compile/AST,5 JSON parse. Seven local fake-native/tool/integrity/exclusive command controls passed, including exact0/17/7 exit preservation, diagnostic-before-entry/absent-output, host/worker integrity and existing-directory/package refusal. One explicit in-memory SDK stand-in captured exact resource/mount/env/command and scratch-only receipts. All33 original file pins matched;8 real package inputs and manifest remained unchanged/untracked. Existing77-case comparison is explicitly bound with unchanged simulator domains, not rerun.

Initial CPU invocation quoting failed parsing before execution and was corrected without production/package edits; retained in `cpu_standin_initial_failure.json`. Exact commands, executed control stdin, substitutions and outputs are in `generated_entry_cpu.json`, `standin_cpu_receipt.json` and the test report.

## Root-only remaining work

Independent frozen-source reviews, actual locked-image/SDK/CUPTI binary identity, current personal auth/quota, local NFS source mount and fresh personal cloud destination validation, seal, at-most-one1H800/600s launch/monitoring and raw-data/nativeU1–U4 acceptance. The existing probe API reports no binary SHA256; no speculative helper/source edit was added. CPU stand-ins do not establish actual platform/image/CUDA/runtime behavior. No GPU/platform/auth/network/Docker/StepMind/SSH/cloud action, real submission, automatic retry or paused GLM operation occurred.
