# W128-GRAPH-PAYLOAD-PACKAGE requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded the package-only allocation, immutable reviewed source and root-only launch boundary. |

[Original Request] Prepare, but do not seal or launch, a runnable immutable package in a separate worktree `/data/ycfeng/Frontier/.worktrees/v4-graph-payload-package`, branch `v4-graph-payload-package`, from `a2c4ad87f231a113f0d4d543050e3697a208dcba`. Own only this `package_preparation/` directory and fresh untracked sibling `h800_graph_probe_01/`. Do not change the reviewed probe worktree, source or records; further delegation is prohibited.

[Original Request — primary D-V4-GRAPH-PROBE-PACKAGE] Reuse PRIMARY `profile01/stream_probe/h800_stream_probe_01/{launch.py,config.json}`. Maintenance builder/config at most220 gross lines combined; generated launcher at most100 lines, counted separately. Source copy must match commit `1aa1e4a5ce24bba8aafc8dd9f2fb41236d016077`, SHA256 `a9d6a24e0aff13ef61088d75b5f693b599f42e8c50c2702342fbb84d3754d1a0`. No source edits, new command wrapper, mapper, numerical remap or retry.

[Original Request — resources/destination] One H800/codesign, non-preemptible, replica1, CPU16, mem_gb64,600s; exact image `hub.i.basemind.com/vllm-openai/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`, compat `/usr/local/cuda-13.0/compat`, standard personal cloud mount and local package workspace/NFS. Fresh run root `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/graph_payload_probe/1aa1e4a5_h800_20261024_01`; probe output its `/probe`, diagnostic sink its `/diagnostics`; worker tmp/cache local.

[Original Request — entry semantics] `_diagnostic_start` executes before argparse. Set `FRONTIER_GRAPH_DIAGNOSTICS_DIR` and create only that sibling diagnostic directory before CLI `run --output-dir ... --image-digest ...`. The probe creates its output directory with `exist_ok=False`. Preserve initial nvidia-smi, timeout/exit status, source/manifest hashing, ResourceConfig/spawn lifecycle and launcher_started/result receipts. No speculative binary helper; inspect existing library-hash reporting and document gaps.

## Acceptance and exclusions

1. Full local input manifest/source lock, source/template revisions/hashes/bytes, fixed config/CLI/resource binding and unchanged original package hashes.
2. Actual generated launcher help/dry-run and compile/AST checks. Safe explicit stand-ins may exercise ResourceConfig/spawn only; disclose them. CPU command exercises substitute all output/temp roots to fresh `/data/ycfeng/tmp` and use fake native commands; never write local `/mnt` or invoke actual GPU tools.
3. Explicit existing unchanged-domain77-case binding, not a fresh capture. Task records and test report include Modification History, exact commands, evidence, line budgets, failures and limitations.
4. Commit only package-preparation files; keep generated package untracked and PREPARED_NOT_SEALED. Root owns independent source review, actual-image/SDK/current-auth/quota/NFS/destination checks, seal/launch and raw-data acceptance.

No GPU/platform/auth/StepMind/network/Docker/SSH/cloud operations are authorized for preparation. Primary authority was read from `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/requirements.md:126-131`.
