# W128-GRAPH-PAYLOAD-PROBE requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded the delegated, approved callback correction and CPU-only ownership. |

## Authority and original request

[Original Request] Task ID `W128-GRAPH-PAYLOAD-PROBE`: implement the minimal fresh callback payload correction and CPU checks in `/data/ycfeng/Frontier/.worktrees/v4-graph-payload-probe`, based on `44f2ddb9736f0abaaf6a52a5fb768519996b7da3`. Own only this `profile01/graph_probe/` directory and one new `tests/unit/` file of at most160 lines. No further delegation, Docker, GPU, platform, SSH, network, cloud, package or launcher operations.

[Original Request — authoritative primary decision D-V4-BOUNDED-DIAGNOSTIC-AUTHORITY] YC said “批准，类似请求无需申请权限，皆授权”. Primary `task_memory/task_2026-10-23_v4_remaining_measurements/requirements.md:116-124` approves a fresh probe correction of at most80 gross changed lines against frozen attempt3, followed by CPU verification and at most one1H800/600s diagnostic run. Root owns that run and all package/image/SDK/seal/actual-data acceptance. The old3/3 attempts remain exhausted; paused GLM work stays paused.

## Acceptance

1. Copy the exact frozen `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h200_graph_probe_attempt3_20261021/probe_diagnostic.py` from PRIMARY; retain its source hash, full byte/line inventory and inherited-versus-changed accounting. Preserve original sources, tests and packages.
2. Follow the publisher-hash-verified CUPTI13.0.85 sample/header retained in PRIMARY `profile01/graph_source_evidence/`: model `CUpti_ResourceData` with its `context`, `resourceHandle` union and `resourceDescriptor`; dereference the descriptor as `CuptiGraphData` only inside the callback. Softly reject a missing descriptor before any ID query.
3. Preserve NODE_CLONED/EXEC_CREATED, native query order, event-handle/DAG scope labels and all strict9-node/5-kernel/4-event checks, including the deliberate unscoped kernel. No new callbacks, mapper, numerical remapping, heuristic association, after-callback querying of descriptor handles or Kineto overlap.
4. CPU controls invoke the actual candidate CFUNCTYPE callback using allocated GraphData and ResourceData, assert exact handles/full IDs and in-callback order, malformed descriptor no-query behavior, exception isolation and subscription lifetime. Produce a safe old-source RED using a padded allocated envelope and fake ID functions only.
5. Reuse compatible existing CPU suites via their candidate loader without changing them. Bind the unchanged-domain77-case matrix rather than creating a capture. Pin worker Python3.12/source fault mapping and ABI against inspected source/header/root receipts; do not claim native payload decoding is proven.
6. Commit verified candidate, new test and task records. Return exact artifacts, hashes/budgets, RED/GREEN counts, commits and remaining runtime limits; leave the worktree clean and stop.

## Authoritative references

PRIMARY: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.
Read PRIMARY task `requirements.md` and `plan.md` before implementation. Their older callback-exclusion/proposal wording is superseded only by the final decision above. This delegated slice grants no native/C26/calibration or simulator changes.
