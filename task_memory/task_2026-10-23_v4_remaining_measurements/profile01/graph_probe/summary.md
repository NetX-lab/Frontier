# Graph resource payload correction completion archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Bound verified source/test commit1aa1e4a5 and final artifact inventory; local scope complete. |
| 2026-10-24 | Archived the verified CPU/source-only correction and root runtime handoff limits. |

Task ID: `W128-GRAPH-PAYLOAD-PROBE`.
Worktree/branch: `/data/ycfeng/Frontier/.worktrees/v4-graph-payload-probe`, `v4-graph-payload-probe`.
Base: `44f2ddb9736f0abaaf6a52a5fb768519996b7da3`.
Verified source/test commit: `1aa1e4a5ce24bba8aafc8dd9f2fb41236d016077`; final archive commit returned to root separately.

## Outcome

The fresh probe decodes the official CUPTI13.0.85 ResourceData envelope before reading GraphData. Its missing-descriptor check is soft and runs before ID queries. Only two ctypes classes, additive ABI reporting and envelope decoding differ from the frozen attempt3. Callback selection/order, strict9-node/5-kernel/4-event predicates, event/DAG scopes, deliberate unscoped kernel and Kineto separation are inherited unchanged.

Probe:26/80 gross changed lines (24 added,2 deleted); frozen32887 bytes/647 lines; candidate34078 bytes/669 lines;645 identical inherited lines/32705 bytes. New unit file154/160 lines; five-line source-only compiler control gives159/160 total new CPU-control lines.

- Candidate SHA256: `a9d6a24e0aff13ef61088d75b5f693b599f42e8c50c2702342fbb84d3754d1a0`.
- New test SHA256: `b8077244a04978f5e11c76ee8dd9a8fd3da8cfd04c44b7f271054bae8cb74c5d`.
- Frozen source SHA256: `f7314cfbc70a1ba347339245754e385d24239eff9f24ceb485e5f3fba379a1e0`.

## Deliverables

All task paths below are relative to `task_memory/task_2026-10-23_v4_remaining_measurements/profile01/graph_probe/`:

- `probe_diagnostic.py`: fresh candidate; original frozen source/package unchanged.
- Repository `tests/unit/test_cupti_graph_resource_payload.py`: allocated actual callback, exact-handle/ID, malformed payload, failure-isolation and lifecycle controls.
- `source_delta.diff`, `source_verification.json`: exact inherited/changed accounting, AST invariants and query/source line mapping.
- `cpu_receipts.json`, `cpu_{red,green,inherited}.{log,xml}`: exact environment/commands, separate raw RED/GREEN and71 candidate-loading evidence.
- `abi_source_binding.json`: official-header hashes, exact compiler stdin/command and ctypes/native layout match.
- `fault_source_mapping.json`: retained actual worker3.12.3/root-image/fault bytes, static original-node query mapping and explicitly local3.12.13 bytecode evidence.
- `frozen_input_verification.json`: unchanged37 historical byte pins and referenced source/test hashes.
- `matrix_binding.json`: explicit existing77-case comparison binding, exact comparison hash and unchanged-domain diff; no new capture.
- `owner_cpu_acceptance.json`, `artifact_hashes.json`: scoped owner acceptance and exact per-file SHA256/byte/line inventory; root runtime acceptance remains separate.
- `requirements.md`, `plan.md`, `progress.md`, `checkpoint.md`, `test_report_2026-10-24_graph_resource_payload.md`, this archive: scope, route and verification records.

## Validation

New candidate controls16PASS. Compatible original suite71PASS/5deselected,71 source loads redirected to candidate through its unchanged loader. Those five old bare-GraphData/exact-ABI cases are explicitly replaced by the new controls, not changed. Safe old-source RED:2 expected assertion failures/exit1 using allocated padded envelopes and fake ID functions only. C++ syntax-only official-header ABI assertions passed; all37 historical byte pins and12 attempt3 manifest files matched. Existing77-case matrix is complete/identical with unchanged simulator domains. Source commit1aa1e4a5 passed a clean `git status --porcelain` and empty unchanged-domain diff; source/docs/JSON whitespace checks passed, while verbatim raw pytest/diff formatting was preserved.

## Root handoff and limitations

Root owns independent review, actual locked-image/SDK/package checks, sealing, at-most-one1H800/600s runtime diagnostic and actual-data acceptance. Candidate/source CPU acceptance is not proof of actual driver envelope delivery, callback clone completeness, original exit139 cause, strictU1–U4 runtime success, native serving ownership, timing/performance or consumer admission. Worker3.12.3 bytecode was not rerun by this agent; root's existing receipts plus exact source/header establish the static binding only.

No Docker/GPU/StepMind/SSH/network/cloud/package/launcher operation, native/C26/simulator edit or further delegation occurred. GLM remains paused and old3/3 history unchanged. Commit identifiers and final clean-worktree receipt are returned to root after Git verification; no further local work is selected.
