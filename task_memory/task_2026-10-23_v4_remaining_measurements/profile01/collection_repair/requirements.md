# W128-PROFILE03-COLLECTION requirements

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded the root allocation and acceptance requirements before source edits. |

## Original request and authority

[Original Request] Implement and CPU-verify the approved source-backed worker changes in `/data/ycfeng/Frontier/.worktrees/v4-profile03-collection`, branch `v4-profile03-collection`, base `44f2ddb9736f0abaaf6a52a5fb768519996b7da3`. Keep edits and commits there. Own exactly `tests/comparison/calibration/native_pd_worker.py` (80 gross changed lines) and `tests/unit/test_native_pd_worker.py` (180), total 260; records only under this `collection_repair/` directory.

[Original Request] Add per-role max_iterations through the existing node_plan profiler path using task configuration `role_profiler_max_iterations`, preserving inputs and per-role/lane paths. Use known-peer direct drain paths for both quorum and retained membership. Use `peer_wait_seconds` for the headless post-stop-marker wait. Add separate diagnostic monotonic wait/stop timings without changing `shutdown_order` meanings, native escalation/returns/errors, or soft missing-marker behavior. A finite wait is not a guarantee.

[Original Request] Build meaningful RED/GREEN through the existing test seam: marker20/stop10/peer60, active headless when possible, soft timeout, delayed drain enumeration20 with all known paths present, partial known peers, normal/error native returns and role configuration copying. Package512/64 and records600000 are later package values, not production literals or sealed-input edits.

[Original Request] Use CPU environment `asyncopd-py312`, `PYTHONPATH=.:/data/ycfeng/tmp/d29_py312_cpu_extras01/site`, and `FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69` where needed. Run affected worker/manifest/profile-driver suites separately when fake modules contaminate a shared process. Inspect actual failures first. Bind the existing unchanged-domain77-case matrix by hashes/domain identity, not a new capture.

[Original Request] No other source/native/C26/checker files; no GPU, Docker, StepMind, authentication, network, SSH, cloud, new package or further delegation. Do not touch or review the DP owner's four files. Root owns independent review, integration, actual image/package and GPU run. Commit verified code and records, leave a clean worktree, then stop. Return evidence/proposal before expanding if the design or budget blocks.

## Authoritative inputs

- Primary `requirements.md` lines107–113 and `plan.md` lines41–43 under the parent task in `glm53-dsv4-vllm028`.
- Primary `profile01/profile03_design.md` and `test_report_2026-10-24_lifecycle_cpu.md`.
- Primary `profile01/lifecycle_cpu/{command.json,control.py,receipt.json,profile_capacity.json}`; unchanged worker SHA256 `9ec701ec12f7a5da0f442f994a353f98c5ae1d2880b9feca38106ea536fd5cab`.
- Primary `profile01/stream_probe/matrix_binding.json` for inherited matrix evidence.

## Acceptance and limits

Verify the real worker control flow using CPU stand-ins, input immutability, all four role/lane paths, valid/invalid role values, preserved native outcomes and separate local-clock diagnostic timings. Reuse affected existing suites. Production/test gross diff budgets and final source/record hashes must be explicit. CPU controls do not establish GPU shutdown, cloud filesystem causality, cross-host order, native coverage guarantees, or CPU overhead.
