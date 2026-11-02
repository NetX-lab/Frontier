# W128-PROFILE03-COLLECTION CPU delivery

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Archived the verified two-file source change and its bounded CPU evidence. |

## Outcome

Source commit `20660f4c0776ddd75dc211f1f0819266b48ba23c` on `v4-profile03-collection`, base `44f2ddb9736f0abaaf6a52a5fb768519996b7da3`, is CPU-verified and pending root review. Production30/80 gross changed lines, tests129/180, total159/260. No other source, native, C26, checker, package or DP-owner files changed.

The worker selects task-local role profiler iteration limits through its existing local Torch node plan, copying the profiler configuration and preserving role/lane paths. Drain quorum and retained membership use the same known-peer direct paths. Headless post-stop evidence waits use the existing peer budget. Separate local monotonic diagnostics retain wait/stop boundaries without changing shutdown-order meanings, native signal escalation, native outcomes or soft missing-marker behavior.

## Deliverables

- `/data/ycfeng/Frontier/.worktrees/v4-profile03-collection/tests/comparison/calibration/native_pd_worker.py` — source SHA256 `59cdd58d9fce0d3aa432caeace4411d8fef074f59857260e81c75d3441bcfa1b`.
- `/data/ycfeng/Frontier/.worktrees/v4-profile03-collection/tests/unit/test_native_pd_worker.py` — tests SHA256 `84fa7a4be7a8a957eb62671e652d6e00ff48d153c92671bea7a53fe6528cea4c`.
- Records root: `/data/ycfeng/Frontier/.worktrees/v4-profile03-collection/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/collection_repair/`.
- Exact diff and source budget/commit: `source_changes.diff`, `source_manifest.json`.
- Commands/results and original failures: `test_report_2026-10-24_collection_repair.md`, `final_checks.json`, `baseline_commands.json`, `red_command.json`, `red_worker.log`, `issues.md`, `verification_interruption.json`.
- Existing matrix and primary-evidence hashes: `matrix_binding.json`.
- Requirement/progress/self-check and archive integrity: `requirements.md`, `plan.md`, `progress.md`, `checkpoint.md`, `artifact_checks.json`, `record_hashes.json`.

All record filenames above are relative to the exact records root. `record_hashes.json` enumerates SHA256s for every delivered sibling artifact except itself. Its own hash and the records archive commit are reported separately in the owner response to avoid self-reference.

## Observed validation

- Unchanged baseline worker19PASS and manifest14PASS. RED on the original unchanged worker SHA256 `9ec701ec12f7a5da0f442f994a353f98c5ae1d2880b9feca38106ea536fd5cab`:19PASS/16FAIL, including exact role-selection, delayed-marker and stale-enumeration signals.
- Final frozen suites, separately: worker36, manifest14, collector33, streaming39, cold-start18, canonical HTTP1, calibration P/D68, DP placement75. Every final process exit0, no failures/errors/skips, immutable before/after source hashes.
- Active and naturally completed headless marker20 controls use peer60 despite stop10; late/missing markers remain soft. Delayed/missing known peers still wait or fail at the existing finite deadline. Native error23 and thrown stop errors retain their existing outcomes.
- Existing77-case matrix is explicitly bound to final source and unchanged simulator/input domain trees; comparison SHA256 `7a8728ecc0851373827d9cdc7dfb68c552299e6c54b5fae75bd8cc5429d36ad6`. No new capture.
- One auxiliary verification snapshot assertion failed after only test assertions changed while the initial batch ran. Original error and unretained exit-code limits were retained; frozen reruns recorded all actual final exits. No business behavior change was made for that issue.

## Open work and limits

Root owns independent review, integration, later profile03 package values, actual image checks and GPU/runtime validation. No GPU, external-network, Docker, StepMind, authentication, SSH or cloud operation was performed in this slice. The existing loopback-only CPU HTTP controls were executed.

Finite peer waiting is not a completion guarantee. Native stop remains three timed attempts followed by an unbounded final wait. Local clocks do not establish cross-host order; CPU stand-ins do not establish GPU shutdown, cloud-delay causality, profiler coverage or CPU overhead. Later package512/64 and records600000 remain unapplied. After the records-only archive commit and clean-tree confirmation, this owner stops.
