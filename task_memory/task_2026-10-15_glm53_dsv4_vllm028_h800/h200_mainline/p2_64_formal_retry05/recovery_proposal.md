# Fifth P2-64 submission proposal

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-04 | Prepared and independently checked generation05 after the external generation04 stop; GPU submission remains unapproved. |

## Decision pending

Request exactly one fifth submission of `package_01`, manifest SHA256
`23e26da6c02cc42e54134c12abfdfea298f64191e0f985a860505e374278227c`:
64 H200 (8 replicas x 8 GPUs), all preemptible, `step_main`, timeout 3600 seconds,
ceiling 64 container GPU-hours. No automatic sixth submission.
D-P2-64-FOURTH authorized only the fourth allocation. Ordinary debug approval
continues to cover local fixes and verification; it does not increase this
explicit allocation count. No approval receipt or submission exists for this candidate.

## Evidence and choices

Generation04 `exp-1103-131952-259161` was stopped at 2026-11-03T10:00:43Z
(18:00:43 HKT) by `alimasiyilixiati`, with annotation `release_quota` and a
matching successful stop audit. All 178 available audit rows and 65 historical
queue events are retained. Null startTime, zero retained replicas and the absent
cloud output support never-scheduled status. The launcher observed the stop and
exited afterwards. No complete reference exists. The generic exact-clock receipt
remains null; queue wall time is excluded from GPU-hours.

| Option | Cost and consequence |
| --- | --- |
| Recommended: one same-configuration fifth submission | Up to 64 container GPU-hours; obtains the required new promotion-bound, pair-warmed reference if it completes. Queue delay or another external stop remains possible. |
| Defer the allocation | No additional GPU usage; the candidate and all evidence remain retained, and formal P2-64 acceptance stays open. |

The earlier 24.3 GPU-hour figure is a historical estimate, not a validated
prediction for this pair-warmed candidate. No current quota check or reservation
is claimed by offline preparation. There is no evidence that changing native
code or the workload would prevent a queue-time release_quota stop.

## Reviewable candidate

The builder adds one entry to the unchanged CASES registry. Independent audit
verifies all 47 sealed files; 41 equal generation04 byte for byte. The other six
match exact replacements of package/run/cloud identity, purpose, build time and
the derived overlay digest. Independent `launch.py --check-only` prints
`INPUT_HASHES_PASS_NO_SUBMISSION`. Native `a06a72a4326cba02234df6cfbdc21b78d3a15d26`
and harness `f33306d40443cfc0c44842d6ee044f6703c736e7` remain clean. The promotion
receipt SHA256 remains `d505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596`.

Workload: three repetitions, each with 128 formal requests at 1 QPS,
6144 prompt / 1024 output tokens, and 16 pair-warmup requests before measurement.
Topology: P32+D32, TP8/DP4/EP32/PP1 per role. No simulator or native correction is
included. Evidence: `lead_package_acceptance.json`, `lead_package_audit.py`,
`package_01_build.json`, and the task-level November 4 test report.

## Execution after approval

1. Record the explicit one-submission decision and a package-external approval
   receipt. Check quota through Pi and submit this exact package once using the
   personal local StepMind Python RJobBackend. Preserve required local NFS and
   cloud mounts. End the finite submitting Pi turn normally.
2. Use a distinct Pi label for finite monitoring. Keep the launcher alive while
   queued/running. Never use ds4 --stop on a label that owns a live launcher;
   auxiliary validation errors remain non-blocking soft gates.
3. Verify creator, eight nodes/64 H200 identities, source hashes, API readiness,
   formal request completion and terminal status. Capture container clocks as
   they become available. No replacement run is automatically authorized.
4. On terminal, retain receipt/events/clocks, copy and independently rehash
   available outputs, then use `post_run_command.json`: pair coverage, R5,
   all-request and warm metrics against the already completed Frontier replay.
   Report all three repetitions and four metrics, including failures.
