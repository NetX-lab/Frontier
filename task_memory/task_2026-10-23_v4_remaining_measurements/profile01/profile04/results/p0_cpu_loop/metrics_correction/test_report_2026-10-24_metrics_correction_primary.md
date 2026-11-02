# Primary metrics correction source and complete controls

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 Hong Kong / 2026-10-24 UTC | Rechecked28retained source/control bindings before local commit without repeating any complete control. |
| 2026-10-24 UTC | Accepted source/AST and both separately allocated complete controls before P0 validation. |

Decision: D-V4-P0-AUDIT-METRICS-CORRECTION. Source120/200 and tests175/180 gross lines; no compression or frozen source edits. Both independent source reviews report No findings within their declared coverage.

Environment: local /usr/bin/python3, Python3.10.6, standard library, no conda/dependency changes. The exact commands, UTC times, full source/input/output hashes and inherited cgroup fields are in primary/{unit_receipt.json,complete_controls.json,control_allocation.json} and each control's receipt.

```bash
timeout 60s /usr/bin/python3 -B tests/unit/test_v4_trace_audit_metrics.py -v
/usr/bin/python3 -u -B /data/ycfeng/tmp/glm53_dsv4/p0_metrics_correction_20261025_01/run_controls.py
```

The67-line driver starts each of the two controls once, at most2 concurrently. Each actual audit uses sudo -n systemd-run --scope -p MemoryMax=30G timeout60s and the new pinned adapter; inherited loading-GC, observations and attribution remain unchanged. The recorded cgroup limit is32212254720bytes for both. Initial usage is not peak; maxrss_kib is the auditor's observed process RSS maximum.

Expected: all8unit/AST controls pass; all semantic JSON fields/count/sum/union/hierarchy/leaf/per-stream and decompressed kernel-index rows match retained completed outputs exactly. Only elapsed_seconds/maxrss_kib and fresh output index path/compressed hash are removed for comparison. New index paths and actual compressed hashes are separately verified, with no numeric tolerance or predicate change.

| Check | Actual | Result |
| --- | --- | --- |
| Primary unit/AST | 8PASS/0failures/errors/skips, exit0,1.869s | PASS |
| decodeDP1/rank0 | 142567kernel rows/61useful steps exact; enclosing6.014371s | PASS |
| prefillDP1/rank0 | 570964kernel rows/4useful steps exact; enclosing30.108530s | PASS |
| Full source/AST and frozen pins | Five callsites only; count/sum retain original order; duplicate hierarchy membership and original union recurrence preserved | PASS |
| Independent Standards/Spec | No findings within source/CPU/AST coverage | PASS |
| Unchanged-domain matrix | Existing77identical cases explicitly bound and comparison rehashed; no fresh simulator capture | PASS |

Each exact comparison reuses the frozen stage_audit.py compare command inside its own60s/30GiB scope. Exact commands/results and complete audit/index paths remain in primary/complete_controls.json; selected closed logs/cgroup/observation records are retained beside it. Full raw traces/index bytes remain in their original/fresh scratch paths.

Acceptance: ACCEPTED_CPU_METRICS_CORRECTION_SOURCE, in primary/source_acceptance.json. This settles source/equality prerequisites; performance savings and P0 output are unmeasured. P0rank0 pilot and remaining7conditional once-only ranks are NOT_STARTED. R5 remains24/32 with all8original timeouts preserved. No GPU/native ownership/dependency/cost/reference admission.

Continuation byte verification: primary/continuation_binding.json records28/28PASS in0.031266s, Python3.10.6, sudo systemd30G/timeout60s. It rehashes both delivered sources, five frozen sources, reviewed receipts and both complete JSON/index bindings; new control and affected-rank invocations remain0. Exact inline command is retained in the current session transcript.
