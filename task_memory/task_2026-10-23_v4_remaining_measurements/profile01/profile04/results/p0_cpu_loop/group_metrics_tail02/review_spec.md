# Spec review: group metrics segments

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Hong Kong continuation: reviewed timer, repaired tiny02 and serialized evidence; preserved the original failure. |

Component/phase: group metrics segment timer and tiny02, before the full diagnostic. Reviewer: Codex, independent Spec axis, fresh context.

Task: `task_memory/task_2026-10-23_v4_remaining_measurements`. Fixed point: `947da5ee9ebb279f7978e2cf5e1883622734b654`; inspected HEAD: `b3db1e0efd9d870dfa9e797b740907191b1b256b`.

Read AGENTS.md, task requirements/plan/current progress handoff, accepted group source4dec265e/adapter6dff04f3, baseline ast_closure.json, original/takeover/repair/execution allocations, both captured full-file diffs and manifests, and primary_source_inspection.json. Both diffs match their bound source bytes.

Verified source SHA256; owner paths are relative to this report:

| File | SHA256 |
| --- | --- |
| tests/analysis/v4_trace_audit_group_metrics_timing.py | 9f456317208fa3646fd32cf24d6e92cfe7c9ca08da716b98aabf0b0aa619c426 |
| owner/tiny_control.py | e1f0450a6aebe1ba84944571a40f2a3e5a64d2b4fb92f26d0ddd9ff24031ddf5 |
| owner/tiny_inputs.json | 8a2e90c821cc3286f7533077fb6a52797e8a2ba0e36fadb103bd910c56fd4edc |
| owner/tiny_control02.py | dabfdc8199ea3e6e17ea9c7ef96656801c5c8b973b29c02ca13a6096c0580650 |
| owner/tiny_inputs02.json | df1c94280684d4c604644c4171b04316228868b4f5991f0d511c09c487f9a8a2 |

(a) Missing/partial requirements: no active-source omission found. Full diagnostic measurements remain pending primary validation, rather than a source finding.

(b) Unrequested scope: none found. Timer112/140 and tiny02 114/115 lines comply. Four global and six per-group boundaries preserve the original sorted arguments/returned list, expressions, evaluation order, count/sum/union, groups and raw IDs. No per-event logging, extra sort, GC or loader change.

(c) Implemented-but-wrong behavior, CLOSED: original tiny_control.py:40–42 assumed memory.max; tiny_result.json remains FAIL with checks=[] and no AST evidence. Task requirements.md:300 requires “exact command/cgroup/source bindings.” The separately allocated control02:40–44 follows requirements.md:306, “reuse the four existing v1 sensor expressions”; line92 compares the complete six-row membership sequence.

Verification: inspected owner/tiny_result02.json (SHA256 1b4074b85ac02cd5594ce70d2f2782c214670daf5687ce949ea8bdcd853202e7):32/32 serialized expected==actual/statusPASS, actual32212254720 bytes, elapsed0.132296s. Complete/source/union/pipeline AST values match pinned references; floathex, stable ties/duplicates, forwarding/list identity and copied evidence agree. Execution hashes match active bytes.

Limits: static source and serialized primary evidence only; no reviewer runtime, tests, raw traces or child agents. No remaining active Spec finding. Full-input performance, completed/censored diagnostic timings and R5/native/cost admission are unverified here. Check-progress self-check kept the review on its assigned route.
