Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Hong Kong continuation: reviewed active control02; retained original failure and remediation. |

Component: group-metrics segment diagnostics. Phase: Standards review. Reviewer: Codex.
Fixed point: `947da5ee9ebb279f7978e2cf5e1883622734b654`; HEAD: `b3db1e0efd9d870dfa9e797b740907191b1b256b`.
Read `git diff 947da5ee...HEAD` within scope and complete added-file diffs/source, including both fixtures.
Standards: user-supplied AGENTS working instructions; repository `AGENTS.md:783-801`; current task requirements/plan/progress handoff; D-V4-P0-GROUP-METRICS-SEGMENTS, PRIMARY-TAKEOVER, and D-V4-P0-GROUP-SEGMENTS-TINY-CGROUP-REPAIR.

Paths beginning owner/primary are relative to `task_memory/task_2026-10-23_v4_remaining_measurements/profile01/profile04/results/p0_cpu_loop/group_metrics_tail02/`.

| Inspected path | SHA256 |
| --- | --- |
| tests/analysis/v4_trace_audit_group_metrics_timing.py | 9f456317208fa3646fd32cf24d6e92cfe7c9ca08da716b98aabf0b0aa619c426 |
| owner/tiny_control02.py | dabfdc8199ea3e6e17ea9c7ef96656801c5c8b973b29c02ca13a6096c0580650 |
| owner/tiny_inputs02.json | df1c94280684d4c604644c4171b04316228868b4f5991f0d511c09c487f9a8a2 |
| owner/tiny_control.py | e1f0450a6aebe1ba84944571a40f2a3e5a64d2b4fb92f26d0ddd9ff24031ddf5 |
| owner/tiny_inputs.json | 8a2e90c821cc3286f7533077fb6a52797e8a2ba0e36fadb103bd910c56fd4edc |
| owner/tiny_result.json | 2216990134aac3b12729fcd26a7a9cc75ebe38ac19a187bd107027a707089ac7 |
| owner/tiny_result02.json | 1b4074b85ac02cd5594ce70d2f2782c214670daf5687ce949ea8bdcd853202e7 |
| primary/source_review_inputs02.json | c51f42f22e0d2316c55bcc4990dbe0ee96585b257a02b5a4660659ef63b4e023 |
| primary/full_active_source_diff02.txt | ff134c407db68d1b7f686f2fdb99630bf2f3481bb369c570022567aa8a63caca |
| primary/primary_source_inspection.json | 86e9ca795567c55ed076b63c174868ba14ae1550824274f6f62574e003669097 |

Findings: no open hard Standards violation or actionable heuristic smell in active files. Historical documented breach, `owner/tiny_control.py:40-42`: `line.startswith('0:')` and `'memory.max'` reinvented a v2-only sensor despite the pinned existing v1 sensor. AGENTS working instructions, Engineering: "no second way to do what the codebase already does"; repository `AGENTS.md:798`: "free of hard-coded logic or duplicated local reinventions." Original FAIL/checks0 provides no AST acceptance.

Remediation: active `owner/tiny_control02.py:40-44` reuses the existing controller/mount-root sensor without a v2 fallback; line92 checks the complete six-row membership sequence. Original script/input/failure remain immutable.

Static evidence: timer20-31 calls the original sorted once with original arguments and returns the same list; 61-71 preserve computation/union/order/group fields with sparse boundaries. Diagnostic forwarding and pinned AST navigation are justified, not heuristic findings. Caps: timer112/140, control02 114/115.

Verification: inspected primary's retained Python3.10.6 result02:32/32 consistent PASS checks, actual32212254720bytes,0.132296s; stripped AST, exact fixtures, forwarding and immutable pins pass. Original tiny1/1 spent; repair1/1 spent; diagnostic0/1. Reviewer performed no runtime/tests, raw-trace/index reads, platform/network actions, Git mutation, source edits or child delegation. Full-input performance, R5 completion and native admission remain unverified.
