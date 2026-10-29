# P0 metrics correction Spec review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 UTC | Completed independent source and AST Spec review. |

Phase: D-V4-P0-AUDIT-METRICS-CORRECTION, before primary complete controls and P0 execution. Reviewer: independent Codex Spec reviewer.
Fixed baseline: `0f2fb6cdffbbe86f2fdc06e5141e3e80c227c2b9`. Both additions are absent at this commit and remain untracked; the two requested no-index diffs returned the expected exit 1.

| Inspected source | Verified SHA256 |
| --- | --- |
| `tests/analysis/v4_trace_audit_metrics.py` | `6dff04f36dbcd8a66a298e4f6ae88d382bb29813c8a98f8fea91372b271b788a` |
| `tests/unit/test_v4_trace_audit_metrics.py` | `76734ee80dadfcc49111c01218db23e665ee1f085de892e1fdd832c30538ede9` |
| `/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/audit.py` | `7ed0ea45a61d5150e2f45bbc1cdf245f2331e40a351c235228b209b064ede0c8` |
| `/data/ycfeng/tmp/glm53_dsv4/p0_audit_cpu_20261024/stage_audit.py` | `f7f1253ce9ebae5dcb3b60e2aeb97d7f17f58f34a7d68af56611918f809b9fe4` |
| `/data/ycfeng/tmp/glm53_dsv4/p0_gc_observation_20261024/load_without_gc.py` | `6b91e25fc592d4f1340ef22e6d6e73c4dd8ed6e7ce6239f7a113d946a74edd02` |
| `/data/ycfeng/tmp/glm53_dsv4/p0_gc_observation_20261024/observe_gc.py` | `3ae64dce5e1498418c3489d48bb74393f08c4dbbcb95231bab4e11a414a5a83c` |
| `tests/comparison/calibration/forward_device_timeline.py` | `fe4c3f82500df79ce097ce762808bcaf2b78040e41cbab942ccc746a723b3a52` |

Inspected AGENTS.md; task requirements.md, plan.md and progress.md; all sources above in full; and owner/{owner_delivery.json,verification_receipt.json,input_source_bindings.json,source_diff.patch,test_report_2026-10-25_metrics_correction.md}. Also inspected owner control bindings, unit receipt, negative control and progress records. The complete 307-line patch matches both current additions exactly. Budgets: source 120/200, tests 175/180, owner maintained Markdown 96/100.

Findings: **No actionable findings within source/AST coverage.** The five metrics expressions cover per-stream, total, status, hierarchy and leaf groups. Source lines 38–55 retain original group count, trace-order builtin sum, duplicate memberships, one stable `(ts, ts+dur)` sort and the unchanged union recurrence. Attribution, overlap/reversal, hierarchy/leaf/per-stream semantics and loader/GC/output policy remain unchanged.

Independent verification: 31 source/hash/AST assertions PASS using `timeout 60s /usr/bin/python3 -B -` (inline check in the review transcript), Python 3.10.6. Removing exactly five substitutions, the batch/function insertion and pipeline composition restores the complete staged/original auditor and loading-GC pipeline ASTs. All five frozen source pins match before/after. Owner's eight unit controls cover the required empty, overlap, tie, float-order and repeated-membership cases; their PASS is source-only evidence.

Validation limits: decodeDP1/rank0 and prefillDP1/rank0 exact complete controls, the later P0 pilot and remaining ranks are deliberately PENDING_PRIMARY, not owner PASS or review findings. Full-output equality, runtime, peak memory and performance improvement remain unverified. No full-input invocation, GPU/platform operation, source edit, delegation or Git write occurred; only this review is written.
