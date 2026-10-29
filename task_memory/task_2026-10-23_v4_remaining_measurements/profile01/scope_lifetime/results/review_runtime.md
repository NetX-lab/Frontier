## Modification History

| Date (UTC) | Change |
| --- | --- |
| 2026-10-24 | Reviewed frozen runtime evidence and completed primary audit. |
| 2026-10-24 | R2: independently closed three R1 audit omissions; retained exact hashes and scalar verification. |

Reviewer: Codex, independent runtime-evidence reviewer. Phase: S5 R2, `exp-1025-021228-212042`; primary owns GPU/platform/final acceptance.

Finding verdict: all three R1 omissions CLOSED; no new material finding within inspected coverage. Preserve [R1](review_runtime_r1.md).

Inspected: current main/scope requirements, plans, progress and summaries; whole final 152/190-line audit and frozen 252-line probe; source binding/CPU acceptance; frozen callback ABI/IDs/subscriber/target extraction and real ServingProfile.range; original package seal/inventory; all five local raw files; R2 allocation/result/receipt/logs; execution binding, platform_02, its 32-line log and original launcher records. All 55 inspected artifact hashes and scalar evidence are in [review_runtime_r2.json](review_runtime_r2.json).

Key SHA256 identities; raw paths are relative to `results/raw`, package means the original executable worktree:

| Artifact | SHA256 |
| --- | --- |
| Final audit | `539da5af58b362d0ecdea09af8e9c6c3157f21af7f530fb12c3629ffce658f93` |
| Frozen probe | `bc638fbdf0111a8ce78383d182d41e19a5799bacc6cc5a657aa79b1a34672685` |
| Package manifest | `9ff1abec5fa2451baa2a285bdf081011eba9a2741448ad39a64b9a27aae4198e` |
| `primary_raw_audit_r2.json` | `a7dcdcbed3ac1fa50fe8fcd97a1142184df117faf3fa34bff8b90cefbbdf3584` |
| `raw_audit_receipt_r2.json` | `a5b768ba05179d5a0eeceb3159390ef7bd3c397823eb380bf7ea9fd8fe39851c` |
| `diagnostics/stages.jsonl` | `ee426997af0349e1510d3113525f1e6700781da4c536385e64744136aca6818e` |
| `diagnostics/fault.log` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `probe/probe_report.json` | `e988ee758ae7e08a52ecd996277b8b574f7113511f58d3ef9d4ddad77812e586` |
| `probe/observe_trace.json` | `fe8c03c857ec412fbb5dd02335b2454e4ac3b25d3fa88b72d504c0786a6fbee8` |
| `probe/control_trace.json` | `a3815882be30a2ac83b50577b508e74cb5211cc06e66fb81327cf60d5c8f32bc` |

- R1 errors CLOSED: audit line47 explicitly rejects observe.errors; raw errors/fault are empty.
- R1 pairing CLOSED: lines66–72 require five starts and exact payload/sequence pairing. Raw observe pairs:9/11,12/14,15/17,18/20,21/23.
- R1 close CLOSED: lines124–127 require unique successful close before unique Kineto start; raw observe37<38 and control51<52, both true.

Verification: Python3.10.6, `timeout 60s python3 -B`, 73/73 scalar checks and 17/17 rejection checks using the three actual repaired AST expressions. No auditor/probe import or rerun. Primary R2 receipt records60/60 PASS,5direct joins,6unmatched,0.122845s; execution binding20/20 PASS. Five downloads and18frozen inputs match hashes.

P1: payload copies and ID/type queries remain inside callbacks13/20/18. P2: thread equalities are observations. P3: each selected source occupies exactly one monotonic window with exact scope equality. P4: default keep_graph=False; one discarded raw_cuda_graph call per graph records the expected RuntimeError boolean after capture_end; no source-handle query/dereference; raw_exec matches copied callback identity. Full frozen-source inspection supplements the audit's limited one-call AST check. P5/P6: five complete created(node,fullID)→clone(originalNode,originalNodeID)→replay fullID joins under one target launch.

Control.verdict remains UNVERIFIED; control supports five-kernel topology/exact output identity only. Six unmatched-created causes are unknown. Host driver release is UNRECORDED; API13000/CUPTI130001 are telemetry.

Permitted conclusion: frozen P1–P6 are supported for this sealed image/H800/1024-element probe, allocation spent1/1, Succeeded/exit0/restart0. Primary should cite combined source/raw review when deciding acceptance. Native serving, general ownership strategies, timing/non-perturbation, dependencies, costs and reference admission remain excluded.
