# D29 trace serialization repair requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record the parent assignment and single-path acceptance boundary. |

## [Original Request]

TaskID: `D29-TRACE-SERIALIZATION-REPAIR`.
Parent assigns implementation from `767da5437609301a6183dbfdcfd87b2402a32007`
in `/data/ycfeng/Frontier/.worktrees/step5-d29-trace-serialization`, branch
`feat/step5-d29-trace-serialization`.

> Source ownership ONLY frontier/metrics/trace_store.py, <=20gross, no newtests or runtimeflags/schema/cache.
> Decision: remove the internal one-caller, unexported undocumented TraceEvent.to_dict entirely; remove asdict import; at existing flush write site serialize declared dataclass fields filtering top-levelNone exactly as before. Prefer fields() to retain declared key set/order.
> Do not retain another serializer/helper or change BaseEvent.to_dict.
> All existing producers already normalize metadata toJSONprimitives; no supported nested-dataclass payload. Flusherror bufferrestore/exceptionbehavior unchanged. Do not expose aliased-return method.
> Verify byte-identical598-event trace replay (8,353,308B sha3ea6239cea9295e5d532722200c78238c9e01ecb58b32383fe8c8aaa20b7e004); realTraceStore existing test_measured_attention_root_metrics.py and relevant measured ledger tests have1knownbaseline mapper allow_structural failure, retain baseline comparison.
> Demonstrate source-based improvement with no runtime diagnostic flag and preserve fullproofs. Adapt probe source binding explicitly to your worktree/current source with recorded hashes; never silently run frozen79 and callitnewsource. Script rewrites only new scratch; source/test fixtures/inputs read-only.
> Do focusedverification+byte differential+exactAST/numstat/whitespace, commit scoped code, and records under trace_serialization_repair/.
> Primary owns independent review, strict77/full-unit on finalfrozen union, CLI andacceptance. Do not runfullCLI/GPU/platform/network, nopeercherry-picks/unrelatededits/newagents.

## Authority and acceptance

Read root `requirements.md`, `plan.md`, and `AGENTS.md`. The authoritative
allocation is `../repairs/trace_serialization_scope_amendment.json`: one new
CORE path, at most 20 gross; fixed-probe donor ceiling 65 -> 45 (41 measured on
frozen79); CORE4404/source9942/global10000 unchanged. Scope arithmetic must be
rederived from Git, not copied from the seam proposal.

Preserve declared fields and their order, omission of top-level `None` only,
complete metadata and exact JSONL bytes, error propagation, and restoration of
the pending buffer before concurrent additions. Verify the complete actual
layer60 wave: 208 distinct `(ep_id, name)` pairs, all durations and physical
proofs unchanged. Do not substitute layer61 proof objects for layer60 objects.

Run memory-heavy checks under `timeout 60s` and a 30 GiB memory scope. Limit
captured trace reads to the retained 8 MiB tail; do not scan the full trace or
CSV inputs. Existing retained fixtures are read-only; new scratch scripts and
raw outputs live under `/data/ycfeng/tmp/d29_trace_serialization_repair/`.

This deliverable is focused source verification, not whole-workflow acceptance.
