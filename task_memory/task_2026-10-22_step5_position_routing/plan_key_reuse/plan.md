# D29 derived plan-key reuse plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Plan the bounded two-path implementation and exact-output verification. |

## Sequence

Accepted snapshot/budget -> interface RED -> two-path implementation -> {focused tests, actual-wave parity and timing} -> {Standards, Spec reviews} -> parent verification -> integration/final114-path accounting -> frozen regression and full-workflow gates.

Use an isolated worktree based on the allocation commit. Preserve the completed snapshot worktree as a baseline. Extend the existing identity/isolation regression rather than create a new suite. Keep production `cache_key` calls live while counting actual encodings separately. The accepted wave driver must select its cold plan for output checks; do not substitute a reporting sample or add a diagnostic byte adapter to candidate production.

Evidence to reuse: `../immutable_plan_preparation/test_report_2026-10-24_parent.md`, `../emission_immutable_work/{report.md,parent_receipt.md}`, `../run_input_snapshot/wave/`. The parent domain probe additionally proves unchanged repeated bytes/bytearray TypeErrors and detached public dictionaries. The config control remains UNQUALIFIED, the fields conversion is rejected, and private trace encoding is unallocated.

CPU commands use Python3.10.6, `-B`, `runuser brainpp`, single BLAS threads and bounded memory/time. Retain original fixtures/caches, source pins and full output; only scratch artifacts may be created. Record exact commands, Git-derived gross/net/path counts, hashes, observed results and limits. No new full CLI until the combined residual work has been evaluated.
