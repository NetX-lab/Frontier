# WARM-ACCOUNTING completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-11-01 | Archived two bounded CPU classifications and correction proposals. |

## Overview

Task WARM-ACCOUNTING reused WARM-RESIDUAL but did not repeat decomposition or
FIFO sweeps. Confirmed NIXL structural-to-wire size mismatch (6,581,376B per
handle,0.417203ms wire-price effect) and EP-dispatch registry gap-count mismatch
(129 native kernels/43layers versus registry1;0.046836ms direct P-step effect).
Both are existing-path correction proposals, not implementations or macro TTFT
fixes. Host-zero and per-worker READ timing remain observation/approval limits.

## Deliverables and validation

All files are under
`task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/warm_accounting_20261101/`.

- `test_report_2026-11-01_warm_accounting.md`: decisive source/evidence, proposals,
  exact observation gaps and independent validation cost.
- `payload.py`, `inspect_current.py`, `segments.py`, `gap_verify.py`, `validate.py`,
  `run.sh`: bounded CPU-only inspection and reproducible commands.
- `payload_summary.json`, `wave64_handle_stats.json`, `lifecycle32_handle_stats.json`,
  `current.json`, `p2_64_segments.json`, `p2_32_segments.json`, `gap_verify.json`,
  `runner_rows.json`, `validation.json`: extracted evidence and checks.
- `requirements.md`, `plan.md`, `progress.md`, `checkpoint.md`: scoped task records.

Observed validation:7/7 PASS, Python3.10.6, requested environment and30GiB
virtual-memory limit. Four native source files checked against supplied hashes;
three others have no individual worker hash but are in the clean pinned checkout.
No GPU, full replay, modeling/table/source edits, main-record/old-report edits or
commit. Effect estimates cover wire/device pricing only; no new TTFT claim.

## Open work

Primary/YC decides whether to implement the two small fixes. Future READ/host
mechanisms require matched worker/scope timing and relevant approval. Primary
owns archive/commit and final task acceptance. The new directory is ignored by
.gitignore:171; selected artifacts need explicit force-staging for archival.
