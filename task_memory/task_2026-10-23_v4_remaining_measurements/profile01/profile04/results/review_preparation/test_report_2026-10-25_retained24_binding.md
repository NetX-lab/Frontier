# Retained 24-rank receipt review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Preserved the completed read-only binding and checked its serialized acceptance limits. |

The retained `retained24_binding.json` records a primary read-only hash binding at 2026-10-24T20:55:12.597459Z, 2026-10-25 Hong Kong. Its 0.9599414258264005-second operation hashed closed output/index/command/status/log artifacts and read their existing final JSON; it did not parse raw traces, decompress indices or invoke an auditor/control. The previous in-session command text was not retained, so this report does not invent a reproducible original command.

Current command: `/usr/bin/python3 -` from `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`; standard-library inline receipt read with `json.loads(Path(.../retained24_binding.json).read_text())`. It checks sorted `(role,dp,rank)` keys, sums of kernel/annotation counts, actual 32,212,254,720-byte limits, at least three useful steps per rank, role-specific counts and all zero-invocation/false-admission fields. Python 3.10.6, `/usr/bin/python3`, no conda. Exact expected/actual values and input hash are retained in `retained24_serialized_review.json`.

Expected and observed: 24 unique completed original keys, 7,758,672 kernels, 102,248 exact annotations, P1/D0/D1 useful counts 4/61/61, all actual 30GiB, zero new trace/control invocations, zero new complete P0 outputs, no full R6/R7, reference or cost admission. All eight serialized checks PASS. No raw, kernel-index or prior audit driver was re-read or executed during this review.

This accepts only the retained receipt consistency and preserves its earlier hash observation. R5 remains 24/32; full R6/R7 stays BLOCKED, and ownership, dependency, whole-DP, transport and reference/cost limits remain unchanged. SHA256 equality is not a new numeric measurement.
