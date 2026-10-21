# D5 T1 PD client TTFT

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-20 | Completed and committed the scoped endpoint, fixture correction and CPU verification; recorded remaining native/T2 limits. |

## Overview

Implemented the PD first-output endpoint through the existing Request and metric
distribution owners, without execution/timing changes. The public historical
prefill-completion `ttft` contract remains intact; `client_ttft` is exported only
for `pd-disaggregation`. The inspected first-output timestamp is
`first_decode_token_completed_at`; the similarly named
`decode_first_token_completed_at` is the second output and is preserved.

## Deliverables

Code commit: **6db2cd9**, `fix: expose PD client TTFT at the first D output`,
on `feat/step5-pd-structural`, after ticket 20's `d65a4b9`.

- `frontier/entities/request.py`: `Request.client_ttft` and an explicit historical `ttft` docstring.
- `frontier/metrics/constants.py`: `RequestMetricsTimeDistributions.CLIENT_TTFT`.
- `frontier/metrics/metrics_store.py`: PD-only distribution initialization and completion export, including seconds/milliseconds ground-truth fields.
- `tests/unit/test_pd_client_ttft.py`: endpoint, historical metric, export and actual NIXL admission-order regressions.
- `tests/unit/test_mla_core_native_op_tracing.py`: supply required `sys_arch="co-location"` in the existing metric-store fixture.
- `docs/general/module-size-exceptions.md`: metric semantics, scope limits and retained module boundary.
- `task_memory/task_2026-10-20_pd_client_ttft/test_report_2026-10-20_pd_client_ttft.md`: commands, observations and verification limits.

## Observed verification

- New regressions: 9 passed.
- Corrected affected tests: **764 passed, 8 skipped** (Torch absent).
- Full unit suite: **5,581 passed, 70 skipped, 103 failed, 29 collection errors**;
  all 132 failing IDs exactly match the known stream-consolidation reference,
  with no new or missing IDs.
- Step structural: **4 cases / 26 requests**; GLM structural: **2 cases / 10
  requests**; legacy DeepSeek-v3 PD: **offline/online / 2 requests**. All old
  metric cells and simulated timing artifacts are identical; the new endpoint
  equals the first D stage completion minus arrival. GLM op-trace bodies are
  byte-identical; only their pre-existing header wall timestamps differ.
- Legacy matrix versus `8585748`: **77/77 original outputs identical**, comprising
  60 unchanged cases and 17 PD cases with only the approved extra CSV column;
  902 artifacts total, 519/519 cache names equal. Independent client interval
  reconstruction passes all 179 legacy PD requests. The unchanged raw
  comparator reports 60 identical / 17 expected schema differences.
- Source AST/whitespace, protected-path, 1,996-line Request and tested-file
  SHA-256 checks pass. Selected evidence is retained in this task's `evidence/`.

## Open work and limits

The scoped simulator T1 endpoint code/test line is complete. Parent progress
and ticket 14/17 notes are updated. Whole-ticket 14 and 17 acceptance remains open:
native normalization/pairing, same-gate client metric mapping, native lifetime
and wire/content qualification are not supplied by this code change. T2 tail
pricing still needs ticket 10 profiles and ticket 24 representative-cell
assessment. No GPU operation or external publication was performed.

Candidate evidence was collected with a recorded dirty source patch and an
existing cache: this does not close the aggregate clean-cache/provenance gate
or any native predictive/performance budget. DeepSeek-v3 uses its supported
legacy connector `none`; the pre-existing NIXL/latent-MLA rejection is retained
and is not repaired or bypassed. No DeepSeek-V4 PD support is asserted.
