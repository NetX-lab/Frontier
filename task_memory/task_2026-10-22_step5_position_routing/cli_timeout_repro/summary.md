# D29-CLI-TIMEOUT-REPRO summary

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Archive the diagnosis phase; implementation and full workflow remain parent-owned. |

## Outcome

A deterministic seconds-scale RED-capable loop is established at the actual
main07 position query/pricing seam on frozen e41/3dac. No source/test change or
full CLI attempt was made.

- Unprofiled production pricing:3.453/3.376s,1056 repeated layer descriptions,
  RED exit1.
- Single diagnostic variable, reuse the actual fixed model's resolved group
  tuple:0.404/0.351s,zero repeated descriptions, diagnostic GREEN exit0.
- Costs, count proofs and full plan keys are identical, payload SHA256
  `d2c141f088b52063edb539094fe3e0967f35e53303d70ebe2e8fbdbe4c86c952`.
- Actual profile:12 complete88-layer group resolutions per two pricing calls,
 14.438s of15.239s cumulative runtime in that resolver.
- Full original CSV15s bounded loader profile separately shows repeated typed
  model/config validation after9.302s parsing. Repeated collective artifact
  parsing is a secondary runtime hotspot.

## Deliverables

- Exact measured commands, hotspots, allocation estimates and limits:
  `test_report_2026-10-23_timeout_loop.md` in this directory.
- Decision/progress: `requirements.md`, `plan.md`, `progress.md`.
- Phase self-check: `checkpoint.md`.
- Scratch script: `/data/ycfeng/tmp/d29_cli_timeout_repro/runtime_loop.py`.
- Raw final RED/control: `execution-final01/`, `execution-groups-final01/`.
- Profiles/full-input samples: `execution01/`, `execution-groups01/`, `loader01/`,
  `typed01/`, `init01/` under the same scratch root.
- Frozen source digest receipt: scratch `source_receipt.json`.
- Parent business/input terminal evidence: root
  `resumed_parent/main07_timeout_receipt.json`, commits33288957/44c3008d.

## Limits and handoff

Observed main07 progress is only initial PREFILL scheduling: seven records per
mode, last simulation time0.0/0.25, no final metrics or stage/lane/transfer
ledgers. The loop is not old-state substitution, a reduced-workload acceptance,
native evidence, or proof that2400s will pass after one repair. It samples one
actual lane/layer with verified paired-cache setup excluded from timing.

Primary owns hypothesis ranking, repair authorization, path budgets, review and
original complete-workflow validation. Suggested CORE repair seam is repeated
model-scoped group selection in the existing predictor/workload mixins;
estimated60–105 source/test gross lines require root allocation. Backend JSON and
typed initialization are secondary findings, not implicitly included fixes.

Systemd scopes were unavailable; bounded probes used timeout60s and30GiB
RLIMIT_AS, with peak RSS about1.14GiB. All outputs/cache paths were relocated
before CLI parsing. No production/test or frozen input edits, dependency/GPU/
platform/network operations, extra agents, or complete CLI retries occurred.
