# D20 actual main handoff

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archive the first reproducible source blocker; execution remains incomplete. |
| 2026-10-21 | Complete the approved policy correction and hand off missing request-free measured coverage. |
| 2026-10-21 | Verify the separately assigned request-free capability; complete actual-main validation remains in progress. |

Task ID: D20-ACTUAL-MAIN. Status: BLOCKED.

Deliverable: `tests/integration/run_step5_measured_main.py`, a real subprocess CLI runner with explicitly synthetic compute and READ inputs. Exact commands, parsed configurations, input requests, logs and status receipts are under this directory's `evidence/`.

Completed code slice: `frontier/execution_time_predictor/step5_cost.py` translates the supported global policy to the existing phase-profile vocabulary and loads eager D-tail/miss coverage alongside graph decode. No new profile enum or fallback. The fixture and focused tests now use the public `full_decode_only` policy rather than the unsupported global `full` shortcut.

The separately authorized request-free capability is now implemented using the existing typed workload, source scopes and shared cache. Canonical stage-boundary and attention consumers share one builder. Explicit has_attention_metadata0 coverage is distinct from ordinary decode; no block table/scored history or skipped attention work is invented. Projections, norms, zero-fill, dense FFN and stage-boundary work retain positive measured-fixture costs. Missing identity/coverage remains an error. Parent owns the ordinary rows.py producer-column seam.

Observed request-free validation:41 focused cases pass in27.89s; the actual CLI initialization regression passes in60.70s and explicitly requires zero completions. A broader affected/structural check reports271PASS/1SKIP/1FAIL; the obsolete structural test still expects all measured construction to raise NotImplementedError and was not changed. Full actual-main run08 is in progress against explicit synthetic6145-token, uniform expert-load and complete decode-context coverage. There is no complete-execution verdict yet.

Open work: complete offline/online runs, first-D query16/context6128, completed6144/1024 lifecycle, one completed3.5ms READ, release and final metrics. No complete-execution, native-admission or accuracy claim. Fixture capacity8192 differs from canonical7168 and is intentionally retained as a declared test limitation. No77-case matrix, GPU/auth/network or new agents were used.
