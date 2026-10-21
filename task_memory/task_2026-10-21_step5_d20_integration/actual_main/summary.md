# D20 actual main handoff

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archive the first reproducible source blocker; execution remains incomplete. |
| 2026-10-21 | Complete the approved policy correction and hand off missing request-free measured coverage. |

Task ID: D20-ACTUAL-MAIN. Status: BLOCKED.

Deliverable: `tests/integration/run_step5_measured_main.py`, a real subprocess CLI runner with explicitly synthetic compute and READ inputs. Exact commands, parsed configurations, input requests, logs and status receipts are under this directory's `evidence/`.

Completed code slice: `frontier/execution_time_predictor/step5_cost.py` translates the supported global policy to the existing phase-profile vocabulary and loads eager D-tail/miss coverage alongside graph decode. No new profile enum or fallback. The fixture and focused tests now use the public `full_decode_only` policy rather than the unsupported global `full` shortcut.

Observed validation:100 affected manager/public-predictor/input-union/bundle tests pass in57.60s. One unmodified real-main CLI initialization regression passes in64.11s; it explicitly stops before request arrival and asserts zero completions, so it is not complete-execution evidence. The final full offline and online attempts both pass initialization and enter runtime, then reject idle-DP DummyForwardBatch measured attention coverage. Source scopes exist at attention/step5.py:497, but current request-owned profiles and runtime identity do not cover that workload. Parent must assign that distinct measured compute/profile capability before full request acceptance.

Open work: measured request-free coverage/binding, new offline/online runs, first-D query16/context6128, completed6144/1024 lifecycle, one completed3.5ms READ, release and final metrics. No complete-execution, native-admission or accuracy claim. Fixture capacity8192 differs from canonical7168 and is intentionally retained as a declared test limitation. No77-case matrix, GPU/auth/network or new agents were used.
