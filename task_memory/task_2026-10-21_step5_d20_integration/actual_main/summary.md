# D20 actual main handoff

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archive the first reproducible source blocker; execution remains incomplete. |

Task ID: D20-ACTUAL-MAIN. Status: BLOCKED.

Deliverable: `tests/integration/run_step5_measured_main.py`, a real subprocess CLI runner with explicitly synthetic compute and READ inputs. Exact commands, parsed configurations, input requests, logs and status receipts are under this directory's `evidence/`.

Observed validation: both offline and online parse the required P16/D16 TP8/DP2/EP16 topology and then fail during shared predictor initialization. Global policy `full_decode_only` reaches the profile graph-mode map accepting only none/piecewise/full. Parent must fix the production boundary at shared_prediction_model_manager.py:465 -> step5_cost.py:22 -> step5.py:70. No production file was changed here.

Open work: parent fix, new offline/online runs, first-D query16/context6128, completed6144/1024 lifecycle, one completed3.5ms READ and final metrics. No complete-execution, native-admission or accuracy claim. Fixture capacity8192 differs from canonical7168 and is intentionally retained as a declared test limitation.
