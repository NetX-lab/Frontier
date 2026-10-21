# D20 actual main progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Capture first offline/online actual CLI failures and parent handoff. |

Task ID: D20-ACTUAL-MAIN.

- completed: inspected D20 records/current D21 handoff, current generated main help, D19 old input guide, supplied fixture and READ unit fixture. Correct P16/D16 and v0.28.0 placement are explicit.
- completed: added `tests/integration/run_step5_measured_main.py`. The child is the unmodified `python -B -m frontier.main` entrypoint, not an in-process simulator wrapper. Synthetic timings, skipped CPU overhead and8192-versus7168 limitation remain explicit. Individual batch metrics and bounded stage summary avoid retaining all EP wave operator rows for1024 outputs.
- failed: offline actual main exits1 in2.040627269s before processing requests; graph policy mismatch below. Exact command/config/log retained under evidence/offline.
- failed: online actual main independently reaches the same source error in2.684166906s before processing requests; exact command/config/log retained under evidence/online.
- blocked: complete outputs, observed D tail/context, completed transfer and final metrics depend on parent source correction. No execution/accuracy/native PASS is claimed.
- pending: rerun both modes in new output directories after parent fix; inspect normal metrics/batch/transfer records and strengthen the runner's success assertions against their actual schema.
- completed: verified runner AST, all14 retained artifact hashes, both exact parsed topology/policy/capacity settings and both expected error strings. Initial tracked diff whitespace check passed; staged check flags only the original terminal blank line in each raw log, retained without changing evidence bytes. Authored code/document checks exclude these two logs. Receipt source HEAD is21f29728e2bfedf5713d602557ba96766e75e327. This verifies reproduction/artifact integrity, not the blocked success path.

## Exact parent source handoff

`shared_prediction_model_manager.py:465-467` passes global CLI value `full_decode_only` to `step5_cost.py:20-22`. That immediately calls `step5_measurement_type`, whose `step5.py:38-42,67-72` map accepts only profile modes none/piecewise/full. Result: `ValueError: Unsupported Step graph_mode='full_decode_only'`. Legal public CLI policy and physical profile graph mode are different vocabularies. Both commands fail while initializing the shared manager, before lazy runtime delegation or transfer events can be exercised.

Requested parent action: fix the policy-to-measurement boundary, retaining eager P/D-tail KERNEL_ONLY and FULL pure-decode CUDA_EVENT behavior. Do not change the CLI policy to unsupported `full` or patch the simulator in the runner. Source owner also needs a current-CLI regression case; existing fixture generation injects global `full` and therefore did not expose this boundary.
