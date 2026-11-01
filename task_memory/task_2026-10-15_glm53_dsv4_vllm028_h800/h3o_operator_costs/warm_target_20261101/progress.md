# WARM-READING progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Completed CPU-only opener identification, six paired warm readings, and output verification. |

Task ID: WARM-READING. Status: completed.

| Plan step | Status | Evidence |
| --- | --- | --- |
| 1. Mechanism/event identification | completed | openers.json; canonical32 4 warmup/0 formal; canonical64 11/5; lifecycle/wave opener sets PASS_EXACT. |
| 2. Reusable warm-reading script | completed | tests/analysis/e2e_warm_reading.py (100 lines); unchanged gate.py functions; six exact original and six exact empty-set value reproductions. |
| 3. Six readings and metric matrix | completed | readings/*_warm_reading.json; readings/metrics.csv, 192 rows; 96 warm PASS, 3 unchanged all-request FAIL. |
| 4. Acceptance and handoff | completed | output_verification.json, test_report_2026-11-01_warm_reading.md, command_receipts.json, checkpoint.md. |

Change rationale and verification:

- Request-owner inventory is retained even in clean references. Joining inventory decode DP ownership to proxy remote-engine identity permits event/request binding without a latency threshold. Expectation: unique formal candidates and exact reproduction of lifecycle/wave first-pair opener sets. Observed: PASS on every formal event; warmup-only candidates remain explicitly unbound when ambiguous.
- Warm reading filters each native repeat and its corresponding Frontier copy by that repeat's evidenced opener IDs. PD-POOL sums both sides' metric numerators and denominators. Expectation: repeat01 retains123 for P2-64, later repeats128; P2-32 unchanged. Observed:379/384 pooled request-repeat counts, all affected and unaffected groups independently checked.
- First iteration kept the harness's equivalent no-op Frontier tuple representation. Acceptance inspection identified that explicit pooled counters should sum both sides even for empty exclusions. Final code sums those counters and preserves the unchanged gate's binary-float ratio rounding; ordinary triple pooling otherwise differs at about1e-13. Final outputs are in readings/; first-iteration root-level outputs are retained unchanged as history, not authoritative deliverables.
- B29 matching rows already existed; retained B29 reading hashes matched exactly. No normalize step or new replay was needed.

Execution environment: /data/ycfeng/envs/frontier-py310/bin/python, Python3.10.6; PYTHONDONTWRITEBYTECODE=1, TMPDIR=/data/ycfeng/tmp, PYTHONPATH=<repo>:<repo>/tests/analysis; inherited ulimit -v31457280. Exact commands, timestamps, git HEAD/dirty state, logs and output hashes are recorded in command_receipts.json. Final run_analysis.py and verify_outputs.py exited0 and their outputs were inspected; a successful exit alone was not used as acceptance evidence.

Scope preserved: no frontier/ or tests/comparison/e2e_gate/ changes, no published input/reading/row/run overwrite, no GPU/platform operation, no commit, no edits to the parent task's documents. Initial unrelated requirements.md modification and untracked .cortexkit/, .pi/, remaining_measurements/ were preserved.

No unfinished delegated work. Limitations:1-second log resolution; warmup event-to-request ambiguity is retained, but no formal binding is ambiguous; paired retained-row reading is not counterfactual stronger-warmup scheduling; full-domain formal qualification remains outside this deliverable. Next action: primary agent accepts the evidence and handles its separately assigned skill/task-record update.
