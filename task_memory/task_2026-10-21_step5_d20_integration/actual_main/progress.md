# D20 actual main progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Capture first offline/online actual CLI failures and parent handoff. |
| 2026-10-21 | Fix the approved global-policy boundary; verify100 affected tests and actual CLI initialization; diagnose the next separate blocker. |

Task ID: D20-ACTUAL-MAIN.

## Current state after approved policy correction

- completed: parent follow-up explicitly authorizes this agent to fix the Step graph-policy boundary. Existing scheduler `vllm_v1_iteration_policy.py:94-103` already defines full_decode_only as FULL only for a uniform decode capture hit, otherwise NONE. No shared normalization utility exists; profile enum remains none/piecewise/full.
- completed: `step5_cost.py` now translates the three supported public policies at the manager-facing boundary. D loads KERNEL_ONLY for initial prompt-tail/graph-miss work and CUDA_EVENT for graph decode. P remains KERNEL_ONLY; FFN-only roles remain excluded. No flags, fallback, policy/profile enum or unrelated production change.
- completed: removed the unsupported global `full` fixture shortcut from manager/public predictor and integration fixture setup. Updated union tests now require both P and eager D coverage, still exclude FFN roles and retain separate event coverage. Profile rows still correctly use physical `full`.
- completed: pre-fix policy test recorded5FAIL/7PASS. Focused manager suite14PASS/9.88s; final affected manager/public predictor/input-union/bundle suite100PASS/57.60s.
- completed: `tests/integration/test_step5_measured_main.py` executes the actual CLI with full_decode_only/v0.28.0 and asserts clean initialization/final output. It intentionally uses existing time_limit1 before arrival10 and requires zero completed requests. Final result1PASS/64.11s. This tests the graph-policy boundary only, not request completion.
- failed: full offline `/data/ycfeng/tmp/d20_actual_main_h80032_03` exits1/23.585099s; full online `/data/ycfeng/tmp/d20_actual_main_h80032_04` exits1/18.785726s. Both now pass shared-manager initialization and enter `Simulator.run`, then reject the first request-free DP dummy forward.
- blocked: complete6144/1024, first D16/cached6128, NIXL lifecycle/release and final request metrics require measured request-free coverage. No dummy-forward bypass, zero timing, topology weakening or profile substitution was applied.
- verification limits: one broad225-case attempt exceeded120s after partial progress, with no final verdict and no leftover matching pytest process. It was replaced by the scoped100-case affected check, not retried as a campaign. The new CLI test's first attempt used fractional time_limit although the existing CLI accepts int; its second exposed a test-only numeric enum-key assumption. Both were corrected; actual initialization had already returned0 on the second. Raw failed logs remain in scratch, selected final evidence is under evidence/policy_fix.
- completed: final receipt/AST check passed all14 new retained hashes, both parsed-config hashes and unchanged topology/policy, both exact runtime blocker strings and the changed Python ASTs. Authored changes pass `git diff --check`; raw logs remain byte-exact evidence.

## Next separate source handoff

Call chain: `replica_stage_schedule_event.py:180` -> `dp_dummy_forward.py:78,21` -> normal public MoE predictor -> `predict_step5_attention_time` -> `step5_batch_features`, which explicitly raises `ValueError: Step measured request-free attention coverage is unavailable` for DummyForwardBatch.

`entities/batch.py:1331-1338` defines the actual idle-DP workload: one token, no request or attention metadata, attention kernels/cache writes skipped, projections/norms/MoE still run. `attention/step5.py:497` already names the distinct request-free source scopes. Current synthetic attention rows cover request-owned6144/16/1 shapes, not this workload; the measured feature/schema path does not represent a request-free identity. Reusing the regular one-token request cost would price skipped attention work incorrectly. Parent should assign the measured compute/profile owner to add exact request-free coverage and its runtime binding, preserving source scopes and fail-fast coverage. This exceeds the newly approved graph-policy fix; do not infer new producer/modeling authority.

Next actions: (1) parent integrates this verified policy slice and assigns the request-free coverage blocker; (2) after that delivery, resume the existing unchanged full offline/online command into fresh output directories; (3) finish tail/context, transfer, release and final metrics assertions. Parent owns its77-case candidate rerun; this agent launched no matrix or GPU operation.

## Initial reproduction archive

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
