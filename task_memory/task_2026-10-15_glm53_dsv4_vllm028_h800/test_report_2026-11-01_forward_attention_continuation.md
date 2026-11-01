# Forward-attention qualification and independent-pass replay continuation

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Continuation inspected the four original live CPU replay processes; terminal metrics remain pending while comparison collection is prepared. |
| 2026-11-01 | Recovered the verification command receipt after a shared-exec variable collision; numerical output hashes unchanged, original start timestamp explicitly not retained. |
| 2026-11-01 | Median3 A6/forward completed and verified against original folds, references and omissions; three complete classes PASS, four P2 outcomes pending. |
| 2026-11-01 | Supplement business verification, complete three-run tables, current-table forward qualification and review complete; four CPU replays and median3 qualification running. |

## Scope and expected results

YC approved `D-A6-FORWARD-ATTENTION` and `D-A6-PASS-REPLAYS` in `requirements.md`. Expected: full-model attention qualification against measured layer windows on unchanged KFold 5 / seed 1234 assignments, median absolute error <=5% and p90 <=10%; explicit workload exclusions; all scope/phase diagnostics retained. Four CPU P2 replays vary only the attention table generation and compare TTFT, TPOT, E2E and THPT with B29 and all three clean native repeats. These runs do not authorize table publication, simulator changes, new GPU submissions, or relaxation of E2E gates.

Environment: repository `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`; local verification uses `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6), `PYTHONPATH=$PWD:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `TMPDIR=/data/ycfeng/tmp`. No conda activation is used.

## Supplement terminal and collection

- Job `exp-1101-143721-849401`: personal creator `i-fengyicheng`, one non-preemptible H200, `step_main`, node `gpu-h200-1080.lgcm.sh.istep.fun`, platform and replica Succeeded, container exit 0 / restart 0. Worker Python 3.12.3, vLLM 0.28.0. Exact platform receipt: `h3o_operator_costs/a6_three_run_20261101/platform_final.json`.
- Container interval: 06:37:43--07:33:32 UTC, 3,349 seconds = 0.9302778 GPUh. Launcher log create-to-success interval: 3,375.649 seconds = 0.9376803 GPUh. These are observed intervals, not a billed-usage reading. `launch_command.json` explicitly records that the original detached wrapper and shell exit code were not captured; its elapsed value is reconstructed from launcher log timestamps.
- Command: `env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp timeout 900s /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_three_run_20261101/collect_selected.py`.
- First attempt FAIL: `AssertionError: running` from the cloud phase status, despite platform/container success. No output directory was created. Source inspection confirmed the worker atomically replaces `status.json` after archival. A subsequent directory enumeration and read returned `succeeded` with the complete archive list, without changing the worker, collector, or cloud bytes. A cross-client cached directory entry is an inference; the read transition is observed.
- Unchanged collector rerun PASS: 11 files / 440,302,960 bytes, each local SHA-256 equals its selected remote SHA-256. Raw/download root: `/data/ycfeng/tmp/glm53_dsv4/a6_supplement_passes_20261101_download`; persistent receipt: `h3o_operator_costs/a6_three_run_20261101/download.json`. Completed profiler traces remain under the approved personal cloud prefix, not downloaded wholesale.
- Business verifier initial attempt did not test the workload: `FileNotFoundError: repeatability_result.json`. This was a missing execution prerequisite, not a measurement failure; the unchanged verifier passed after its producer completed.
- Repeatability producer PASS_IDENTITY_AND_GEOMETRY: 864/864 planned steps, 4,320 new layer-group rows, no missing/outside/geometry/identity/count mismatch, and 29,376 original supplement medians reproduce the published input. Exact command/inputs: `h3o_operator_costs/a6_three_run_20261101/repeatability_command.json`; result: `repeatability_result.json`. Exit 0, 61.111929 seconds.
- Business verification PASS: platform/container/launcher terminal success, graph-mode counts match the frozen prediction (engine 0 FULL 288 + PIECEWISE 288; engine 1 PIECEWISE 288), bridge PASS, all 864 planned traces nonempty and archived. Trace names/sizes are checked remotely; only the 11 selected files have remote/local SHA checks, not every complete trace. Command/result: `business_command.json` / `business_verification.json` in the same directory; exit 0, 11.878027 seconds.
- The prior approximate remaining budget was 2.160175 GPUh. Subtracting the observed container interval leaves approximately 1.229897 GPUh; conservatively using the larger launcher interval leaves approximately 1.222495 GPUh. Neither value is a platform bill or a new run authorization.

## Independent-pass timing coherence

`tests/analysis/v4_attention_multipass_table.py` now emits `pass1_all` / `pass2_all` in addition to its existing diagnostic median-of-three tables. Each independent pass supplies its kernel, signed gap, and layer-window statistics together; workload identity columns and unrelated tables remain unchanged. This prevents a synthetic combination of one replay's concurrent-kernel timing with another replay's offsets. The manifest reports whether every published row has an independent-pass replacement.

Command: `env PYTHONPATH="$PWD:$PWD/tests/analysis" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_v4_attention_multipass_table.py -q -p no:cacheprovider`.

Expected: coherent kernel/gap/window replacement; unchanged input and unshared rows; incomplete timing schemas rejected. Actual: **3 PASS**, 0 FAIL, 0 SKIP (0.09 seconds).

Full generation and preflight PASS:

- `build_three_run_rows.py` produced three complete 7,105-row generations: the existing 989 workloads plus 432 supplement workloads in each generation. Exit 0, 66.437260 seconds; exact command: `h3o_operator_costs/a6_three_run_20261101/build_rows_command.json`.
- `v4_attention_multipass_table.py` consumed the combined files, not the 989-workload files alone. All 1,421 workloads / 7,105 rows match in all three inputs; `original_shared` is byte-identical to the published table. The median3 generation replaces 96,620 kernel-time medians, preserving every other column. Exit 0, 20.422085 seconds; exact command: `h3o_operator_costs/a6_pass_replays_20261101/tables_command.json`.
- Each pass has 1,298,690 timing fields equal to that pass's measured rows, 296,965 count fields equal to the reference, and 760,235 unchanged non-timing fields. The first inline audit failed on a tuple key because JSON represents tile scheduling as a Boolean and CSV as a string; the audit then reused the existing `row_key` normalization, without changing data or the generator.
- All four P2 config bindings PASS: only the attention table's digest changes from B29. Compute, MoE, kernel gaps, runner block/replay and collectives match every baseline consumer digest/row count. Dummy mode is off and no `dsv4_step_cost.json` sits beside an input. Runner siblings were copied byte-identically.
- Evidence: `h3o_operator_costs/a6_pass_replays_20261101/{rows_manifest.json,tables_manifest.json,preflight.json}`. Scratch inputs: `/data/ycfeng/tmp/glm53_dsv4/a6_pass_replays_20261101/{rows,tables}`.

| Table | Rows | SHA-256 |
| --- | ---: | --- |
| Reference | 7,105 | `01a9d51b4389db15c0d61eaa7347482e6a99df421540929abc7f0c55052365eb` |
| Median-of-three | 7,105 | `bb240084c1e80742693caa90110079a0ba807fe01d7eba357594cf1463216566` |
| Independent pass 1 | 7,105 | `7c54e9d12743ffe1f218d019a5fc33aa34eb738850585cd1a7477249416709f6` |
| Independent pass 2 | 7,105 | `6d10a6996c07613a02ea9aa5577fe30ac3ad65397ceda0572843b02f2efc68ef` |

Method limit: median3 changes only kernel-time targets; its gaps and measured-window reference remain the original generation. This is the approved diagnostic A6 reconstruction, not an independent-pass table or a published replacement. The independent-pass sensitivity tables replace kernel/gap/window statistics together.

## Forward-attention implementation and current input

Changed files: `tests/analysis/v4_attention_phase_heldout.py` and `tests/unit/test_v4_attention_phase_heldout.py`. Layer multiplicities come from the configured model's specs and existing tile-schedule planner owner, not literals. The aggregate joins canonical workload units to their recorded folds, rejects fold/row identity conflicts, and keeps missing/duplicate/unexpected groups NOT_EVALUABLE. Measured `layer_window_stats.median`, not composed scope medians, supplies the qualified reference. Scope and phase diagnostics remain separate from E2E acceptance.

Primary command: `env PYTHONPATH="$PWD:$PWD/tests/analysis" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_v4_attention_phase_heldout.py tests/unit/test_v4_attention_heldout.py tests/unit/test_v4_attention_multipass_table.py -q -p no:cacheprovider`. Actual: **31 PASS**, 0 FAIL, 0 SKIP, 3.85 seconds. Log: `/data/ycfeng/tmp/a6_supplement_passes_20261101_01/forward_attention_primary_tests.log`. This completed before the server restart and was not repeated.

Retained current aggregation: `h3o_operator_costs/a6_forward_attention_20261101/current_aggregation.json`, SHA-256 `4c254346b33fc58e2396507d3697c39c394cd35c84ddf16765ac586e5acd42d3`. It applies the implemented `aggregate_forward_attention` to retained predictions (no retraining), with the unchanged loaded 7,105-row fold assignment validated. Input/provenance hashes are recorded in the JSON.

| Forward class | Complete workloads | Median absolute error | p90 absolute error | Verdict |
| --- | ---: | ---: | ---: | --- |
| FULL/decode | 464 | 0.629624% | 1.667818% | PASS |
| PIECEWISE/decode | 469 | 0.577179% | 1.646428% | PASS |
| PIECEWISE/first_tail | 440 | 0.653529% | 1.886652% | PASS |

Of 1,421 recorded workload units, 1,373 are complete/evaluable, 24 partially evaluable, and 24 completely skipped. FULL/dummy, NONE/dummy, NONE/prefill and PIECEWISE/dummy have no complete evaluable workload and remain NOT_EVALUABLE. **Overall attention qualification is NOT_EVALUABLE**, not a full-domain PASS. E2E remains NOT_ASSESSED. Absolute predicted/measured attention times and signed errors for each complete workload are in the JSON. These findings do not qualify the omitted workloads.

The retained row JSON had only the earlier selected-scope diagnostics. The full updated main invocation completed after restart: exit 0, 322.008842 seconds, 6,923 evaluated rows and 182 explicit coverage skips. It reuses identical-input/fold trainer scratch and reproduces the three forward-class values above while retaining every scope diagnostic. Command: `h3o_operator_costs/a6_forward_attention_20261101/current_command.json`; full output: `current_phase_heldout.json` (SHA-256 `d17258d6f5cec99a8042b76aae6e89a7dfbaba779173e75465cca3eae9ca0274`); compact record: `current_summary.json`. Its purpose was completing diagnostics absent from the old selected-scope output, not repeating completed aggregation or measurements.

Phase review B-FORWARD-REVIEW (`ses_e69f04c05ffeKSDHTXKtDmR7O7`) inspected the four-file source diff, requirements/plan/tickets 05/09, tests, current main output and table manifest: **No actionable findings**. The reviewer did not repeat tests or launch workloads. Primary's full actual table-field and input-binding assertions are additional evidence; final median3/replay results were not yet available to the reviewer.

Verified code commits: `21ab3b31` (forward qualification and focused tests), `fb8670ac` (coherent independent-pass tables and focused tests), `e033c2cc` (the already-executed supplement completion scripts). None changes `frontier/` or published inputs. The current full all-scope JSON and 10 MB supplement repeatability JSON remain local evidence; their output hashes are retained in the committed command receipts, with current aggregation/summary and table/input manifests selected for Git. Root requirements/plan were re-read before this milestone checkpoint.

## Active CPU runs

Four replays started once at 07:53:42 UTC, supervisor PID 223761:

| Pass / case | Original timeout PID | Command receipt |
| --- | ---: | --- |
| pass1 / P2-32 | 223796 | `h3o_operator_costs/a6_pass_replays_20261101/pass1_all_p2_32_command.json` |
| pass1 / P2-64 | 223794 | `h3o_operator_costs/a6_pass_replays_20261101/pass1_all_p2_64_command.json` |
| pass2 / P2-32 | 223800 | `h3o_operator_costs/a6_pass_replays_20261101/pass2_all_p2_32_command.json` |
| pass2 / P2-64 | 223798 | `h3o_operator_costs/a6_pass_replays_20261101/pass2_all_p2_64_command.json` |

Each runs the unchanged `a10/a10_driver.py` with exactly one `--field random_forrest_execution_time_predictor_config_atten_kernel_only_input_file=<pass CSV>` override; K19, 10.3/11.4 ms delays and all other bindings remain fixed. Four private empty model-cache roots; 60 GiB address-space limit per process, baseline single-thread BLAS/OpenMP/MKL settings. Preflight had created only `metrics/.../config.json`: the initial launch assertion incorrectly required absent roots and failed before any CLI call. Inspected all four roots, proved no cache/receipt/run output existed, then used the existing driver's supported config-only roots. No original simulation was repeated or deleted.

The existing random-forest factory uses `random_state=0` in `frontier/execution_time_predictor/shared_prediction_model_manager.py` and `random_forrest_execution_time_predictor.py`; the comparison does not introduce a new fitting seed. Completed tests/review and current output were not repeated after committing unchanged bytes.

Run/log/gate roots: `/data/ycfeng/tmp/glm53_dsv4/a6_pass_replays_20261101/{runs,logs,gates}`. The original supervisor waits and invokes the unchanged `a10_gate.py` once after each successful run, against all three pinned native clean repeats. Gate exit success alone does not imply numerical PASS. B29 P2-64 repeat01 TTFT is already FAIL at 12.254940665%; all per-repeat results must remain visible.

Continuation observation at 09:21 UTC: exact original supervisor/timeout PIDs remain alive; driver PIDs 223797/223795/223802/223801 show 94.3–94.6% CPU and about 1.4 GB RSS each after 1h28m. Current driver receipts are INITIALIZED, so no final metric is available. Commands used `ps -o pid,ppid,stat,etime,%cpu,rss,args --sid <recorded timeout PID>` and scoped JSON/log reads. Existing simulation initialization warnings are retained in the logs; these process observations do not establish numerical correctness. No workload was restarted.

Median3 A6/forward is complete, not an active process: A6 fits/scope diagnostics exit 0 in 2,379.995336 seconds (07:54:32–08:34:12 UTC); forward qualification exit 0 in 216.090562 seconds (08:34:12–08:37:48 UTC), with the same table/fold cache and 30 GiB limit. Exact commands/environment/source/output hashes: `h3o_operator_costs/a6_forward_attention_20261101/{median3_a6_command.json,median3_forward_command.json}`. Scope errors stay diagnostic; no new formal table is published. The original supervisor's two stages finished once and must not be relaunched.

## Median3 verification and comparison

Direct verification PASS: all 1,421 workload coordinates/row indices/fold IDs, KFold5/seed1234, thresholds, config input hash and configured layer-group counts match the original A6. Every one of the 1,373 complete forwards retains exactly the same measured reference and 43-layer weighting. All 182 row skips and 48 workload omissions are identical (24 partial / 24 fully unevaluable); E2E remains NOT_ASSESSED and overall attention NOT_EVALUABLE. All output/table hashes match the completed command receipts. No successful exit alone is used as qualification evidence.

| Forward class | Complete workloads | Median error | p90 error | Original p90 | Change (percentage points) | Maximum error | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FULL/decode | 464 | 0.679554% | 1.628628% | 1.667818% | -0.039190 | 7.223122% | PASS |
| PIECEWISE/decode | 469 | 0.727630% | 1.932193% | 1.646428% | +0.285765 | 3.705850% | PASS |
| PIECEWISE/first_tail | 440 | 0.911188% | 2.215912% | 1.886652% | +0.329259 | 3.915296% | PASS |

No complete workload exceeds 10%; signed mean relative errors are -0.151280%, -0.086793%, -0.094774%. The four other forward classes remain unevaluable. Median3 does **not** uniformly improve accuracy: PIECEWISE p90 rises modestly, though all complete-class limits pass. This supports robustness of the forward qualification to this target reconstruction, not a demonstrated improvement in ordinary serving metrics or permission to publish it.

Exact verification body/environment: `median3_verification_command.json` in the same directory. Selected evidence: `median3_verification.json` (source hashes, comparisons, maximum absolute errors in microseconds and scope diagnostics), `median3_summary.json`, and `median3_forward_workloads.csv` (1,373 predicted/actual values in microseconds, signed errors and absolute relative errors). Relative-error denominators were positive in all complete rows; none are undefined. Full local output `median3_forward_attention.json` has SHA-256 `c48cf5692ec6748022bcd2482979d5998290e7c9a83e5e2542304111d258cabd`; it is retained locally, not duplicated into Git.

The existing median3 generator replaces kernel-time medians only; gap/window reference remains original. The independent pass1/pass2 replay tables replace all timing statistics together. This difference is retained in the comparison record and cannot be hidden by a phase PASS.

Record-format check: ordinary `git diff --check` passed before adding new artifacts. `git diff --cached --check` then reported the standard Python CSV writer's CRLF endings as trailing whitespace (exit 2). The CSV bytes and recorded hash were preserved. `git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check` passed (exit 0), retaining the default whitespace checks and recognizing CRLF. No source/config file or numerical evidence was changed for this format check.

Verification-receipt RCA: the original inline `exec` shared the parent namespace. Its `for label,r` loop replaced the parent `r` command record with the median3 A6 result, making the final receipt a 2,924,993-byte result dump. The original retained invocation and surviving exit/elapsed/finish/output hashes recovered a correct 8,276-byte command receipt; the wrong prior bytes remain in Git commit `b568dc67` and their SHA is in the recovery block. The verifier took 0.872703 seconds and finished at 08:41:09 UTC; its original start timestamp was lost and is explicitly `null`, not reconstructed. All three numeric output hashes match before/after recovery. No qualification, data generation or P2 process was rerun, and no source/published input changed.

## Pass-comparison collector verification

`tests/analysis/v4_attention_pass_comparison.py` reads retained evidence only. It requires the four original terminal commands and their nested `gate` receipts, checks actual table/config/reference bindings, and emits six readings, 96 pooled/per-repeat metric rows, eight cross-generation spreads and eight three-run native variability summaries. It preserves signed delta separately from absolute error. Spread is `(max prediction - min prediction)` divided by B29 prediction and by pooled native value; native sample SD/CV are descriptive, not confidence intervals.

- Author command: `PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/a6_supplement_passes_20261101_01/collector_focused_check.py`. **17 checks PASS**: CLI/compile, both retained B29 readings (32 metric rows), arithmetic/identity negative cases and rejection of incomplete input/output overwrite. B29 P2-64 repeat01 TTFT remains 12.254940665% FAIL.
- Primary corrected a draft-only schema assumption before use: the existing supervisor embeds `gate` in each command receipt, rather than creating separate gate-command files. Command: `env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/a6_supplement_passes_20261101_01/collector_gate_schema_check.py`. **PASS** for all four nested schemas with explicit in-memory terminal fixtures. This is a schema check, not evidence of completed replays.
- Primary directly checked the four available runtime configs and seven-table sets against the collector's B29 substitution logic: **4/4 PASS**, only the attention generation differs. This does not substitute for checking final artifacts.
- Phase review `A6-P2-COMPARISON-REVIEW` (`ses_e69881453ffe47mqrOTlBhRn4w`) found one P2 issue: a new pass reading's digest/verdict was not compared with the nested gate record. The two-line correction also binds the run receipt digest. Command: `env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/a6_supplement_passes_20261101_01/collector_terminal_binding_check.py`. Matched retained B29 bytes pass; changed receipt digest, reading digest and verdict each reject: **4 checks PASS**. The reviewer confirmed its sole finding resolved, without rerunning checks. Source is 179 lines; SHA-256 `a914c441a6e31c4895ce5408361fdf8307f849abb416e3420a530b8dd59a0137`.
- Persistent preparation evidence: `h3o_operator_costs/a6_pass_replays_20261101/{collector_verification.json,collector_terminal_binding_check.json}`; exact scripts remain in the temporary directory above. Final 96-row execution remains pending. Preparation checks use explicit in-memory completion envelopes only; no new gate, normalizer or replay was invoked.

## Pending acceptance

1. Four complete CPU P2 replay/gate results, all-repeat metric and cross-pass sensitivity comparison; original processes only. Median3 qualification is finished and verified.
2. Retain findings, review any new defect and obtain a further decision before formal table promotion; do not close unrelated qualification gaps.

Tickets 05/09/11/19 and formal bundle/E2E acceptance remain OPEN. Attention measurements come from the retained single-H200 graph-replay setup; the CPU sensitivity comparison does not establish an unmeasured real-serving schedule distribution or qualify H800. No prefill, hardware-generalization, or other scope is added here.
