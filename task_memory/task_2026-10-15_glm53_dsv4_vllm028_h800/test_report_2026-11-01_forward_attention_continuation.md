# Forward-attention qualification and independent-pass replay continuation

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | All four original replays and final 96-row comparison complete; P2-64 repeat01 TTFT fails for every generation, with other rows passing. |
| 2026-11-01 | Both P2-32 original replay/gates complete and verified; report partial cross-table sensitivity while original P2-64 runs continue. |
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

## P2-32 completed comparison — 10:04 UTC terminals

Original pass1/pass2 replay elapsed times are 7,804.565654s and 7,829.273651s, both exit 0; their once-run gates also exit 0 with PASS for each of the three clean references. A6-P32-PARTIAL (`ses_e6966ea38ffer5zhQD62hilFYn`) verified 41/41 read-only assertions using the retained comparator's `metric_rows`, `table_hashes`, and `verify_reading`: terminal digests, native CSV identity, request population, table hashes, and config identity. Primary independently matched both actual receipt/reading SHA-256 values to their command records and checked all three repeat verdicts and pooled values with `/data/ycfeng/envs/frontier-py310/bin/python -` (`PYTHONDONTWRITEBYTECODE=1`, repository cwd; no conda activation). No gate or replay was repeated.

| Metric (unit) | Native pooled | B29 prediction (error) | Pass1 prediction (error) | Pass2 prediction (error) | Prediction max−min / B29 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TTFT (ms) | 364.029650 | 344.878991 (5.2607%) | 343.688275 (5.5878%) | 343.616678 (5.6075%) | 1.262313 ms / 0.3660% |
| TPOT (ms) | 19.918966 | 19.926786 (0.03926%) | 19.917612 (0.006796%) | 19.936668 (0.08887%) | 0.019055 ms / 0.09563% |
| Request E2E (ms) | 20741.131945 | 20729.981427 (0.05376%) | 20719.405655 (0.10475%) | 20738.827565 (0.01111%) | 19.421910 ms / 0.09369% |
| Token throughput (tokens/s) | 6231.415496 | 6225.735862 (0.09115%) | 6224.157666 (0.11647%) | 6224.263915 (0.11477%) | 1.578195 tokens/s / 0.02535% |

Absolute pooled errors (B29/pass1/pass2): TTFT 19.150658/20.341374/20.412972 ms; TPOT 0.007820/0.001354/0.017701 ms; E2E 11.150518/21.726289/2.304379 ms; throughput 5.679634/7.257830/7.151581 tokens/s. All reference denominators are positive. Worst per-repeat errors are TTFT repeat03: B29 5.6072%, pass1 5.9331%, pass2 5.9527%. There are no FAIL metrics among the 48 P2-32 pooled/per-repeat rows. This is partial evidence, not the final six-reading/96-row comparison.

The two new tables leave all six other profiling inputs, workload bytes and effective config unchanged except the attention path and run-root paths. All three native CSVs match B29 byte for byte. Observed P2-32 sensitivity is below 0.37% of B29 for every metric; this supports small attention-measurement influence on this fixed case, not a confidence interval or a general accuracy guarantee. TTFT still underpredicts native by about 5.3–5.6%, so small cross-table spread does not eliminate that gap.

At 10:13 UTC, original P2-64 drivers 223795/223801 were still active (96.5%/96.3% CPU, 2h19m elapsed). The existing final collector is now queued on supervisor 223761's exit using blocking `pidfd_open`/`select`; `h3o_operator_costs/a6_pass_replays_20261101/comparison_command.json` records the exact collector argv, environment, source hash and output `comparison.json`. It reads the four original terminal/gate records once and never launches a workload. Numeric FAILs remain in that comparison. Its completion is pending.

## Final independent-pass comparison — 10:34 UTC

All four original replays and their once-run gates finished. P2-64 pass1/pass2 elapsed 9,636.368776s / 9,619.667357s, exit 0. Both numerical gates report FAIL, not an execution failure. `completion.json` records the final supervisor completion at 10:34:20 UTC. The queued collector executed once after that event:

```bash
env PYTHONPATH="$PWD:$PWD/tests/analysis" PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp \
  timeout 1200s /data/ycfeng/envs/frontier-py310/bin/python \
  tests/analysis/v4_attention_pass_comparison.py \
  --task-root "$PWD/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800" \
  --output "$PWD/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a6_pass_replays_20261101/comparison.json"
```

Environment: repository cwd, Python 3.10.6, no conda activation. Exact absolute argv/environment and script hash are in `comparison_command.json`. Expected six retained readings, 96 metric rows, eight cross-table spreads, eight native variability summaries, unchanged non-attention inputs/config/clean-reference bytes, and original terminal hashes. Actual: **PASS_EVIDENCE_CHECKS**, exit 0 in 1.273275s, finished 10:34:21 UTC. This is evidence-integrity PASS, not numerical or formal qualification PASS. Output SHA-256 `032b832741e1f49fc8915369604e0f7faf8a27bcbc5071a109f9c8763a4f0920`; raw log `/data/ycfeng/tmp/a6_supplement_passes_20261101_01/pass_comparison.log`, SHA-256 `1c2bd1b79a5b2e98b91bf215fcec1f0d6dd36c945f854c6e4cc24aee81d89b94`.

### P2-64 values and preserved failure

| Metric (unit) | Native pooled | B29 prediction (error) | Pass1 prediction (error) | Pass2 prediction (error) | Prediction max−min / B29 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TTFT (ms) | 741.216351 | 684.313827 (7.67691%) | 700.581587 (5.48217%) | 699.733815 (5.59655%) | 16.267761 ms / 2.37724% |
| TPOT (ms) | 25.870531 | 25.295337 (2.22336%) | 25.271534 (2.31536%) | 25.293018 (2.23232%) | 0.023802 ms / 0.09410% |
| Request E2E (ms) | 27206.769532 | 26561.443186 (2.37193%) | 26553.361048 (2.40164%) | 26574.490911 (2.32398%) | 21.129863 ms / 0.07955% |
| Token throughput (tokens/s) | 5963.206318 | 5975.251570 (0.20199%) | 5967.966913 (0.07983%) | 5966.889413 (0.06176%) | 8.362157 tokens/s / 0.13995% |

Absolute pooled errors (B29/pass1/pass2): TTFT 56.902525/40.634764/41.482536 ms; TPOT 0.575194/0.598997/0.577513 ms; E2E 645.326345/653.408484/632.278621 ms; throughput 12.045252/4.760595/3.683095 tokens/s. All denominators are positive.

The sole failing row in each P2-64 generation is **repeat01 TTFT**, native **779.888727 ms**:

| Generation | Predicted (ms) | Absolute error (ms) | Relative error | Verdict |
| --- | ---: | ---: | ---: | --- |
| B29 | 684.313827 | 95.574901 | 12.254941% | FAIL |
| pass1_all | 700.581587 | 79.307140 | 10.169033% | FAIL |
| pass2_all | 699.733815 | 80.154912 | 10.277737% | FAIL |

All 93 other pooled/per-repeat rows pass, including every P2-32 row. P2-64 repeat02/repeat03 TTFT errors are respectively 5.676%/4.727% for B29 (exact values in JSON), 3.43363%/2.46238% for pass1 and 3.55048%/2.58041% for pass2. No clean run or request was excluded; pooled PASS does not override these failures.

### Native variability and interpretation

Three independent clean repetitions are retained at each scale. Native TTFT means are P2-32 **361.974764 / 364.748496 / 365.365689 ms** (range 3.390925 ms, sample SD 1.806142 ms, CV 0.496152%) and P2-64 **779.888727 / 725.492280 / 718.268047 ms** (range 61.620681 ms, sample SD 33.685485 ms, CV 4.544622%). Native sample CVs for TPOT/E2E/throughput are P2-32 0.190358%/0.190983%/0.043357% and P2-64 1.606546%/1.456283%/0.240975%. These are descriptive statistics from three runs, not confidence intervals or proof that a representative mean/tail has converged. A stable ordinary-run accuracy claim remains unestablished, particularly near the P2-64 TTFT threshold; retained numeric FAILs remain authoritative for this comparison.

The measured table-generation effect is small for seven of eight case/metric combinations (at most 0.366017% of B29). The exception is P2-64 TTFT at 2.377237% (16.267761 ms), still smaller than its remaining repeat01 gap but material enough not to call all user metrics insensitive. The two new P2-64 predictions differ by only 0.847772 ms; the larger spread is against B29. This experiment changes the complete attention table, including prefill/dummy rows and same-pass gap statistics, so it does not attribute that TTFT change specifically to CSA FULL decode overlap, nor isolate the residual cause. No simulator change or table selection is justified solely by which generation is nearer the gate.

Durable evidence under `h3o_operator_costs/a6_pass_replays_20261101/`: `comparison.json`, `comparison_command.json`, `completion.json`, and all four original terminal command records. `comparison_metrics.csv` exports all 96 rows (SHA-256 `e38a6c8a2717e64d9a891411b36843209f1773ca328f52951af0171b725db732`); `native_variability.csv` exports all eight descriptive summaries (SHA-256 `b644311824e20c1958b63c3ce121b0782690981d1d938e10ebc7d382c370bc8e`). These are direct JSON-row exports using Python `csv.DictWriter`, with no metric recomputation or filtering.

## Completion and remaining acceptance

The approved forward-qualification implementation, supplement validation, median3 verification, and four diagnostic sensitivity replays/comparison are complete. Final actual-evidence review A6-PASS-FINAL-REVIEW (`ses_e69531224ffeMg6U8NpSEOZTF9`) found **no actionable evidence defect** and accepted the diagnostic deliverables. The reviewer inspected requirements, plan/progress, terminal JSON, collector, actual readings/configs/normalized rows and comparison CSV; independently recomputed 96 metrics, eight spreads and eight variability summaries. All three repeat01 TTFT failures were retained. Full large-CSV byte hashes rely on the actual collector; source equivalence relies on baseline's recorded diff paths, not a new independent complete source-diff verification. Final report/status edits were checked by the primary, not reviewed as final text by that reviewer.

The reviewer highlighted that new P2-64 TTFT predictions are only **1.318267 / 2.166039 ms** below the lower acceptance boundary, 701.899855 ms (90% of repeat01 native TTFT). The observed 16.267761 ms cross-generation variation therefore matters to a near-threshold verdict, despite most metrics being stable. No rounding, pooled result or run exclusion changes either FAIL.

Primary acceptance additionally matched the final JSON hash to `comparison_command.json`, checked six readings/96 rows/eight spreads/eight variability records and exactly three retained P2-64 repeat01 TTFT failures, and round-tripped all 96 metric CSV and eight variability CSV rows to the JSON without value differences. `git diff --check` passed. No source or measured input was changed in this close-out, so completed regression tests were not repeated.

The next substantive issue is the remaining P2-64 TTFT gap and reference variability; this comparison alone does not choose its correction or authorize new measurements. Formal table promotion requires a further decision, and no generation is promoted here.

Tickets 05/09/11/19 and formal bundle/E2E acceptance remain OPEN. Attention measurements come from the retained single-H200 graph-replay setup; the CPU sensitivity comparison does not establish an unmeasured real-serving schedule distribution or qualify H800. No prefill, hardware-generalization, or other scope is added here.
