**1. Assignment and outcome.**

B29 implements the two measured serving-path delays approved by decision 46 of `requirements.md`, following the supplied B29 assignment and A10 report sections 3.6 and 5.4. Outcome: **implemented**. Implementation and verification are reported here; final review, acceptance, staging and commits remain with Claude.

Working root (`ROOT`): `/data/ycfeng/Frontier/.worktrees/b29-ingress-delays`. Read-only comparison worktree (`BASE`): `/data/ycfeng/Frontier/.worktrees/b29-base`. Both started at `6fd8bc5a04c8202dfec7581bafd67ea829ae7448`. Scratch (`S`): `/data/ycfeng/tmp/codex-implement/6d8aed77-5e60-463c-87d4-b0f7a492b026/b29`. Interpreter (`PY`): `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6. Records (`B29`): `ROOT/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b29`. In this report `A10` means the neighboring `h3o_operator_costs/a10` directory.

Writable scope was limited to `frontier/config/config.py`, `frontier/config/kv_cache_transfer_config.py`, `frontier/simulator.py`, `frontier/events/cluster_batch_end_event.py`, `frontier/events/request_arrival_event.py` if needed, `tests/unit/`, `docs/cli/README.md`, `A10/a10_driver.py`, and own records under `B29/`. The arrival handler needed no change. All temporary files, cache copies and simulator outputs were written to scratch. No tracked edits predated this assignment. The unrelated untracked/ignored directories named in the brief were left intact. No commits, Git state changes, dependency installations, network/GPU operations, messages, or broader task-record updates were performed.

The implementation preserves D1–D7: top-level ingress and common KV-config handoff fields both default to zero; finite nonnegative validation occurs in each config; offline batch ingress is rejected; one arrival factory serves both initialization paths; handoff shifts only the existing KV start event; zero values introduce no conditional branch; measured H200 values are driver inputs. Existing `nixl_read_completion=engine_step`, transfer pricing, target admission and release behavior remain in use. Diagnostic `--field` values override driver defaults, but collide with another explicitly supplied field or an explicit CLI option. This interpretation permits the diagnostic variants requested in the assignment.

**2. Changes and reasons.**

Every changed source file is listed below. The new B29 evidence files are inventoried in `artifact_inventory.json`; they are task-local verification records, not simulator inputs.

| File | Symbols / candidate lines | Change and reason |
| --- | --- | --- |
| `frontier/config/config.py` | imported helper; `SimulationConfig.request_ingress_delay_ms`, line 146; `__post_init__`, line 381 | Adds the serving-front-end field beside `simulation_mode`, path-specific millisecond help, finite nonnegative validation, and a configuration-time rejection for nonzero offline batch ingress. Generated offline arrivals remain eligible because they use arrival events. |
| `frontier/config/kv_cache_transfer_config.py` | `math` import; `_require_finite_non_negative`, line 25; `BaseKVCacheTransferConfig.request_handoff_delay_ms`, line 50; `__post_init__`, line 110 | Adds the common connector-independent handoff field and validation. The helper composes finite checking with the existing nonnegative helper and is used only for the new fields, preserving existing fields' validation behavior. |
| `frontier/simulator.py` | `_init_parallel_events`, line 411; new `_request_arrival_event`, line 421; `_init_event_queue`, line 1347 | Both arrival paths now use one factory computing `request.arrived_at + request_ingress_delay_ms / 1000`. Request client arrival is retained and event creation order is unchanged. |
| `frontier/events/cluster_batch_end_event.py` | `_apply_prefill_output`, line 603 | Changes the sole KV start construction to `time + kv_pred._config.request_handoff_delay_ms / 1000`. No transfer-predictor arithmetic or lifecycle consumer was changed. |
| `docs/cli/README.md` | history line 7; serving-path subsection beginning line 416 | Documents both paths, flags, zero defaults, units, client `arrived_at`, offline batch restriction and deployment-specific measurement requirement. |
| `A10/a10_driver.py` | constants lines 26 and 31; `operator_fields`, line 36; `resolved_inputs`, line 60; new `parse_args`, line 208; CLI entry lines 238–240 | Supplies 10.3/11.4 ms defaults with evidence comments; adds `--without-ingress` and repeatable `--field` using `h4a4.parse_field`; rejects explicit conflicts; records each role's CC backend type and reads `summary_csv_path` only when that field exists. The existing override-diff receipt records diagnostic fields. |
| new `tests/unit/test_serving_path_delays.py` | 176 lines; tests and isolated-run helper listed below | Adds 22 cases for defaults, validation, flattening, connectors, both arrival paths, exact lifecycle shifts and output identity. Runs real DES cases in isolated child processes to keep entity IDs identical for byte comparisons. |
| `tests/unit/test_dsv4_a10_tools.py` | additions lines 249–322 | Adds 11 cases for driver defaults/zero mode, explicit duplicates and option conflicts, backend variants, and actual receipt serialization with a stubbed simulator. All previous test assertions remain intact. |
| `tests/unit/test_pdd_parallel_initializer.py` | added line 94 | Adds `request_ingress_delay_ms=0.0` to the existing `SimpleNamespace` configuration fixture so it represents the expanded config interface. This is the only changed pre-existing test line; no pre-existing assertion was changed or weakened. |

Executed `git diff --stat`:

```text
 docs/cli/README.md                                 | 16 +++++
 frontier/config/config.py                          | 17 +++++
 frontier/config/kv_cache_transfer_config.py        | 17 +++++
 frontier/events/cluster_batch_end_event.py         |  2 +-
 frontier/simulator.py                              | 13 ++--
 .../h3o_operator_costs/a10/a10_driver.py           | 51 +++++++++++++--
 tests/unit/test_dsv4_a10_tools.py                  | 74 ++++++++++++++++++++++
 tests/unit/test_pdd_parallel_initializer.py        |  1 +
 8 files changed, 179 insertions(+), 12 deletions(-)
```

The new test module is untracked, so ordinary `git diff --stat` omits its 176 lines. Including it, the source change totals 355 insertions / 12 deletions in nine files. B29 records are ignored by the existing Git rules; Claude controls any staging of these records. `static_checks.json` records syntax, line counts, whitespace and path checks. Every changed Python source module and every B29 Python script is at or below 2,000 lines.

All following B29 files are new assignment records. Each replay directory contains only small outputs/receipts/logs, with caches retained exclusively in scratch.

| New record path(s), relative to B29 | Purpose |
| --- | --- |
| `artifact_inventory.json`, `report.md` | Final report and file inventory/content hashes. |
| `b28_config_check.json`, `b28_snapshot_p2_32_base.json`, `b28_snapshot_p2_32_candidate.json`, `b28_snapshot_p2_64_base.json`, `b28_snapshot_p2_64_candidate.json`, `receipt_config_comparison.json` | B28 base/candidate config snapshots and actual receipt/config identity results. |
| `base_zero_audit_co-location.log`, `base_zero_audit_pd-disaggregation.log`, `zero_base_identity.json`, `zero_base_identity_audit.json` | Executed independent base-to-candidate byte identity, hashes, exact audit commands and logs. |
| `check_b28_config.py`, `check_zero_base_identity.py`, `config_snapshot.py`, `final_checks.py`, `ingress_values.py`, `receipt_config_check.py`, `regression_compare.py`, `replay_check.py`, `run_final_regression.py`, `run_pytest.py`, `run_replays_native_env.py`, `run_smoke.sh` | New evidence checkers and copies of executed scratch runners; symbols/commands described in section 3. |
| `environment.json`, `static_checks.json` | Interpreter/package environment and executed syntax/size/diff-scope checks. |
| `focused.log`, `focused.xml`, `focused_command.json`, `focused_r2.log`, `focused_r2.xml`, `focused_r2_command.json`, `focused_r3.log`, `focused_r3.xml`, `focused_r3_command.json` | Focused test command records, logs and JUnit results, including superseded fixture attempts. |
| `ingress_values.txt` | Five measured terms with source lines, quote differences and asserted sums. |
| `initial_replay_status.json`, `replay_p2_32_partial.json`, `replay_results.json`, `replay_results.txt`, `replay_status_native_env.json`, `trace_selection.json` | Trace selection, successful/failed replay statuses, joined timing results, and retained P2-32 interim evidence. |
| `regression_base.log`, `regression_base.xml`, `regression_base_command.json`, `regression_base_continue.log`, `regression_base_continue.xml`, `regression_base_continue_command.json`, `regression_base_final.log`, `regression_base_final.xml`, `regression_base_final_command.json`, `regression_base_helper.xml`, `regression_base_helper_command.json`, `regression_candidate.log`, `regression_candidate.xml`, `regression_candidate_command.json`, `regression_candidate_final.log`, `regression_candidate_final.xml`, `regression_candidate_final_command.json`, `regression_candidate_helper.xml`, `regression_candidate_helper_command.json`, `regression_comparison.json`, `regression_comparison.txt`, `regression_comparison_initial.json`, `regression_nonpasses.json`, `regression_selection.json`, `regression_targets.json`, `regression_targets_initial.json` | Selection manifests, base/candidate command records, raw logs/XML and node-level comparisons/nonpasses. |
| `smoke.log` | Executed 12-of-12 dummy smoke matrix evidence. |
| `source_diff.patch` | Reviewed tracked source diff; the new unit module remains separately in the working root. |
| `initial_replay_attempts/p2_32_baseline/`: `command.json`, `receipt.json`, `run.log` | Retained replay evidence; original success/failed-attempt status is preserved. |
| `initial_replay_attempts/p2_32_delayed/`: `command.json`, `receipt.json`, `run.log` | Retained replay evidence; original success/failed-attempt status is preserved. |
| `initial_replay_attempts/p2_64_baseline/`: `command.json`, `receipt.json`, `run.log` | Retained replay evidence; original success/failed-attempt status is preserved. |
| `initial_replay_attempts/p2_64_delayed/`: `command.json`, `receipt.json`, `run.log` | Retained replay evidence; original success/failed-attempt status is preserved. |
| `replays/p2_32_baseline/`: `command.json`, `config.json`, `frontier_transfer_ledger.jsonl`, `metrics_ground_truth.jsonl`, `receipt.json`, `request_metrics.csv`, `run.log`, `system_metrics.json` | Retained replay evidence; original success/failed-attempt status is preserved. |
| `replays/p2_32_delayed/`: `command.json`, `config.json`, `frontier_transfer_ledger.jsonl`, `metrics_ground_truth.jsonl`, `receipt.json`, `request_metrics.csv`, `run.log`, `system_metrics.json` | Retained replay evidence; original success/failed-attempt status is preserved. |
| `replays/p2_64_baseline/`: `command.json`, `config.json`, `frontier_transfer_ledger.jsonl`, `metrics_ground_truth.jsonl`, `receipt.json`, `request_metrics.csv`, `run.log`, `system_metrics.json` | Retained replay evidence; original success/failed-attempt status is preserved. |
| `replays/p2_64_delayed/`: `command.json`, `config.json`, `frontier_transfer_ledger.jsonl`, `metrics_ground_truth.jsonl`, `receipt.json`, `request_metrics.csv`, `run.log`, `system_metrics.json` | Retained replay evidence; original success/failed-attempt status is preserved. |

The D4 consumer audit found no transfer-info object, ledger entry or request start timestamp written between prefill completion and handling the delayed start. The future event timestamp is used by normal DES ordering; downstream consumers receive the actual start-handler time:

| Consumer, inspected source | What it records / does after this change |
| --- | --- |
| `vllm_v1_engine_replica_scheduler.py:663–677`, `complete_kv_transfer_for_requests` at line 499 | Marks the completed prefill request pending transfer and retains KV allocations throughout the handoff gap and transfer. Resource release still occurs at transfer completion. |
| `KVCacheTransferStartEvent.handle_event`, lines 164–180 | Structural NIXL selects and reserves the decode target when the delayed start runs. Existing capacity deferral may postpone actual start further. Other connectors retain their existing target admission path. |
| `KVCacheTransferInfo` construction, `kv_cache_transfer_start_event.py:228–251` | `transfer_start_time=self.time` is the actual delayed start, possibly later if existing capacity deferral applies. |
| Metrics start callback and optional operation trace, start event lines 255–262 | The transfer ledger's `start_ts_s` and operation-trace start use that same actual start. No earlier ledger is materialized at prefill end. |
| `Request.on_kv_cache_transfer_start`, start event line 265; request lines 975–984 | Records the actual start, keeping the earliest start across multiple request transfers as before. |
| Engine-step NIXL register/post/poll and `KVCacheTransferEndEvent`, lines 111–142 | The existing engine-step mode still chooses READ-post/completion alignment. Transfer end, ledger end and request transfer-completion timestamps follow that path. Measured transfer duration is actual end minus actual start; the handoff gap is excluded. |
| Transfer-end source release, lines 144–159 | Frees the prefill allocations only at actual completion and records the structural NIXL release timestamp there. |
| PD-AF KV arrival and downstream M2N lifecycle | PREFILL→DECODE_ATTN arrival follows the delayed KV completion. Subsequent attention/FFN transfers follow their existing lifecycle. No consumer of an earlier anticipated KV start was found. |
| unchanged `RequestArrivalEvent.handle_event`, lines 35–39; `Request.on_arrival`, lines 1169–1195 | Calls `on_arrival(self.time, ...)` and records queue entry and cluster-arrival history at the delayed event time. Scheduling/wait accounting uses that queue entry; ground truth records delayed `time_s` with the original client `arrived_at`. |

`Request.ttft` (`request.py:953–961`) and the CSV/JSON export (`metrics_store.py:2981–2982`) retain their existing endpoint of prefill completion minus client arrival. The added ingress therefore appears there; the later handoff does not. Existing TTFT component fields keep their definitions, and the handoff is not added to KV transfer duration. A10's existing serving comparison (`A10/remainder/frontier_remainder.py:30–35`) separately uses the first decode output timestamp; this report uses that same definition and reports both endpoints below.

**3. Verification.**

All commands had timeouts. Test and smoke processes used `ulimit -v 31457280` (30 GiB); warm replays used `ulimit -v 62914560` (60 GiB). Package/environment details are in `environment.json`. Test-template `PYTHONPATH` selects NumPy 1.24.3, pandas 2.0.1, SciPy 1.10.1 and sklearn 1.2.2. Warm replay caches require the interpreter's already-installed native packages: NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3 and sklearn 1.7.2. No package was installed or upgraded.

Focused command, executed from scratch:

```bash
timeout 3700 "$PY" "$S/run_pytest.py" "$ROOT" focused_r3   tests/unit/test_serving_path_delays.py   tests/unit/test_dsv4_a10_tools.py   tests/unit/test_pdd_parallel_initializer.py
```

**PASS: 47 cases in 29.51 seconds**: 22 serving-delay cases, 23 A10 cases (11 added), and two initializer cases. New/changed-test coverage totals 33 added cases plus the two existing initializer cases. No skip, xfail or weakened assertion was introduced. Focused R1 initially failed five subprocess cases because the fixture used the nonexistent generator name `trace`; this was corrected to `trace_replay`. R2 passed 45 cases; two additional coverage cases produced the final 47-case pass. These attempts are superseded by `focused_r3.xml` and its log. Their exact expanded commands and results are retained as `focused_command.json` / `focused.xml` / `focused.log` and the corresponding `focused_r2` files. The initial regression collection command is retained as `regression_base_command.json` with its log/XML; `regression_base_continue_command.json` records the continued run, and final commands supersede both.

| Added test function, candidate line | Cases | Demonstrated acceptance condition |
| --- | ---: | --- |
| `test_delay_defaults_are_zero`, serving module line 24 | 1 | Both config defaults are exactly 0.0. |
| `test_invalid_delays_rejected`, line 32 | 8 | Negative, NaN, positive infinity and negative infinity fail for each field. |
| `test_ingress_rejected_in_offline_batch_mode`, line 39 | 3 | Each supported architecture rejects nonzero offline batch ingress at config construction. |
| `test_flat_flags_round_trip`, line 45 | 1 | Both exact flat flags parse and reconstruct into their designated config homes. |
| `test_handoff_delay_independent_of_connector`, line 58 | 3 | `none`, `p2p_nccl` and `nixl` accept the common handoff field. |
| `test_parallel_arrivals_use_delayed_factory`, line 64 | 1 | Parallel initialization emits delayed PREFILL events without changing client arrivals. |
| `test_online_ingress_delays_queue_entry_and_ttft`, line 141 | 2 | Real co-location/PDD online runs preserve client arrival, shift arrival event/queue entry/ground truth, and increase `Request.ttft` by exactly ingress. |
| `test_pdd_online_handoff_shifts_transfer_and_decode`, line 154 | 1 | Real PDD online run shifts prefill by d and transfer start/end/pure decode first token by d+h. |
| `test_explicit_zero_delays_preserve_output_bytes`, line 168 | 2 | Unset versus explicit zero fields give byte-identical request CSV and system JSON for PDD and co-location. |
| `test_serving_delay_options_and_receipt_diff`, A10 module line 253 | 4 | Both cases use measured constants by default and both fields at zero with `--without-ingress`; override diff is correct. |
| `test_field_duplicates_and_option_conflicts_fail`, line 277 | 5 | Duplicate fields and conflicts with trace, run-ahead or either zero-delay option fail clearly. |
| `test_field_variants_record_backend_and_override_diff`, line 284 | 1 | Builds `p2_32` with ASTRA analytical CC, verifies no `summary_csv_path` is present, and records backend types/field diff without running a simulator. |
| `test_field_override_is_written_in_receipt`, line 300 | 1 | Driver writes the diagnostic seed/CC fields and per-role backend selection into the completed receipt. |

The exact-shift fixture has one request at client time 0.5 s, prompt 8/output 3, Llama2, PP1/DP1, dummy operator cost 125 ms, ingress 125 ms, handoff 250 ms, no graphs/CPU overhead and `kv_connector=none`. The 4,194,304-byte KV payload and configured bandwidth 0.268435456 Gbps with zero latency yield 125 ms transfer time. Costs, arrival and shifts are binary-rational multiples of 0.125 s; one request prevents queue interactions, and this connector has no engine-step receive alignment. Strict equality therefore tests exact shifts rather than accepting numerical tolerance.

Independent zero-delay comparison also executed the same helper against actual base imports for one PDD and one co-location run, and compared each with the candidate's explicit-zero run. The recorded audit command is `ulimit -v 31457280; timeout 300 "$PY" "$S/check_zero_base_identity.py"`. It runs each base helper with an inner 120-second timeout and records the exact base PYTHONPATH, candidate output paths and byte comparisons in `zero_base_identity_audit.json`. **PASS: both `request_metrics.csv` and `system_metrics.json` are byte-identical in both architectures.** Hashes are in `zero_base_identity.json`. This goes beyond comparing candidate unset/zero values.

Measured-value extraction command:

```bash
timeout 60 "$PY" "$B29/ingress_values.py" > "$B29/ingress_values.txt"
```

**PASS.** Every required term exists; the checker reads the mean and source line, uses decimal sums, checks 10.30/11.4 ms and checks the driver constants. No printed source value has extra precision relative to the assigned quote:

| Term | Measured mean (ms) | Evidence line |
| --- | ---: | --- |
| Client dispatch→proxy arrival, P2-64r n379 | 0.6 | `A10/remainder/reference_split_stdout.txt:3` |
| Proxy arrival→API server arrival, n128 | 3.11 | `A10/k19/ingress_split_wave-20261030-01.txt:2` |
| API server arrival→`add_enqueue`, n128 | 6.59 | same file, line 3 |
| **Ingress sum** | **10.30** | Exact decimal sum of those three means. |
| Prefill update→prefill response, n128 | 2.3 | `A10/remainder/remainder_stdout.txt:8` |
| Prefill response→decode `add_enqueue`, n128 | 9.1 | `A10/remainder/pairs_stdout.txt:19` |
| **Handoff sum** | **11.4** | Exact decimal sum of those two means. |

The 1.63 ms prefill engine-loop wait and 10.4 ms decode engine-loop wait are excluded for the reasons in D7. P2-32 uses the same measurements because it has no `add_enqueue` capture and shares the engine build, proxy and API server. Driver literals represent the checked decimal sums; using binary `2.3 + 9.1` directly would select the adjacent float below literal 11.4.

Regression selection and execution:

`regression_targets.json` contains 179 targets: every Python module recursively under `tests/unit` containing any assigned selection string, union B24's list, with redundant node selectors removed when a whole module is selected. B24's required list is read from the main worktree path supplied in the brief; it was not present in the base/candidate task records. The base has 178 existing targets; the single new serving-test module exists only on the candidate and accounts for intended additions.

Executed final selection with the same runner/pytest options on both roots:

```bash
ulimit -v 31457280
timeout 3800 "$PY" "$S/run_final_regression.py" "$BASE" regression_base_final
timeout 3800 "$PY" "$S/run_final_regression.py" "$ROOT" regression_candidate_final
```

`run_pytest.py` expands each target to an absolute path, uses the assigned environment, disables bytecode/cacheprovider, runs from scratch with a read-only `data` symlink, and emits command JSON, JUnit XML and logs. The expanded pytest command has this form, with actual target paths stored in `regression_*_final_command.json`:

```bash
ulimit -v 31457280
env PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   VLLM_BS_SOURCE_ROOT=/data/ycfeng/tmp/opencode/vllm-v0.28.0   TMPDIR=<private-work-tmp> FRONTIER_TMP_ROOT=<private-work-tmp>   PYTHONPATH=<root>:<root>/tests/unit:<root>/tests/comparison/calibration:<root>/tests/analysis:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages   timeout 3600 "$PY" -m pytest <absolute-selected-paths> -q   --continue-on-collection-errors -p no:cacheprovider   --basetemp="$S/tmp/<label>" --junitxml="$S/<label>.xml"
```

The first base collection stopped on three missing-`aiohttp` errors. Adding `--continue-on-collection-errors` to **both** sides retains those errors and executes the remaining selection; it does not skip or suppress a test. A selection audit added the helper module and the new serving-test module after the initial common pass; the final complete-selection pass removes dependence on that staged execution.

Final whole-selection outcomes (JUnit test nodes):

| Selection | PASS | FAIL | ERROR | Existing SKIP |
| --- | ---: | ---: | ---: | ---: |
| base | 4926 | 128 | 64 | 44 |
| candidate | 4959 | 128 | 64 | 44 |

Both runs additionally report five existing passing subtests. Raw pytest exits are 1 on both sides. Final logs/XML and expanded commands are retained under B29.

Comparison command:

```bash
timeout 60 "$PY" "$B29/regression_compare.py" "$S" "$B29/regression_comparison.json"
```

**PASS: 5,162 common nodes have identical outcomes and normalized failure/error messages; zero removed nodes, zero changed common outcomes, and exactly 33 intended PASS additions.** The full node additions are:

```text
tests/unit/test_dsv4_a10_tools.py::test_field_duplicates_and_option_conflicts_fail[options0]
tests/unit/test_dsv4_a10_tools.py::test_field_duplicates_and_option_conflicts_fail[options1]
tests/unit/test_dsv4_a10_tools.py::test_field_duplicates_and_option_conflicts_fail[options2]
tests/unit/test_dsv4_a10_tools.py::test_field_duplicates_and_option_conflicts_fail[options3]
tests/unit/test_dsv4_a10_tools.py::test_field_duplicates_and_option_conflicts_fail[options4]
tests/unit/test_dsv4_a10_tools.py::test_field_override_is_written_in_receipt
tests/unit/test_dsv4_a10_tools.py::test_field_variants_record_backend_and_override_diff
tests/unit/test_dsv4_a10_tools.py::test_serving_delay_options_and_receipt_diff[False-p2_32]
tests/unit/test_dsv4_a10_tools.py::test_serving_delay_options_and_receipt_diff[False-p2_64]
tests/unit/test_dsv4_a10_tools.py::test_serving_delay_options_and_receipt_diff[True-p2_32]
tests/unit/test_dsv4_a10_tools.py::test_serving_delay_options_and_receipt_diff[True-p2_64]
tests/unit/test_serving_path_delays.py::test_delay_defaults_are_zero
tests/unit/test_serving_path_delays.py::test_explicit_zero_delays_preserve_output_bytes[co-location]
tests/unit/test_serving_path_delays.py::test_explicit_zero_delays_preserve_output_bytes[pd-disaggregation]
tests/unit/test_serving_path_delays.py::test_flat_flags_round_trip
tests/unit/test_serving_path_delays.py::test_handoff_delay_independent_of_connector[nixl]
tests/unit/test_serving_path_delays.py::test_handoff_delay_independent_of_connector[none]
tests/unit/test_serving_path_delays.py::test_handoff_delay_independent_of_connector[p2p_nccl]
tests/unit/test_serving_path_delays.py::test_ingress_rejected_in_offline_batch_mode[co-location]
tests/unit/test_serving_path_delays.py::test_ingress_rejected_in_offline_batch_mode[pd-af-disaggregation]
tests/unit/test_serving_path_delays.py::test_ingress_rejected_in_offline_batch_mode[pd-disaggregation]
tests/unit/test_serving_path_delays.py::test_invalid_delays_rejected[request_handoff_delay_ms--1.0]
tests/unit/test_serving_path_delays.py::test_invalid_delays_rejected[request_handoff_delay_ms--inf]
tests/unit/test_serving_path_delays.py::test_invalid_delays_rejected[request_handoff_delay_ms-inf]
tests/unit/test_serving_path_delays.py::test_invalid_delays_rejected[request_handoff_delay_ms-nan]
tests/unit/test_serving_path_delays.py::test_invalid_delays_rejected[request_ingress_delay_ms--1.0]
tests/unit/test_serving_path_delays.py::test_invalid_delays_rejected[request_ingress_delay_ms--inf]
tests/unit/test_serving_path_delays.py::test_invalid_delays_rejected[request_ingress_delay_ms-inf]
tests/unit/test_serving_path_delays.py::test_invalid_delays_rejected[request_ingress_delay_ms-nan]
tests/unit/test_serving_path_delays.py::test_online_ingress_delays_queue_entry_and_ttft[co-location]
tests/unit/test_serving_path_delays.py::test_online_ingress_delays_queue_entry_and_ttft[pd-disaggregation]
tests/unit/test_serving_path_delays.py::test_parallel_arrivals_use_delayed_factory
tests/unit/test_serving_path_delays.py::test_pdd_online_handoff_shifts_transfer_and_decode
```

The comparator retains raw messages and traceback details. For matching it normalizes worktree/scratch roots, run labels, wall clocks, process object addresses, the moved offline-PDAF config log line (534→551), redundant pytest-truncated `CompletedProcess` / `TempPathFactory` reprs, clipped banner length/PYTHONPATH label, and unittest's root-length-dependent truncated PYTHONPATH diagnostic. Every cosmetic difference was inspected; exception types, assertion values, stderr and return statuses remain compared. `regression_nonpasses.json` lists all shared nonpassing nodes by module, and `regression_comparison.json` retains both sides' full messages. The two whole-suite pytest processes exit nonzero because of shared failures; the regression comparison gate is reported separately from a clean-suite pass.

| Shared failing/error module | Both-side counts | Observation |
| --- | --- | --- |
| `test_colocation_release_review_contracts.py` | 10 FAIL | Missing legacy scripts/executable `python`. |
| `test_dp_placement_harness.py` | 6 FAIL | Missing `aiohttp`. |
| `test_dsv4_attention_runtime.py` | 1 FAIL | Expected single-row training log absent. |
| `test_dsv4_h200_tables.py` | 1 FAIL | Existing table count assertion: 7,105 vs 4,945. |
| `test_dsv4_pilot_cases.py` | 1 FAIL | Missing P1h package model config. |
| `test_examples_documentation_contracts.py` | 2 FAIL | Existing README surface assertions. |
| `test_gdn_hybrid_e2e_increment14ab.py` | 1 FAIL | Existing routing-details target mismatch. |
| `test_glm53_structural_execution.py` | 19 ERROR | Fixture changes cwd to read-only source; default `outputs` cannot be created. |
| `test_mixed_layer_decode_ffn_scheduling.py` | 1 FAIL | Relative source path absent from scratch cwd. |
| `test_module_split_boundaries.py` | 1 FAIL | Existing event/owner export-order assertion. |
| `test_moe_routing_conservation.py` | 8 FAIL | Relative source paths absent from scratch cwd. |
| `test_moe_share_expert_operator_families.py` | 1 FAIL | Relative profiling source path absent from scratch cwd. |
| `test_native_cold_start.py` | 1 ERROR | Missing `aiohttp` at collection. |
| `test_native_groundtruth_check.py` | 42 ERROR, 21 SKIP | 42 Python 3.10 `TarFile.extractall(filter=...)` setup errors; 21 existing skips. |
| `test_native_pd_canonical_http.py` | 1 ERROR | Missing `aiohttp` at collection. |
| `test_native_reader_binding.py` | 14 FAIL | Existing pinned harness hash mismatch. |
| `test_native_streaming_replay.py` | 1 ERROR | Missing `aiohttp` at collection. |
| `test_pdaf_config_contract.py` | 2 FAIL | Subprocess changes cwd to read-only source and fails creating default output. |
| `test_pdaf_deferred_trace_contract.py` | 1 FAIL | Subprocess cannot create default output under read-only source. |
| `test_pdaf_examples.py` | 3 FAIL | Example subprocess selects an environment missing `plotly`. |
| `test_pdaf_parity_reference_observer_bootstrap.py` | 51 FAIL | Pinned reference checkout path absent. |
| `test_pdd_public_surface_docs.py` | 3 FAIL | Existing README release wording assertions. |
| `test_profiling_runtime_boundary.py` | 1 FAIL | Existing profiling-import boundary assertion. |
| `test_replica_identity_contract.py` | 15 FAIL | Relative source paths absent from scratch cwd. |
| `test_stage_execution_context.py` | 1 FAIL | Existing probe scheduler lacks `_full_stage_replica_schedulers`. |
| `test_v4_attention_supplement.py` | 1 FAIL | Existing supplement count assertion: 0 vs 1. |
| `test_v4_trace_audit_interval_metrics.py` | 1 FAIL | Original-environment expectation conflicts with the required PYTHONPATH. |
| `test_vllm_dp_load_balancer.py` | 2 FAIL | Relative prefix-cache fixture absent from scratch cwd. |

The unavailable `aiohttp`, missing pinned PD-AF reference tree, Python 3.10 `TarFile.extractall(filter=...)` incompatibility, absent legacy/reference assets and tests that change cwd to read-only source are outside B29's authorized fixes. Existing contract/data/assertion failures are also identical on base. Starting from scratch cannot redirect a test that explicitly changes cwd to `REPO_ROOT`; no write-scope workaround or test weakening was added.

Dummy matrix command:

```bash
timeout 1900 bash "$S/run_smoke.sh" > "$S/smoke.log" 2>&1
```

**PASS: 12 of 12** dense/MoE, co-location/PDD/PD-AF, offline/online cases. The runner uses the explicit Python interpreter, 30 GiB limit, private scratch cwd/data symlink and private metrics/cache/CC-cache paths. The inner command is `timeout 1800 bash "$ROOT/examples/architecture/run_dummy_smoke_matrix.sh" -- --metrics_config_cache_dir "$S/smoke/cache" --measured_cc_backend_config_cache_dir "$S/smoke/cc_cache"`. `smoke.log` ends with “Completed 12 profiling-independent dummy smoke case(s).”

Warm short-trace replay:

Source trace: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/data/processed_traces/canonical_fixed_6144_1024_req128_qps1/trace.csv`. Selection: preserve the header and first eight data rows unchanged, in source order. Scratch trace: `S/trace_first8.csv`; requests arrive at 0,1,…,7 seconds with 6,144 prompt and 1,024 output tokens. `trace_selection.json` records the rule and identity. Output lengths were not reduced.

Fresh cache copies used `timeout 120 cp -a /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/<case>_cold_g1hs_r2/cache "$S/<case>_<variant>_native_env/cache"` for both cases/variants. Each copied cache had exactly 1,758 files. The source directories were read-only and were never written; inventory was performed on scratch copies only.

Executed outer command:

```bash
ulimit -v 62914560
timeout 7400 "$PY" "$S/run_replays_native_env.py"
```

Each of four inner runs used `timeout 3600 "$PY" "$ROOT/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_driver.py" <case> "$S/<case>_<variant>_native_env" --trace "$S/trace_first8.csv" --without-observer`, with `--without-ingress` appended for baseline. Actual expanded commands are in each retained `command.json`. The executor kept at most two replays active at once. `PYTHONPATH` was `ROOT:ROOT/A10` to use native installed package paths compatible with the warm caches.

The first four initialization attempts with the test-template package paths failed with `ModuleNotFoundError: numpy._core`. Those paths select NumPy 1.24 ahead of NumPy 2 needed by the retained serialized caches. Using the same interpreter's native installed packages resolved initialization without dependency changes. The successful runs use fresh scratch copies and are recorded separately from those failed attempts.

**PASS: four FINISHED receipts, all processes exit 0, and all 32 run/request offset pairs meet the checks.**

| Run | Exit | Wall seconds |
| --- | ---: | ---: |
| p2_32 baseline | 0 | 1164.543 |
| p2_32 delayed | 0 | 1139.247 |
| p2_64 baseline | 0 | 1546.137 |
| p2_64 delayed | 0 | 1555.983 |

Replay-check command:

```bash
timeout 60 "$PY" "$B29/replay_check.py" "$S" "$B29/replay_results.json"   > "$B29/replay_results.txt"
```

The checker joins arrival/completion ground truth, KV transfer ledger and request CSV for all eight requests in each run, verifies client arrival, checks baseline offsets at zero and default offsets at 10.3/11.4 ms, and decomposes serving TTFT shifts. It checks the unchanged CSV `ttft` against prefill completion minus client arrival separately. Serving TTFT below means `first_decode_token_completed_at - arrived_at`; “first output” is that timestamp, while “pure decode token” is `decode_first_token_completed_at`, the following decode token. These are simulator completion timestamps, without an added egress model.

| Case / request | Serving TTFT baseline→default (ms) | Δ (ms) | Arrival offset baseline/default (ms) | Handoff offset baseline/default (ms) | First output baseline→default (s) | Pure decode token baseline→default (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p2_32 / 0 | 623.535277→645.235277 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 0.623535277→0.645235277 | 0.642962991→0.664662991 |
| p2_32 / 1 | 325.860866→347.560866 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 1.325860866→1.347560866 | 1.345290015→1.366990015 |
| p2_32 / 2 | 320.574951→342.274951 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 2.320574951→2.342274951 | 2.340269379→2.361969379 |
| p2_32 / 3 | 318.791550→340.491550 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 3.318791550→3.340491550 | 3.338473763→3.360173763 |
| p2_32 / 4 | 319.643263→341.343263 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 4.319643263→4.341343263 | 4.339258567→4.360958567 |
| p2_32 / 5 | 320.610837→342.310837 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 5.320610837→5.342310837 | 5.340228957→5.361928957 |
| p2_32 / 6 | 321.649218→343.349218 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 6.321649218→6.343349218 | 6.341287546→6.362987546 |
| p2_32 / 7 | 322.433538→344.133538 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 7.322433538→7.344133538 | 7.342052019→7.363752019 |
| p2_64 / 0 | 817.464194→839.164194 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 0.817464194→0.839164194 | 0.842547019→0.864247019 |
| p2_64 / 1 | 446.994950→468.694950 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 1.446994950→1.468694950 | 1.472079209→1.493779209 |
| p2_64 / 2 | 481.926536→503.626536 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 2.481926536→2.503626536 | 2.507010796→2.528710796 |
| p2_64 / 3 | 519.909958→541.609958 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 3.519909958→3.541609958 | 3.545287426→3.566987426 |
| p2_64 / 4 | 561.101433→582.801433 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 4.561101433→4.582801433 | 4.586455353→4.608155353 |
| p2_64 / 5 | 602.308967→624.008967 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 5.602308967→5.624008967 | 5.627662887→5.649362887 |
| p2_64 / 6 | 643.293112→664.993112 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 6.643293112→6.664993112 | 6.668647032→6.690347032 |
| p2_64 / 7 | 658.694798→680.394798 | 21.700000 | 0.000/10.300 | 0.000/11.400 | 7.658694798→7.680394798 | 7.684040244→7.705740244 |

p2_32: serving TTFT mean increase 21.700000000 ms, range 21.700000000–21.700000000 ms; largest residual from 21.7 ms 1.28572708e-11 ms.

p2_64: serving TTFT mean increase 21.700000000 ms, range 21.700000000–21.700000000 ms; largest residual from 21.7 ms 8.97060204e-12 ms.

No request differs from 21.7 ms by more than one measured first decode step; no outlier explanation is required. The measured decomposition and each threshold remain in `replay_results.json`.

Canonical CSV TTFT remains a separate prefill endpoint:

| Case | Canonical CSV TTFT baseline mean (ms) | Default mean (ms) | Mean Δ (ms) |
| --- | ---: | ---: | ---: |
| p2_32 | 209.954995 | 220.254995 | 10.300000 |
| p2_64 | 404.740733 | 415.040733 | 10.300000 |

B28 configuration identity commands:

```bash
timeout 180 "$PY" "$S/check_b28_config.py"
timeout 60 "$PY" "$B29/receipt_config_check.py" "$S" "$B29/receipt_config_comparison.json"
```

**PASS both cases:** all 782 pre-existing flat fields and complete `added_fields` / `override_diff` match the base B28 driver after normalizing run roots and removing only the two new fields at zero. Actual baseline replay nested config dumps and receipt configuration sections also match the base snapshots. Receipt metadata now includes the required per-role CC backend types; volatile status/timings/source-dirty paths are not configuration fields. Baseline is the actual `--without-ingress` replay, as requested.

Final static command: `timeout 60 "$PY" "$B29/final_checks.py"`. It checks every changed source Python module and every B29 Python script with Python 3.10 AST parsing, the 2,000-line gate, `git diff --check`, and changed-path scope. Final results and exact `git diff --stat` are in `static_checks.json`.

**4. Remaining work and decisions for Claude.**

The assigned implementation and required checks are finished. No new regression or replay blocker remains. Claude owns final review/acceptance, staging/commits, and any broader task-record updates. A clean whole-suite run would require the out-of-scope environment/reference/source-layout prerequisites above; shared existing failures were retained and matched against base as acceptance requires. No materially different modeling interface or behavior was adopted. Execution deviations were limited to continuing through shared collection errors, splitting then finalizing the selected-test list, using native installed package paths for warm caches, and explicitly reporting A10 serving TTFT separately from the unchanged canonical CSV metric.

Observed limitations are the shared regression failures/errors described above and the existing canonical TTFT endpoint. The serving-path delays are fixed deployment inputs, not per-request distributions. Capacity pressure and engine-step alignment may change their observed first-output effect; the recorded replay decomposition records those effects instead of fitting an offset. The measured means are reused for P2-32 under D7; no independent P2-32 enqueue measurement was invented. Positive-delay PD-AF and capacity-pressure matrices were not added beyond the assigned checks; consumer behavior was inspected and the existing selected regression/dummy matrix exercised the unchanged zero-default paths.

**5. Review handoff.**

Claude should inspect the common field homes/validation, the single arrival factory and sole shifted KV start, driver option conflict handling and receipt backend facts, and the new strict-equality/byte-identity tests. Review the evidence comments and `ingress_values.txt` against the five source lines. Review `regression_comparison.json` for node-level additions and all shared nonpasses, and `replay_results.json` for every request and its timing decomposition. Rerun the focused 47-case command and the finalized selection/matrix as needed in an environment with the stated prerequisites; use private scratch paths for runners and caches. Copied runner scripts under B29 are evidence copies of scripts executed from scratch and should be copied into a scratch directory before reuse if they derive outputs from their own location. No final acceptance, commit, staging or task-record decision has been made by this executor.
