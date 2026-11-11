# Test report 2026-11-11 — verification of the Codex review corrections

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-11 | Created: verification of the four corrections applied after the Codex review of the final delivery (F001-F004), including the extra stale assertion the F002 fix exposed. |

## Scope

Verifies the corrections applied to the findings of
[the Codex review of the final delivery](codex-review-7053879f-03a9-4e51-ae74-c6bbab2891dd/review.md)
(verdict `changes_required`, 0 P0, 0 P1, 1 P2, 3 P3). Nothing in the simulator's executable path changed, so no
numerical result is expected to move; the test-side corrections are expected to change only which modules can be
collected and two stale expectations.

## Environment

- Worktree: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, branch `feat/glm53-dsv4-vllm028`.
- Interpreter for every check: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6).
- Every command ran from the worktree root with `PYTHONPATH=$PWD` and `FRONTIER_TMP_ROOT=/data/ycfeng/tmp`;
  `--basetemp` directories were created beforehand under `/data/ycfeng/tmp/frontier_review_fix/`.
- CPU and local artifacts only: no GPU, platform, cloud-storage, StepMind or RJob operation.
- Commits under verification: `c5f06a9b` and `01cc26fc` for the test corrections; the record corrections are
  documentation only.

## 1. F002 — collection independence of the helper-importing unit modules

Expected: every unit module that imports a helper by name collects on its own; the full `tests/unit` collection gains
the previously erroring modules and loses none.

Command, run with the conftest in place and again with it renamed away:

```bash
/data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit --collect-only -q -p no:cacheprovider
```

| State | Collected | Error modules |
| --- | --- | --- |
| with `tests/unit/conftest.py` | 8,176 | 15 |
| without it | 7,879 | 25 |

The 10 repaired modules are `test_dsv4_kernel_identity_check.py`, `test_native_dp_wave_timeline.py`,
`test_v4_attention_coverage.py`, `test_v4_attention_heldout.py`, `test_v4_attention_multipass_table.py`,
`test_v4_attention_package_difference.py`, `test_v4_attention_phase_heldout.py`,
`test_v4_attention_supplement.py`, `test_v4_collective_kernels.py` and `test_v4_replay_census.py`. The set of newly
introduced errors is empty. The 15 remaining error modules are the pre-existing missing-package cases recorded under
I-018. **PASS.**

The review reported five modules. Measuring the whole directory showed the same defect in five more, including
`test_dsv4_kernel_identity_check.py`, which imports from `tests/comparison/calibration` rather than `tests/analysis`
and had been collecting only because an earlier module imported `native_pd_worker`, whose own import inserts that
directory into `sys.path`. The conftest therefore resolves both helper directories.

Shadowing check, run with the test directories removed from `sys.path`: neither helper directory contains a module
name that `importlib.util.find_spec` already resolves — 0 collisions across 30 `tests/analysis` and 32
`tests/comparison/calibration` modules. **PASS.**

Standalone runs, one module per pytest process:

| Module | Result |
| --- | --- |
| `test_dsv4_kernel_identity_check.py` | 42 passed |
| `test_dsv4_pilot_cases.py` | 53 passed |
| `test_native_dp_wave_timeline.py` | 19 passed |
| `test_v4_collective_kernels.py` | 51 passed |
| `test_v4_attention_coverage.py` | 126 passed in 150.03 s |
| `test_v4_attention_heldout.py` | 8 passed |
| `test_v4_attention_package_difference.py` | 2 passed |
| `test_v4_attention_phase_heldout.py` | 20 passed |
| `test_v4_attention_supplement.py` | 9 passed, 1 warning |
| `test_v4_attention_multipass_table.py` | 3 passed |
| `test_v4_replay_census.py` | 17 passed |
| `test_dsv4_h200_tables.py` | 13 passed |

363 tests, all PASS. Order independence: the 11 modules other than `test_v4_attention_coverage.py` (excluded only for
runtime) give **237 passed** in declaration order and **237 passed** reversed. **PASS.**

The single warning is a pandas `DtypeWarning` for columns 49-50 of
`attention_graph_replay_kernel_only.csv`, raised inside `frontier/execution_time_predictor/dsv4_predictor.py:56`. It
predates these changes. Cause: 65 of the 7,105 rows are `phase=dummy` DP dummy-forward rows that leave `is_prefill`
and `is_mixed_batch` blank, so pandas infers `object` for those two columns. No lookup is affected, because the
predictor matches `decode`, `prefill` and `first_tail` phases. Recorded as an observation; the table is not changed.

## 2. F004 — stale row count for the promoted attention table

Expected: the test expectation equals the promoted table's actual row count and its manifest entry.

```bash
/data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_h200_tables.py -q -p no:cacheprovider
```

`attention_graph_replay_kernel_only.csv` holds 7,105 rows — the 4,945 G1h-02 rows plus the 2,160 supplement rows
promoted in `fdb6e032` — and `h3o_operator_costs/h200_tables/manifest.json` records the same count. The expectation
was 4,945. After the correction: **13 passed**, with all four tables rehashing to their manifest `sha256` entries.
**PASS.**

## 3. The stale assertion exposed by the F002 fix (I-021)

Expected after making the module collectible: all its tests pass, or the failure is explained and corrected at its
cause.

First run after the conftest fix: `test_probe_c_writer_failure_is_closed_by_projected_union` failed `assert 0 == 1`.
Cause: the test was written at `60dfafe7` (2026-10-29 17:59) against the pre-supplement measured writer range
`(1.0, 7.0)`, so the reported csa first-tail padded16/own11/writes8 coordinate lay outside the measured ranges and had
to be closed by the projected union. The promotion `fdb6e032` at 20:09 the same day widened the measured range to
`(1.0, 16.0)`, so the coordinate is now measured directly and the union adds nothing. The test is retargeted and
renamed to `test_probe_c_writer_failure_is_closed_by_the_promoted_supplement`: it asserts the measured range
`(1.0, 16.0)` and that neither the measured ranges nor their union with the projected rows reports a miss. Module:
**9 passed. PASS.** The coverage conclusion is stronger than the original test asserted, not weaker.

## 4. F003 — Git provenance of the promotion receipt

Expected: the auditor records verified provenance instead of a claim, and the rerun stays complete.

The auditor had hard-coded `task_memory is ignored; no Git creation timestamp proof is available`. `task_memory/` is
gitignored, but the receipt was force-added, so Git does hold its creation time:

```bash
git log --follow --diff-filter=A --format='%H %cI' -- \
  task_memory/.../formal_p2_20261102/promotion/receipt_p2_32_to_p2_64.json
# 1a7950ed5ba9b766a5322313eb7e0f50fa76a2e4 2026-11-02T07:39:13+00:00
```

| Quantity | Value |
| --- | --- |
| Receipt `issued_utc` | `2026-11-02T07:38:49.523696+00:00` |
| Adding commit date | `2026-11-02T07:39:13+00:00` (24 s later) |
| Committed blob sha256 | `d505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596` |
| Current file sha256 | identical |
| generation08 submission | `2026-11-07T16:28:57Z` |

`review_20261108_h200_mainline/audit_qualification.py` now computes the adding commit, its committer date and the
committed blob digest, and asserts them in the new check `promotion receipt committed unchanged before submission`.

```bash
/data/ycfeng/envs/frontier-py310/bin/python \
  task_memory/.../review_20261108_h200_mainline/audit_qualification.py
# CHECKS 907 PASS 907 FAIL 0        (11.4 s; log: qualification_audit_20261111.log)
```

Expected 906 prior checks plus the new one, all PASS; actual **907/907 PASS, 0 FAIL**. **PASS.** The earlier
`qualification_audit.log` and `qualification_audit_initial.log` are kept as history, so the superseded statement
remains visible in its original run output. The ordering conclusion is unchanged and now has independent Git
corroboration.

## 5. Regression checks

Expected: unchanged behaviour outside the corrections, including the three known README documentation-contract
failures.

```bash
/data/ycfeng/envs/frontier-py310/bin/python -m pytest \
  tests/unit/test_pdd_public_surface_docs.py tests/unit/test_examples_pdd_scripts.py \
  tests/e2e/test_colocation_examples_layout.py -q -p no:cacheprovider
# 3 failed, 17 passed

/data/ycfeng/envs/frontier-py310/bin/python -m pytest \
  tests/unit/test_e2e_gate.py tests/unit/test_native_pd_worker.py -q -p no:cacheprovider
# 83 passed in 8.62s
```

The three failures are the same pre-existing identifiers as the recorded baseline —
`test_top_level_docs_advertise_supported_pdd_without_upcoming_claims`,
`test_public_docs_require_sequential_pdd_release_mode` and
`test_public_docs_state_internal_parallel_pdd_boundary_and_current_evidence` — all owned by the release-docs surface
and outside this task's scope. Unchanged, therefore not a regression. **PASS for the regression intent.**

## 6. F001 — record corrections and their cross-checks

No executable check applies; each correction was cross-checked against the artifact the review named.

| Correction | Cross-check |
| --- | --- |
| Ticket 19 GPU-hour box: waiver is the current state; the original criterion reads as dated history | `requirements.md` D-GPU-HOURS-NOT-REQUIRED; all measured figures and known gaps preserved verbatim, and the duplicated generations 05-08 sentence reduced to one occurrence |
| Ticket 19 exclusion statement | `openers` is empty in both admitted readings (`p2_32_warm_reading.json`, generation08 `warm_reading.json`); the five IDs belong to `p2_64_warm_reading.json` under the superseded `dsv4-h200-p2-64-canonical-20261025-01-repeat01` |
| Ticket 19 localization claim narrowed to "no further round after final qualification" | D-WARM-TARGET (`requirements.md:435-451`) and D-WARM-TWO-FIX / D-WARM-EP-DEFER (`:406-433`) are cited in place, so the earlier rounds and their verdicts stay visible |
| Ticket 17 Status: the closed H800 32 P0 case no longer holds current priority | D-H800-CLOSED; the sentence now records when that priority lapsed |
| Ticket 17 Milestones: per-instance dispositions added, 9 H200-established boxes ticked, the full-model effective-runtime box left open with a stated reason, the two H800 `P0 remaining` boxes left unchecked | Evidence recomputed for this correction, listed in section 7 |
| `plan.md`, `summary.md`, `progress.md` titles: "H800 follows" -> "H800 closed" | D-H800-CLOSED |

## 7. Evidence recomputed for the ticket 17 milestone dispositions

All five canonical workload files under `data/processed_traces/canonical_fixed_6144_1024_req128_qps1/` rehash to the
`inputs_sha256` entries of both runs' `run_manifest.json`, so both admitted cases consumed the checked-in artifact
itself rather than a copy:

| File | sha256 | Match |
| --- | --- | --- |
| `trace.csv` | `2d557d7d...3c49ec2` | MATCH |
| `request_ids.json` | `318cd1ed...8b2511f6` | MATCH |
| `warmup_pool.csv` | `55a970c5...5038b99e` | MATCH |
| `manifest.json` | `5c3acb69...c78c55d3` | MATCH |
| `SHA256SUMS` | `594c3cb6...623e013b9` | MATCH |

From each case's retained `run/client_requests.jsonl` (432 rows per case):

| Quantity | P2-H200-32 | P2-H200-64 |
| --- | --- | --- |
| segment x repeat | 3 x (16 warmup + 128 formal) | 3 x (16 warmup + 128 formal) |
| formal HTTP status / finish reason / decode tokens | `{200}` / `{length}` / `{1024}` | `{200}` / `{length}` / `{1024}` |
| formal tokens per repetition | 786,432 in / 131,072 out | 786,432 in / 131,072 out |
| per-repeat arrival offsets (floor) | exactly `0..127` s | exactly `0..127` s |
| maximum send lateness | 0.002006 s | 0.002036 s |
| warmup / formal transport ID overlap | 0 | 0 |
| idle gap between repetitions | 1.339 s, 1.293 s | 1.467 s, 1.580 s |

`template_request_id` carries the stable template and `request_id` the repetition-qualified transport ID, which is the
rule the milestone states. From `run_check_status.groundtruth_clean.json`, both cases return PASS for `checkout`,
`inputs_unchanged`, `harness_pinned_and_client_rows` (128 client rows per repeat), `requests_and_idle_lanes` (7
boundaries over all 2 and all 4 DP lanes), `native_workers` (4 and 8 distinct nodes, both roles, every configured DP
lane), `rank_runtime_graph` (32 and 64 physical ranks, 6 phases per rank, per-rank eager prefill forwards and decode
PIECEWISE/FULL replay counts), `native_lifecycle` and `terminal_status`. `platform_terminal_receipt.json` records
`preemptible_label` `no` for the 32-GPU case and `yes` for the 64-GPU case, matching the required resource classes.

## Verification limits

- Observations above are recomputations of retained local artifacts. No native or GPU run was repeated, so native
  runtime behaviour beyond what those artifacts record is outside this report.
- `test_v4_attention_coverage.py` is excluded from the order-independence pair only because of its 150 s runtime; it
  passes standalone.
- The three README documentation-contract failures are unchanged, not fixed; they belong to the release-docs owner.
- The 15 remaining collection errors in `tests/unit` need packages the minimal environment omits (I-018), so those
  cases stay unknown rather than passing.
- Full-model effective-runtime evidence in the ticket 01 sense was never collected for the H200 cases and is excluded
  by the dummy-weight policy; its milestone box stays open with that reason recorded.
