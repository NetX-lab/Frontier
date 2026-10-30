**1. Assignment and outcome**

Outcome: **partial**. The opt-in DP dummy host run-ahead scheduler, config, A10 integration, documentation and CPU verification are complete within the writable scope. One confirmed placement-policy hook defect under K remains in a file outside that scope. Claude retains review, acceptance, commits, task records and the lead's 77-case fidelity matrix.

Working root: `/data/ycfeng/Frontier/.worktrees/b26-dp-dummy-run-ahead`. Branch-point HEAD: `627253d5c71b313888a34891ad15bf922b278762`. Read-only base: `/data/ycfeng/Frontier/.worktrees/b26-base`. Scratch: `/data/ycfeng/tmp/codex-implement/7218b11a-e3b0-49e9-a533-3c76428f66c2/b26`. Interpreter: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6).

Read evidence: working-root AGENTS.md (release, parallel semantics, size and starting-test gates); main-tree requirements 42/43, A10 report sections 3.1–3.5, `h3o_operator_costs/a10/wave/` depth/FIFO evidence and I-005; pinned native `DPEngineCoreProc.run_busy_loop`/`step_with_batch_queue` and GPU model-runner dummy/DP dispatch paths. Section 3.5 supersedes the older reading of host dummy time. I-005 concerns lost native START_DP_WAVE observer records; the complete forward records supplied the measured queue-depth diagnosis.

Settled behavior: `max_forwards_in_flight=None` follows the legacy path. An exact int ≥1 opts in only on MoE attention-DP dummy lanes with PP1. K counts real and dummy device forwards. Host iteration n returns after n−K ends and every peer forms n; existing batch-output ownership, admission, 32-host-iteration wave sync and connector rules remain. Ordinary implementation choices: the specified child mixin after cleanup, one FIFO for unfinished device forwards, and the existing DP release-event path for peer wakes.

Writes are confined to the changed-file table below: permitted config/scheduler/events, integration files, the A10 unit file, requested docs and task-local A10/report artifacts. Other allowed paths needed no edits. Stage execution context, pre-MoE rooms, base cluster scheduler and ClusterBatchEndEvent are unchanged. Pre-existing `.cortexkit/`, `.pi/` and `remaining_measurements/` were preserved. No GPU, network, installations, Git mutations, writes under data/, or changes to external A10 jobs/roots were performed.

**2. Changes and rule mapping**

| Changed file | Lines | Behavior / reason |
|---|---:|---|
| `frontier/config/replica_scheduler_config.py` | 768 | Optional K, help/omit_if_none metadata, exact-int validation (108, 348). |
| `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py` | 1692 | Cleanup/mixin extraction; host schedule guard, legacy-only output idle-clock update (150, 525, 590). |
| `frontier/scheduler/replica_scheduler/vllm_dp_busy_loop.py` | 381 | NEW: single device FIFO and extracted busy loop; depth/peer releases, host iteration ends, wave-start wake and zero-cost jump (28, 94, 105, 117, 145, 218, 307, 337, 350). |
| `frontier/events/batch_stage_end_event.py` | 237 | Final-stage real device release before output pop; legacy-only dummy-end READ report (169, 190). |
| `frontier/events/replica_schedule_event.py` | 255 | Existing peer release path plus host follow-up and inline host-end NIXL reports (100). |
| `tests/integration/test_vllm_dp_run_ahead_runtime.py` | 393 | NEW: 22 validation/runtime/base-equivalence/depth/lockstep/wave/pairing/READ/zero-cost cases. |
| `tests/integration/run_dsv4_structural.py` | 498 | Child module in source identity (57); stale runner/cache-zero metadata validator correction (200, 208). |
| `tests/unit/test_dsv4_a10_tools.py` | 248 | P2-32/P2-64 assertions for both-role K and receipt override (45). |
| `docs/general/glm53-dsv4-vllm028.md` | 354 | One row next to max_concurrent_batches/V1 queue admission (157); README unchanged. |
| `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_driver.py` | 203 | Measured constant 19 citing §3.5 observation 4 and exp-1030-144848-164721; base scheduler field and --without-run-ahead (21, 29, 197). |
| `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_observer.py` | 311 | Formation time/index, request IDs and queued-ahead count; preserves per-stage-event identity (166). |
| `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b26/dp_iteration_methods.diff` | 149 | NEW: corresponding DP method diff against branch-point HEAD. |
| `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b26/report.md` | 441 | NEW: durable self-contained execution report; also returned verbatim to Claude. |

The main scheduler started at 1,918 lines. Cleanup removed four stale scheduling-policy comments and unused `math`/`DummyForwardBatch` imports. Feature state/methods would still have crossed the limit, so the DP busy loop moved to the assignment's named child. Final main/child counts are 1,692/381. `VLLM_DP_SYNC_INTERVAL` remains available from the main module through an intentional `__all__` re-export.

| Rule | Implementation and evidence |
|---|---|
| (a) | K dummy formation records device work and starts the connector step without waiting for GPU end. ReplicaScheduleEvent supplies host follow-ups. Idle-host and actual-runtime tests exercise it. |
| (b) | `_host_forward_wait` checks FIFO length >K after formation. Real/dummy device ends remove its oldest record. A blocked launch has already formed its forward: K+1 records can temporarily exist while that launch is held; every host iteration end has ≤K unfinished forwards, following the stated n−K rule. K=1/3/19 tests reach exactly K at host returns and verify pacing by device ends. Real capacity releases before output pop. |
| (c) | The same guard compares `_batch_creation_counter` with `_dp_group()` peers. The lane completing the cohort posts peer IDs in `_dp_sync_released_lane_ids`; consume_dp_sync_release wakes them. A batch-held lane stops idle peers one forward ahead; requests arriving during depth/peer waits enter the following forward. |
| (c2) | Starting a wave posts peer wake IDs at that time under K. None keeps the existing room wake. The wave-start test inspects the events. |
| (d) | Blocking batch, admission limit, in-flight output IDs, held outputs and dummy-after-empty-pop paths keep their existing logic. The new guards add to them. Batch-block/device-release and existing queue/DP suites exercise these paths. |
| (e) | An open K iteration counts and sets its end time only when the host returns after applicable waits. The existing 32-step sync can stop a wave with queued GPU work. The wave-tail test observes 19 dummy forwards left and a later prefill joining behind them. |
| (f) | No stage-context/rendezvous changes. Actual DES tests at K=1/3/19 verify per-lane formation/device order and equal provisional/cohort identities for every shared EP room. A10 separately checks the preceding device-end gap. |
| (g) | `next_pass_start` remains the sole READ-wire jump owner and runs at host iteration end. Fast host return does not prove zero GPU cost: all lanes must witness device completion at that same formation time. Tests verify the zero-cost jump and the positive-cost counterexample. |
| (h) | One DeviceForward FIFO owns unfinished real/dummy forwards in formation order. The dummy count is derived under K; the legacy scalar remains only with None, preserving old consumers/private-field tests. The existing batch-output queue represents output ownership, rather than a duplicate device tracker. |
| (i) | `_last_iteration_end` follows host return under K. Dummy-found READs report then, before the next pass. All K READ posting uses formation time; real-stage invocation skips the already-started host connector pass. The DECODE/engine_step test finds a READ at 4 ms while depth-blocked and reports it at 6 ms with that idle-clock value. Host idle excludes queued dummies and includes batch/depth/peer/sync waits. |
| (j) | Direct dummy-count/end readers were inspected: busy-loop guard/count/idle methods and BatchStageEndEvent. ReplicaScheduleEvent consumes releases; the observer already identifies every stage event and now records formation depth. Row/end counts agree. The placement policy's indirect GPU-group report-key reader has the defect documented below and remains unfinished. |

A test-only compatibility correction was necessary for the structural acceptance commands. The unchanged predictor now emits runner provenance `runner scope kernel and memcpy medians; dummy mode uses its configured duration`, while the old validator demanded the substring `synthetic`; its cache-zero workload also no longer has `native_h2d_cost` (H2D writes have a separate runner scope). The read-only base validator reproduces the provenance failure on the saved artifacts. Corrected only the permitted integration validator: duration, physical extent, assignment order, native-store-traffic and embedding-boundary checks remain. All 20 cache-evidence validator tests pass, including corruption rejection. Simulator prices were not changed.

**3. Verification**

Common environment (A10 uses only checkout + checkout/tests/analysis in PYTHONPATH, matching run_a10.sh):

```bash
REPO=/data/ycfeng/Frontier/.worktrees/b26-dp-dummy-run-ahead
B26=/data/ycfeng/tmp/codex-implement/7218b11a-e3b0-49e9-a533-3c76428f66c2/b26
PY=/data/ycfeng/envs/frontier-py310/bin/python
export PYTHONPATH=$REPO:$REPO/tests/unit:$REPO/tests/comparison/calibration:$REPO/tests/analysis
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=$B26/tmp FRONTIER_TMP_ROOT=$B26/tmp
```

Tests/analysis: `ulimit -v 31457280`; simulations: `ulimit -v 62914560`. Every process command had a timeout. Tests run from `$B26/check-cwd`, which has a read-only data symlink and a copied prefix fixture, avoiding default writes into the checkout. Outputs and caches are in scratch.

Acceptance 1/2 and named pytest checks: executed `timeout 600s zsh $B26/verify.sh`, yielding **320 PASS / 1 known FAIL** in 178.58 s. Required files plus new tests contribute **300 PASS / 1 FAIL**; the extra cache-evidence validator contributes 20 PASS. Log/JUnit: `$B26/handoff-real-final-tests.log`, `$B26/handoff-real-final.xml`. The exact executable script follows:

```bash
#!/bin/zsh
ulimit -v 31457280
REPO=/data/ycfeng/Frontier/.worktrees/b26-dp-dummy-run-ahead
B26=/data/ycfeng/tmp/codex-implement/7218b11a-e3b0-49e9-a533-3c76428f66c2/b26
export PYTHONPATH=$REPO:$REPO/tests/unit:$REPO/tests/comparison/calibration:$REPO/tests/analysis
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=$B26/tmp FRONTIER_TMP_ROOT=$B26/tmp
cd $B26/check-cwd
files=(
 tests/integration/test_vllm_dp_dummy_pass_runtime.py
 tests/integration/test_vllm_dp_placement_runtime.py
 tests/unit/test_dp_dummy_forward_pricing.py
 tests/unit/test_vllm_v1_engine_idle_time.py
 tests/unit/test_dsv4_dp_sync.py
 tests/unit/test_forward_sync_state.py
 tests/unit/test_nixl_read_completion.py
 tests/unit/test_dsv4_structural_execution.py
 tests/unit/test_cluster_scheduler_dp_lanes.py
 tests/unit/test_vllm_dp_load_balancer.py
 tests/unit/test_monolithic_mixed_forward_sync.py
 tests/unit/test_dsv4_a10_tools.py
 tests/integration/test_vllm_dp_run_ahead_runtime.py
 tests/unit/test_dsv4_cache_zero_driver.py
)
paths=()
for file in $files; do paths+=($REPO/$file); done
/data/ycfeng/envs/frontier-py310/bin/python -m pytest $paths -q -p no:cacheprovider --basetemp=$B26/pytest-handoff-real-final --junitxml=$B26/handoff-real-final.xml
```

| Test file | PASS | FAIL |
|---|---:|---:|
| `test_vllm_dp_dummy_pass_runtime.py` | 14 | 0 |
| `test_vllm_dp_placement_runtime.py` | 18 | 1 |
| `test_dp_dummy_forward_pricing.py` | 14 | 0 |
| `test_vllm_v1_engine_idle_time.py` | 1 | 0 |
| `test_dsv4_dp_sync.py` | 27 | 0 |
| `test_forward_sync_state.py` | 11 | 0 |
| `test_nixl_read_completion.py` | 9 | 0 |
| `test_dsv4_structural_execution.py` | 35 | 0 |
| `test_cluster_scheduler_dp_lanes.py` | 23 | 0 |
| `test_vllm_dp_load_balancer.py` | 89 | 0 |
| `test_monolithic_mixed_forward_sync.py` | 25 | 0 |
| `test_dsv4_a10_tools.py` | 12 | 0 |
| `test_vllm_dp_run_ahead_runtime.py` | 22 | 0 |
| `test_dsv4_cache_zero_driver.py` | 20 | 0 |

Known failure: `tests/integration/test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward`. Its fixture expects lane 2's first forward to be dummy; actual lane 2 forms real work. Reproduced in base with K unset (`base-known-tests.log`). The test was not edited or treated as a new regression.

The new exact-base test runs the same dummy-DP scenario with separate base/current production imports and compares every lane/index/start/end tuple for exact equality: PASS. The lead's 77-case fidelity matrix was NOT_RUN by this executor.

Acceptance 3: the P2-32 and P2-64 built-config tests assert K=19 in the base field, both PREFILL/DECODE schedulers and receipt changes. The finished K replay config/receipt records 19; the None replay uses `--without-run-ahead`.

Acceptance 5, structural/smoke/release starting checks: from `$REPO`, using the common environment and appropriate memory limit, executed:

```bash
timeout 1200s "$PY" tests/integration/run_dsv4_structural.py --dp 4 --case offline --output-root "$B26/real-final-structural-offline"
timeout 1200s "$PY" tests/integration/run_dsv4_structural.py --dp 4 --case online --output-root "$B26/real-final-structural-online"
PYTHON_BIN="$PY" METRICS_OUTPUT_DIR="$B26/smoke-handoff" timeout 300s bash examples/architecture/run_dummy_smoke_matrix.sh -- --metrics_config_cache_dir "$B26/smoke-cache-handoff"
timeout 60s "$PY" -m pytest tests/unit/test_pdd_public_surface_docs.py tests/unit/test_examples_pdd_scripts.py tests/e2e/test_colocation_examples_layout.py tests/unit/test_e2e_gate.py -q -p no:cacheprovider --basetemp="$B26/pytest-release-handoff" --junitxml="$B26/release-handoff.xml"
```

Offline and online structural: PASS, exit 0 each. Each completes 4 requests and checks 12 stage rows, 1,376 EP waves, 393,284 trace rows, 4,128 real/39,904 idle EP-lane rows, 8 cache-zero invocations and 44 assignments. Maximum stage/phase conservation errors: 2.6645352591003757e-15 / 6.938893903907228e-18 ms. Source hashes are equal before/after. Evidence: `real-final-structural-{offline,online}.log` and their report.json files.

Smoke: PASS, exit 0, all 12 dense/MoE × co-location/PDD/PD-AF × offline/online cases (`smoke-handoff.log`).

Release starting pytest: **57 PASS / 3 pre-existing FAIL** in 2.35 s (`release-handoff.log`, `release-handoff.xml`). All three are README assertions in test_pdd_public_surface_docs.py:

- `test_top_level_docs_advertise_supported_pdd_without_upcoming_claims`
- `test_public_docs_require_sequential_pdd_release_mode`
- `test_public_docs_state_internal_parallel_pdd_boundary_and_current_evidence`

The same three fail in base. Executed base reproduction of that docs file plus the known placement testcase returned 2 PASS / 4 FAIL (`base-known-tests.log`). The archived log names the tests and results; the complete earlier shell invocation was not retained. README edits are expressly outside this assignment.

Acceptance 4, non-dummy V4 A10 short replay: input is the first 32 canonical requests, keeping arrival/prefill fields and setting num_decode_tokens=2. Scratch trace: `$B26/trace32.csv`; SHA-256 **`88f0b339e9a696858729fbf5edc46408b537903766fe8b34ee9effd7fb90e4a3`**.

Before each final run, executed `cp -a /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_64_warm_g1hs_r2/cache <run-root>/cache`. This is a byte copy, not hard links. Source: 1,758 files / 324,536,393 bytes. Each copy has the same file count and total bytes, no size differences or removed names, and zero source/run shared inodes. Final cache additions and receipts are below (`cache_copy_check.json`, `short_check.json`).

Final commands (launched independently with checkout CWD, simulation memory limit and the A10 environment above):

```bash
# CWD /data/ycfeng/Frontier/.worktrees/b26-base; PYTHONPATH=base:base/tests/analysis
timeout 7200s "$PY" -u task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_driver.py p2_64 "$B26/final-base" --trace "$B26/trace32.csv"
# CWD $REPO; PYTHONPATH=$REPO:$REPO/tests/analysis
timeout 7200s "$PY" -u task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_driver.py p2_64 "$B26/final-unset" --trace "$B26/trace32.csv" --without-run-ahead
timeout 7200s "$PY" -u task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_driver.py p2_64 "$B26/final-k19-r3" --trace "$B26/trace32.csv"
# Analysis memory limit:
timeout 30s "$PY" "$B26/summarize_a10.py"
```

| Baseline artifact | Base SHA-256 | K-unset SHA-256 | Byte equality |
|---|---|---|---|
| `request_metrics.csv` | `df160df153c4100cebefb533239701a1cbce2b5c4f944abe8f5d93870905d1ae` | `df160df153c4100cebefb533239701a1cbce2b5c4f944abe8f5d93870905d1ae` | PASS |
| `system_metrics.json` | `4291f30a09e2581555d894c19bc8028ea16e5c3e62a6e35b609a27586a32193c` | `4291f30a09e2581555d894c19bc8028ea16e5c3e62a6e35b609a27586a32193c` | PASS |

Direct Python read_bytes() equality also passed for the 24,386-byte request CSV and the 5,671-byte system JSON, recorded in baseline_direct_bytes.json.


| Run | Status | Observer rows / stage ends / stale ends | Cache files / added | Simulator wall s |
|---|---|---|---|---:|
| `final-base` | FINISHED | 6144 / 6144 / 0 | 1758 / 0 | 2637.733 |
| `final-unset` | FINISHED | 6144 / 6144 / 0 | 1758 / 0 | 2660.410 |
| `final-k19-r3` | FINISHED | 8064 / 8064 / 0 | 1758 / 0 | 3531.826 |

All finished receipts identify non-dummy V4 predictors and no bound diagnostic step-cost table. Observer rows equal non-stale stage ends; cache names have no additions. Arrival comes from the trace; stage time/depth comes from observer rows joined by request ID to request_metrics.csv. Prefill side is arrival→P device end; TTFT is the existing CSV millisecond contract. In these outputs it equals prefill-side time: Request.ttft (frontier/entities/request.py:953) returns prefill_completed_at minus arrived_at, and MetricsStore._get_ttft_export_value (frontier/metrics/metrics_store.py:2981) exports it unchanged. The same behavior is in the read-only base and matches the AGENTS.md canonical contract. These values exclude the D-side remainder discussed in the motivating A10 report; comparing them directly to its 741.2 ms client-visible TTFT would mix metric contracts. Gap is P start minus the immediately preceding forward end on that lane, including dummies. Full precision, formation indices and predecessor ends are retained in short_check.json.

**K unset: `/data/ycfeng/tmp/codex-implement/7218b11a-e3b0-49e9-a533-3c76428f66c2/b26/final-unset`**

| Request | Arrival s | P lane | Formed s | P start s | P end s | Ahead | Gap µs | Prefill side ms | TTFT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0 | 0.000000000 | 0.000000000 | 0.273940305 | 0 | — | 273.940305 | 273.940305 |
| 1 | 1.0 | 1 | 1.005342584 | 1.005342584 | 1.282123504 | 0 | 0.000000 | 282.123504 | 282.123504 |
| 2 | 2.0 | 2 | 2.000000000 | 2.000000000 | 2.276322918 | 0 | 717876.496061 | 276.322918 | 276.322918 |
| 3 | 3.0 | 3 | 3.007725197 | 3.007725197 | 3.281574001 | 0 | 0.000000 | 281.574001 | 281.574001 |
| 4 | 4.0 | 0 | 4.000000000 | 4.000000000 | 4.273940305 | 0 | 718425.999101 | 273.940305 | 273.940305 |
| 5 | 5.0 | 1 | 5.005342584 | 5.005342584 | 5.282123504 | 0 | 0.000000 | 282.123504 | 282.123504 |
| 6 | 6.0 | 2 | 6.000000000 | 6.000000000 | 6.276322918 | 0 | 717876.496061 | 276.322918 | 276.322918 |
| 7 | 7.0 | 3 | 7.007725197 | 7.007725197 | 7.281574001 | 0 | 0.000000 | 281.574001 | 281.574001 |
| 8 | 8.0 | 0 | 8.000000000 | 8.000000000 | 8.273940305 | 0 | 718425.999101 | 273.940305 | 273.940305 |
| 9 | 9.0 | 1 | 9.005342584 | 9.005342584 | 9.282123504 | 0 | 0.000000 | 282.123504 | 282.123504 |
| 10 | 10.0 | 2 | 10.000000000 | 10.000000000 | 10.276322918 | 0 | 717876.496061 | 276.322918 | 276.322918 |
| 11 | 11.0 | 3 | 11.007725197 | 11.007725197 | 11.281574001 | 0 | 0.000000 | 281.574001 | 281.574001 |
| 12 | 12.0 | 0 | 12.000000000 | 12.000000000 | 12.273940305 | 0 | 718425.999101 | 273.940305 | 273.940305 |
| 13 | 13.0 | 1 | 13.005342584 | 13.005342584 | 13.282123504 | 0 | 0.000000 | 282.123504 | 282.123504 |
| 14 | 14.0 | 2 | 14.000000000 | 14.000000000 | 14.276322918 | 0 | 717876.496061 | 276.322918 | 276.322918 |
| 15 | 15.0 | 3 | 15.007725197 | 15.007725197 | 15.281574001 | 0 | 0.000000 | 281.574001 | 281.574001 |
| 16 | 16.0 | 0 | 16.000000000 | 16.000000000 | 16.273940305 | 0 | 718425.999101 | 273.940305 | 273.940305 |
| 17 | 17.0 | 1 | 17.005342584 | 17.005342584 | 17.282123504 | 0 | 0.000000 | 282.123504 | 282.123504 |
| 18 | 18.0 | 2 | 18.000000000 | 18.000000000 | 18.276322918 | 0 | 717876.496065 | 276.322918 | 276.322918 |
| 19 | 19.0 | 3 | 19.007725197 | 19.007725197 | 19.281574001 | 0 | 0.000000 | 281.574001 | 281.574001 |
| 20 | 20.0 | 0 | 20.000000000 | 20.000000000 | 20.273940305 | 0 | 718425.999106 | 273.940305 | 273.940305 |
| 21 | 21.0 | 1 | 21.005342584 | 21.005342584 | 21.282123504 | 0 | 0.000000 | 282.123504 | 282.123504 |
| 22 | 22.0 | 2 | 22.000000000 | 22.000000000 | 22.276322918 | 0 | 717876.496065 | 276.322918 | 276.322918 |
| 23 | 23.0 | 3 | 23.007725197 | 23.007725197 | 23.281574001 | 0 | 0.000000 | 281.574001 | 281.574001 |
| 24 | 24.0 | 0 | 24.000000000 | 24.000000000 | 24.273940305 | 0 | 718425.999106 | 273.940305 | 273.940305 |
| 25 | 25.0 | 1 | 25.005342584 | 25.005342584 | 25.282123504 | 0 | 0.000000 | 282.123504 | 282.123504 |
| 26 | 26.0 | 2 | 26.000000000 | 26.000000000 | 26.276322918 | 0 | 717876.496065 | 276.322918 | 276.322918 |
| 27 | 27.0 | 3 | 27.007725197 | 27.007725197 | 27.281574001 | 0 | 0.000000 | 281.574001 | 281.574001 |
| 28 | 28.0 | 0 | 28.000000000 | 28.000000000 | 28.273940305 | 0 | 718425.999106 | 273.940305 | 273.940305 |
| 29 | 29.0 | 1 | 29.005342584 | 29.005342584 | 29.282123504 | 0 | 0.000000 | 282.123504 | 282.123504 |
| 30 | 30.0 | 2 | 30.000000000 | 30.000000000 | 30.276322918 | 0 | 717876.496065 | 276.322918 | 276.322918 |
| 31 | 31.0 | 3 | 31.007725197 | 31.007725197 | 31.281574001 | 0 | 0.000000 | 281.574001 | 281.574001 |

Prefill-side mean **278.490182 ms**, range **273.940305–282.123504 ms**. TTFT mean **278.490182 ms**, range **273.940305–282.123504 ms**. Queued prefills: 0; maximum absolute preceding-end gap: 0.000000000 µs; ≤1 µs check: PASS.

Prefill-side series (request order, ms): 273.940, 282.124, 276.323, 281.574, 273.940, 282.124, 276.323, 281.574, 273.940, 282.124, 276.323, 281.574, 273.940, 282.124, 276.323, 281.574, 273.940, 282.124, 276.323, 281.574, 273.940, 282.124, 276.323, 281.574, 273.940, 282.124, 276.323, 281.574, 273.940, 282.124, 276.323, 281.574.

**K=19: `/data/ycfeng/tmp/codex-implement/7218b11a-e3b0-49e9-a533-3c76428f66c2/b26/final-k19-r3`**

| Request | Arrival s | P lane | Formed s | P start s | P end s | Ahead | Gap µs | Prefill side ms | TTFT ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0 | 0.000000000 | 0.000000000 | 0.273940305 | 0 | — | 273.940305 | 273.940305 |
| 1 | 1.0 | 1 | 1.000000000 | 1.029722660 | 1.306503580 | 2 | 0.000000 | 306.503580 | 306.503580 |
| 2 | 2.0 | 2 | 2.000000000 | 2.062285935 | 2.338608853 | 3 | 0.000000 | 338.608853 | 338.608853 |
| 3 | 3.0 | 3 | 3.000000000 | 3.094391208 | 3.368240012 | 4 | 0.000000 | 368.240012 | 368.240012 |
| 4 | 4.0 | 0 | 4.000000000 | 4.124022366 | 4.397962671 | 6 | 0.000000 | 397.962671 | 397.962671 |
| 5 | 5.0 | 1 | 5.000000000 | 5.153745026 | 5.430525946 | 7 | 0.000000 | 430.525946 | 430.525946 |
| 6 | 6.0 | 2 | 6.000000000 | 6.186308301 | 6.462631220 | 8 | 0.000000 | 462.631220 | 462.631220 |
| 7 | 7.0 | 3 | 7.000000000 | 7.218413575 | 7.492262378 | 9 | 0.000000 | 492.262378 | 492.262378 |
| 8 | 8.0 | 0 | 8.000000000 | 8.248044733 | 8.521985038 | 11 | 0.000000 | 521.985038 | 521.985038 |
| 9 | 9.0 | 1 | 9.000000000 | 9.277767392 | 9.554548313 | 12 | 0.000000 | 554.548313 | 554.548313 |
| 10 | 10.0 | 2 | 10.000000000 | 10.310330668 | 10.586653586 | 13 | 0.000000 | 586.653586 | 586.653586 |
| 11 | 11.0 | 3 | 11.000000000 | 11.342435941 | 11.616284745 | 15 | 0.000000 | 616.284745 | 616.284745 |
| 12 | 12.0 | 0 | 12.000000000 | 12.372067099 | 12.646007404 | 16 | 0.000000 | 646.007404 | 646.007404 |
| 13 | 13.0 | 1 | 13.000000000 | 13.401789759 | 13.678570679 | 17 | 0.000000 | 678.570679 | 678.570679 |
| 14 | 14.0 | 2 | 14.000000000 | 14.434353034 | 14.710675953 | 18 | 0.000000 | 710.675953 | 710.675953 |
| 15 | 15.0 | 3 | 15.003236864 | 15.466458307 | 15.740307111 | 19 | 0.000000 | 740.307111 | 740.307111 |
| 16 | 16.0 | 0 | 16.008487947 | 16.471709390 | 16.745649695 | 19 | 0.000000 | 745.649695 | 745.649695 |
| 17 | 17.0 | 1 | 17.000000000 | 17.000000000 | 17.276780920 | 0 | 254350.305444 | 276.780920 | 276.780920 |
| 18 | 18.0 | 2 | 18.000000000 | 18.032563275 | 18.308886194 | 2 | 0.000000 | 308.886194 | 308.886194 |
| 19 | 19.0 | 3 | 19.000000000 | 19.064668548 | 19.338517352 | 3 | 0.000000 | 338.517352 | 338.517352 |
| 20 | 20.0 | 0 | 20.000000000 | 20.094299707 | 20.368240012 | 4 | 0.000000 | 368.240012 | 368.240012 |
| 21 | 21.0 | 1 | 21.000000000 | 21.124022366 | 21.400803287 | 6 | 0.000000 | 400.803287 | 400.803287 |
| 22 | 22.0 | 2 | 22.000000000 | 22.156585642 | 22.432908560 | 7 | 0.000000 | 432.908560 | 432.908560 |
| 23 | 23.0 | 3 | 23.000000000 | 23.188690915 | 23.462539718 | 8 | 0.000000 | 462.539718 | 462.539718 |
| 24 | 24.0 | 0 | 24.000000000 | 24.218322073 | 24.492262378 | 9 | 0.000000 | 492.262378 | 492.262378 |
| 25 | 25.0 | 1 | 25.000000000 | 25.248044733 | 25.524825653 | 11 | 0.000000 | 524.825653 | 524.825653 |
| 26 | 26.0 | 2 | 26.000000000 | 26.280608008 | 26.556930927 | 12 | 0.000000 | 556.930927 | 556.930927 |
| 27 | 27.0 | 3 | 27.000000000 | 27.312713281 | 27.586562085 | 13 | 0.000000 | 586.562085 | 586.562085 |
| 28 | 28.0 | 0 | 28.000000000 | 28.342344440 | 28.616284744 | 15 | 0.000000 | 616.284744 | 616.284744 |
| 29 | 29.0 | 1 | 29.000000000 | 29.372067099 | 29.648848020 | 16 | 0.000000 | 648.848020 | 648.848020 |
| 30 | 30.0 | 2 | 30.000000000 | 30.404630375 | 30.680953293 | 17 | 0.000000 | 680.953293 | 680.953293 |
| 31 | 31.0 | 3 | 31.000000000 | 31.436735648 | 31.710584451 | 18 | 0.000000 | 710.584451 | 710.584451 |

Prefill-side mean **508.665159 ms**, range **273.940305–745.649695 ms**. TTFT mean **508.665159 ms**, range **273.940305–745.649695 ms**. Queued prefills: 30; maximum absolute preceding-end gap: 0.000000000 µs; ≤1 µs check: PASS.

Prefill-side series (request order, ms): 273.940, 306.504, 338.609, 368.240, 397.963, 430.526, 462.631, 492.262, 521.985, 554.548, 586.654, 616.285, 646.007, 678.571, 710.676, 740.307, 745.650, 276.781, 308.886, 338.517, 368.240, 400.803, 432.909, 462.540, 492.262, 524.826, 556.931, 586.562, 616.285, 648.848, 680.953, 710.584.

The 32-request series demonstrates the queue effect; its mean is not the 128-request mean predicted by FIFO. Claude should compare the full replay against the ~509–515 ms prediction.


Acceptance 6: all changed files are ≤2,000 lines. Executed `timeout 30s` with `git diff --check`: PASS, no output. Python 3.10 `/data/ycfeng/tmp/unused_imports.py` scanned all ten changed Python files: PASS, every unused list empty. Exact scanner invocation (CWD `$REPO`, analysis limit) was:

```bash
timeout 30s "$PY" /data/ycfeng/tmp/unused_imports.py frontier/config/replica_scheduler_config.py frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py frontier/scheduler/replica_scheduler/vllm_dp_busy_loop.py frontier/events/batch_stage_end_event.py frontier/events/replica_schedule_event.py tests/integration/test_vllm_dp_run_ahead_runtime.py tests/integration/run_dsv4_structural.py tests/unit/test_dsv4_a10_tools.py task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_driver.py task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_observer.py
```

`VLLM_DP_SYNC_INTERVAL` is an intentional main-module __all__ export.

Earlier attempts: first structural executions failed C17 because source cleanup occurred during the run; second executions exposed the stale validator. Reassessment against the unchanged predictor/base validator justified the test correction; final executions pass. A combined 180-s pytest command timed out near completion; the 600-s rerun supplies complete counts. Earlier A10 runs had insufficient timeouts, and K-only fixes to host READ reporting, zero-cost detection and real-forward READ posting required superseding scratch runs. Only exact scratch-owned A10 driver processes were stopped; external A10 jobs/roots were not touched. Final base/unset runs loaded before the final K-only connector correction; the None path is unchanged and their full byte comparison checks it. Superseded logs/configurations and lightweight diagnostics were retained; 2,114,289,397 bytes of the first failed structural runs' op/EP traces were removed from scratch to stay below 10 GB (`scratch_cleanup.json`). They are not final evidence. Scratch footprint remains below 10 GB.

**4. Remaining work and decisions for Claude**

Required additional writable path: **`/data/ycfeng/Frontier/.worktrees/b26-dp-dummy-run-ahead/frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py`**. This owning file is absent from the execution context/allowed paths, so it was not edited.

Observed evidence: `_next_report_key` (167) reads `stage_execution_context.joinable_forward_group_id`, which follows GPU progress. A bounded real-runtime probe produced eight admissions with None and zero host/report mismatches, versus seven admissions with K=19 and two mismatches. At 0.12400000000000005 s on lane 1, host formation index=23, report key=4, GPU group=4. At 0.6899999999999991 s on lane 0, formation index=25, report key=24. Evidence: `/data/ycfeng/tmp/codex-implement/7218b11a-e3b0-49e9-a533-3c76428f66c2/b26/placement-key-probe.json` and `.log`. The reader audit used `timeout 30s rg -n "_dummy_forwards_in_flight|on_dummy_forward_end" --glob "*.py" frontier tests task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10`. Remaining direct readers are legacy placement/idle-time/NIXL test fixtures, which retain their None-path scalar behavior. The runtime end callback under K means a host-wait release; the derived count still means unfinished dummy device parts.

Inference: report keys no longer name the native host iteration and can affect ordering/stale-load handling. Request-metric impact was not quantified; A10 uses round-robin placement and does not resolve this defect. Passing legacy load-balancer tests are not proof of K correctness.

Recommended action: authorize that file and derive K admission keys from the actual host formation identity, preserving None keys and held/completion-key ownership. Add a K runtime regression for successive admissions/completions and stale reports, then rerun placement/load-balancer checks. A stage-context semantic change or base-class indirection would obscure the correct owner and broaden the scope.

Remaining with Claude: reconcile the canonical CSV TTFT with the client-visible metric used in the motivating A10 report before full TTFT fidelity comparison; decide treatment of the known placement fixture/three README failures; rerun the 77-case fidelity matrix and full P2-32/P2-64 workload; compare the complete 128-request prefill-side series/mean against FIFO (~509–515 ms); review/accept or request changes, then commit/update records. Acceptance and commits are not claimed here.

**5. Review handoff**

Inspect the mixin extraction and method diff below; verify device-capacity release before output pop, counter lockstep/event reuse, host-end NIXL reports, zero-cost evidence and legacy gating. Review A10 field inheritance, measured-input comment, receipt override and observer batch/epoch identity. The structural validator adjustment is a separate test-only correction supported by the unchanged predictor and base failure.

Rerun `timeout 600s zsh /data/ycfeng/tmp/codex-implement/7218b11a-e3b0-49e9-a533-3c76428f66c2/b26/verify.sh`, both structural commands, release starting checks and smoke command after further edits. For A10, use fresh byte copies of the named warm cache and the recorded trace. Inspect raw receipts, gzip streams and short_check.json alongside rounded report values. Limits: PP>1 and non-dummy-DP lanes reject K; CPU host pacing implements the agreed depth/lockstep rule; lost native START_DP_WAVE records are not reconstructed; K placement report keys remain unfinished; the full-workload/77-case fidelity acceptance belongs to Claude.

**DP iteration/dummy-method diff**

The diff compares corresponding methods against the branch point after extraction. Constructor/FIFO helpers and event edits are in the ordinary source diff. Standalone artifact: `/data/ycfeng/Frontier/.worktrees/b26-dp-dummy-run-ahead/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b26/dp_iteration_methods.diff`.

```diff
--- base DP iteration/dummy methods
+++ vllm_dp_busy_loop.py
@@ -13,12 +13,27 @@
             return self._issue_dummy_pass(time)
         if self._iteration_open:
             self._iteration_open = False
+            if self._runs_dp_forwards_ahead:
+                self._last_iteration_end = time
+                self._report_nixl_reads_of_dummy_forward()
             if self._end_iteration():
                 return []
+            if self._runs_dp_forwards_ahead and self._nixl_reads_reported:
+                # ReplicaScheduleEvent applies these connector outputs inline
+                # before the next host pass can schedule their ready requests.
+                self._dp_iteration_resume_at = time
+                return []
+            if self._runs_dp_forwards_ahead and self._dp_wave_running:
+                resume_at = self.next_pass_start(time)
+                if resume_at > time:
+                    self._dp_iteration_resume_at = resume_at
+                    return []
         if not self._dp_wave_running and self.has_unfinished_requests:
             # An engine that receives a request starts a wave on every engine.
             for lane in self._dp_group():
                 lane._dp_wave_running = True
+                if self._runs_dp_forwards_ahead and lane is not self:
+                    self._dp_sync_released_lane_ids.append(lane._replica_local_id)
         batches: List[Batch] = []
         while True:
             formed = self._schedule_iteration(time)
@@ -40,6 +55,9 @@
                 elif self._dp_wave_running:
                     return self._issue_dummy_pass(time)
                 return batches
+            if self._runs_dp_forwards_ahead:
+                self._iteration_open = True
+                return batches
             if self._end_iteration():
                 self._clear_followup_polls()
                 return batches
@@ -50,6 +68,9 @@
         batches = self._start_iterations(super().on_schedule(time), time)
         self._iteration_running_limit = math.inf
         self._claim_nixl_reads(batches)
+        if self._runs_dp_forwards_ahead:
+            for batch in batches:
+                self.on_engine_step_start(time, batch)
         self._in_flight_batch_ids.extend(batch.id for batch in batches)
         return batches
 
@@ -70,15 +91,17 @@
                 if not wave_continues:
                     lane._dp_wave_running = False
                     lane._dp_wave_steps = 0
-            self._dp_sync_released_lane_ids = [
-                lane._replica_local_id for lane in group if lane is not self
-            ]
+            released = [lane._replica_local_id for lane in group if lane is not self]
+            if self._runs_dp_forwards_ahead:
+                self._dp_sync_released_lane_ids.extend(released)
+            else:
+                self._dp_sync_released_lane_ids = released
         return self._waits_at_dp_sync
 
     def consume_dp_sync_release(self) -> List[int]:
         """Lanes whose host this engine's latest pass released from the DP sync."""
         released, self._dp_sync_released_lane_ids = self._dp_sync_released_lane_ids, []
-        return released
+        return list(dict.fromkeys(released))
 
     def _dp_group(self) -> List["VLLMv1EngineReplicaScheduler"]:
         return [
@@ -97,7 +120,10 @@
             batch.engine_idle_time = (
                 math.inf if self._last_iteration_end is None else time - self._last_iteration_end
             )
-            self._last_iteration_end = time
+            if not self._runs_dp_forwards_ahead:
+                self._last_iteration_end = time
+            else:
+                self._record_device_forward(batch, time)
         return batches
 
     def _clear_followup_polls(self) -> None:
@@ -126,10 +152,11 @@
         )
 
     def engine_loop_idle(self) -> bool:
-        """Whether the engine waits for work: nothing in flight and no DP sync to finish."""
+        """Whether the host waits for work, excluding queued run-ahead dummies."""
         return (
             not self._in_flight_batch_ids
-            and not self._dummy_forwards_in_flight
+            and not self._host_forward_wait
+            and not (self._runs_dp_forwards_ahead and self._iteration_open)
             and not self._waits_at_dp_sync
         )
 
@@ -138,7 +165,8 @@
 
         `execute_dummy_batch` is one RPC to every worker of the engine. Each
         stage runs its part once its earlier work ends, with no PP transfer
-        between parts, and the engine waits for every part.
+        between parts. The legacy host waits for every part; run-ahead waits
+        only for its launch depth and DP peers.
         """
         dummy_forwards = []
         for stage_id in range(self._num_stages):
@@ -152,7 +180,11 @@
             self._assign_lane_identity(dummy_forward, self._batch_creation_counter)
             dummy_forwards.append(dummy_forward)
         self._batch_creation_counter += 1
-        self._dummy_forwards_in_flight = self._num_stages
+        if self._runs_dp_forwards_ahead:
+            self._record_device_forward(dummy_forwards[0], time)
+            self.on_engine_step_start(time, dummy_forwards[0])
+        else:
+            self._dummy_forwards_in_flight = self._num_stages
         self._dummy_pass_start = time
         self._iteration_open = True
         self._clear_followup_polls()
@@ -160,6 +192,10 @@
 
     def on_dummy_forward_end(self, time: float) -> bool:
         """Record one ended dummy forward; return whether the engine iterates again."""
+        if self._runs_dp_forwards_ahead:
+            forward = self._unfinished_forwards.popleft()
+            self._last_zero_cost_dummy_end = time if time == forward.formed_at else None
+            return self._iteration_open and not self._host_forward_wait
         self._dummy_forwards_in_flight -= 1
         if self._dummy_forwards_in_flight:
             return False
@@ -180,9 +216,15 @@
         does. An engine that receives work earlier wakes the others through
         its forward's first room.
         """
-        if time > self._dummy_pass_start:
+        if self._dummy_pass_start is None or time > self._dummy_pass_start:
             return time
         group = self._dp_group()
+        if self._runs_dp_forwards_ahead and not all(
+            lane._last_zero_cost_dummy_end == time for lane in group
+        ):
+            # A host return at formation says nothing about GPU cost. Only
+            # device ends at that same formation time witness a zero-cost group.
+            return time
         if not all(lane._only_posted_nixl_reads_unfinished for lane in group):
             return time
         return min(
```
