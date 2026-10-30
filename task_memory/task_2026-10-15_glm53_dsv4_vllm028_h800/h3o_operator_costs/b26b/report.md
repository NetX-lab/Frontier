**1. Assignment and outcome**

**Outcome: implemented.** Both assigned parts and the required checks are finished; final acceptance remains with Claude.

B26b has two outcomes: use host formation indices for `vllm_load_balancing` reports under `max_forwards_in_flight`, and diagnose/fix I-006's late-joining four-lane MoE placement test. Work was CPU-only, using Python 3.10.6 at `/data/ycfeng/envs/frontier-py310/bin/python`.

Working root: `/data/ycfeng/Frontier/.worktrees/b26b-placement-key`; starting HEAD: `5188f0e3dd115b01d22d049b6dd7f7e849d34d97`. Scratch: `/data/ycfeng/tmp/codex-implement/a92368f9-d691-40c5-baae-e0eaba3f807a/b26b`. Read-only B26 scratch: `/data/ycfeng/tmp/codex-implement/7218b11a-e3b0-49e9-a533-3c76428f66c2/b26`.

Allowed paths, relative to the working root: `frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py` and `frontier/scheduler/replica_scheduler/vllm_dp_busy_loop.py`; for Part B only, `frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py`, `frontier/events/batch_stage_end_event.py`, `frontier/events/replica_schedule_event.py`, and `frontier/scheduler/utils/dp_dummy_forward.py`; tests `tests/integration/test_vllm_dp_placement_runtime.py`, `tests/integration/test_vllm_dp_run_ahead_runtime.py`, and `tests/unit/test_vllm_dp_load_balancer.py`; and `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b26b/` for this report. The final diff contains the three source/test files in the change table and this new report. The controlled queue regression was initially added to the allowed unit-test file, then moved into the integration module; the unit-test file finishes byte-for-byte unchanged from HEAD. Scratch contains archives, fixture copies, scripts, caches, and outputs. No pre-existing tracked changes were present (`git status --short` was empty). The instructed `.cortexkit/`, `.pi/`, and `remaining_measurements/` were left alone.

Read: applicable root/ancestor AGENTS.md release, DP-lane, module-size and testing guidance; no applicable CLAUDE.md was found; main-tree `issues.md` I-006, `requirements.md` item 43, B26 report and assignment; the A10 driver/environment wrapper; native engine/coordinator source at commit `2cf0a6915ce544dc493a0990f2ea38d81601128a` under `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`.

Settled behavior: K unset retains the existing production keys and report timing. K uses existing formation identities, counts real and dummy forwards, preserves held-key ownership, and requires no second counter. Part B changes the fixture, with every original assertion retained. Native source required two corrections to the brief's premises: stale reports still apply counts, and I-006 already fails at `5f907f66`. These are documented below. Claude retains final review/acceptance, commits, task records, and any larger fidelity campaign. No Git mutations, dependency changes, network/GPU/job operations, messages, or changes to the external A10/fidelity runs were performed.

**2. Changes and native rules**

**Part A**

| Changed file | Relevant symbols / behavior | Reason |
|---|---|---|
| `frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py` | `on_replica_batch_scheduled` (126-140), `_next_report_key` (175-205) | K admission takes `batch._forward_cohort_provisional_id`; completion without a held admission takes `lane._batch_creation_counter`. K queue-room publication uses the existing `_max_concurrent_batches`. |
| `tests/integration/test_vllm_dp_run_ahead_runtime.py` | `_simulator` accepts a policy; `test_run_ahead_load_reports_follow_formed_forwards_and_held_completions`; `test_run_ahead_decode_queue_reports_host_keys_and_keeps_stale_counts` | The DES regression observes successive admissions/completions, queued predecessors, and held host keys. The controlled two-lane runtime regression covers capacity 2, held-key release, next-forward reports, and an older peer report following a newer step. |

The controlled queue regression reuses the unchanged unit fixture helpers `_policy_scheduler` and `_request`. It runs real DECODE schedulers with K=19 and output capacity 2: three dummy forwards precede real forwards 3 and 4; immediate reports use key 3, full-queue admissions hold key 4, completion releases key 4, and completion without a held key uses next-forward key 5. A peer's older key 4 leaves the latest step at 5 while its changed waiting counts apply. Coverage is split between the two integration tests because the DES arrival fixture produces no older reports; the controlled queue fixture supplies that ordering explicitly.

The formation identity is already assigned from `_batch_creation_counter` by `BaseReplicaScheduler._assign_lane_identity` (470-481), before `_create_batch` increments that counter (517-518). The existing `DeviceForward` FIFO remains the device-work owner; it needs no added counter or index storage. Neither the busy loop nor the device-room code was edited. The legacy `_next_report_key` calculation remains byte-for-byte the same after the K branch. The existing held-key release code is unchanged.

Native report identity: `core.py:2120-2135` stamps counts with `step_counter/current_wave`; `2153-2155` and `2169-2170` publish before/after the engine step; `2216-2219` increments the host counter once per iteration, and `2210-2212` resets it at wave end. Frontier's existing monotonic formation index orders the same wave/step progression without an additional wave counter.

Native publish point: `core.py:647-648` checks the output queue size; `683-689` appends the scheduled batch and returns without waiting while the queue has room; `700-715` otherwise waits and applies the oldest output. K limits unfinished device forwards and does not create output-queue slots. The original pipeline-depth comparison remains correct when output capacity equals PP, but supported explicit unified DECODE capacity 2 at PP1 differs. For K only, the comparison now uses the existing lane capacity; None preserves the old comparison and every legacy publish time. This is the material Part A publish-point choice authorized by work item A3.

The brief asks for a stale report that the balancer drops. The pinned native coordinator does **not** drop its counts: `coordinator.py:390-400` advances only a newer wave/step, `401-413` warns on an older one, and `414-417` still assigns all three counts and marks stats changed. Frontier already has the same rule (`VllmDPLoadBalancer.report`; its existing out-of-order unit test). It was preserved, and the new test explicitly witnesses an older key 4 after key 5. Dropping those counts would change a native contract and require editing an unapproved production file. No such change was made.

Runs that can change: opt-in K `vllm_load_balancing` placements, especially DECODE with output capacity 2; legacy DP placement keys/times remain unchanged. A10 uses `round_robin` and does not enter the changed cluster scheduler.

**Part B**

| Changed file | Relevant behavior | Reason |
|---|---|---|
| `tests/integration/test_vllm_dp_placement_runtime.py` | `LATE_JOIN_TRACE`: third arrival 8 ms -> 10 ms; dummy start/end timestamps and request arrivals in evidence; explicit arrival-inside-dummy assertion (487-491) | Restores the intended four-lane dummy-forward shape after host preparation moved before the first-layer sync. All original placement, first-forward, next-forward and work-conservation assertions remain. |

No Part B production source was changed. Before any possible simulator fix, the scope decision was stated: the failure is a fixture-timing issue; this test change affects its scenario only. It changes neither the DP placement fidelity-matrix runtime paths nor the P2-32/P2-64 round-robin A10 path. Part A's runtime path distinction is above.

The first failing source change is `b0bf0c78eb4b33cd95ce7e28e4cd9bf7b65da4de` (October 8, 2026), **before** `5f907f66` (October 26, 2026). Its parent `2c9d625dfd7d9afac3913f21ab1bed1e19f08b8f` passes the same test. `b0bf0c78` adds `forward_preparation_time = schedule_time + prepare_inputs_e2e_time` and starts the first Prefill/DecodeSyncEvent at `stage_start + preparation + attention` instead of `stage_start + attention`. With the fixture's 1 ms operator durations, first-room entries move from 7/9 ms to 9/11 ms. The 8 ms request therefore arrives before the idle lanes are awakened and correctly forms lane 2's real forward 0. A direct old-trace observer found first-room times 9 ms (lane 0), 11 ms (lane 1), 16 ms (dummy lane 3), and 17 ms (real lane 2).

Main first received that source/test shape in merge `9a650fba342a0da835e162d6f030f056e7bc9211` (October 14, 2026). That archived main tree fails; its main parent `0b6999c7b41766a41e33c8a566eb767cff2b7ea5` has no test with this dummy-forward name. Thus there is no first newly failing commit after `5f907f66`: the proposed lower bound was already failing. Adjacent branch-tree checks isolate the causal source commit, and the main merge establishes when the currently named test first reaches main.

Native behavior still supports the test's intended assertion: `core.py:2153` polls inputs, `2169` steps the engine, and `2178-2182` calls `execute_dummy_batch` before the next loop can poll/schedule. `core.py:927-928` delegates that dummy call to the executor. The fixture is repaired to put its arrival inside that pass under the legacy K-unset path. With the new trace, both placement policies retain `[0,1,2]`; lane 2's dummy starts at 9 ms, request 2 arrives at 10 ms, and its dummy ends at 65 ms. Its real batch starts the next forward. This is a test-setup correction, not a change to native dummy execution semantics.

**3. Verification**

All results below are executed checks. Common environment:

```bash
REPO=/data/ycfeng/Frontier/.worktrees/b26b-placement-key
S=/data/ycfeng/tmp/codex-implement/a92368f9-d691-40c5-baae-e0eaba3f807a/b26b
PY=/data/ycfeng/envs/frontier-py310/bin/python
export PYTHONPATH=$REPO:$REPO/tests/unit:$REPO/tests/comparison/calibration:$REPO/tests/analysis
export PYTHONDONTWRITEBYTECODE=1 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=$S/tmp FRONTIER_TMP_ROOT=$S/tmp
```

Tests/analysis use `ulimit -v 31457280`; simulations use `ulimit -v 62914560`. Every launched shell command has a timeout. Tests run from `$S/check-cwd`, with a data symlink and a byte copy of `examples/fixtures/prefix_cache_shared_session_trace.csv`, to keep generated outputs/caches in scratch. A10 uses `PYTHONPATH=$REPO:$REPO/tests/analysis`, matching `run_a10.sh`. The production patch remained unchanged throughout the probes, structural/smoke checks, and both A10 replays. Only test organization changed during the later checks; the complete required pytest suite was rerun afterward. The executable environment and check scripts are `$S/env.sh` and `$S/verify.sh`.

**Acceptance 1: B26 report-key probe — PASS.**

```bash
# Current production:
timeout 180s "$PY" "$S/probe.py" "$S/probe-after"
# Archived HEAD production; current probe fixture copied into that tree:
PYTHONPATH=$S/trees/HEAD:$S/trees/HEAD/tests/unit:$S/trees/HEAD/tests/comparison/calibration:$S/trees/HEAD/tests/analysis   timeout 180s "$PY" "$S/probe.py" "$S/probe-baseline"
```

The archived production run reproduces all 15 original B26 probe rows exactly. Current None rows: **8/8 exactly equal**, including times, lanes, formed indices, report keys and GPU groups. Current K=19 rows: **7/7 keys equal their host formation indices**. Only two probe fields differ: lane 1 at 0.12400000000000005 s changes key 4 -> 23 (GPU group still 4); lane 0 at 0.6899999999999991 s changes key 24 -> 25 (GPU group still 24). Evidence: `probe-baseline/probe.json`, `probe-after/probe.json`, `probe_comparison.json`, and `probe-baseline-exact.json` in scratch.

**Acceptance 2/5: required pytest suite — PASS, 268 passed / 0 failed / 0 skipped in 176.98 s.**

```bash
timeout 1000s bash "$S/verify.sh"
```

The exact pytest invocation in that script is:

```bash
cd "$S/check-cwd"
timeout 900s "$PY" -m pytest  "$REPO/tests/integration/test_vllm_dp_placement_runtime.py"  "$REPO/tests/integration/test_vllm_dp_run_ahead_runtime.py"  "$REPO/tests/integration/test_vllm_dp_dummy_pass_runtime.py"  "$REPO/tests/unit/test_vllm_dp_load_balancer.py"  "$REPO/tests/unit/test_cluster_scheduler_dp_lanes.py"  "$REPO/tests/unit/test_dp_dummy_forward_pricing.py"  "$REPO/tests/unit/test_vllm_v1_engine_idle_time.py"  "$REPO/tests/unit/test_dsv4_dp_sync.py"  "$REPO/tests/unit/test_forward_sync_state.py"  "$REPO/tests/unit/test_nixl_read_completion.py"  "$REPO/tests/unit/test_monolithic_mixed_forward_sync.py"  "$REPO/tests/unit/test_dsv4_a10_tools.py"  -q -p no:cacheprovider --basetemp="$S/pytest-final" --junitxml="$S/pytest-final.xml"
```

| File | PASS | FAIL |
|---|---:|---:|
| placement runtime | 19 | 0 |
| run-ahead runtime | 24 | 0 |
| dummy-pass runtime | 14 | 0 |
| DP load balancer | 89 | 0 |
| cluster scheduler DP lanes | 23 | 0 |
| dummy-forward pricing | 14 | 0 |
| V1 engine idle time | 1 | 0 |
| DSV4 DP sync | 27 | 0 |
| forward sync state | 11 | 0 |
| NIXL READ completion | 9 | 0 |
| monolithic mixed forward sync | 25 | 0 |
| DSV4 A10 tools | 12 | 0 |

Logs: `$S/pytest-final-relocated.log`, `$S/pytest-final.xml`, `$S/pytest_counts.json`. Initial complete run: 266 PASS / 2 FAIL because the scratch CWD lacked the prefix-cache CSV; copying that named fixture fixed both. Initial isolated queue test from the repository CWD failed with read-only `outputs/`; the new test now uses its pytest temporary CWD. The first draft of the runtime regression demanded queued real admissions on both lanes in a shape where lane 0 had at most one predecessor; that premise was corrected, and the controlled real-DECODE queue test explicitly covers both queues. No production defect was hidden by these harness corrections.

The new regressions were also run against unchanged archived HEAD production, with only the current test files copied in: **expected FAIL, 2 failed** (`$S/baseline-relocated.log`). They catch `assert 4 == 23` for the wrong host key and `assert [] == [(0,3),(1,3)]` for the missing native capacity-2 admission reports. This demonstrates that the new tests distinguish the fix from main. After moving the queue regression into the integration module, the three targeted checks passed together (3 PASS in 6.11 s, `$S/targeted-relocated.log`), and the complete required suite was rerun. Earlier isolated checks also passed (`targeted-second.log`, `unit-new-second.log`).

**Acceptance 3: I-006 archaeology and fix — PASS with corrected historical premise.**

Each tested tree was exported without checkout/bisect/state changes:

```bash
git archive <revision> | tar -x -C "$S/trees/<revision>"
# export bounded by timeout 180s, test bounded by timeout 180s
PYTHONPATH=<tree>:<tree>/tests/unit:<tree>/tests/comparison/calibration:<tree>/tests/analysis  timeout 180s "$PY" -m pytest  <tree>/tests/integration/test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward  -q -p no:cacheprovider --basetemp="$S/history-<revision>-pytest"
```

Executed driver commands: `timeout 1200s "$PY" "$S/history.py" 5f907f66 96d9c1ef 82390e10 826539f7` (its first launcher had an outer 30 s timeout); `timeout 900s "$PY" "$S/history.py" 563dfffd 9a650fba`; `timeout 900s "$PY" "$S/history.py" ced720b3 b0bf0c78 b0bf0c78^ 0b6999c7`; `timeout 900s "$PY" "$S/history.py" HEAD`. Exported trees used their own checked-in test/source and scratch outputs. Selected archives were removed only inside scratch to stay below 10 GB; all test logs remain.

| Revision tried | Test result | Interpretation |
|---|---|---|
| `5f907f66` | FAIL, 1 failed | Same lane-2 real-forward-0 assertion; proposed starting point already fails. |
| `96d9c1ef` | FAIL, 1 failed | Same assertion. |
| `82390e10` | FAIL, 1 failed | Same assertion. |
| `826539f7` | FAIL, 1 failed in 3.41 s | Complete pytest failure is present in the log; the outer launcher timed out before writing its JSONL exit receipt. |
| `563dfffd` | FAIL, 1 failed | Failure predates H4b/NIXL work. |
| `9a650fba` | FAIL, 1 failed | First main merge containing the currently named dummy-forward test. |
| `ced720b3` | PASS, 1 passed | Original dummy-pass implementation/test. |
| `b0bf0c78` | FAIL, 1 failed | Causal host-preparation change. |
| `b0bf0c78^` = `2c9d625d` | PASS, 1 passed | Immediate causal parent. |
| `0b6999c7` | NOT_FOUND, exit 4; no tests ran | Main parent of `9a650fba`; dummy-forward test name absent. |
| archived `HEAD` = `5188f0e3` | FAIL, 1 failed | Original main test before the fixture correction. |
| current working tree | PASS, included in 19/19 placement tests | Fixture restored; original assertions retained. |

Logs/commands: `$S/history.jsonl`, `$S/history-<revision>.log`, `$S/history.py`. Native lines, source cause, side fixed, and affected-run scope are in Part B above.

**Acceptance 5: structural checks and architecture smoke — PASS.**

```bash
timeout 1200s "$PY" tests/integration/run_dsv4_structural.py --dp 4 --case offline --output-root "$S/structural-offline"
timeout 1200s "$PY" tests/integration/run_dsv4_structural.py --dp 4 --case online --output-root "$S/structural-online"
PYTHON_BIN="$PY" METRICS_OUTPUT_DIR="$S/smoke"  timeout 600s bash examples/architecture/run_dummy_smoke_matrix.sh -- --metrics_config_cache_dir "$S/smoke-cache"
```

Both structural commands exit 0. Each completes 4 requests and checks 12 stage rows, 1,376 EP waves, 393,284 trace rows, 4,128 real and 39,904 idle EP-lane rows, 8 cache-zero invocations, and 44 assignments. Maximum stage/phase conservation errors: `2.6645352591003757e-15` / `6.938893903907228e-18` ms. Source identities before/after are unchanged. Evidence: `structural-{offline,online}.log`, their `report.json` and `source_identity.json` files. Smoke exits 0, **12/12 cases**, covering dense/MoE across co-location/PDD/PD-AF offline/online (`smoke.log`). These remain CPU structural/smoke checks, not native latency evidence.


**Acceptance 4: B26 32-request A10 fidelity — PASS.**

```bash
# Each replay used ulimit -v 62914560 and the common environment above.
export PYTHONPATH=$REPO:$REPO/tests/analysis
cp -a /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_64_warm_g1hs_r2/cache "$S/a10-unset/cache"
cp -a /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_64_warm_g1hs_r2/cache "$S/a10-k19/cache"
cp -a /data/ycfeng/tmp/codex-implement/7218b11a-e3b0-49e9-a533-3c76428f66c2/b26/trace32.csv "$S/trace32.csv"
timeout 7200s "$PY" -u task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_driver.py p2_64 "$S/a10-unset" --trace "$S/trace32.csv" --without-run-ahead
timeout 7200s "$PY" -u task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_driver.py p2_64 "$S/a10-k19" --trace "$S/trace32.csv"
# After both commands returned exit 0 with FINISHED receipts:
ulimit -v 31457280
timeout 60s "$PY" "$S/compare_a10.py" > "$S/a10-comparison.log" 2>&1
```

These two replays ran concurrently in private scratch. Each initial cache copy contained 1,758 files totaling 324,536,393 bytes; every source/destination hash matched and no cache inode was shared (`cache_copy_check.json`). Trace SHA256: `88f0b339e9a696858729fbf5edc46408b537903766fe8b34ee9effd7fb90e4a3`. Neither replay writes into the external stage-4 runs or the B26 baseline. Baselines are B26 `final-unset` and `final-k19-r3` under its read-only scratch root.

Both 32-request replays finished. Each requested output is **byte-identical** to its corresponding B26 baseline:

| Run | Artifact | Bytes | Current and B26 SHA256 | Result |
|---|---|---:|---|---|
| K unset | `request_metrics.csv` | 24386 | `df160df153c4100cebefb533239701a1cbce2b5c4f944abe8f5d93870905d1ae` | PASS |
| K unset | `system_metrics.json` | 5671 | `4291f30a09e2581555d894c19bc8028ea16e5c3e62a6e35b609a27586a32193c` | PASS |
| K=19 | `request_metrics.csv` | 23338 | `6a4453dbffa1496618f662bd055f14fe43ffe0037631599b4ab6886dd0b17252` | PASS |
| K=19 | `system_metrics.json` | 5662 | `f4017555a13279d08666b8d363f11cc1742a6c9cf868a30a74adc3183048a678` | PASS |

K unset: initialization 71.536 s, Simulator.run 2645.326 s; 6,144/6,144 complete observer records match exactly, including host/device timing fields. K=19: initialization 73.214 s, Simulator.run 3493.147 s; 8,064/8,064 complete observer records match exactly, including host/device timing fields.

Evidence: `$S/a10-{unset,k19}.log`, their `receipt.json`, `forwards.jsonl.gz` and metrics directories, `$S/a10-comparison.log`, and `$S/a10_comparison.json`. There are no numeric or byte differences to explain. The A10 production path uses round-robin placement, so it does not call the modified load-balancing scheduler; Part B changes only the test fixture. These replays check P2-64's 32-request operator-path reproduction; they do not replace Claude's full fidelity matrix or the larger P2-32/P2-64 campaigns.


**Acceptance 6: hygiene — PASS.**

```bash
timeout 20s git diff --check
timeout 30s "$PY" /data/ycfeng/tmp/unused_imports.py frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py tests/integration/test_vllm_dp_placement_runtime.py tests/integration/test_vllm_dp_run_ahead_runtime.py tests/unit/test_vllm_dp_load_balancer.py
timeout 20s wc -l frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py tests/integration/test_vllm_dp_placement_runtime.py tests/integration/test_vllm_dp_run_ahead_runtime.py tests/unit/test_vllm_dp_load_balancer.py
timeout 20s du -sh "$S"
```

`git diff --check` exited 0. The requested AST unused-import script reported `[]` for every changed Python file. Final line counts:

| Changed Python file | Lines | Limit |
|---|---:|---:|
| `frontier/scheduler/cluster_scheduler/vllm_load_balancing_cluster_scheduler.py` | 215 | 2,000 |
| `tests/integration/test_vllm_dp_placement_runtime.py` | 1405 | 2,000 |
| `tests/integration/test_vllm_dp_run_ahead_runtime.py` | 555 | 2,000 |

Scratch measured 5,040,479,134 bytes (4.69 GiB) at the final hygiene check, below 10 GB; the command output is in `$S/hygiene-final.log`. The new durable report is `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b26b/report.md`. Its contents describe the final implementation and all acceptance evidence; no other task records were changed. No source caches, test bytecode, profiling data, or unrelated source files were written.


**4. Remaining work and decisions for Claude**

The assigned implementation and required CPU checks are finished. No unavailable prerequisite or source blocker remains. The two source-grounded deviations require Claude's review, without expanding this patch:

- Correct Part A's stale-report wording: the pinned coordinator warns and retains the older report's counts while keeping its newer wave/step boundary. This is observed directly at `coordinator.py:390-417` and covered by the new queue test. Dropping counts would be a different behavior and would require scope for the balancer production file.
- Correct I-006's historical lower bound: the archived `5f907f66` tree already fails. The adjacent passing/failing source trees identify `b0bf0c78` as causal; `9a650fba` is the first main merge with the named test. The 8 ms fixture no longer reaches its specified arrival-during-dummy state. Restoring that state at 10 ms preserves the native rule and all original assertions.

Observed fidelity: both short A10 runs and all requested metric artifacts are exact. Inference from the changed call path: round-robin production runs are unaffected; opt-in K load balancing can change placements because host report keys and explicit queue-capacity publication become correct. The CPU checks do not establish native latency accuracy or complete the external fidelity campaign. Claude owns any broader runs, issue/requirement updates, final acceptance, and commits. No approval was assumed or requested for those actions.

**5. Review handoff**

Inspect the scheduler diff for reuse of the existing formation identity, held-key release, K-only queue-capacity publication, and an untouched legacy key calculation. Inspect the two new regressions for successive two-lane admissions/completions, queued dummy predecessors, held key 4 released by completion, next-forward key 5, and native out-of-order count handling. Inspect the Part B trace change and its explicit `dummy_start < arrival < dummy_end` assertion; every previous placement/forward/conservation assertion remains.

The targeted rerun is:

```bash
cd "$S/check-cwd"
timeout 180s "$PY" -m pytest  "$REPO/tests/integration/test_vllm_dp_placement_runtime.py::test_a_request_reaching_a_lane_during_its_dummy_forward_starts_the_next_forward"  "$REPO/tests/integration/test_vllm_dp_run_ahead_runtime.py::test_run_ahead_load_reports_follow_formed_forwards_and_held_completions"  "$REPO/tests/integration/test_vllm_dp_run_ahead_runtime.py::test_run_ahead_decode_queue_reports_host_keys_and_keeps_stale_counts"  -q -p no:cacheprovider --basetemp="$S/claude-review-pytest"
```

This command is proposed with a fresh review basetemp. The same three selected nodes were executed together with `--basetemp="$S/targeted-relocated"` (3 PASS in 6.11 s), then in the final 268-test suite. Use the common environment and scratch fixture described above. For acceptance reproduction, rerun `$S/verify.sh`, the two structural commands, the architecture smoke, and the B26 probe. Existing A10 comparisons can be inspected or rerun with the cache/trace commands above if Claude changes the production patch. Read the archive logs and the native lines when reviewing the two corrected premises. The large historical exports are disposable scratch evidence; durable test receipts, probe JSON, replay comparisons, and this report retain the outcome.
