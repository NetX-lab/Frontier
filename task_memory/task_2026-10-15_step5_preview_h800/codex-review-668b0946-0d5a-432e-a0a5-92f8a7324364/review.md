# Review context

The request is an independent, read-only review of the Step-5 host-step probe, KV reset host-time estimator, and staging changes in commit `f3fb0358fb6d734e1fecdc71cc0855548359f38f`, against parent `242028ef`. The working root is `/data/ycfeng/Frontier/.worktrees/step5-preview`; HEAD resolves to the requested commit. Review date: October 27, 2026. Caller-declared reviewer model: `gpt-6.1-sol-sys-paygo`, reasoning effort `xhigh`.

For source references below:

- `R` means `/data/ycfeng/Frontier/.worktrees/step5-preview`.
- `N` means `/data/ycfeng/tmp/d29_published_sources01/source/vllm`, the caller-supplied native source tree.
- `S` means `/data/ycfeng/tmp/codex-review/668b0946-0d5a-432e-a0a5-92f8a7324364`, the permitted scratch directory.

The reviewed diff contains 12 paths: the task design and progress documents; `canonical_stage.py`, `build_stage.py`, `formal_release.py`, and `native_pd_worker.py`; the two host-probe bootstrap files; `kv_zeroing_host_time.py`; and the three changed unit-test files. Relevant unchanged callers and contracts were also inspected, including reference admission, server-environment construction, the profile loader, Frontier forward-start agreement, and native worker, runner, zeroer, DP-dispatch, sampling, and CUDA-graph code.

The supplied native runtime scope is P/D, TP8 × DP2 per role, PP1, async scheduling, V2 GPU model runner, FULL_AND_PIECEWISE CUDA graphs, mp executor, and MTP off. Image/source byte identity is accepted as caller-provided; it was not independently checked against a running image.

Applicable Frontier `AGENTS.md` instructions were read. No applicable `CLAUDE.md` or additional nested instructions were found for the reviewed paths. The current task's `requirements.md`, current portion of `plan.md`, ticket `tickets/19-graph-host-scheduling.md`, and design §§2, 3.2–3.3, and 4.1–4.3 were reviewed. Material contracts include preserving native arguments/results/exceptions and call counts; keeping diagnostic instrumentation outside references; retaining probe records through shutdown; measuring a component host term rather than a whole-step lookup table; and producing a request-count table when the measured decode trend exceeds 1 ms. See task `requirements.md:161–172`, `plan.md:286–311`, and `design.md:87–128,156–225`.

Excluded from the commit review were the pre-existing staged/unstaged changes under `task_memory/task_2026-10-22_step5_position_routing/` and untracked `.cortexkit/`, `.pi/`, and `tests/integration/step5_pd/attention_census.py`. Repository sources, task records, and Git state were not edited. Temporary reproduction files and test outputs were placed under `S`.

# Assessment

**Verdict: changes_required.** Six actionable findings are supported by source inspection and local reproductions. The named instrumentation seams are present and their import/binding order is sound in the inspected scope. The principal problems are reference isolation, terminal record retention, and estimator validity. A successful job can leave incomplete records or produce a plausible scalar that does not implement the documented measurement contract.

| Requested requirement | Assessment | Evidence and consequence |
|---|---|---|
| Named seams exist and run for workload forwards | Satisfied by source inspection | Native runner, zeroer, DP dispatch, timing collector, and sparse-cache methods all exist. Details below. |
| Hooks bind before the zeroer stores bound methods | Satisfied by source inspection | Class instrumentation runs after module execution; native zeroer binding occurs later during KV setup. |
| Preserve original arguments, results, exceptions, decorators, and call count | Satisfied in inspected call paths | The wrapper calls the decorated original once and isolates callback errors. Focused fake-native tests pass. Actual GPU perturbation is unverified. |
| Keep probe instrumentation outside every clean/reference path | Violated | Explicit diagnostic reference acceptance is possible, F001; inherited activation survives clean environment preparation, F005. |
| Reliably retain and copy records after server shutdown | Violated | The copy loop retains written bytes, but forked workers skip the only terminal flush, F002. |
| Remove only DP peer wait from the host interval | Violated | Whole-function timing includes local preparation and graph-dispatch work, which is subtracted as wait, F003. TP-to-engine availability mapping is separately unverified. |
| CUDA events bound the documented main-stream work | Partially satisfied | Real sample exit follows main-stream postprocessing; dummy execute exit precedes remaining GPU work, F006. No GPU timing check ran. |
| Join DP engines by collective ordinal | Supported within the narrow source scope | Positive-token real and runtime dummy forwards each dispatch once; zero-token connector-only calls return before dispatch. No source-supported ordinal shift was found with MTP off and elastic EP off. Live multi-engine coverage remains unverified. |
| Select only formal measurement steps | Unverified across hosts | Client monotonic-to-wall conversion is internally consistent, but filtering server wall times requires bounded server/client clock offsets. |
| Use the mean for an appropriate scalar profile | Satisfied as the literal design statistic | The code uses the resetting-step mean. Its composition through a maximum still requires native validation. |
| Switch decode to a request-count table above the 1 ms trend threshold | Violated | The output is always scalar, F004. |
| Avoid whole-step lookup tables in prediction | Satisfied in inspected scope | Device spans remain diagnostic; the emitted prediction input is a host component. This does not establish that the component is correctly estimated. |

## Source checks that passed

All requested seams exist in `N`:

| Probe seam | Native definition and relevant caller |
|---|---|
| `GPUModelRunner.execute_model` | `v1/worker/gpu/model_runner.py:1745–1756`; real reset calls at `1273–1280` |
| `GPUModelRunner.sample_tokens` | `v1/worker/gpu/model_runner.py:2113–2117` |
| `KVBlockZeroer.zero_block_ids` | `v1/worker/utils.py:519–523`; side-storage hook invocation at `585–598` |
| Module-global `sync_cudagraph_and_dp_padding` | `v1/worker/gpu/dp_utils.py:44–57`; `dispatch_cg_and_sync_dp` resolves the global at `312–325` |
| `StepTimingCollector.forward_start` | `v1/worker/gpu/async_utils.py:94–97`; called before target forward at `model_runner.py:2022` even when native timing collection is disabled |
| Sparse-cache reset hook and static `_ids` | `models/step5/nvidia/sparse_summary_cache.py:486–488,507–509` |

Native `v1/worker/utils.py:363–365` stores `layer.zero_kv_cache_side_storage` as a bound method during zeroer construction, reached from `model_runner.py:808–817`. The bootstrap instruments the class after its module executes, before this later construction. The worker inherits the installed finder/wrappers under fork, or imports `sitecustomize` on interpreter startup under spawn; process-local probe state is initialized on first wrapped execute at `host_step_probe.py:128–131`. The local multiprocessing reproduction exercised both startup modes.

`host_step_probe.py:223–242` forwards the original positional and keyword arguments, returns the original result, and leaves original exceptions propagating. Native `@torch.inference_mode` and `@step_eplb_after` decorators remain on the original callable. The inspected signatures match the execute and zeroer calls; omitted defaults are handled by the probe's `.get` operations. No reachable signature-binding failure was found in this scope.

The wrapped methods are outside the target model's captured forward: native `model_runner.py:1092–1112` delegates graph capture to the manager, whose capture path invokes raw `model(**model_inputs)` at `cudagraph_utils.py:759` inside the graph context. This inspection did not find probe callbacks being compiled or captured with the target forward.

For real steps, `AsyncOutput` switches to an output-copy stream and restores the main stream (`async_utils.py:137–171,268–277`), after which `sample_tokens` performs main-stream postprocessing (`model_runner.py:2210–2249`). A probe end event at sample exit therefore covers that main-stream work, while intentionally excluding completion of the output-copy stream. `event.query()` is a completion poll, distinct from event synchronization; the installed PyTorch implementation exposes these separately in `torch/cuda/streams.py:205–231`. The probe calls no explicit event/device synchronization. Driver overhead and timing perturbation were not measured.

Within the stated scope, target `execute_model` dispatches once at `model_runner.py:1810–1825`, without reusing a DP sync state. Real zero-token connector-only steps return earlier at `1770–1775`. Runtime dummies enter `_dummy_run` through `gpu_worker.py:1467–1474`, which calls wrapped execute at `model_runner.py:902–911`. Startup profiling also uses `_dummy_run`; direct target graph capture does not use the DP-dispatch seam. Additional inspected dispatch callers belong to speculative-decoding paths excluded by MTP off. Native DP groups join corresponding TP positions across engines (`distributed/parallel_state.py:2271–2287`). These checks support ordinal joining in this scope, but do not replace a live 32-worker ordinal inventory.

# Findings

## F001 — P1 high: A declared host-probe run can pass native reference admission

**Confidence:** High for consumer acceptance; whether the external receipt producer would issue the necessary PASS is unverified.

**Locations:** Changed `R/tests/integration/step5_pd/formal_release.py:19–23,112–120`; stage construction at `R/tests/integration/step5_pd/build_stage.py:198–204`; relevant reference consumer at `R/tests/comparison/e2e_gate/admission.py:227–329`, particularly `296–303`. The formal caller consumes this admission result at `formal_release.py:44–55`.

**Violated contract:** Task design §4.1, `design.md:158–162`, says `host_probe` is diagnostic and never a reference.

**Trigger and causal chain:** Build a host-probe stage and later present its pinned artifacts with a passing image-native clean-run-check receipt. The builder still assigns `mode="clean"`. Adding `host_probe` to `COLLECTION_KINDS` exempts its submission from prerequisites; it does not reject that run when used as a prerequisite/reference. `client_reference` checks older `pd_state_observation` and `serve_observers` flags but never rejects `host_step_probe`, diagnostic `canonical_replay.kind`, or summary `replay_kind`.

**Expected versus actual:** A known probe run must be rejected independently of a receipt claiming clean status. The consumer accepts it if its other receipt, runtime, and artifact checks pass.

**Evidence:** `S/reproduce_review.py` created hash-bound image-native artifacts with config `canonical_replay.kind="host_probe"`, `host_step_probe=true`, and summary `replay_kind="host_probe"`, plus a passing receipt. `client_reference()` returned `status="PASS"`, `reasons=[]`. This is a consumer-level reproduction, not evidence that the actual external clean-run-check producer accepts instrumentation.

**Impact:** Instrumented TTFT/TPOT can be admitted as clean comparison or formal-readiness evidence, defeating the run-kind isolation guarantee.

**Focused correction:** Reject diagnostic probe declarations in `client_reference` and the clean-run receipt producer. Validate config, summary, and effective worker environment consistently. Preserve acceptance of legitimate clean workload references.

**Verification check:** Add a hash-bound negative reference-admission case using a passing receipt and an explicit host-probe config/summary. It must fail; the corresponding clean positive control must pass.

## F002 — P2 medium: Forked workers skip the terminal probe flush and lose records

**Confidence:** High; reproduced with actual multiprocessing and completed fake CUDA events.

**Locations:** [Probe terminal registration](/data/ycfeng/Frontier/.worktrees/step5-preview/tests/integration/step5_pd/host_probe_bootstrap/host_step_probe.py:88), with pending retention at lines `110–151` and the only terminal drain at `212–220`; `R/tests/comparison/calibration/native_pd_worker.py:293–299`; estimator completeness checks at `R/tests/integration/step5_pd/kv_zeroing_host_time.py:75–92`.

**Native evidence:** `N/envs.py:955–957` defaults worker multiprocessing to fork. `N/utils/system_utils.py:122–177` can force spawn under specified conditions, so the production choice is not assumed. Workers use that context at `N/v1/executor/multiproc_executor.py:723–756`. CPython's fork child returns through `os._exit(code)`, skipping Python `atexit` callbacks: `/data/ycfeng/envs/cpython-3.12.13-linux-x86_64-gnu/lib/python3.12/multiprocessing/popen_fork.py:62–73`. Native cleanup explicitly invokes worker/runner shutdown at `multiproc_executor.py:979–986` and `gpu_worker.py:1643–1662`, but neither closes this probe.

**Violated contract:** Design §4.1, `design.md:169–173`, requires durable worker records and post-stop copying. The implementation retains completed records in memory until a later execute or terminal callback.

**Trigger:** The selected worker start method is fork, and the worker shuts down after its last execute. `self.last` always retains the final step; additional records may remain pending. Native shutdown synchronizes the device at `model_runner.py:2364–2367`, but no probe drain follows it.

**Expected versus actual:** All completed host records should survive ordinary worker shutdown. In the scratch reproduction, marking all fake events complete yielded sequences `[1,2,3,4]` under fork versus `[1,2,3,4,5]` under spawn. A separate Python 3.12 reproduction also lost the terminal sequence under fork.

**Impact:** The gzip loop copies only bytes already written. A missing tail can silently pass `role_processes`, which checks rank population, generation, and error records but no terminal completion marker. Depending on later dummy/connector steps, omitted records can include formal samples. This establishes tail loss, not total loss of all job records.

**Focused correction:** Drain and close the probe through native runner/worker shutdown, using the runner's existing synchronization. Make closure idempotent and resolve completed event timings before closing. Add a terminal record/count that allows the estimator to distinguish complete from truncated files; do not rely solely on `atexit`.

**Verification check:** Execute real fork and spawn processes with the final events complete, exit through native-like cleanup, and verify all sequences and terminal counts. Check that copied gzip files match the drained source records. Include a truncated-file negative estimator case.

## F003 — P2 medium: Whole-function duration subtracts local CPU work as DP peer wait

**Confidence:** High; source-proven boundary error with a deterministic estimator counterexample.

**Locations:** [Probe sync timestamps](/data/ycfeng/Frontier/.worktrees/step5-preview/tests/integration/step5_pd/host_probe_bootstrap/host_step_probe.py:196), lines `196–204,254–257`; `R/tests/integration/step5_pd/kv_zeroing_host_time.py:109–113,136–138`; native `N/v1/worker/gpu/dp_utils.py:66–77,79–210`.

**Violated contract:** Design §4.3, `design.md:203–212`, intends `excess` to remove the time an engine waited for peers while retaining local reset, preparation, and intrinsic collective cost.

**Trigger and causal chain:** Engines incur different local work inside `sync_cudagraph_and_dp_padding`. The probe times the entire function. Native blocking `dist.all_reduce` is only line `77`; the function also allocates/populates a CPU tensor at `66–73`, reduces and interprets the result, and performs local graph dispatch at `189–195`. The estimator treats every duration above the minimum engine duration as peer wait and subtracts it from launch latency. Local graph-dispatch branches and ordinary process scheduling can therefore be charged to the wrong category.

**Expected versus actual:** Local CPU work must remain in the engine's host term. Consider simultaneous collective entry, 0.2 ms intrinsic collective cost, and 2 ms additional post-collective padding work on DP0:

| Engine | Actual peer wait | Required host interval | Estimator host interval | Reported wait |
|---|---:|---:|---:|---:|
| DP0 | 0 ms | 8.2 ms | 6.2 ms | 2.0 ms |
| DP1 | 0 ms | 6.2 ms | 6.2 ms | 0 ms |

The existing estimator produced these values in `S/reproduce_review.py`; see `S/reproduction_results.log`, `NON_COLLECTIVE_WORK`.

**Impact:** Local host cost is underestimated and the DP-wait report is incorrect. This directly changes the profile used by Frontier's `max_i(a_i+h_i)` composition.

**Focused correction:** Timestamp the actual DP CPU `all_reduce` boundaries while retaining tensor preparation and result/graph dispatch in the host interval. Continue to label duration-minus-minimum as an estimate: unequal collective return overhead can still confound exact wait cancellation and needs qualification.

**Verification check:** Use a native-like sync fixture containing local work before and after the collective. Simultaneous collective arrival must produce zero estimated peer wait and retain both engines' distinct local costs. Also test genuine staggered arrival with equal local work.

## F004 — P2 medium: The estimator never emits the required request-count table

**Confidence:** High; reproduced through the estimator CLI and its real output construction.

**Locations:** [Unconditional scalar emission](/data/ycfeng/Frontier/.worktrees/step5-preview/tests/integration/step5_pd/kv_zeroing_host_time.py:237); request-count statistics at lines `157–176`; design §4.3 at `R/task_memory/task_2026-10-15_step5_preview_h800/design.md:216–219`, reinforced by §3.3 at `125–128`. Existing loader support is in `R/frontier/execution_time_predictor/kv_block_zeroing_host_time.py:77–98,100–115`.

**Violated contract:** A decode host-time trend exceeding 1 ms over the observed request-count range must switch the profile to `by_num_requests`, with bounded interpolation.

**Trigger:** Valid resetting samples show a decode mean of 5.2 ms at four requests and 9.2 ms at forty requests. The report correctly records both distributions, but profile construction always executes:

```python
roles = {role: {"time_ms": report["roles"][role]["host_ms"]["mean"]} for role in ROLES}
```

**Expected versus actual:** This fixture should emit a sorted decode table containing the two measured means. The CLI instead exited successfully and emitted `decode={"time_ms":7.2}`. Evidence: `S/reproduction_results.log`, `REQUEST_TREND`.

**Impact:** The profile can overcharge low-concurrency decode steps and undercharge high-concurrency steps while reporting the evidence that required a different representation.

**Focused correction:** Implement the documented threshold and emit count-specific mean entries in the loader's `by_num_requests` format. If the trend decision needs an explicit statistical qualification policy, stop scalar emission pending that disposition instead of silently ignoring the requirement.

**Verification check:** Add CLI cases below and above the 1 ms threshold. Verify scalar output for the qualified constant case, a table for the dependent case, correct interpolation through the existing loader, and failure outside the table's measured count range.

## F005 — P2 medium: Clean server preparation preserves inherited probe activation

**Confidence:** High for the environment leak; runtime activation additionally requires a bootstrap path available on inherited `PYTHONPATH`.

**Locations:** [Host-probe environment preparation](/data/ycfeng/Frontier/.worktrees/step5-preview/tests/comparison/calibration/native_pd_worker.py:91), lines `91–100,187–201`; relevant scrubbing at `R/tests/comparison/dp_placement_pp/vllm_replay.py:334–339`; preflight at `R/tests/integration/step5_pd/build_stage.py:140–148`; verifier at `canonical_stage.py:111–116`; bootstrap at `host_probe_bootstrap/sitecustomize.py:6–12`.

**Violated contract:** Design §4.1 requires probe configuration, staged bootstrap, server environment, and run kind to agree, and permits activation only for `host_probe`.

**Trigger and causal chain:** Start a clean/workload worker from an environment containing valid `STEP5_HOST_STEP_PROBE` JSON and a prior probe bootstrap directory on `PYTHONPATH`. `server_env` scrubs older VLLM diagnostics but preserves this new variable. When the config has no `host_step_probe` key, `host_probe_environment` returns the inherited environment unchanged. Runtime uses actual `os.environ`, whereas preflight starts with `{}` and the verifier checks only `server_env_set` additions.

**Expected versus actual:** Clean preparation should remove inherited activation. The scratch reproduction returned the original probe JSON and `reported_changes={}` for a workload config. At Python startup, the inherited `sitecustomize` consequently installs hooks.

**Impact:** A clean run can be instrumented with incorrect role/output settings while its recorded environment additions give no indication of the probe. This is distinct from F001's acceptance of an explicitly declared diagnostic run.

**Focused correction:** Remove inherited `STEP5_HOST_STEP_PROBE` for clean server and client construction, then enable it afresh only for an explicitly allowed host-probe kind. Record or validate the effective activation environment, rather than assuming an empty preflight environment represents runtime inheritance.

**Verification check:** Prepare and launch a clean fake-native server with inherited probe JSON and bootstrap `PYTHONPATH`; no hooks or probe files should appear. An explicit host-probe control should still produce correctly attributed files.

## F006 — P3 low: Dummy end events precede remaining main-stream GPU work

**Confidence:** High for the ordering defect; its GPU timing magnitude is unmeasured.

**Locations:** [Dummy end-event placement](/data/ycfeng/Frontier/.worktrees/step5-preview/tests/integration/step5_pd/host_probe_bootstrap/host_step_probe.py:145), lines `145–151`; native `N/v1/worker/gpu/model_runner.py:902–911,918–927,970–972`; runtime dummy caller `N/v1/worker/gpu_worker.py:1467–1474`.

**Violated contract:** Design §4.2, `design.md:184`, describes this event as the end of the dummy step's main-stream work.

**Trigger:** A runtime dummy executes on the stated PP1, MTP-off path. The probe records its end event when `execute_model` returns. Native `_dummy_run` then continues and computes:

```python
sample_hidden_states = hidden_states[input_batch.logits_indices]
```

This GPU advanced-indexing work occurs after the recorded event, even without a speculator.

**Expected versus actual:** The end event should follow all dummy main-stream work. It currently excludes the indexing tail. The next `gap_ms` can include that tail, and `prior_pending` can report false after the probe event completes while the dummy tail is still executing.

**Impact:** The device-gap and pending-work cross-checks can misclassify dummy-to-real transitions. The direct host timestamps are not changed by this event-placement error.

**Focused correction:** Record the dummy end event at `_dummy_run` completion, keeping the original call and result intact. Coordinate this with terminal draining in F002.

**Verification check:** Use a fake dummy runner that queues an additional event after execute, and confirm the probe end follows it. When GPU validation is available, compare the event boundary with the actual dummy tail on the main stream.

# Verification

CPU tests used `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6). The additional fork reproduction used the existing Python 3.12.13 environment. No software was installed or upgraded. No native GPU workload or CUDA timing experiment ran.

The test environment disabled bytecode and pytest cache writes and directed temporary output to scratch. Commands below use these concrete definitions:

```sh
REVIEW_ROOT=/data/ycfeng/Frontier/.worktrees/step5-preview
REVIEW_SCRATCH=/data/ycfeng/tmp/codex-review/668b0946-0d5a-432e-a0a5-92f8a7324364
REVIEW_PY310=/data/ycfeng/envs/frontier-py310/bin/python
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$REVIEW_SCRATCH"
export PYTHONPATH="$REVIEW_ROOT"
export WANDB_DISABLED=true
export VIDUR_DISABLE_WANDB=1
cd "$REVIEW_ROOT"
```

## Executed checks

| Check | Expected | Observed | Status |
|---|---|---|---|
| Scope/revision and Git status inspection | Requested commit/parent; distinguish unrelated worktree state | HEAD matched `f3fb0358`; unrelated changes were excluded; final status matched initial status | PASS |
| `git diff --check 242028ef f3fb0358` | No whitespace errors | Exit 0, no diagnostics | PASS |
| Three focused changed test files | Tests pass | 54 passed in 11.34 s | PASS |
| Worker and existing host-profile tests | Tests pass | 28 passed, 3 failed in 32.08 s | FAIL; baseline-equivalent worker failures |
| Parent worker baseline | Determine whether lifecycle failures were introduced here | 1 passed, same 3 failed in 34.86 s | Baseline equivalence established |
| Scratch review reproductions | Exercise adverse cases missing from focused tests | Script exited 0 and reproduced F001–F005; conditional TP counterexample also recorded | Findings confirmed at stated scope |
| Native seam, binding, dispatch, and capture-path inspection | Verify exact supplied-source contracts | Definitions and callers inspected; no native workload executed | Source inspection completed |
| Native 32-GPU end-to-end probe and timing qualification | Establish live records, overhead, hook counts, stream timing, and ordinal coverage | Not executed under review-only constraints | NOT_RUN |

Focused test command:

```sh
"$REVIEW_PY310" -B -m pytest \
  tests/unit/test_step5_host_step_probe.py \
  tests/unit/test_kv_zeroing_host_time_estimator.py \
  tests/unit/test_step5_priority_runs.py \
  -q -p no:cacheprovider --basetemp="$REVIEW_SCRATCH/pytest"
```

Related test command:

```sh
"$REVIEW_PY310" -B -m pytest \
  tests/unit/test_native_pd_worker.py \
  tests/unit/test_kv_block_zeroing_host_time.py \
  -q -p no:cacheprovider --basetemp="$REVIEW_SCRATCH/pytest-worker"
```

The three failures were `test_worker_separates_auxiliary_report_faults_from_native_failure` cases `report_write_failure`, `native_exit_at_replay_completion`, and `native_shutdown_error`. Each failed the missing `native_health_200` assertion after `TimeoutError: Native P/D rendezvous or workload deadline exceeded`.

The parent worker was exported read-only from Git into scratch:

```sh
git show 242028ef:tests/comparison/calibration/native_pd_worker.py \
  > "$REVIEW_SCRATCH/review_parent_native_pd_worker.py"
```

The unchanged worker test was copied into scratch with only its import redirected to that parent module, then executed:

```sh
"$REVIEW_PY310" -B -m pytest \
  "$REVIEW_SCRATCH/test_parent_native_pd_worker.py" \
  -q -p no:cacheprovider --basetemp="$REVIEW_SCRATCH/pytest-parent"
```

It produced the same three failures and error pattern. They are not attributed to this commit. The baseline log is [parent_worker_results.log](/data/ycfeng/tmp/codex-review/668b0946-0d5a-432e-a0a5-92f8a7324364/parent_worker_results.log).

The adverse-case reproduction command was:

```sh
"$REVIEW_PY310" -B "$REVIEW_SCRATCH/reproduce_review.py"
```

The script uses the checked-in fake-native fixture, actual fork/spawn multiprocessing, the real estimator functions/CLI, real environment preparation, and real reference admission. Its source and results are [reproduce_review.py](/data/ycfeng/tmp/codex-review/668b0946-0d5a-432e-a0a5-92f8a7324364/reproduce_review.py) and [reproduction_results.log](/data/ycfeng/tmp/codex-review/668b0946-0d5a-432e-a0a5-92f8a7324364/reproduction_results.log).

The focused bootstrap tests primarily exercise fake native modules. Their passing results establish wrapper behavior for those fixtures; they do not establish CUDA-event accuracy, 23-hook coverage in the native model, absence of measurement perturbation, or the live production process start method.

# Open questions and next actions

## Unverified modeling and runtime questions

**Q1 — What native engine availability corresponds to Frontier's `a_i`?** Even after F003 is corrected, `max_r(L_r−E_r−F_r)` is not generally an engine-ready interval measured from a single engine availability. Frontier records one arrival per lane at `forward_start_agreement.py:94` and composes it at `112–120`; native DP coordination joins engines independently at each TP position and does not establish a common first-sync timestamp across TP ranks.

A conditional counterexample, granting first-sync as an availability proxy and setting excess to zero, is:

| TP rank | First-sync F | Launch L | Local duration |
|---|---:|---:|---:|
| 0 | 0 ms | 8 ms | 8 ms |
| 1 | 5 ms | 10 ms | 5 ms |

If engine availability is `max(F)=5` and readiness is `max(L)=10`, the required engine interval is 5 ms. The estimator returns 8 ms, recomposing readiness as 13 ms. `TP_ENGINE_ORIGIN` in the scratch log reproduces the algebra. This is not counted as a confirmed finding because the native-to-Frontier availability mapping has not been established. Define that mapping and measure rank-origin skew. If this availability definition is adopted, derive readiness and availability on a common engine clock; do not assume a maximum of durations is equivalent to a duration between engine-level boundaries.

**Q2 — Is first `_ids` return the exact device-idle origin?** Native `utils.py:585–598` passes a CUDA slice to `_ids`, whose `.tolist()` blocks for its transfer; on the main-stream path, return is evidence that preceding main-stream work has completed. It is not the exact timestamp of previous-step completion. Current-step request-state writes (`model_runner.py:1244–1248,1757–1764`) and the block-ID H2D copy precede it, and already-idle steps may have a gap before it. No earlier explicit device synchronization was identified in the inspected scoped request-update path, but implicit synchronization and the exact completion interval were not measured. Establish this with a native timeline or bounded completion witnesses, distinguishing waited and already-idle steps. Also describe `gap_ms` as an end-to-launch main-stream interval: reset/preparation GPU work can occur inside it, so it is not literally an interval of total GPU inactivity.

**Q3 — Are server and replay-host wall clocks sufficiently aligned?** `formal_windows()` at estimator `65–72` correctly converts client monotonic times into the client's wall-clock domain using that phase's offset. Filtering server `time.time_ns()` at probe `136` and estimator `121–123` additionally assumes aligned wall clocks and no relevant realtime jumps. Record clock offset/error bounds for each server against the replay host, or use explicit phase/request correlation. Without that evidence, startup, sweep, or boundary steps may be incorrectly included or excluded. This was not reproduced as an actual clock failure.

**Q4 — Does a mean scalar reproduce the aggregate effect after maximum composition?** The mean is the statistic explicitly requested by design §4.3 and implemented by the estimator. However, `E[max_i(a_i+h_i)]` is generally different from applying `max` to per-role mean host times. Native variability and correlation with engine availability/request count must establish whether the scalar approximation is adequate. First fix F003 and F004; then validate on held-out clean repeats. No numerical bias from this approximation was measured in this review.

**Q5 — What does the external clean-run-check producer enforce?** F001 proves the reference consumer accepts a known probe given a passing receipt. The actual producer was not located or audited in this review. Inspect its diagnostic config, summary, and effective-environment rejection rules, and add an end-to-end negative admission test. A producer-side safeguard would reduce reachability but would not make the consumer's present acceptance check correct.

**Q6 — What are the live collection coverage and perturbation bounds?** A native qualification run still needs to establish the selected fork/spawn method, one process per rank, terminal completeness, 23 reset hooks on appropriate steps, equal collective coverage, and event ordering/overhead. The existing perturbation calculation labels a profile `PERTURBED` rather than blocking emission, as the design specifies. Its comparison is meaningful only with a verified clean run of the same case. No new GPU job was submitted during this review.

## Corrections in dependency order

1. Enforce reference and effective-environment isolation, F001 and F005.
2. Fix terminal draining and completeness checks, F002; correct the dummy event boundary, F006.
3. Measure actual collective boundaries and retain local CPU work, F003. Resolve Q1/Q2 before treating the resulting interval as Frontier's engine-level host term.
4. Implement the documented request-count profile selection, F004.
5. Rerun the focused CPU negative cases, then qualify native record coverage, clock alignment, and perturbation before consuming the profile for prediction.

The confirmed findings are actionable without a GPU job. Remaining CUDA, clock, and modeling questions require additional evidence; passing the inspected unit tests alone does not settle them.
