# H4b-(a) native D-side handoff measurement

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 | Created: question, trace findings, scheduler observer, records, soft gate, harness base, package changes, CPU check, cost, acceptance and limitations. |

**Authority.** D-H4A-CHECKPOINT (H4b-(a) first), D-GPU-SHORTAGE-REPORT, D-INPUT-PERSISTENCE and D-MAINLINE-LOOP-CAP-DSV4 in [requirements.md](../requirements.md). The run is DIAGNOSTIC. It does not replace native06 as the P0 reference, and its client metrics do not enter any gate.

**Case.** Identical to native06: P0 on 32 H800, P16 + D16, each role one Replica with TP8, attention DP2, EP16, PP1; workload 68f891b (per repetition 16 warmups and 128 formal requests of 6144/1024 tokens at QPS 1, three repetitions). The only difference is the scheduler observer below.

## 1. Question

H4a left a −81.22 ms TTFT gap in the D-side handoff: from the proxy's P response to the proxy's first D byte, native takes 123.49 ms and Frontier 42.28 ms ([H4a report section 7](../test_report_2026-10-25_h4a_p0_diagnostic.md)). H4b-(b) must model this interval from measured inputs. It needs to know where the 123 ms goes:

| Part | From → to | Process |
| --- | --- | --- |
| a | proxy P response → D API server receives the request | proxy, network, D API server |
| b | D API server → D engine core `add_request` | API server, ZMQ, engine-core input thread |
| c | `add_request` → first `schedule` that sees the request (it enters WAITING_FOR_REMOTE_KVS and the NIXL READ is requested in that step's connector metadata) | engine-core loop |
| d | that step → the `update_from_output` that reports the request in `finished_recving` | NIXL READ, worker polling once per step, TP aggregation |
| e | that update → the `schedule` that promotes the request and schedules its single recomputed token (k=1) | engine-core loop, batch queue |
| f | that schedule → the `update_from_output` that emits the first token | k=1 step: CPU dispatch and GPU execution |
| g | first-token update → proxy's first D byte | output thread, ZMQ, API server, HTTP stream |

**What the profile02 traces already show** (idle-lane samples, profiled repeat, scratch analyses `/data/ycfeng/tmp/glm53_dsv4/h4b/window_cpu.txt`, `window_gpu.txt`, `cross_lane.txt`):
- The k=1 step comes at the 4th engine iteration after the request appears. The first 3 iterations run no forward on that lane apart from the KV-connector work.
- The idle lane's iteration cadence is about 19 or 10 ms. It is probably paced by the busy lane through the DP dummy batch and the EP synchronization.
- The k=1 step runs in piecewise graph mode (`model_dispatch_run_pw_graph`), with 26-42 ms of CPU dispatch. At the same time the other lane is forced into piecewise mode.
- The next GPU step to the first byte takes about 5.5 ms. The arrival part (a + b) is at most about 10 ms.

These samples come from the profiled repeat, where the profiler slows host work. They name the parts but do not give unperturbed values, and they give no per-request scheduler times. native06 has no per-request engine-core timeline either: vLLM's request stats do not record WAITING_FOR_REMOTE_KVS, `finished_recving` or the promotion step.

## 2. Mechanism

vLLM 0.28 builds the engine-core scheduler from `SchedulerConfig.get_scheduler_cls()`, which resolves the `--scheduler-cls` qualname (`vllm/config/scheduler.py:170-191`). EngineCore constructs it with keyword arguments `vllm_config, kv_cache_config, structured_output_manager, include_finished_set, log_stats, block_size, hash_block_size` (`vllm/v1/engine/core.py:148-161`) and calls its `shutdown()` from `EngineCore.shutdown()` (`core.py:750-756`). Every transition in section 1 from c to f passes through three scheduler methods in the engine-core process: `add_request`, `schedule` and `update_from_output`.

**Observer class.** `native_scheduler_lifecycle.LifecycleObservedScheduler` subclasses `vllm.v1.core.sched.async_scheduler.AsyncScheduler`, the class that native06 used (`async_scheduling` true and no `scheduler_cls`). The batch queue (`EngineCore.step_with_batch_queue`, depth 2) belongs to EngineCore and does not depend on the scheduler class. The overrides are:
- `add_request(request)`: calls `super()`, then records the request.
- `schedule(*args, **kwargs)`: times the `super()` call, then records the step from the returned `SchedulerOutput`.
- `update_from_output(scheduler_output, model_runner_output)`: copies the worker-side KV connector fields first, because the scheduler later aggregates its own connector stats into the same object. It then times the `super()` call and records the step's results from the returned outputs.
- `shutdown()`: calls `super()`, then flushes and closes the file.

Each override passes its arguments unchanged and returns `super()`'s result unchanged. Exceptions from `super()` propagate unchanged. Only the observation code runs inside `try/except`. The observer never calls a scheduler or connector method that changes state. In particular it does not call `connector.get_kv_connector_stats()`, which resets the scheduler-side stats.

**Activation.** The engine config gains `scheduler_cls: "native_scheduler_lifecycle.LifecycleObservedScheduler"`. `vllm_replay.server_command` passes it as `--scheduler-cls`. The module is importable in every engine-core process, because `pd_replay.instance_env` already puts the staged `tests/comparison/calibration/` directory on `PYTHONPATH` for the `native_full_structure.FullModelInventory` worker extension, and engine-core processes inherit the server environment. The observer applies to both roles, that is 4 engine cores: P dp0, P dp1, D dp0, D dp1.

**Output.** The observer reads the existing `FRONTIER_NATIVE_SERVING_EVIDENCE_DIR` (the node directory on the cloud volume) and `FRONTIER_NATIVE_ROLE`, which `native_pd_worker.run` already sets for this worker extension. It writes `<node>/scheduler_lifecycle/dp{rank}_pid{pid}.jsonl`. If the directory variable is missing, it records nothing, as `native_full_structure._install_native_serving_inventory` does. Writes are buffered. The file is flushed after at most 2 s, in `shutdown()` and at interpreter exit.

## 3. Records

One JSON object per line. `t` is `time.time()` and `mono` is `time.monotonic()` at the start of the event; `dur_ms` is the duration of the `super()` call. Empty lists and absent values are left out.

| `event` | Fields |
| --- | --- |
| `start` | `schema` (`native_scheduler_lifecycle.v1`), `role`, `dp_rank`, `host`, `pid`, `t`, `mono`, scheduler class, `async_scheduling`, connector name |
| `add` | `t`, `mono`, `request_id`, `external_req_id`, `arrival_time` (set by the API server on its host), `prompt_tokens`, `max_tokens`, `do_remote_prefill`, `do_remote_decode`, `remote_request_id` from `kv_transfer_params` |
| `schedule` | `step` (`current_step`), `t`, `mono`, `dur_ms`, `tokens` (total scheduled), `new` (`[request_id, num_computed_tokens, num_scheduled_tokens]` per new request), `cached` (number of cached requests), `waiting_remote_kv` (request ids in WAITING_FOR_REMOTE_KVS after the call), `recv` and `send` (request ids in the NIXL metadata `reqs_to_recv` and `reqs_to_send`), `finished` (`finished_req_ids`), `preempted` |
| `update` | `step` of the scheduler output (mapped from the output object's identity at `schedule`), `t`, `mono`, `dur_ms`, `tokens` (scheduled in that step), `finished_recving`, `finished_sending`, `kv_stats` (copy of the worker-side `kv_connector_stats.data`: `transfer_duration`, `post_duration`, `bytes_transferred`, `num_descriptors` and the failure counters), `first_token` (request ids whose first output token is in this update), `done` (request ids finished in this update) |
| `error` | the first 5 observer tracebacks, then only a count |

`first_token` is derived from the returned `EngineCoreOutputs`: a request whose output carries new token ids and that has not emitted before. The observer keeps one set of emitted request ids and removes an id when its request finishes.

**Volume.** A D lane runs about 8,000 steps per repetition. The `schedule` and `update` records are about 150-300 bytes each, so about 15 MB per D engine core per run and less on P. Every scheduled step is kept, because the step cadence (section 1, parts c-e) is the quantity being measured.

**Overhead.** Each record builds one small dict and one `json.dumps` call, measured in the CPU check (section 6). The expected cost is tens of microseconds per step against steps of 10 ms or more, under 1%. The run's client metrics are compared with native06 (section 8) to check for a visible slowdown.

## 4. Analysis plan

Per formal D request, joined to the proxy record through `remote_request_id` (and `request_id` as a cross-check):
- **Single engine-host clock:** c, d, e and f from `mono`, together with the number of engine iterations in c-e, and the NIXL `transfer_duration` of the request's READ against d.
- **Clock-free remainder:** `(decode_first_byte_monotonic − prefill_response_monotonic)` on the proxy host minus `(first-token update − add)` on the engine host gives a + b + g without any cross-host clock.
- **Cross-host split:** a, b and g from the `t` fields against the proxy's wall times. This split depends on host clock synchronization. The sign of `arrival_time − prefill_response_time` bounds the offset between the proxy host and the D head host in one direction.
- **Lane state:** whether the request's lane was idle or decoding when it arrived, because the idle-lane cadence and the busy-lane cadence differ.

By-product, not part of H4b-(a) acceptance: unprofiled step intervals per composition (k, m) on all four engine cores. They can later replace the profiled-repeat decode samples (substitutions S4 and S5) in H3r.

## 5. Harness base and changes

native06 ran harness commit `b62c155a` from the worktree `v4-balanced-native-model` (branch `feat/v4-balanced-native-model`, still at that commit and left unchanged). Since then the mainline has changed 4 of the staged harness files (`native_full_structure.py`, `native_pd_worker.py`, `pd_replay.py`, `vllm_replay.py`: profile windows and ordered peer shutdown, 138 insertions and 28 deletions). So that the measurement differs from native06 only by the observer, H4b-(a) uses a new clean worktree `/data/ycfeng/Frontier/.worktrees/h4b-lifecycle-harness` on a new branch `h4b-lifecycle-harness` created at `b62c155a`. The branch carries exactly these changes:

| File | Change | Lines |
| --- | --- | ---: |
| new `tests/comparison/calibration/native_scheduler_lifecycle.py` | the observer class (sections 2 and 3) | ~170 |
| `tests/comparison/dp_placement_pp/vllm_replay.py` | add `"scheduler_cls"` to the v0.28 `server_command` passthrough tuple | 1 |
| new `tests/comparison/calibration/check_native_scheduler_lifecycle.py` | CPU check (section 6), run in the vLLM 0.28 environment | ~120 |

No `frontier/` file changes. Simulator behavior, including GLM, co-location, PDD and PD-AF results, is unaffected. After the run, the branch is merged into `feat/glm53-dsv4-vllm028`. `b62c155a` is an ancestor of the mainline, and the mainline's `vllm_replay.py` changes touch other lines, so the merge is expected to be clean.

## 6. CPU check

`check_native_scheduler_lifecycle.py` runs with `/data/ycfeng/envs/oneshot-opd-py312/bin/python` (vllm 0.28.0). The installed wheel matches NATIVE `2cf0a69` byte for byte for the scheduler, engine-core, NIXL and outputs files (`/data/ycfeng/tmp/glm53_dsv4/h4b/wheel_vs_native.txt`). The check uses the vLLM unit-test helpers from the NATIVE checkout (`tests/v1/kv_connector/unit/utils.py`):
1. Drive the remote-prefill lifecycle of `test_remote_prefill_lifecycle.py::test_basic_lifecycle` (add, schedule into WAITING_FOR_REMOTE_KVS, `finished_recving`, promotion, the k=1 step, the first token, finish) once on `AsyncScheduler` and once on `LifecycleObservedScheduler`. Every `SchedulerOutput` and every returned `EngineCoreOutputs` must compare equal.
2. Drive one remote-decode (P-side) lifecycle the same way, with `finished_sending`.
3. Check the JSONL file: a `start` record, `add`, the WAITING_FOR_REMOTE_KVS step with the request in `recv`, the `finished_recving` update, the promotion step with `[id, num_tokens − 1, 1]`, `first_token`, `done`, and no `error` record.
4. Inject an observer failure (an unserializable field) and check that scheduler results are unchanged and one `error` record is written.
5. Time 10,000 `schedule` + `update` pairs with and without the observer on the same inputs and report the per-step difference.

## 7. Package

`build_package.py` in this directory builds `package_01/` from the record copy of native06 (`../v4_p0_native_20261021/candidate_h800_p0_06/`), which is byte-identical to the launched package. It copies the package, then applies only these changes:

| Item | Change |
| --- | --- |
| `config.json` path fields | `workspace`, `script`, `harness_dir`, `model_dir`, `case_dir`, `observer_bootstrap`, `overlay_manifest` and every `tool_hashes` key point into `package_01/` |
| `config.json` `harness_worktree` | the new clean worktree; `harness_files` restaged from the branch commit with `git show`, plus the observer module |
| `config.json` `tool_hashes` | rehashed for the restaged harness files; the observer module added |
| `config.json` `expected_source_hashes` | 3 pins added from NATIVE `2cf0a69`: `vllm/v1/core/sched/async_scheduler.py`, `vllm/config/scheduler.py`, `vllm/v1/outputs.py` (the existing 74 pins already cover `scheduler.py`, `output.py`, `engine/core.py`, `core_client.py` and the NIXL schedulers) |
| `config.json` `run_tag`, `job_tag`, `output_dir` | `dsv4-h800-p0-lifecycle-20261025-01`; `/mnt/codesign-exp/ycfeng/frontier/glm53_dsv4_vllm028/t18/dsv4_h800_p0_lifecycle_20261025/case/runs/groundtruth_clean/dsv4-h800-p0-lifecycle-20261025-01` |
| `engine.json`, `case/inputs/engine.json`, `case/inputs/native_config.json` | `scheduler_cls` added; nothing else |
| `overlay.json` | `staged_dir` and the leaf source path point into `package_01/` |
| `case/manifest.yaml` | `case_id` `dsv4-h800-p0-lifecycle-fixed128-qps1`, new `run_generation`, `diagnostic_only` true, `e2e_reference_admission` false |
| `source_lock.json`, `evidence_readiness.json` | the harness commit, the new pins and the diagnostic purpose |

The builder then compares every file of `package_01/` with native06 and fails unless the only differences are the ones listed above. It runs the package's own `seal.py`, then `launch.py --check-only`, which must print `INPUT_HASHES_PASS_NO_SUBMISSION`. `launch.py`, `submit_native_diagnostic.py`, `native_dummy_bootstrap.py`, the workload trace, the model metadata, the image digest and every timeout stay identical.

## 8. Cost, submission and acceptance

**Cost.** One job: 4 replicas × 8 H800 = 32 GPUs on `codesign`, non-preemptible, with the same image, resources and 3600 s job timeout as native06. Expected wall time is about the same as native06 (startup plus three repetitions). This is the first H4b GPU job; D-MAINLINE-LOOP-CAP-DSV4 counts it.

**Submission.** Read-only quota check at 2026-10-25T11:26:44Z (`prelaunch/quota_20261025T112642Z.json`): `codesign` total 128 H800, non-preemptible use 108, so 20 free, and no node with 8 free GPUs. Per the handbook, the job is submitted when the package is ready and the platform queues it. The quota is read again just before submission. If the queue delays the route, YC is told at once (D-GPU-SHORTAGE-REPORT). The launcher stays alive until the job is terminal. Stopping uses `rjob_stop.sh`.

**Acceptance.**
1. Platform status Succeeded on all 4 replicas. The existing run checks (`run_check_status.groundtruth_clean.json`) pass as they did for native06: 3 repetitions × 128 formal requests complete with HTTP 200 and full outputs.
2. Observer files exist for all 4 engine cores. Each has a `start` record and no `error` record. Every formal D request has `add`, a WAITING_FOR_REMOTE_KVS step, a `finished_recving` update, a promotion step and a `first_token` record.
3. The run's client TTFT, TPOT, E2E and THPT are computed with the canonical client tools and compared with native06 r1-r3 and their spread. This comparison is reported, not gated. If a metric falls outside native06's range, the report states it as a possible observer effect.
4. The parts a-g and the iteration counts are reported per lane state, with the clock-free remainder and its cross-host split (section 4).
5. The results and the analysis are copied into this directory, with cloud paths and hashes, before H4b-(b) starts.

**Expectation before the run.** This is not a fit. From the traces: c is at most one engine iteration; c + d + e covers about 3 iterations; f is about 30-40 ms; and c to f together are about 75-105 ms. The clock-free remainder a + b + g is about 10-25 ms.

## 9. Limitations

- The observer sees scheduler events, not GPU time. GPU-level detail of the k=1 step stays with the profile02 traces.
- Cross-host splits (a, b, g) depend on host clock synchronization, which is not measured. Only the clock-free sum is exact.
- One run. Its spread across repetitions is reported, but there is no second run.
- The observer adds a small host cost per step. Section 6 measures it on CPU, and acceptance item 3 compares the run's client metrics with native06.
