## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Verified routing/admission controls and fresh same-run evidence; clarified completed first-divergence RCA versus incomplete full-case feedback attribution. |

# DP workflow RCA diagnostic controls

Status: PASS for the bounded controls and completed fresh same-run route/snapshot/enqueue/scheduler/batch validation. The first DP inversion RCA is complete and workflow parity fails as reproduced; attribution of execution-time and snapshot feedback across the full100-request closed loop remains INCOMPLETE. No new route/snapshot/enqueue collection is pending for the first-divergence finding. Canonical evidence is analysis/dp-workflow-rca/same-run-01/same_run_join.json, fresh_same_run_evidence.json, and fresh_boundary_controls.json. Historical commands referencing same-run-provisional are retained as execution receipts; their traces and mappings were verified byte-for-byte equal to the finalized same-run-01 files.

## Execution

Working directory: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
Python: /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python, Python 3.13.13, conda dev-vidur-v03-hopper-e2e.

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp FRONTIER_LOG_LEVEL=ERROR /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/integration/issue26_dp_workflow_rca_observe.py --command /data/ycfeng/tmp/issue26-shared-forward-numerical-01/runtime/command.json --output /data/ycfeng/tmp/issue26-dp-workflow-prefix-01 --routes 8 > /data/ycfeng/tmp/issue26-dp-workflow-prefix-01.log 2>&1
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/integration/issue26_dp_workflow_rca_controls.py --source /data/ycfeng/tmp/vLLM-BS/vllm/v1/engine/core_client.py --observation /data/ycfeng/tmp/issue26-dp-workflow-prefix-01/observation.json --mapping task_memory/task_2026-09-07_issue26_ttft_h200/analysis/historical-replay-02-clean/request_id_map.csv --comparison task_memory/task_2026-09-07_issue26_ttft_h200/analysis/cpu-shared-forward-01/batch_comparison.json --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/dp-workflow-rca/source_controls.json
git diff --check
```

The diagnostic deliberately reuses only the current-task full-run trained caches. It retains all real case settings and operator predictions, writes a fresh output directory, and stops after eight observed routing decisions. It does not complete or accept E2E metrics. Production source is unchanged.

## Criteria and results

1. Actual Frontier routing must reproduce the existing full-run prefix and expose the exact counts consumed before local increments. PASS: eight routes observed; the first seven owners are DP0, DP0, DP1, DP1, DP0, DP1, DP0 in engine-queue order. First report time 65.79007690374297 ms reproduces the full run's first stage endpoint.
2. Under identical measured counts, pinned vLLM source selection and current Frontier selection must agree. PASS: 8/8 choices match, including local waiting increments and tie-breaking in the supported single-frontend case.
3. Change request order alone across the first close pair; preserve counts and operator durations. PASS: swapping the request labels assigned to the two route slots produces client5 -> DP1 and client6 -> DP0, equal to the isolated old diagnostic owners. No load report occurs between these two slots. This demonstrates sufficiency only; the old diagnostic's actual route order remains unrecorded.
4. Determine whether later member mismatches can propagate without another admission-policy change. PASS as a label-level control: swapping only client5/client6 labels in the existing Frontier decode membership makes all client7 through client15 prefill member vectors agree with the isolated old diagnostic. This is not a simulator rerun and does not align KV progress or prove later admission timing equality.
5. Whitespace validation: PASS, no output from git diff --check. No failed commands or runtime exceptions occurred in these two diagnostics; ObservationComplete is the declared bounded stopping mechanism.

## Evidence

- analysis/dp-workflow-rca/frontier_prefix_observation.json: exact command, eight route states and changed engine reports.
- analysis/dp-workflow-rca/source_controls.json: source-method choices, pair-order counterfactual and all member-label controls.
- Raw log: /data/ycfeng/tmp/issue26-dp-workflow-prefix-01.log.

At client6 / Frontier5, routing time is 1.0252374149858952 s and counts are [[0,3],[0,2]]: scores 3 and 2 select DP1. At client5 / Frontier6, 0.200871844 ms later, counts are [[0,3],[1,2]]: scores 3 and 6 select DP0. The last publication is 972 ms for both routes. The later owner swap is therefore compatible with changed request order even when operator timings remain identical; exclusive attribution to accumulated operator error is not supported by the old artifacts.

No simulation error or latency acceptance metric is produced by these control checks. The fresh full-run numeric comparison remains in test_report_2026-09-08_shared_forward_full_case.md. New same-run routing and admission evidence is required to determine the actual observed cause, including frontend snapshot delivery and coordinator phase after warmups.

## Admission observation extension

The same observer now records each returned scheduler batch and cumulative prior scheduled tokens. Repeated the exact observe command above with output `/data/ycfeng/tmp/issue26-dp-workflow-prefix-02` and log `/data/ycfeng/tmp/issue26-dp-workflow-prefix-02.log`. PASS: all eight route dictionaries equal generation01 exactly, and41admission records were collected. The first admissions are request0 prefill at0s, request0 one-token decode at0.06579007690374297s, and request1 prefill plus request0 decode at0.11778284745306514s; prior scheduled request0 tokens in that mixed batch are4097. Selected evidence: analysis/dp-workflow-rca/frontier_admission_observation.json. This confirms observation did not alter the measured prefix.

## First-forward-only measured substitution

Direct inequality check PASS: replacing only Frontier's first forward duration with the old diagnostic measured 77.125022888184ms still ends before the unchanged second queue arrival 81.208001822233ms by 4.082978934050ms. Since GlobalBatchEndEvent directly creates ReplicaScheduleEvent at the same simulation timestamp, this duration-only substitution still schedules one pure decode before request1 becomes visible. No actual predictor scaling or modified simulation was needed to settle that condition. Evidence: analysis/dp-workflow-rca/first_forward_admission_counterfactual.json; values are read from the saved Frontier observation and DP0/TP0 first formal batch in historical-replay-02-batch-validation.json.

The pinned vLLM CUDA batch scope ends before compute_logits, sampling and synchronous bookkeeping. EngineCore then processes scheduler output, publishes output/load state, optionally executes DP finish synchronization, and drains new input before its next schedule. These intervals include both GPU and host/transport work; they are not collectively labelled CPU overhead. Their actual duration is not derived by subtracting unrelated runs. The inequality does not establish that the same ordering occurred in the new run.

## Fresh same-run trace analyzer

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/integration/issue26_dp_workflow_rca_trace.py --run task_memory/task_2026-09-07_issue26_ttft_h200/analysis/h200-rca-01/runtime/batch --command /data/ycfeng/tmp/issue26-shared-forward-numerical-01/runtime/command.json --output task_memory/task_2026-09-07_issue26_ttft_h200/analysis/dp-workflow-rca/same-run-01
```

PASS:400unique routes and400unique enqueues join400completed clients;100formal requests each join exactly one prefill batch on each of4TP workers of their selectedDP. All400route choices match the current Frontier weighted-load selector under the actual recorded vLLM counts;399state transitions additionally match the snapshot_receive/local-increment history, with the first route establishing initial state. Scheduler log token maps match actual batch membership exactly. Route timestamp <= enqueue timestamp <= first prefill scheduler timestamp holds for all100formal requests. Both provisional replay traces and mappings compare byte-for-byte equal to the fully validated traces/mappings.

Actual same-run first reversed pair: client5 routes beforeclient6 by2.023212146ms, butclient6 is enqueued beforeclient5 by0.048529357ms. Counts at route5 are[[0,3],[0,2]], choosingDP1; route6 observes the local increment[[0,3],[1,2]], choosingDP0. These are direct observations, not server-arrival inference. Evidence: analysis/dp-workflow-rca/fresh_same_run_evidence.json and same-run-01/same_run_join.json.

The first formal route sees stale frontend counts[[0,0],[0,1]]. Inspection of every schedule record after that monotonic timestamp finds zero warmup requests. This proves the observed stale count is not evidence that a warmup request is still being scheduled; residual dummy-loop activity is not measured by these request schedule records.

The CPU processes temporarily remained in D/wait_on_page_bit_common during Python imports, delaying startup by minutes. They continued without restart, environment changes, or altered timing inputs; the analyzer completed with exit0. This was file-read waiting, not a failed calibration execution.

## Fresh paired DES boundary controls

```bash
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp FRONTIER_LOG_LEVEL=ERROR /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/integration/issue26_dp_workflow_rca_observe.py --command task_memory/task_2026-09-07_issue26_ttft_h200/analysis/dp-workflow-rca/same-run-provisional/enqueue_command.json --output /data/ycfeng/tmp/issue26-dp-new-enqueue-01 --routes 8 > /data/ycfeng/tmp/issue26-dp-new-enqueue-01.log 2>&1
PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp FRONTIER_LOG_LEVEL=ERROR /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/integration/issue26_dp_workflow_rca_observe.py --command task_memory/task_2026-09-07_issue26_ttft_h200/analysis/dp-workflow-rca/same-run-provisional/route_command.json --output /data/ycfeng/tmp/issue26-dp-new-route-01 --routes 8 > /data/ycfeng/tmp/issue26-dp-new-route-01.log 2>&1
```

Both processes exit0/PASS after exactly8routes; first7prefill admissions are retained because the8throute is the declared stop boundary. Enqueue-input matches6/8DP owners and5/7prefill member multisets; route-input matches8/8DP owners and7/7prefill member multisets. Both retain identical prediction inputs, and their two close-pair counts are identical; only arrival boundary/time ordering differs. This isolates the first owner swap from a required operator-duration change. Expected limitation also observed: request0prior scheduled tokens in the second mixed batch are4097(enqueue-input),4098(route-input), versus4096(vLLM). Therefore improved DP ownership alone is not full workflow parity. See analysis/dp-workflow-rca/fresh_boundary_controls.json and rca.md. No E2E metric is accepted from these prefix controls.
