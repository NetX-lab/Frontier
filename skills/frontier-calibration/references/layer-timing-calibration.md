# Optional Workflow Calibration with Measured Layer Timings

Use this procedure within `workflow-gap-analysis` to check Frontier's workflow
before refining operator/kernel timing. Measured full-model layer service times
reduce compute, communication, and CPU timing-estimation errors, so remaining
differences can be investigated as workflow behavior. They do not eliminate
measurement uncertainty or prove correctness outside the exercised cases.

## 1. Check Whether the Experiment Can Run

Confirm that the available, authorized allocation can run the target model at
its actual GPU count, device type, interconnect, and TP/PP/DP/EP layout. For
example, a 32-GPU target needs the matching 32-GPU deployment; extrapolating a
smaller run does not isolate that target's workflow. Follow the operator's
platform rules and existing resource authorization.

Inspect the selected engine's probes in `engine-integration.md` and the Frontier
timing input path before collecting anything. Map native or compiler-fused
scopes to model layers; unavailable layer boundaries leave this optional
experiment unsupported, even when clean E2E collection works. `BaseExecutionTimePredictor.predict_stage_execution_time`
and `StageExecutionTime` expose layer results and stage-owned work, but do not
by themselves provide a measured-timing importer. Identify the actual adapter,
its accepted timing boundaries, and the scheduler/event consumers. If a probe
or adapter is missing, record the concrete gap and scope its implementation
under the existing change rules; do not invent a CLI flag or use dummy mode.

When resources, authorization, or timing support are unavailable, record the
reason this optional check was not run and continue the ordinary calibration
path. Reuse adequate existing observations of the same case generation.

## 2. Measure the Whole Model at the Target Scale

Freeze the case with `semantic-alignment-check`: model/dtype, physical rank
placement and parallel groups, request trace, scheduler limits, graph mode,
routing, prefix state, and the clean run's observed KV capacity and allocation mapping. Collect clean ground
truth and the needed timing observations through `groundtruth-run-check` and
`groundtruth-run`, on the same node group. Keep their run tags separate. Apply
`contract.md`, Calibration Objective, to the clean repetitions and timing
samples; one favorable run or lane mode cannot stand for typical behavior.

Measure every physical layer across the full model and all participating
stages/ranks. Cover prefill and decode, the batch/token/context shapes the
workload exercises, mixed steps when enabled, and the KV-pressure cases needed
to exercise preemption and cache reuse. Retain repeated samples and their
spread; one representative layer multiplied by the layer count is insufficient.
Record each sample's layer, phase, rank/stage, batch token and context state,
graph/routing state, units, timing boundaries, and source run.

Account for compute, communication, CPU launch/preparation, and non-layer work
such as embedding, sampling, output handling, and transfers. Keep work charged
once per step or stage in that scope. Reconcile layer and stage spans with the
observed forward/step elapsed time, including overlap and unexplained time.
Use low-overhead observations that preserve asynchronous execution; serialized
`op_timing` or synchronized `kernel_timing` sums are not full-layer elapsed time.
Apply the existing clean/probe admission rules where applicable, and record
instrumentation cost separately. Missing coverage, unexplained residuals that
could affect the conclusion, or timing distortion leave the check
`INSUFFICIENT_EVIDENCE` until resolved.

## 3. Substitute Times While Frontier Runs Its Own Workflow

Use a separate diagnostic label and explicit timing inputs, with the same
request trace and semantic settings as the clean reference. Record the adapter,
source observations, coverage, and command receipt. Select measured service
times from the simulator's current layer, phase, batch/context state, and
rank/layout; a shape without a matching measured scope is a coverage gap.

Replace only duration estimates at the existing execution boundaries. Keep
Frontier's topology, participant groups, event dependencies, batch formation,
KV allocation/reuse/freeing, and preempt/resume decisions active. Ground-truth
batch decisions, absolute event timestamps, and KV transitions are comparison
evidence, not instructions replayed into Frontier.

Use a layer aggregate only where it preserves those dependencies. Keep separate
measured communication and host spans where their placement or overlap affects
execution. A communication event retains its participants and dependencies when
its duration is measured. Queueing, barriers, and waits caused by those
dependencies remain simulator outcomes; exclude them from service times that
would charge them again. Remove the predicted compute, comm, CPU, and kernel-gap
terms covered by each measured span. Charge step/stage overhead once. If the
adapter cannot preserve a relevant boundary, record the limitation and leave
that workflow claim unverified instead of flattening it into one scalar.

## 4. Compare Metrics and Workflow Evidence

Compare the diagnostic simulator's formal-request rows with clean ground truth
using the contract's definitions for TTFT, TPOT, request E2E, request throughput,
and token throughput. Report both absolute values, absolute errors, and relative
errors; a zero reference makes relative error undefined. Use the existing
per-metric 10% criterion and verified metric conversion. Apply identity checks
and only the mode/dispatch treatment justified for this engine and case.
Store this as a diagnostic comparison; it never closes the native E2E gate.

Also compare topology/placement, the first formal batch, subsequent step
membership and order, queue/admission decisions, KV block use and release,
cache hits, preemption/resumption, and completion counts. Investigate the first
reproducible divergence using event/source anchors and measurement variability.
A metrics pass alone cannot validate unexercised preemption or cache paths;
name which behaviors the case actually tested.

Write `analysis/workflow_gap_table.csv`, `analysis/workflow_gap_summary.md`, and
`analysis/workflow_gap_status.json` under the case's diagnostic generation. The
summary identifies the measured inputs, uncovered states, remaining timing
uncertainty, and the comparison's scope. A workflow check passes only when its
metrics pass, timings are admitted and cover the run, and no unexplained
workflow mismatch remains. Attribute a mismatch to `simulation_logic` only
after excluding input, measurement, and ground-truth defects. Route a supported
workflow correction through `workflow-repair`, then repeat this check.

## 5. Refine Timing After the Workflow Check

Keep the passing measured-timing run as the workflow reference. For the same
case, progressively replace its measured layer spans with measured or predicted
compute ops/kernels, communication ops, and CPU overhead through the existing
`op-gap-analysis`, `residual-lane-classification`, and CPU entries. Collect
missing fine-grained coverage through their supplements. Preserve their
approval and CPU-repair ordering rules; this experiment does not publish CPU
tables or open CPU repair early.

At each replacement, keep other timing inputs fixed, use a fresh diagnostic
label, and compare against both the measured-timing reference and clean ground
truth. Record which boundaries changed and the resulting metric deltas. An
adverse change directs analysis to that timing source; load-sensitive scheduling
can also change as a consequence, so check the events before assigning cause.

Finish with a fresh native-prediction `simulator-run` and `e2e-metrics-gap` using
`run_gate.py gate`. A workflow pass with measured times supports only that
experiment; final accuracy still requires the normal predictor and clean gate.
