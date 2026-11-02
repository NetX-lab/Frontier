# Frontier

Frontier is a discrete-event simulator of LLM serving deployments. It predicts request latency and throughput, and is calibrated against native vLLM runs.

## Language

### Deployment

**Replica**:
One complete GPU pod that serves requests as one unit, with its own tensor, data, expert, and pipeline parallel groups.
_Avoid_: serving group, instance, engine

**DP lane**:
One data-parallel request owner inside a Replica; every request belongs to exactly one DP lane.
_Avoid_: DP rank (when request ownership is meant)

### Ground truth

**Native run**:
A real vLLM serving run on GPUs whose measurements are the ground truth Frontier is compared with.
_Avoid_: real run, truth run

**Dummy-weight native run**:
A native run whose model weights are randomly initialized instead of loaded from a checkpoint, while the full model structure, kernels, communication, and memory allocation still execute.
_Avoid_: dummy run, fake model

**DP dummy forward**:
The forward pass an idle DP lane executes so that its expert-parallel peers can complete their collectives.
_Avoid_: dummy batch, idle forward

**Dummy timing mode**:
Frontier's fixed-time predictor mode, valid for structural tests only and never as calibration evidence.
_Avoid_: dummy mode (unqualified), dummy predictor

### Evidence tiers

**Fixture replay**:
Re-running the accuracy gate on stored, already-normalized readings; it proves the tooling, not the simulator.

**Trace-conditioned cost replay**:
Predicting the cost of batches whose composition, shapes, and routing were taken from a native run.

**Controlled workflow replay**:
Driving Frontier with controlled arrivals and fixed timings to check one admission or lifetime rule.

**Predictive online E2E**:
A Frontier run that forms its own queues, batches, and completion times from a workload, model configuration, deployment, and an independently calibrated predictor bundle, with no input taken from the native run it is compared with.
_Avoid_: E2E (for any of the replay tiers)

### Metrics

**Internal TTFT**:
Time from a request becoming visible to the queue to the completion of its prefill.
_Avoid_: TTFT (unqualified) in PD scenarios

**Client TTFT**:
Time from client submission to the first token visible to the client, measured on one client clock.
_Avoid_: first-token latency
