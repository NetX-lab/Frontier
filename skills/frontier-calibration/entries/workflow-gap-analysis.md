# Workflow Gap Analysis

**When to use:** Use when checking simulator workflow with measured layer timings before operator/kernel calibration, or when queueing, step mix, batches, or lane behavior diverge.

Read `../references/contract.md`, `../references/evidence-tiers.md`, and
`../references/pitfalls.md` (Reading the Gate, Analysis and Correction).

## Optional: Check Workflow with Measured Layer Timings

For a new model or a workflow-first investigation, use
[the layer timing procedure](../references/layer-timing-calibration.md) when
the target GPU scale is available and collection is authorized. It substitutes
measured service times while Frontier still decides topology, scheduling, KV
state, and preemption. This branch needs the procedure's clean reference and
coverage checks, not an existing operator error budget or native-predictor run.
If its prerequisites are unavailable, record why it was not run and continue
the ordinary calibration path without claiming workflow isolation.

## Analyze a Collected Gap

Require formal IDs, admitted workflow observations from the selected engine
on the same cell and node group as the clean reference, Frontier stage/event
traces of the gated label, and the error budget. Use the verified observation
mapping in `engine_binding`; missing workflow probes leave this analysis
unverified without invalidating an independently supported E2E comparison.

For the existing vLLM profile, evidence includes the `schedule_timing` log and
`tests/comparison/dp_placement_pp/extract_vllm_placement.py` (placement,
engine iterations, dummy passes, joined by correlation id) and
`compare_placement.py`; the DP x PP mode labels from
`tests/comparison/e2e_gate/dp_pp_modes.py`; `alignment.csv` from
`op_gap_analysis.py` when an op run exists.

Compare arrival visibility, the wait before the first step, token budgets,
admission, preemption, phase changes, batch formation, per-request step-class
counts (prefill alone, prefill with decodes, several prefills, pure decode by
batch size), KV availability, DP synchronization and dummy passes, PP stage
hand-offs, transfers, and completion bookkeeping. Start from the first valid
non-empty formal scheduler batch with ordered ids and token vectors. Separate
client dispatch skew (`dispatch-align-trace`) from internal control flow, and
for DP x PP cells compare within the same lane mode before comparing across
modes when the case validates that conditioning. Keep the ordinary clean
comparison alongside it.

Record every row as `MATCH`, `MISMATCH`, `UNSET`, or `UNSUPPORTED` with values,
source anchors, confidence, affected phase, error-source class, and Frontier
owner. Require the shared first-batch and three-aligned-batch predicates before
calling operator comparison meaningful; otherwise route to `op-supplement`.

Write `analysis/workflow_gap_table.csv`, summary, and status. This entry is
read-only and does not permit a code change.
