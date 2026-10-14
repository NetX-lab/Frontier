# Workflow Gap Analysis

**When to use:** Use when the error budget attributes a gap to queueing or step mix, or Frontier and vLLM batches, lane modes, or timing paths diverge and the control flow must be explained.

Read `../references/contract.md`, `../references/evidence-tiers.md`, and
`../references/pitfalls.md` (Reading the Gate, Analysis and Correction).
Require formal ids, a passing `schedule_timing` run of the same cell and node
group as the clean reference, Frontier stage and event traces of the gated
label, and the error budget.

Evidence: the vLLM scheduler log; for DP cells,
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
modes.

Record every row as `MATCH`, `MISMATCH`, `UNSET`, or `UNSUPPORTED` with values,
source anchors, confidence, affected phase, error-source class, and Frontier
owner. Require the shared first-batch and three-aligned-batch predicates before
calling operator comparison meaningful; otherwise route to `op-supplement`.

Write `analysis/workflow_gap_table.csv`, summary, and status. This entry is
read-only and does not permit a code change.
