# Operator Gap Analysis

**When to use:** Use when the error budget attributes a gap to device price, so matched step classes need a per-operator or per-kernel comparison.

Read `../references/contract.md` (Operator Comparability Predicate),
`../references/evidence-tiers.md`, and `../references/tool-boundaries.md`.
Require formal ids, a passing clean run and passing `op_timing` and
`kernel_timing` runs of the same trace on one node, a Frontier `--op-trace` run
under its own label, lane and rank provenance, and profiling coverage.

Native side: `tests/comparison/calibration/op_gap_analysis.py --clean-run
--op-run --kernel-run --request-ids --output-dir analysis/op_gap/` measures
batch windows, scope times per shape class, host gaps (op scope less kernel
scope), peer state across DP ranks, dummy passes, and the measurement cost per
decode-cycle class. It counts a TP all-reduce nested in a projection scope once
and lists join problems instead of dropping rows. Add
`forward_device_timeline.py` output from a `device_timeline` run when device
idle between kernels matters, and `flashinfer_decode_plan_bench.py` when graph
and eager attention plans may differ.

Normalize scheduler, batch, phase, lane, and operator identity internally and
keep `analysis/op_trace_normalized.jsonl` as an intermediate artifact. Select
the first valid formal batch and require at least three distinct aligned
logical batches covering every active phase. For each matched key compute
signed `gap_ms`, absolute gap, and relative gap with a positive finite
denominator. Read each step class with its measurement family (CUDA-event
tables for eager steps, kernel-only tables plus `kernel_gap.csv` for
graph-replayed steps) and compare padded widths with vLLM's. Report the
`kernel_timing` run's scope times only; its step times are distorted by the
device syncs.

Preserve missing, conflicting, and excluded rows; route missing coverage or a
grid gap to `op-supplement`. Assign each attributed part an error-source class:
a table or coverage problem is `measurement_input`, a wrong family, width, or
pricing rule is `simulation_logic`.

Write `analysis/op_gap_summary.md` and `analysis/op_gap_status.json` beside the
tool outputs. This entry only reports evidence; it never changes predictors,
profiling inputs, or Frontier code.
