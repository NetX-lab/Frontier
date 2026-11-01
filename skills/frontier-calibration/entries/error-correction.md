# Error Correction

**When to use:** Use when decided Frontier-vLLM analysis artifacts identify corrections that must be applied, revalidated, and republished.

Read `../references/contract.md`, `../references/repair-approval.md`, and
`../references/call-graph.md`. This is the only composite that coordinates code
changes. Select one decided domain at a time: semantic alignment, workflow,
operator, residual lane, or CPU. Each domain needs a complete analysis and the
user's recorded decision before its entry runs.

Write `repairs/repair_plan.md` with the decisions, their order, and one ticket
per correction. For each domain: call the domain entry, which records its
code-change marker, lands one commit, and runs the regression set; rerun the
affected cells; refresh the domain analysis; and obtain a fresh decision before
the next domain. Run workflow and operator corrections before any CPU
correction of the same metric. A `groundtruth_defect` returns to `parity-run`
with the new ground-truth commit and a new run generation.

Close with the gate:

1. Rerun every claimed cell the changes can reach and gate it with
   `e2e-metrics-gap`; cells the changes cannot reach must stay byte-identical.
2. Write `repairs/rerun_gate_table.csv` and `repairs/rerun_gate_summary.md` with
   absolute values and relative errors before and after.
3. Republish: update the cells' recorded Frontier commands and revisions in
   `tests/comparison/e2e_gate/cases/scenario_matrix.json`, the pins in
   `artifacts.json`, and `cases/published_readings.json`, and confirm
   `pytest tests/unit/test_e2e_gate.py -q` replays the fixtures.

Readings pin the Frontier commit that produced them, so a calibration branch is
merged, never rebased, and its readings stay separate from the PR stack under
test. The case closes when every applicable metric of every claimed cell is
within `0.10`, repeated clean references are sufficient, the claimed scope
matches ordinary-user evidence, and the diagnostic evidence is valid; a remaining failure is
reported with its class and the user's decision to defer it. This composite
never uses a result artifact as its own permission to change code.
