# Operator Supplement

**When to use:** Use when operator analysis or the coverage check lacks a phase, shape, grid point, kernel gap, or device generation in the profiling tables.

Read `../references/contract.md` (Supplement Integrity),
`../references/tool-boundaries.md`, and `../references/pitfalls.md` (Admission
and Tables). Require an explicit missing tuple list from `op-gap-analysis` or
the coverage check. A supplement is measurement work: it runs when the user's
plan for the task includes it, as a planned ticket.

Plan the tuples from the formal batches: first formal batch, active phases,
in-flight endpoints, and the dense default grid of each profiler
(`docs/profiling/README.md`). Filter the attention grid against device memory
on the CPU first. Profile on the case's device generation, TP size, dtype, and
measurement family; label analytical rows instead of borrowing a neighboring
shape. Include `python -m frontier.profiling.kernel_gap.main` for graph-replayed
steps and the collective or KV benchmarks when the plan needs fits.

Run the plan with `tests/comparison/calibration/run_profiling_plan.py` on a GPU
worker (it writes `manifest.json` and `COMPLETE`), then publish with
`tests/comparison/calibration/merge_profiling_csvs.py --run-dir --output-dir
--table <name> <source> ... [--base-dir]` into a fresh directory. The merge
retains every base row, records repeated identities, rejects conflicts and rows
without measurements, checks source hashes, and writes `merge_receipt.json` and
`COMPLETE` last. Keep the plan as `e2e_profiling/profiling_plan_<purpose>.json`
and publish to `e2e_profiling/tables/<table set>/<model>/`, naming the set by
purpose, device, scope, and date plus a letter (for example,
`e2e-h200-t42-dense-20261007a`).

A new table set gets a new Frontier settings file. Re-run the coverage check,
then return to a fresh `op-gap-analysis` and fresh Frontier runs. This entry
changes data only and never authorizes a code change.
