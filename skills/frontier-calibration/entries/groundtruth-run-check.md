# Ground-Truth Run Check

**When to use:** Use when a ground-truth run needs its preflight before launch, or its evidence check after the run before any analysis reads it.

Read `../references/contract.md` (Ground-Truth Checkout, Run Modes) and
`../references/pitfalls.md` (Ground-Truth Collection). The mode switches are
those of `tests/comparison/dp_placement_pp/vllm_replay.py`; the check itself is
case-local (`../references/tool-boundaries.md`, Case-Local Tools).

## Preflight

Before launch, verify and record in `runs/groundtruth_<mode>/<run_tag>/run_manifest.json`:

- the checkout tuple: branch `feature/frontier-comparison-instrumentation`,
  recorded commit, clean tree, diff artifact and its SHA-256, overlay patch
  state; a local commit carries its ahead count;
- the mode, its engine file, and the mode predicate: `clean` writes request
  metrics only, every diagnostic mode leaves request metrics off;
- at least three warmups covering every prompt-length class, drained before
  the first formal arrival; the compile-cache warm pass for eager MoE;
- formal and warmup request ids, the cell trace hash, and fresh output paths;
- the job grouping: the clean run and the runs that feed tables share one job;
  static-port jobs run alone.

A failed preflight stops before `groundtruth-run`.

## Post-Run Check

After the run, check every run tag and write
`analysis/run_check_status.groundtruth_<mode>.json` with one PASS or FAIL per
check and its evidence:

- checkout and overlay unchanged; inputs unchanged since launch; harness
  scripts pinned; GPU type and count as declared; terminal status `COMPLETE`
  with exit 0;
- mode predicate honored; every formal request served once with its trace
  output length; warmups present;
- the KV block count derived from the server log (the source of Frontier's
  block count for this cell);
- PDD: both connector checks and a complete `pd_metrics.py` join;
- DP > 1: no request split across engines; decode graphs: capture complete;
- `cpu` and its variants: every engine step in execution order, every trace
  request, and stage records covering every forward;
- `device_timeline`: `device_timeline.json` covering rank 0 of the world size.

A run's status is PASS only when every check passes. Every passing clean run
of a cell that meets the case's routing status is an admissible reference for
the gate. CPU-probe admission into a table is a separate rule owned by
`cpu-overhead-supplement`.

This entry changes no code or data. A PASS admits the run as evidence; it says
nothing about numeric parity or whether enough independent clean runs exist.
The case-level reference-sufficiency check follows `contract.md`, Calibration
Objective; it does not discard ordinary slow runs that passed this check.
