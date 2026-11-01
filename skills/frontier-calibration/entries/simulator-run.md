# Simulator Run

**When to use:** Use when a frozen Frontier case and cell are ready to produce simulator metrics and diagnostic traces.

Read `../references/contract.md` (Case Binding, Command Receipts) and
`../references/pitfalls.md` (Frontier Runs). Require the same manifest and a
passed semantic check.

Build one `python -m frontier.main` command from the case's native engine configuration and verified mapping and
its current-generation Frontier settings file, passed by explicit argument.
Bind:

- the cell trace, or the dispatch-aligned member trace named by the manifest
  for a DP x PP or dispatch study;
- the KV capacity and lifecycle mapping from the cell's passing clean run,
  including a block count when applicable;
- every profiling table path, the CPU-overhead tables, and `kernel_gap.csv`
  when the settings name them, so the checked-in default tables never load;
- non-dummy prediction and a predictor cache shared by the runs on the same
  tables (`--cache-dir` or `metrics_config_cache_dir`).

A published cell replays through `tests/comparison/e2e_gate/run_gate.py
frontier --cell <case_id>/<cell> --label <label>`, which checks every pinned
input hash and the member trace hash. A new label runs through the case runner
(`../references/tool-boundaries.md`, Case-Local Tools).

Write into a fresh `$FRONTIER_TMP_ROOT/<task>/<case_id>/<label>/` with
`run_manifest.json` (command, Frontier commit and dirty state, input hashes,
seed) before launch. Verify the exit code, a log free of fallback warnings
(for example, a CPU-overhead table without rows for the model prices zero),
and every declared metric. An `--op-trace` run is diagnostic and gets its own
label.

This entry executes a case only; it never changes Frontier or reference-engine source.
