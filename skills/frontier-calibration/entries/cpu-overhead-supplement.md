# CPU Overhead Supplement

**When to use:** Use when CPU analysis reports missing or unadmitted host rows and the user approved the measurement work for the task.

Read `../references/contract.md`, `../references/tool-boundaries.md`, and
`../references/pitfalls.md` (Ground-Truth Collection, Admission and Tables).
Require an explicit missing tuple list from `cpu-overhead-analysis`, its
predicate in `../references/call-graph.md`, and the user's recorded approval.
Plan each tuple as `model, batch, TP, prefill tokens, decode tokens, scheduling
mode, precision, phase`.

Collect `cpu` mode runs in the same job and node as their paired clean run,
after the warm pass, with probes isolated from E2E and operator
instrumentation. Admit each probe run with `tests/comparison/e2e_gate/run_gate.py
admit --rule <rule> --probe <run> --reference <clean run> ...`; the ratio must
lie in 0.97-1.03:

| Rule | Cells | Probe measurement | Reference |
| --- | --- | --- | --- |
| `client_e2e` | all cells | mean client E2E | the cell's clean run in the same job |
| `send_span` | PDD prefill, with `client_e2e` | median summed per-layer KV-send span of isolated prefills | mean over the job's clean runs |
| `isolated_prefill` | DP x PP cells | the first formal request's isolated prefill step | median of the job's CPU-probe cohort |

A run that misses its rule is rejected, or published only with an
`accepted_deviation` the user decided; it never enters silently.

Produce the tables with `python -m frontier.profiling.cpu_overhead.vllm_cpu_probe`
(one `--cpu_probe_logs` per serving instance, `--num_pipeline_stages`,
`--engine_idle_edges_ms 0 100 500`; the published recipes in
`tests/comparison/e2e_gate/artifacts.json` show complete argument lists) into a new
`inputs/cpu_overhead_<generation>/` directory with `provenance.json`: producer
commit and arguments, every input log's SHA-256 with its run check and
admission result, and the output SHA-256. Logs from vLLM-BS commits that lack
the request-phase and KV-load fields are rejected by the producer; rebuild old
tables with `run_gate.py cpu-table --producer-tree`. Never edit a published
table.

Return to `cpu-overhead-analysis` with a new settings generation and fresh
Frontier runs. This entry changes data only and never authorizes a code change.
