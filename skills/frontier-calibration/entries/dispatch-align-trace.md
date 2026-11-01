# Dispatch Align Trace

**When to use:** Use when client dispatch or frontend routing delay changes which requests a DP lane or step sees, as in DP x PP cells and online cases with visible dispatch skew.

Read `../references/contract.md`, `../references/tool-boundaries.md` (Case-Local
Tools), and `../references/pitfalls.md` (Ground-Truth Collection). Frontier
replays the cell trace's intended arrivals; the reference sees each request only after
the client dispatches it and the frontend routes it to a DP engine. When that
delay decides lane placement or step membership, validate the engine-specific
dispatch/route observations and the clock mapping before constructing a trace.
Record whether the comparison uses intended arrivals, observed arrivals, or a
modeled delay distribution. A distribution used for prediction needs evidence
that it represents normal behavior and validation on fresh runs. Keep the
ordinary-user comparison visible alongside diagnostic replay.

## Existing vLLM DP x PP Recipe

Use the following recipe only for a case that has validated its dispatch
abstraction and the existing reader/replay schema. Other engines need their
own supported observations and mapping; a frontend route timestamp is not
a universal engine output.

Measure the pool from the cell's passing clean runs: the interval from client
dispatch to the frontend's route of each formal request. Delays above the
API-server stall bound (`stall_ms`) are stalls that Frontier does not model and
stay out of this diagnostic pool under that declared bound. They remain in
the ordinary clean-reference population unless independently invalid. Each Frontier ensemble member draws one delay per request
from the pool with its own seed (the published DP x PP cells use seeds 1-5) and
adds it to that request's `arrived_at`; every other trace field stays as it
was. The case runner does the draw (`--route-delay-seed`) and writes the
member trace as `trace.csv` in that Frontier run directory.

Record the draw in the run receipt and, for a published cell, in the scenario
matrix entry of that Frontier run: `route_delay` (`seed`, `groundtruth_runs`,
`samples`, `stalled_samples`, `stall_ms`, `measured_ms`), the drawn
`route_delay_ms`, and `member_trace_sha256`. `run_gate.py frontier` rebuilds the
member trace from these and fails when its SHA-256 differs. Before the run,
compare the member trace with the base trace: the same ids, token counts, and
order. Reject a duplicate id, a negative or non-finite delay, or any change
other than `arrived_at`.

This entry changes a data trace only. It never closes the E2E gate, and a
replay heartbeat or harness change belongs to `groundtruth-run`.
