# Dispatch Align Trace

**When to use:** Use when client dispatch or frontend routing delay changes which requests a DP lane or step sees, as in DP x PP cells and online cases with visible dispatch skew.

Read `../references/contract.md`, `../references/tool-boundaries.md`, and
`../references/pitfalls.md` (Ground-Truth Collection). Frontier replays the
cell trace's intended arrivals; vLLM sees each request only after the client
dispatches it and the frontend routes it to a DP engine. When that delay
decides lane placement or step membership, Frontier needs the measured delays.

Take the delays from the cell's passing clean runs: the interval from client
dispatch to the frontend's route of each formal request. Delays above the
API-server stall bound are stalls that Frontier does not model; leave them out
of the pool and record the bound. For each ensemble member, draw one delay per
request from the pool with a recorded seed (the published DP x PP cells use
seeds 1-5) and add it to that request's `arrived_at`. Generate the base trace
with `tests/comparison/dp_placement_pp/make_trace.py` when the cell has none.

Write `analysis/dispatch_aligned_trace.csv` per member and
`analysis/dispatch_alignment_manifest.json` with the pool source runs, the
bound, the seeds, and each member trace's SHA-256; the scenario matrix records
the same hashes and `run_gate.py frontier` checks them. Re-read each member
trace and compare ids, token counts, priorities, and payload with the base
trace. Reject a duplicate id, a negative or non-finite delay, or a changed
request.

This entry changes a data trace only. It never closes the E2E gate, and a
replay heartbeat or harness change belongs to `groundtruth-run`.
