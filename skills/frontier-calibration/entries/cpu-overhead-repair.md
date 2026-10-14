# CPU Overhead Repair

**When to use:** Use when a decided CPU analysis proves a scoped Frontier host-side pricing correction, such as a missing or misordered host term in a step.

Read `../references/contract.md`, `../references/repair-approval.md`, and the
CPU edge in `../references/call-graph.md`. Require that predicate, a complete
CPU analysis with a reproducible `simulation_logic` part, `correction_state=pending`,
and the user's recorded decision with its decision ID; the human review record
follows `repair-approval.md`. Keep the change within the named Frontier
host-pricing owner and the CPU-overhead term schema
(`frontier/profiling/cpu_overhead/validation.py`); a table without the new
term must price exactly as before, so public examples stay unchanged.

Write the code-change marker (`change_marker`) with the decision ID and scope
before editing. Apply one scoped Frontier change as one commit, run focused CPU
tests and the regression set, rerun the affected cells with a fresh clean E2E
reading, then write `repairs/cpu_repair_receipt.json` with changed files,
commit, commands, before and after term values, the decision ID, and validation
status. Require fresh CPU and E2E analysis before another code change.
