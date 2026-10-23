# D29 timeout diagnosis plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record and close the diagnosis-only feedback-loop scope. |

Dependency: source/terminal census -> real-path RED -> ranked hypotheses ->
single-variable observations -> bounded handoff. Implementation and complete
CLI acceptance are outside this plan.

| Step | Acceptance | State |
| --- | --- | --- |
| Bind source, fixture and failure | Use e41/3dac and actual main07 records; distinguish observations from extrapolation | completed |
| Build a real timing/call-count loop | Agent-runnable, seconds-scale, repeated deterministic RED at production query/pricing seam | completed |
| Profile and test one-variable hypotheses | Actual production methods, unchanged points/models/plans, retain result equality and measurements | completed |
| Report actionable owner seam | Exact commands, source references, diagnostic limits and allocation estimate | completed |

Verification uses the original stored CLI configuration with all output/cache
paths relocated before parsing. It loads the retained paired model bundle with
existing SHA256 and model-metadata checks, then invokes production position
query/pricing on the actual P6128 source and idle peer's physical one-row frame.
The loop skips expensive Simulator initialization through an existing test
construction seam; the relevant production methods, real Cluster/backend,
exact cost/count proofs and plan keys are unchanged.

Do not continue adding probes after this signal is established. Parent decides
the next repair allocation and owns original full-workflow verification.
