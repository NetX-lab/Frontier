# Plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Created the isolated-reader implementation and CPU verification plan. |

Dependency: A -> B -> C -> D; inventory audit is independent of B.

| Step | Scope | Acceptance |
| --- | --- | --- |
| A | Native/Batch/CSV contract and clean worktree | Pinned native revision; evidenced decode/tail distinction; available layout identified. |
| B | Small attention-only reader extension | Exact ongoing q1 selection and receipt; existing prefill and all serving guards preserved; no shared-core edits. |
| C | Focused existing V4 tests and real accepted tables | Both methods, six roles, context boundaries, exact latencies/provenance and meaningful typed negatives pass. |
| D | Audit strict77, run eligible matrix, commit/archive | No GLM-specific case runs; clean committed-source strict comparison when eligible; scoped diff <=500 changed lines; records and qualifications retained. |

Parallelism: source/layout investigation and inventory audit use independent read-only local tools. No further delegation is needed for this bounded slice.

Verification: Python3.10.6 `/data/ycfeng/envs/frontier-py310/bin/python`, CPU only, local temp root `/data/ycfeng/tmp/opencode-v2-tmp/opencode`, `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`. Reuse `tests/unit/test_dsv4_graph_measured.py` and other V4-only workload/source tests. Existing strict driver `tests/e2e/refactor_fidelity/measure_commit.py`, baseline `/data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/baseline`. Never run global unit suite.
