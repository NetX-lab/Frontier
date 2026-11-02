# W128-V4-MOE-PACKAGE plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Established the two-file preparation scope and root-only runtime gates. |

## Dependency graph

`corrected root/native/producer evidence -> two-file builder/plan -> fresh package -> {exact dry runs, fresh entry controls, original/input hashes, relevant existing tests,77-matrix binding} -> source commit -> records commit -> root reviews/image/SDK/seal/submission`

No parallel agents. CPU tests and independent source inspections may run concurrently. Native checkout is read-only `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69` at2cf0a69.

## Steps and acceptance

1. Read actual root corrected records and inspect existing launch/runtime mechanisms. Reuse the existing Q1 trace-retaining worker and W4 launcher; never use the historical trace-skipping worker or add a second worker implementation.
2. Implement only build_package.py/profile_plan.json (<=300 gross lines). Stage committed base56e9405d frontier/data/config, retained launcher/worker/wheels, corrected root input records and native source pins. All native/image pins are preparation inputs, not completed runtime gates.
3. Generate only a new path. Run existing MoE main --dry_run twice from the frozen snapshot, verify exact four canonical rows/method and separate directories. Pin all package inputs and originals; refuse a pre-existing destination.
4. Run fresh launcher and worker entry controls without preloads; stop before platform/runtime operations. Check worker source-based trace retention and SDK mem/preemptible semantics through source/CPU stand-ins only. Run existing MoE/routing/tracer tests; bind the existing77-case comparison to unchanged simulator domains.
5. Re-read requirements.md, run check-progress, record exact evidence/limits/line count, and commit source separately from records. Root still owes image, SDK/platform, quota, local mount, seal, GPU execution and retained-data acceptance.
