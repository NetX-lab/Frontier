# case_init — stage_admission_case_001

Immutable record. Written 2026-09-23, before the first GPU run.

| Field | Value |
| --- | --- |
| `case_id` | `stage_admission_case_001` |
| `run_generation` | 1 |
| `requesting_user` | `i-fengyicheng` |
| `reviewer_identity` | `i-fengyicheng` |
| `auto_recycle` | `false` |
| Ground-truth checkout | `/data/ycfeng/Frontier/.real-engine/vLLM-BS` |
| Ground-truth branch | `feature/frontier-comparison-instrumentation` |
| Ground-truth commit | `494b9f327036d4493034a9b37ebb343354884e01` |
| Ground-truth remote tip | `ea95f571e20937c7c908c6d59ddd1cd6bf9268f1` (local is one unpushed commit ahead) |
| Tree dirty | `false` |
| Diff artifact | `inputs/groundtruth_local_commit.diff`, SHA-256 `84fc24db0e2411268a93f8be7ca5f8e4e5072ea09d86063cac2cfb98381feb2c` |
| Fork changes over `upstream-v0.10.2` (`01efc7ef7`) | `inputs/fork_changed_files.txt`, SHA-256 `be638438ff661e1178f0c5ab68da7b9d249ebf8e641dcb1f6cd15d128c0abdaa` |
| Weight mode | `dummy`, no real weight download |
| Mode | instrumented only; no clean E2E run (plan D-8) |
