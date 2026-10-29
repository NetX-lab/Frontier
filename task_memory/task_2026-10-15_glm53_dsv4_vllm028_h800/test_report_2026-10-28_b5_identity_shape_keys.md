# Test report: stage 1a assignment B5, kernel identity keyed by step shape (2026-10-28)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-28 | Created: lead review and verification of Codex assignment B5 (`a1c6212e`). |

## Scope

- Assignment: [h3o_operator_costs/codex_assignment_B5_identity_shape_keys.md](h3o_operator_costs/codex_assignment_B5_identity_shape_keys.md).
- Codex report: [h3o_operator_costs/codex-implement-a1c6212e-bb73-48bd-b5e5-e0dc95cd4dda/implementation.md](h3o_operator_costs/codex-implement-a1c6212e-bb73-48bd-b5e5-e0dc95cd4dda/implementation.md) (outcome `implemented`).
- Base: `42a86605`. The assignment named `56e06be2`, the HEAD when it was written; `42a86605` is the later records-only commit, so the base is correct.
- Changed files (all inside the assigned scope):
  - `tests/comparison/calibration/dsv4_kernel_identity_check.py`: native and pilot shape keys (step class, mode, padded rows, eager rows), `NO_REFERENCE`, status and verdict priorities, shape coverage, `--margin_cases` removed.
  - `tests/analysis/v4_runtime_census.py`: `ForwardKernels.scheduled_tokens` from the parsed step range.
  - `tests/unit/test_dsv4_kernel_identity_check.py`: new and updated tests; the file is also reformatted to PEP 8 spacing.

## Lead review

- Pilot padded rows equal `cudagraph_dispatch.num_tokens` and are checked against the unique grid x of `_fused_q_kv_rmsnorm_kernel`; native padded rows come from the same kernel and must be unique. Eager rows are `None` in FULL, the padded rows for a dummy, and the scheduled rows otherwise. Matching uses the whole key; deduplication uses the key plus the kernel multiset.
- Lead corrections (direct, small):
  - the pilot dummy rule tested `spec.phase == PHASE_DUMMY and not requests`; a dummy step always has empty requests, so the extra condition was removed;
  - the module docstring was reflowed to the file's line width.
- Analysis scripts: `--margin_cases` removed from `h3o_operator_costs/evidence/p1h_bridge/package_03/run_analysis.sh` and from the scratch copy `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/run_analysis.sh`.

## Checks (after the lead corrections)

Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), `PYTHONPATH=$W:$W/tests/unit:$W/tests/comparison/calibration:$W/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `WANDB_DISABLED=true`, `VIDUR_DISABLE_WANDB=1`, `ulimit -v 31457280`.

| Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| Unit and regression suites | `pytest tests/unit/test_dsv4_kernel_identity_check.py tests/unit/test_dsv4_composition_check.py tests/unit/test_dsv4_graph_bridge_check.py tests/unit/test_dsv4_composition.py tests/unit/test_dsv4_graph_replay_profiling.py tests/unit/test_dsv4_attention_profiling.py -q -p no:cacheprovider` | all pass | 296 passed | PASS |
| Identity, P2-32 | `dsv4_kernel_identity_check.py --config /data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/bridge_config.json --native_trace_dirs .../dsv4-h200-p2-profile-20261025-01/artifacts/torch_traces --output /data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b5_review/identity_p32.json` | 46 PASS, 129 NO_REFERENCE, 0 FAIL, verdict PASS, exit 0 | 46 / 129 / 0, PASS, exit 0 | PASS |
| Identity, P2-64 | the same with `dsv4-h200-p2-64-profile-20261025-01` | 54 PASS, 121 NO_REFERENCE, 0 FAIL, verdict PASS, exit 0 | 54 / 121 / 0, PASS, exit 0 | PASS |
| Cold imports | Codex run: `import dsv4_kernel_identity_check`, `import v4_runtime_census`, no torch, vllm or cuda modules loaded | exit 0 | exit 0 | PASS (Codex run; the lead's edits do not change imports) |

Shape coverage (both trace sets): 7 referenced and 16 unreferenced pilot shapes. Referenced at P2-64: decode FULL 1 and 16, dummy FULL 1, dummy NONE 1, dummy PIECEWISE 1, first tail alone PIECEWISE 1, prefill NONE 6144. P2-32 has decode FULL 2 instead of decode FULL 16. The 11 P2-64 PIECEWISE steps at 16 padded rows that failed under the three-part key are now `NO_REFERENCE` (pilot eager rows 8 and 11, native 9 and 13).

Evidence copied to `h3o_operator_costs/evidence/p1h_bridge/package_03/`: `identity_shape_p32.log`, `identity_shape_p64.log`, and `identity_shape_{p32,p64}.coverage.json` (verdict, status counts, shape coverage and cases from the full reports in `/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/b5_review/`).

## Limits

- The check proves kernel containment only at the shapes the retained native traces contain. The 16 unreferenced shapes per trace set remain unqualified by it; the stage 4 end-to-end replays qualify them.
- CPU only; no GPU run.
