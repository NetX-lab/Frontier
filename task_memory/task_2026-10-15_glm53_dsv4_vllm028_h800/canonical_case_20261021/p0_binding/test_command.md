# Existing fixed-workload tests

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded the existing byte/count/actual Frontier loader checks only. |

Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. /data/ycfeng/envs/frontier-py310/bin/python -m pytest -q -p no:cacheprovider tests/unit/test_canonical_fixed_manifest.py -k 'not builder' --junitxml=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/canonical_case_20261021/p0_binding/fixed_tests.xml
```

Expected:3PASS; existing exact checksums,128-row identities/arrivals/lengths/disjoint warmups and actual Frontier loader without clipping. The already-accepted byte-rebuild test is deselected: this task creates no second manifest. All native evidence checks use the original-run APIs in `command.md`; no new test or source file.
