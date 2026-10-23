# D29 artifact bytes repair completion archive

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archive the verified allocated repair and root-owned acceptance limits. |

TaskID `D29-COLLECTIVE-ARTIFACT-BYTES`, session `ses_e9669de64ffeH614F6iNYIqgeb`.
Sole-writer worktree `/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-bytes`,
branch `fix/d29-artifact-bytes`; code commit
`2f2d9339d5bce600fbb0a3f78aebbcf6f71be696`, based on `2cfbb344`.

Both existing artifact caches now read current raw bytes on every query/prediction
and decode only for changed-byte JSON parsing. Previous successful cache state
and every remaining source/numerical check stay unchanged. The existing test
hook reads bytes and returns its original parser marker after decoding; all20
backend assert ASTs and all other test files remain unchanged.

Deliverables in this directory:
- `requirements.md`, `plan.md`, `progress.md`, `checkpoint.md`: allocated scope,
  execution, observed failures/resolutions and phase route.
- `code.diff`: review-ready three-file source/test diff.
- `receipt.json`, `baseline_binding.json`, `candidate_binding.json`: exact source,
  committed byte identity, original/derived script SHA and evidence paths.
- `test_report_2026-10-24_artifact_bytes.md`: exact commands, observed values,
  acceptance conditions and limits.

Observed validation: baseline78 PASS/127.120s; candidate78 PASS/124.657s, no
FAIL/ERROR/SKIP. Actual complete layer60 has16 EP,80 reads/wave unchanged, two
cold/zero warm artifact parses and identical full208 events/costs/proofs/keys,
pricing payload and ledger. Warm pricing median1.037333→0.325204s (3.1898x),
with candidate samples0.421348/0.310525/0.325204s. Timing is not a full-run result.
Original fixture03/cache names/sizes/mtimes stay unchanged for all262 files;
selected model bytes match their original bundle SHA. Existing replacement/error
seams and64 output-key negatives pass; original3,902-byte late-error prefix remains.

Git step: backend10, workload14, test8, total32/36 gross (source24/test8),
net+2. Retained against `1aee0af`:84/91,197/323,235/235. No additional source/test
paths, owner/global allowance changes, submodule pin changes or peer writes.
The retained old-hook assertion failure is instrumentation-only, not a baseline
correctness defect. The final code matches measured candidate SHA exactly.

Root-owned remaining work: independent review, final union/regressions and source
freeze, full CLI/workflow, native/GPU/G3/E2E. This component does not admit them.
No active jobs, delegation, full CLI/trainer, CSV scan, GPU, network or dependency
operation was launched. The records commit is the commit containing this archive;
its exact ID is returned with the code commit in the task delivery.
