# Test report: spec-decode live batch (2026-09-24)

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-24 | Created for `e2c2937` (draft PR 39). |

## Scope

- Candidate `specfix`: `.worktrees/spec-decode-live-batch` at `e2c2937` (`fix/spec-decode-live-batch-metadata`, stacked on `67783c7`). Base `g3bfix`: the gate outputs of this branch at `df8ebc6`; `67783c7` adds only docs.
- Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6), `WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1`, tree under test as working directory and `PYTHONPATH`.
- Expectation: the new cases fail before the fix and pass after it; no other result changes, because a live copy of a batch with spec metadata is built only under preemption at PP4 with speculative decoding, which no gate cell exercises.

## Commands

```bash
T=/data/ycfeng/Frontier/.worktrees/spec-decode-live-batch
R=/data/ycfeng/tmp/spec-decode-live-batch/red_67783c7   # git archive of 67783c7 plus the new test file
python -m pytest tests/integration/test_vllm_v1_decode_preemption_runtime.py -q -p no:cacheprovider   # in $T and in $R
python -m pytest tests/unit/test_stage_execution_context.py tests/integration/test_vllm_v1_decode_preemption_runtime.py -q -p no:cacheprovider   # in $T
G=/data/ycfeng/tmp/issue26-correctness-pr/followups_20260924/gates
bash $G/run_side.sh $T specfix
BASE_TREE=/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr BASE_LABEL=g3bfix bash $G/compare.sh $T specfix
```

The PP4 case was chosen from a 72-cell probe (`/data/ycfeng/tmp/spec-decode-live-batch/probe/pp4_spec_cells.py`, outputs `pp4_red.json` and `pp4_green.json`): the smallest dense cell that raises on `67783c7` and completes with the fix.

## Results

| Check | Pass condition | Actual | Verdict |
| --- | --- | --- | --- |
| New integration case on `67783c7` | fails | 1 failed (`dense_pp4_spec_decode`: `planned_draft_tokens_per_request length mismatch: expected=1, got=2` at `global_batch_end_event.py:173` -> `batch.py:1076`), 4 passed | PASS |
| New cases with the fix | pass | 5 passed; with the unit file 35 passed | PASS |
| Unit suite | 0 regressions | 0 regressions, 0 new failures. 3806 passed, 84 failed, 10 errors, 51 skipped (base 3808, 84, 10, 50). Before-only: 3 `test_collective_sim_zero_payload` tests, skipped as a module here because the optional collective-sim binary is not built in this worktree. After-only: the new unit test and that module skip | PASS |
| Integration suite | 0 regressions | 0. 38 passed (base 37); after-only: the new case | PASS |
| Fidelity matrix | 74 of 74 identical | 74 identical, 0 mismatched, 0 failures; exit 1 from the provenance note only | PASS |
| Examples | 16 of 16 identical | 16 of 16 | PASS |
| Stage-admission matrix | every cell PASS | 51 of 51 PASS | PASS |
| W9-05 probe | 72 of 72 identical | 72 of 72 | PASS |

Evidence: `gates/compare_specfix_vs_g3bfix/`, `gates/{g3bfix,specfix}/probe/probe.json`.

## Limits

- Dummy timing and checked-in profiles only.
- The probe also found N1 (scheduler frontier keeps rejected drafts; see `progress.md`, CROSS-correctness-0). It is pre-existing, independent of this fix, and not addressed here; the chosen case does not hit it.
