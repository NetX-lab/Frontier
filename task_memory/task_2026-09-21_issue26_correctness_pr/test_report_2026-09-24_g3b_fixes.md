# Test report: G3b dispositions A-E (2026-09-24)

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-24 | Created for commits `dfb0b25`, `20f0f94` and `df8ebc6`. |

## Scope

- Candidate `g3bfix`: this worktree at `df8ebc6` (fix/issue26-correctness-pr). Base `pr35base`: `.worktrees/rerun-review` at `99db5a1`. Between them only docs and the three commits differ.
- Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (Python 3.10.6), `WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1`, each run with the tree under test as working directory and `PYTHONPATH`.
- Expectation: no suite regressions; fidelity, examples, stage-admission matrix, W9-05 probe, C2 and KV sweep identical to the base. The three commits change only the report hooks after stale drops and deferred releases, dead guards, and the comparison harness, and none of these gates route through a changed report at PP4 with preemption.

## Commands

```bash
G=/data/ycfeng/tmp/issue26-correctness-pr/followups_20260924/gates
W=/data/ycfeng/Frontier/.worktrees/issue26-correctness-pr
bash $G/run_side.sh $W g3bfix > $G/g3bfix.driver.log 2>&1
BASE_TREE=/data/ycfeng/Frontier/.worktrees/rerun-review BASE_LABEL=pr35base bash $G/compare.sh $W g3bfix
bash /data/ycfeng/tmp/issue26-correctness-pr/followups_20260924/g3bfix_c2kv/run.sh
```

`run_side.sh` runs the unit and integration suites (`w9_01_stage_admission_ordering/composition_run_suites.sh`), the fidelity matrix (`tests/e2e/refactor_fidelity/run_matrix.py run --jobs 8 --clean-cache`), the 16 architecture examples, the stage-admission matrix groups G3b, G9 and G10, and the W9-05 probe. `g3bfix_c2kv/run.sh` runs `step9_p5/c2_pp1_policy_matrix.py` (run on both trees, then compare) and `w9_05/kv_pressure_sweep.py` on both trees.

## Results

| Gate | Pass condition | Actual | Verdict |
| --- | --- | --- | --- |
| Unit suite | 0 regressions, 0 new failures | 0 and 0. 3808 passed, 84 failed, 10 errors, 50 skipped (base 3828, 84, 10, 51). Before-only: 23 removed guard and seam tests, 2 construction tests renamed by their match string, 1 module-level skip of `test_collective_sim_zero_payload` in the base tree (collective-sim not built there; 3 tests run here) | PASS |
| Integration suite | 0 regressions | 0. 37 passed (base 33); after-only: the 4 new tests | PASS |
| Fidelity matrix | 74 of 74 identical | 74 identical, 0 mismatched, 0 failures. Exit 1 from the provenance note only (untracked `outputs/metrics/meta_llama_llama_2_7b_hf/`) | PASS |
| Examples | 16 of 16 identical | 16 of 16 | PASS |
| Stage-admission matrix | every cell PASS | every cell PASS | PASS |
| W9-05 probe | 72 of 72 cells identical | 72 of 72 | PASS |
| C2 PP=1 policy matrix | 24 of 24 identical | 24 of 24 (`c2 compare exit 0`) | PASS |
| KV-pressure sweep | 72 of 72 identical, every request complete | 72 of 72; all drained; 206 decode and 38 prefill preemptions; 0 short outputs | PASS |

Evidence: `gates/compare_g3bfix_vs_pr35base/{suites_unit.json,suites_integration.json,fidelity.log,examples.log,stage-matrix.log}`, `gates/{pr35base,g3bfix}/probe/probe.json`, `g3bfix_c2kv/{c2_compare.log,kv_sweep/*/kv_pressure_sweep.json}`.

## Limits

- The gates show that nothing outside the fixed paths changed. The fixed paths themselves are covered by the new integration tests (stale-drop and terminal-release reports at PP4), which pass here. Their failure on `c577222` is Grok's turn-1 red/green evidence, reviewed by the lead but not re-run.
- Every gate here uses dummy timing or the checked-in profiles; none is a native comparison.
