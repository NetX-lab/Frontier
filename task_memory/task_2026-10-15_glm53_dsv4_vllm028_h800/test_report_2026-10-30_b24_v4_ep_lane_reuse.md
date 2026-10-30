# Test report: B24, one prediction per distinct V4 EP lane in a wave (stage 4)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-30 | Created: lead review of B24 (Codex request `9de7af1c`), the CPU-torch rerun, the 77-case fidelity matrix of main `901bc091`, and the start of the full-length identity check. |

## Scope

B24 removes repeated work from the EP wave prediction of the V4 operator path ([assignment](h3o_operator_costs/codex_assignment_B24_v4_ep_lane_reuse.md)). `predict_ep_wave_phase_times` ran one full V4 FFN prediction for each EP participant, 32 per wave on P2-64. It now predicts each distinct key once per wave and gives the five phase times to the other participants with the same key. The key is the source DP index plus every `EPLaneWorkload` field except `ep_id` and the contiguous `owned_expert_ids`. A check fails when the dataclass gains a field that the key does not cover. Lane construction, SP assignment and workload logging stay per participant. Runs that capture lane records keep one prediction per lane.

- Branch `b24-v4-ep-lane-reuse`, commit `a7a7cbf5`; merged into main as `901bc091`.
- Codex report: `h3o_operator_costs/b24/report.md` (audit of every rank-dependent input in `h3o_operator_costs/b24/audit.md`); scratch `/data/ycfeng/tmp/codex-implement/9de7af1c-59f2-4ff4-a3ca-bfa1bf52e247/`.
- Environment: `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6), `PYTHONDONTWRITEBYTECODE=1`, BLAS/OMP/MKL threads 1, `ulimit -v 31457280` for tests and 62914560 for simulations.

## 1. Lead checks

| Check | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| Torch-dependent tests that the Codex environment skipped, the new tests and the changed consumer test | `pytest tests/unit/test_moe_rank_routing.py tests/unit/test_glm_moe_profiling.py::test_expert_parallel_layout_accepts_only_contiguous_aligned_maps tests/unit/test_v4_ep_lane_reuse.py tests/unit/test_dsv4_operator_consumer.py` with CPU torch on `PYTHONPATH` (`/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages`); log `/data/ycfeng/tmp/b24_lead/torch_tests.log` | all pass, none skipped | 79 passed, 0 skipped | PASS |
| New tests after the lead's docstring edit | `pytest tests/unit/test_v4_ep_lane_reuse.py` | 13 passed | 13 passed | PASS |
| Fidelity matrix | `bash /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/run_fidelity_b24.sh` (77 cases, worktree `.worktrees/fidelity-candidate-901bc091` at main `901bc091`, output `stage4/fidelity_b24/`) against `stage4/fidelity/candidate_b6c28858` | 77 of 77 identical | 77 identical, 0 mismatched, 0 failures, 0 cache-name differences | PASS |
| Hygiene | `wc -l`, `/data/ycfeng/tmp/unused_imports.py`, `git diff --check` | at or below 2,000 lines, no unused imports, clean | `expert_parallel.py` 871 lines; no unused imports; clean | PASS |

Lead edit: the function docstring keeps the statement that lanes are predicted and logged in participant order, and adds the reuse rule.

## 2. Codex checks (from its report, not rerun by the lead)

| Check | Result |
| --- | --- |
| S1: P2-32, eight requests, cold | no differences in parsed artifacts or cache names |
| S2: P2-64, eight requests, warm copies | no differences; 22,102 predictions for 176,128 participants over 5,504 waves |
| S3: S1 with operation metrics on | no differences |
| Speed, full P2-64 trace, 900 s window from time 0, base and candidate side by side | 8.64 against 17.40 simulated seconds, 2.02x |
| Regression selection | 882 passed |
| Dummy smoke matrix | 12 of 12 |

## 3. Full-length identity check (running)

The K=19 A10 replays on main `901bc091` (`stage4/a10/p2_{32,64}_{cold,warm}_k19b24`, `stage4/run_a10_k19_b24.sh`, detached worktree `.worktrees/a10-k19-901bc091`) and on main `50ed2061` (`stage4/a10/p2_{32,64}_{cold,warm}_k19`) differ only by B24. Their metrics must be byte-identical. Codex estimated about 2.2 h per replay with B24, against 5.6 h before.
