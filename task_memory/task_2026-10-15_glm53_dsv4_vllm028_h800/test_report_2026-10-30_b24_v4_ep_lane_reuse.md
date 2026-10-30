# Test report: B24, one prediction per distinct V4 EP lane in a wave (stage 4)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-30 | Section 3: P2-32 full-length identity PASS (cold and warm); A8 census counts with and without B24. |
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

## 3. Full-length identity check

The K=19 A10 replays on main `901bc091` (`stage4/a10/p2_{32,64}_{cold,warm}_k19b24`, `stage4/run_a10_k19_b24.sh`, detached worktree `.worktrees/a10-k19-901bc091`) and on main `50ed2061` (`stage4/a10/p2_{32,64}_{cold,warm}_k19`, detached worktree `.worktrees/a10-k19-50ed2061`) differ only by B24. Their metrics must be byte-identical. Codex estimated about 2.2 h per replay with B24, against 5.6 h before.

Scripts and outputs are copied to `h3o_operator_costs/b24/identity/`: `k19_identity.py` byte-compares every file under `metrics/` and `collectives.json`; `k19_identity_detail.py` lists the differing `config.json` keys and `collectives.json` leaves and compares the decompressed `forwards.jsonl.gz`.

| Case | Command | Expected | Actual | Result |
| --- | --- | --- | --- | --- |
| P2-32 warm | `python k19_identity.py stage4/a10/p2_32_warm_k19 stage4/a10/p2_32_warm_k19b24`, then `k19_identity_detail.py` on the same roots (`k19_identity_p2_32.txt`, `k19_identity_p2_32_detail.txt`) | `request_metrics.csv`, `system_metrics.json`, `metrics_ground_truth.jsonl`, `frontier_transfer_ledger.jsonl` and `op_precision_metadata.csv` identical; `config.json` differs only in run paths; `collectives.json` differs only in predictor call counts and their totals; forwards equal | 5 metrics files identical; `config.json` 3 of 913 keys differ, all run-root paths (`cc_backend_config.cache_dir`, `metrics_config.cache_dir`, `metrics_config.output_dir`); `collectives.json` 50 of 648 leaves differ, 25 `call_count` and 25 `predicted_total_ms`; forwards equal | PASS |
| P2-32 cold | same, cold roots | same | same as warm: 5 metrics files identical, the same 3 path keys, 50 leaves (`call_count` and `predicted_total_ms`), forwards equal | PASS |
| P2-64 warm and cold | same, P2-64 roots | same | running: the `50ed2061` replays were still writing `metrics_ground_truth.jsonl` at 15:40 | pending |

`collectives.json` counts predictor calls, not simulated collective calls, so B24 lowers the counts by design. The A8 census of the two warm P2-32 runs shows the size (`a8_p2_32_warm_k19.txt`, `a8_p2_32_warm_k19b24.txt`): 19,436,010 calls without B24 against 6,310,644 with it, all exact and 0 violations in both. The census of the earlier run on main `9a4f3471` (before B26) counted 19,438,100. The drop to 6.3 million therefore comes from B24's reuse, not from the B26 host run-ahead, which changed the count by 2,090 calls. The A8 census of a B24 run therefore reports predictions, not calls; its exactness and violation results are unaffected because each reused prediction is the same priced point.

The gate of the warm run without B24 (`stage4/a10_gate/p2_32_warm_k19_post/gate_stdout.txt`) gives the same values as with B24: TTFT 335.1145 against 364.0296 ms (−7.94%), TPOT 19.9269 against 19.9190 ms, E2E 20,720.30 against 20,741.13 ms, throughput 6,226.14 against 6,231.42 tokens/s, PASS, routing MATCH.
