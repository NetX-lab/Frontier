# Test report: one sync waiting room and one sync kind per cluster (W3-R5, W3-R6)

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-24 | Created: review of the Grok G1 commit, gates against `6d59946`, merge into PR 35. |

## Subject

- Decision Q5 (a) (`requirements.md`, 2026-09-24):
  - `initialize_sync_waiting_rooms` sets only `_sync_waiting_room` and `_sync_kind`.
  - The alias rooms, `uses_shared_forward_room`, the three-way switches and the per-kind open-step partition are removed.
  - No numeric change is allowed.
- Commit `341970d` on `refactor/sync-waiting-room` (parent `6d59946`), 19 files, 147 insertions and 291 deletions.
- Grok task G1 wrote it:
  - Prompt: `/data/ycfeng/tmp/grok_runs/prompts/g1_w3.md`.
  - Report: `/data/ycfeng/tmp/issue26-correctness-pr/followups_20260924/w3/g1/report.md`.
  - G1 finished the partial Opus draft (backup `followups_20260924/w3/partial_opus_implement.diff`).
- Merged into `fix/issue26-correctness-pr` as `1978b72` (merge, no rebase) and pushed.

## Review of the diff (lead)

| Point | Finding |
| --- | --- |
| Room and kind | `SYNC_KIND_BY_CLUSTER_TYPE` in `sync_state.py` maps MONOLITHIC to `"forward"`, PREFILL to `"prefill"` and DECODE to `"decode"`. The constructor calls the initializer only for those three roles, as before. A dense model gets `None`, so `enter_layer_sync` keeps the dense-model `ValueError`. |
| Handler selection | `enter_layer_sync` now picks the wave handler by `_sync_kind`; before, it used the entering batch's phase. The two are equal on every reachable path. `replica_stage_schedule_event.py:173-184` sends PREFILL batches only to the prefill sync path and DECODE batches only to the decode sync path. On MONOLITHIC the shared room already selected the forward handler. Before the change, an entry whose phase disagreed with its role failed on a `None` room, and no event produces one. |
| DECODE_ATTN / DECODE_FFN | Neither role gets the attributes. DECODE_FFN MoE returns in its own branch (`replica_stage_schedule_event.py:472`) before the `DecodeSyncEvent` branch at `:482`. The `direct_batch is None and self._sync_kind == "forward"` checks short-circuit for direct batches. |
| One open-step map | Each cluster scheduler owns one `ForwardSyncState`, and before the change each cluster used exactly one kind. One map per cluster therefore binds the same keys. |
| Removed guard | `model_config is not None` is dropped. `ReplicaConfig` always builds `model_config` from the model name (`frontier/config/replica_config.py:127`), so the guard covered a state that cannot occur. |
| Tests | Fixtures call `initialize_sync_waiting_rooms` or set the two attributes. The per-kind assertions become one `_open_steps == {}` check, which still fails if any binding is left open. No test was removed, and the JUnit ids are unchanged. `test_prefill_and_decode_share_monotonic_identity_allocator` was restored under its own name. |
| Harness | `tests/e2e/stage_admission_matrix.py` reads `getattr(cluster_scheduler, "_sync_waiting_room", None) or {}`. The harness walks every cluster, including PD-AF roles without the attribute. |
| Quality gates | No new flag, no hard-coded value, no filler names. The one new table is the role-to-kind classification. `base_cluster_scheduler.py` is 1936 lines. |

## Gates against base `6d59946`

**Commands.**
- `bash gates/run_side.sh .worktrees/sync-waiting-room w3`, then `bash gates/compare.sh .worktrees/sync-waiting-room w3`.
- The scripts are under `/data/ycfeng/tmp/issue26-correctness-pr/followups_20260924/gates/`.
- Run from 2026-09-24T08:30:55Z to 08:37:50Z at `341970d`.

**Environment.** Python `/data/ycfeng/envs/frontier-py310/bin/python` (3.10), with `WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 FRONTIER_TMP_ROOT=/data/ycfeng/tmp PYTHONDONTWRITEBYTECODE=1`.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Unit suite (JUnit) | 0 regressions, no id changes | Before and after: passed 3828, failed 84, skipped 51, error 10. Regressions 0, new failures 0, `only_before` 0, `only_after` 0 (`compare_w3/suites_unit.json`) | PASS |
| Integration suite (JUnit) | 0 regressions | Before and after: passed 33, skipped 22, error 5 (absent PD-AF Reference checkout). Regressions 0 (`compare_w3/suites_integration.json`) | PASS |
| Fidelity matrix | 74 of 74 identical | 74 identical, 0 mismatched, 0 failures, complete comparison (`compare_w3/comparison.json`). Compare exit 1 comes only from the provenance flag, which the untracked `outputs/metrics/meta_llama_llama_2_7b_hf/` directory raises. `git status` shows no tracked change | PASS |
| Release examples | 16 of 16 identical | 16 of 16 identical (`compare_w3/examples.log`) | PASS |
| Stage-admission matrix G3b, G9, G10 | All PASS | 51 of 51 PASS, compare exit 0 (`compare_w3/stage-matrix.log`) | PASS |
| W9-05 KV-pressure probe | 72 of 72 drained | 72 drained, 0 double-scheduled, 0 short outputs (`gates/w3/probe.log`) | PASS |
| Merged tree `1978b72` | Focused tests pass; removed names absent | 185 passed: the ten G1 unit files, `test_collective_sim_zero_payload.py` and `test_monolithic_mixed_forward_runtime.py`. `git grep` of the removed names outside `task_memory` is empty | PASS |

G1 also ran its own checks. Six MoE configurations compared byte-identical with BASE: monolithic `attn_dp` 1 and 2 × PP 1 and 2, one PDD run and one PD-AF run, each with a non-empty EP ledger. G9 scored 11 of 11 by the matrix rules (G1 report, section 4). The lead's gate run above supersedes those as the acceptance evidence.

## Limits

- The change is structural. Identical outputs over the matrix, which includes dummy-timed and trained cases, show no numeric effect on the covered configurations; configurations outside the matrix are covered by the code argument in the review table.
- Historical task records still name the removed attributes as the record of the earlier design.
