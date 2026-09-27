# Test report: spec-decode live batch (2026-09-24)

## Modification History

| Date | Change |
| --- | --- |
| 2026-09-24 | Added the N1 fix `fd096ea` (rejected drafts off the vllm_v1 frontier) and its gates. |
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
- The probe also found N1 (scheduler frontier keeps rejected drafts; see `progress.md`, CROSS-correctness-0). It is pre-existing and independent of this fix; the chosen case does not hit it. It is fixed by `fd096ea`, below.

## N1: rejected drafts off the vllm_v1 frontier (`fd096ea`)

- Candidate `n1fix`: `.worktrees/spec-decode-live-batch` at `fd096ea`. Base `specfix`: the gate outputs of `e2c2937` above. Same environment.
- Change: `VLLMv1EngineReplicaScheduler._roll_back_rejected_drafts`, called from `on_batch_end`, subtracts each live row's rejected drafts from `_scheduled_num_computed_tokens_by_request`, as vLLM 0.10.2 does in `update_from_output` (`.real-engine/vLLM-BS/vllm/v1/core/sched/scheduler.py:1296-1308`).
- Expectation: the frontier tests fail before and pass after; results change only where a speculative step rejects drafts and the extra frontier reaches a block boundary or `max_model_len`. Among the gates, only speculative-decode cases can differ.

```bash
R=/data/ycfeng/tmp/spec-decode-live-batch/red_e2c2937   # git archive of e2c2937 plus the new test file
python -m pytest tests/integration/test_vllm_v1_spec_decode_frontier.py -q -p no:cacheprovider   # in $T and in $R
bash $G/run_side.sh $T n1fix
BASE_TREE=$T BASE_LABEL=specfix bash $G/compare.sh $T n1fix
P=/data/ycfeng/tmp/spec-decode-live-batch/n1_pdd   # observer shim: PYTHON_BIN=$P/py.sh, run from each tree root
PYTHONPATH=$P PYTHON_BIN=$P/py.sh N1_OBSERVE_OUT=$P/<side>/observe.json METRICS_OUTPUT_DIR=$P/<side>/metrics bash examples/architecture/pdd/offline/moe_spec_dec.sh
```

| Check | Pass condition | Actual | Verdict |
| --- | --- | --- | --- |
| Frontier test on `e2c2937` | fails | 2 failed (dense and MoE DP2 EP2: run stops with non-empty scheduler state) | PASS |
| Frontier test with the fix | passes | 2 passed | PASS |
| Unit suite | 0 regressions | 0 regressions, 0 new failures; 3806 passed, 84 failed, 10 errors, 51 skipped on both sides | PASS |
| Integration suite | 0 regressions | 0; 40 passed (base 38); after-only: the 2 new tests | PASS |
| Examples | 16 of 16 identical | 16 of 16 (includes `co-location_offline_moe_spec_dec`) | PASS |
| Stage-admission matrix | every cell PASS | every cell PASS | PASS |
| W9-05 probe | 72 of 72 identical | 72 of 72 | PASS |
| Fidelity matrix | identical except explained speculative-decode cases | 73 identical, 1 mismatched (`pdd_spec_dec_offline`), 0 failures | PASS (explained below) |

`pdd_spec_dec_offline` (`examples/architecture/pdd/offline/moe_spec_dec.sh`: 8 requests of 256 + 32 tokens, 128 blocks of 16, ngram k=2, committed 2):

| Quantity | `specfix` | `n1fix` |
| --- | ---: | ---: |
| DECODE preemption events | 1 | 0 |
| Mean request E2E (ms) | 10272.921 | 9989.171 |
| Frontier minus tokens over the 120 speculative DECODE step ends | 1 to 15, growing by one per step | 0 at every step |
| Preempted request 6 at preemption | frontier 289, tokens 278 (19 blocks instead of 18) | not preempted |

- The observer runs reproduce the gate metrics exactly (E2E mean and preemption count above). The other ledger and request-metric differences (one stage-ledger record with 6 instead of 7 requests; `request_decode_tokens_at_preemption_mean` `0.0` vs `0`) follow from the removed preemption.
- Observation: the DECODE frontier starts at the request's tokens, while MONOLITHIC starts one behind (`_get_scheduler_num_computed_tokens`, `vllm_v1_kv_allocation.py:94-109`). N1 keeps that starting offset unchanged; the co-location test asserts `frontier == tokens - 1`, and on DECODE the gap stays at its starting value.
- Evidence: `gates/compare_n1fix_vs_specfix/`, `gates/{specfix,n1fix}/probe/probe.json`, `/data/ycfeng/tmp/spec-decode-live-batch/n1_pdd/{base,n1}/`.
- Limits: dummy timing only; the DECODE path is checked by the fidelity case and the observer, not by a dedicated test.
