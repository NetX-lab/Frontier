# D20-PD CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record the first verified computed-context slice and remaining measured integration work. |

## Environment

- Worktree: `/data/ycfeng/Frontier/.worktrees/step5-measured-pd`, branch `feat/step5-measured-pd`, initial6db2250.
- Python: `/data/ycfeng/envs/frontier-py310/bin/python`,3.10.6/GCC11.3.0; pytest9.1.1. System `python` is absent; system `python3` lacks pytest; the former R2 Python3.12 environment lacks sklearn. No dependency installation was performed.
- CPU only; no GPU, authentication, platform, cloud, SSH or push operations.

## Computed context: RED -> GREEN

Expected: actual NIXL transfer and four-lane D scheduler execute nine outputs with contexts4104..4112, modeled transfer103687956 bytes, and empty final allocation/frontier tables. This reuses shared6cc09dc, not Step native-tail qualification.

```sh
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_measured_pd.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d20_pd_context_red02
```

RED:1 failed in7.52s; actual contexts4105..4113. The earlier `red01` failed on an incomplete test DP context and was corrected before the product patch.

```sh
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_measured_pd.py tests/unit/test_step5_structural_pd.py tests/unit/test_step5_pd_placement.py tests/unit/test_glm53_structural_pd.py tests/unit/test_pd_client_ttft.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d20_pd_context_green01
```

GREEN:70 passed in74.19s, including offline/online Step lifecycle. Transfer bytes, output counts and final ownership balances meet expectations. No native timing or full-matrix fidelity claim.

## Measured transfer and recomputation

- Request RED:1 failed (`d20_pd_request_red_tail`), then14 failed/1 passed (`d20_pd_request_red_validation_fixed`). GREEN:26 dedicated and84 affected. Delegated command:

```sh
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_pd_transfer_prompt_recompute.py tests/unit/test_pd_client_ttft.py tests/unit/test_request_preempted_step.py tests/unit/test_request_spec_decode_zero_commit.py tests/unit/test_vllm_v1_recompute_pricing.py tests/unit/test_kv_transfer_completion_contract.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d20_pd_request_green_regression
```

- Measured constructor RED:1 failed in5.52s (`d20_pd_measured_red01`), rejected by the original dummy-only scope.
- Integrated attempts exposed the max-token attribute name and shared zero-query guards. After approval and inspection of the allocation call chain, the measured route passed1/1 in9.03s (`d20_pd_measured_green04`).
- Lifetime RED: missing cost released the D pool but left a frontend owner. A second failure was an invalid zeroing assumption: this fixture has no zeroing prologue. Only the observed frontend defect was fixed.
- Budget RED:1 passed/1 failed in10.99s (`d20_pd_budget_red01`), proving budget rejection left a source claim. Existing prefix/budget validation was moved into preflight.

Final command:

```sh
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_step5_measured_pd.py tests/unit/test_pd_transfer_prompt_recompute.py tests/unit/test_kv_transfer_completion_contract.py tests/unit/test_step5_pd_placement.py tests/unit/test_step5_structural_pd.py tests/unit/test_glm53_structural_pd.py tests/unit/test_pd_client_ttft.py tests/unit/test_step5_cache_scheduler.py -q -p no:cacheprovider --tb=short --basetemp=/data/ycfeng/tmp/d20_pd_measured_green07
```

**PASS:196 tests in139.73s.** Previous run:194 in132.35s; two compatibility/budget cases were then added. `git diff --check` passed.

| CPU expectation | Observed | Result |
| --- | --- | --- |
| No analytical substitution in grouped measured mode | Analytical timing patched to raise; measured route succeeds | PASS |
|4105 tokens, prefix4080, full D tail |first-D query25/context4080, prefill phase, eager | PASS |
| Covered READ completion means GPU-ready |3.5ms; D cannot schedule beforehand | PASS |
| Prefix-only reservation without invented query |1020 modeled blocks; no compute-token mutation | PASS |
| Step component bytes and units |103051776 bytes/worker lane, explicit prefix label | PASS |
| Sampling |tail emits1, then8 single-token steps; TTFT at tail completion | PASS |
| Ownership |P holds until READ; final allocations/frontiers empty | PASS |
| Wrong/missing coverage |prompt/TP/device/alignment/readiness/identity/duration/budget rejected; cost failure unwinds owners | PASS |
| Legacy behavior |dummy Step and shared non-grouped GLM retain contracts | PASS |
| Ordinary zero-token calls |rejected; only explicit positive remote frontier admits zero-query grouped planning | PASS |

## Unfinished acceptance

Small actual `frontier.main` is **NOT_RUN**: the computation/bundle owner must supply the documented coverage/cost hooks and admitted inputs. Protected computation predictor/loader files are unchanged; missing supplier coverage fails rather than substituting analytical timing.

P compute remains whole-prompt execution. CPU results do not resolve P-side native backoff or T2 pricing. The6144->6128+16 candidate, H200 computation/tail/mixed costs, READ-to-GPU-ready calibration, native content/readiness/reference qualification and formal four-metric accuracy remain outstanding. No R2 verdict is read as admission.
