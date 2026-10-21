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

## Unfinished acceptance

Measured admission/provider consumption, native prompt-prefix/first-D recomputation and their negative controls are in progress. Small actual `frontier.main` measured execution depends on the computation owner's integrated provider/bundle interface. Real H200 inputs, native tail/readiness/content qualification, early T2 pricing and formal four-metric accuracy remain outstanding. No R2 claimed PASS is used as admission.
