# Native balanced binding CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | W128 recorded focused identity, reader and native observer checks. |

Task: W128-V4-BALANCED-NATIVE-BINDING. Workdir: `/data/ycfeng/Frontier/.worktrees/v4-balanced-binding`.
Environment: Linux, `/data/ycfeng/envs/asyncopd-py312/bin/python`, Python3.12.13; no conda activation. No GPU/native runtime, platform, auth, SSH, cloud or Docker operation. Native reference source is read-only.

## Commands and observations

```bash
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_reader_binding.py -k balanced_strategy -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-balanced-binding-identity03
env PYTHONDONTWRITEBYTECODE=1 FRONTIER_VLLM_BS_ROOT=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 NATIVE_READER_BINDING_CHECKER=/data/ycfeng/Frontier/.worktrees/v4-balanced-binding/task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py /data/ycfeng/envs/asyncopd-py312/bin/python -B -m pytest tests/unit/test_native_serving_inventory.py tests/unit/test_native_v2_serving_inventory.py tests/unit/test_native_reader_binding.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-balanced-binding-final01 --junitxml=/data/ycfeng/tmp/opencode-v2-tmp/opencode/v4-balanced-binding-final01.xml
git diff --check
git diff --numstat
```

| Expected passing condition | Observed | Result |
| --- | --- | --- |
| Actual registered strategy class/module/byte identity and all routers match | Three focused controls PASS,0.26s | PASS |
| Missing routing evidence rejects rather than defaults | Missing identity control rejects | PASS |
| Same-name replacement/module/package origin cannot pass strategy identity | PathFinder observes winning package origin; strict actual origin/SHA checker rejects | PASS |
| Explicit balanced native clean environment, inherited selection removed otherwise | Actual server_env assertions pass | PASS |
| Existing reader environment/origin/shadow controls remain valid |46 reader tests PASS,51.47s before final source refresh | PASS |
| Native observer/dispatch/origin controls and unchanged native return/errors | Combined119 tests PASS,62.85s | PASS |
| Non-routing deterministic predicates remain unchanged | AST equality for provenance/inputs/replay/workers/lifecycle/terminal | PASS |
| Bound and formatting | Five code/test/binding files,+107/-2=109 changed lines; diff check clean | PASS |

The combined119-test run preceded the last test-only change that replaced a fixture origin string with actual PathFinder resolution. The affected three tests were rerun afterwards and passed. Initial73-test observer run had72PASS/1FAIL because its AST assertion allowed only the former dispatch predicate change. The new staged-strategy import guard also changes check_native_canonical; its test now allows that exact function and separately checks unchanged non-routing predicates. This was a test expectation correction, not relaxed runtime admission.

Evidence: XML path above. The final new controls use fake vLLM host metadata plus the real committed strategy function/collector/checker; they do not exercise native CUDA kernels or certify runtime backend selection.

## Identity and verification limits

- Strategy SHA256: `6562ec1c7e1ee7a8021d71a1a698c36f126307f00681320b07397b53c4c3f4ea`.
- Frozen external R2 SHA256 remains `29523700ac8492fa6e192ce272f8ab5a7a48d1528bd4064417e26e811533d10a`.
- Old capacity128 prefill adapter SHA256 remains `31c7c5c8f15680894967fdab45ef3d3e0d2d12359b17848c7602abfc020b7bdc`.
- Native/C26 were inspected read-only, not edited. Old native05/prefill packages were not edited, staged, sealed or requalified.
- Frontend matching-input validation, new prefill adapter/producer identity, actual router/env native origins, fresh-generation pins and native FP8/GPU proof are pending. Exact paths and scope boundary are in design.md.
- No latency, CPU overhead, GPU resource, four-metric calibration or package-readiness claim.
