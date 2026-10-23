# W128 private attention input verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded final CPU validation and pending fixed-source/root gates. |

## Environment and expectations

Checkout: `/data/ycfeng/Frontier/.worktrees/v4-attention-input`, baseline `972f0cb1`, branch `feat/v4-attention-input`.
Python: `/data/ycfeng/envs/frontier-py310/bin/python`, Python3.10.6. No dependency installation or GPU operation.
Every command uses `PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode`.

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Full affected unit module | Existing144 and new sidecar controls pass with no skips/errors |194passed,0fail/error/skip,82.93s | PASS |
| Original accepted explicit API |50CUDA_EVENT+44KERNEL_ONLY exact medians |94exact; serving_admission=false | PASS |
| Current writer + automatic selected input | Same94 layouts/medians without caller layout |94exact using explicitly labelled test admission fixture | PASS |
| Preparation trust boundary | Fresh output, source hashes valid, no admission emitted |candidate02 has no `.admission.json`; existing-output and unsupported acceptance reject | PASS |
| Sidecar/evidence refusals | Missing/malformed/stale sidecar; missing/altered admission/review/CSV/raw/plan/model/summary; raw/job lineage; provenance/layout/routing/method/phase/graph reject |48 parameterized typed controls plus existing controls passed in full194 run | PASS |
| Legacy reader |102positive/72typed negative; no median drift |102/72; exact equality to retained SHA-pinned prior102 receipt | PASS |
| Source scope |2core/180changed,3non-core/350target |2core94changed;3non-core285changed | PASS |
| Fixed-source77 | One clean candidate with complete artifacts |Pending capture after code commit | PENDING |

Counts are not summed across iterations. The194 count is the last complete test-module invocation. First182 runs were intermediate iterations before adding all evidence controls. The automatic94 is a second API path, not94additional measured observations.

## Exact commands

```bash
/data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_dsv4_graph_measured.py -q -p no:cacheprovider --junitxml=/data/ycfeng/tmp/v4_attention_input_final_units.xml
/data/ycfeng/envs/frontier-py310/bin/python tests/comparison/calibration/prepare_dsv4_attention_input.py --accepted-prefill /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json --prefill-package /data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill_balanced --output /data/ycfeng/tmp/glm53_dsv4/v4_attention_input_20261023_w128_candidate02
/data/ycfeng/envs/frontier-py310/bin/python tests/integration/check_dsv4_measured_scopes.py --accepted-prefill /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json --prefill-package /data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill_balanced --output task_memory/task_2026-10-23_v4_remaining_measurements/input_bundle/explicit_comparison.json
/data/ycfeng/envs/frontier-py310/bin/python tests/integration/check_dsv4_measured_scopes.py --accepted-prefill /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json --prefill-package /data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run/task_memory/task_2026-10-22_v4_h800_prefill/h800_prefill_balanced --attention-input /data/ycfeng/tmp/glm53_dsv4/v4_attention_input_20261023_w128_cli_fixture02 --output task_memory/task_2026-10-23_v4_remaining_measurements/input_bundle/automatic_fixture_comparison.json
/data/ycfeng/envs/frontier-py310/bin/python tests/integration/check_dsv4_measured_scopes.py --attention-root /data/ycfeng/tmp/glm53_dsv4/v4_attention_cwd02_results/P6144 --compute-root /data/ycfeng/tmp/glm53_dsv4/h200_dsv4_compute_results_20261021/9f58dbe --output task_memory/task_2026-10-23_v4_remaining_measurements/input_bundle/legacy_comparison.json
git diff --check
```

The CLI fixture was copied separately fromcandidate02 with `shutil.copytree`, then bound using `tests.unit.test_dsv4_graph_measured.write_test_input_admission()`. Its review text is TEST FIXTURE ONLY; it never qualifies the production candidate. The pytest fixture invokes `prepare_attention_input()` imported from this worktree only. No external worktree executable was run.

## Evidence and limits

`verification_receipt.json` pins all five source files, original acceptance/retained files, exported candidate bytes, selected log/XML outputs and legacy inputs. Accepted source/package bytes were revalidated after preparation. The prior legacy receipt SHA is `7cfcc6cf315062825543dd9c8f0887233e0e447aeb68f5582a80293036ce2ae4`.

Final candidate: `/data/ycfeng/tmp/glm53_dsv4/v4_attention_input_20261023_w128_candidate02/dsv4_attention.json`.
Test-only automatic fixture: `/data/ycfeng/tmp/glm53_dsv4/v4_attention_input_20261023_w128_cli_fixture02/`.
Raw logs: `/data/ycfeng/tmp/v4_attention_input_{final_units,prepare_final,explicit_final,automatic_final,legacy_final}.log`.

Observations establish byte binding, independent-builder layout equality and CPU exact lookup only. They do not establish GPU completeness, stream overlap, physicalTP8, graph equivalence or serving totals. Root must independently review actualcandidate02 and supply its isolated-only admission, review source, and perform strict77 baseline comparison. Native06 R4 is unchanged.
