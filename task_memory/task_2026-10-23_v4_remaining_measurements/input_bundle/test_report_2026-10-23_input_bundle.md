# W128 private attention input verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded final CPU validation and pending fixed-source/root gates. |
| 2026-10-23 | Archived sole77candidate,451rehashes,519cache names, actual writer rebuild and non-timing-only layout evidence. |

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
| Fixed-source77 candidate | One clean candidate with complete artifacts |77/77successful cases;451artifacts rehashed;519cache filenames;1287source files match before/after | PASS CAPTURE |
| Strict baseline comparison | Root compares retained baseline/artifacts |Not performed by this component | ROOT PENDING |

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

## Final fixed-source capture

Code commit: `4a1b53e6033134c59b4561306eb2d586bd3d8300`.
Before the sole invocation, HEAD and clean status were asserted and all1287tracked non-task file bytes were compared to committed blobs. The collective-sim gitlink was recorded unchanged. After the run, clean status and all1287source digests were checked again. Each77case record has that same clean source/harness revision. No source or records were edited during capture.

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/frontier-py310/bin/python tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/v4-attention-input --label candidate --output-root /data/ycfeng/tmp/glm53_dsv4/v4_attention_input_4a1b53e6_20261023 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 --clean-cache --continue-on-failure
```

Observed77successful cases/77artifact directories/451actual artifact files rehashed against per-case receipts/519retained cache filenames. `matrix_receipt.json` contains source gate, command, artifact hashes, cache list and exact original paths. Raw log: `/data/ycfeng/tmp/v4_attention_input_strict77.log`. Root strict comparison remains pending; candidate capture PASS is not IDENTICAL or baseline admission.

## Additional decisive checks

- `non_timing_layout_receipt.json`: removed all282call/timing records from the289raw records. The7engine/step records alone reproduce both six-layer sidecar layout maps exactly.
- `post_capture_verification.json`: actualcandidate02 without admission yields typed FileNotFoundError-backed rejection for both automatic methods; explicit API remains compatible on the same bytes (CUDA_EVENT0.9120000004768372ms/KERNEL_ONLY0.898909ms for selected sparse scope).
- Current repository `build_dsv4_rows` plus `write_rows_csv` rebuilt both original tables byte-identically under `/data/ycfeng/envs/asyncopd-py312/bin/python` (Python3.12.13). CUDA_EVENT SHA52571225ce8ddd0d2e87615f80f2aaadd980cbd8b2c10d460a1c75dff7c2af5f; KERNEL_ONLY SHAfe22f22cf4c21bcad93c8ccd463115a607ace65193d900925b8eaa2e6cf69b43. Source imports came only from thisworktree.
- Initial Python3.10 writer-byte assertion failed on fiveCUDA_EVENT `.std` last-bit differences, with every median/metadata cell unchanged. Python3.12 full-consumer import lacked `fasteners`; existing lightweight profiling CLI/rows writer was used directly, no installs. The original errors, fivecell differences, commands' log paths and final rebuild receipt are preserved in `post_capture_verification.json`. No source patch, numeric adjustment or original rewrite followed.
- Consumer package versions: numpy2.2.6, pandas2.3.3, scipy1.15.3, scikit-learn1.7.2, pytest9.1.1, fasteners0.20, ddsketch3.0.1. Python3.10 remains the consumer/test/matrix environment.
