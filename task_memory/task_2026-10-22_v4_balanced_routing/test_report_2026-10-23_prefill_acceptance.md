# Balanced prefill primary source acceptance

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Verified C runner-value fix, actual consumer controls and affected regression before integration. |

Source:8c2502c; records:b9b90e1. Worktree `/data/ycfeng/Frontier/.worktrees/v4-h800-prefill`; interpreter `/data/ycfeng/envs/frontier-py310/bin/python`3.10.6. No conda activation or dependency changes.

## Commands and results

Root executed the exact bash block in `test_report_2026-10-23_runner_values.md` from that worktree using `zsh -c`. Producer and actual `read_isolated_scope` rejected the same-name wrong runner class; valid balanced receipt, balanced consumer and absent-selector legacy consumer passed. All five values were true and the command exited0. Raw output: `/data/ycfeng/tmp/glm53_dsv4/root_prefill_runner_r1_20261023.log`.

The outer reader initially attempted `json.loads` on all stdout and failed because the existing architecture warning precedes the JSON. Root parsed the retained object beginning after the first newline followed by `{`; it contained exactly five true booleans. The warning and original bytes were preserved; the control was not repeated. This was a receipt-reader error, not a production failure.

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_dsv4_attention_profiling.py tests/unit/test_dsv4_graph_measured.py -q -p no:cacheprovider -k 'exact_prefill or exact_isolated or attention_rejects_unqualified or attention_does_not_median or attention_layout' --junitxml=/data/ycfeng/tmp/glm53_dsv4/root_prefill_runner_r1_20261023.xml > /data/ycfeng/tmp/glm53_dsv4/root_prefill_runner_r1_affected_20261023.log 2>&1
```

Expected all22 affected checks to pass; XML observed22 tests,0 failures,0 errors,0 skipped: **PASS**.

Root also ran `git show 8c2502c:<path>` for all five C-owned files and compared the output bytes to the working files, then `git diff --numstat a71fcd3 8c2502c -- <the same five paths>`: exact bytes and five files/190 routing changed lines, **PASS**. The five-path list is in `diff_count.json`; final builder selections remain outstanding.

## Review and limits

Independent `W128-V4-BALANCED-PREFILL-R1-REVIEW` APPROVE closes the original runner-values P2. Accepted scope is source/CPU integration. Fixture0.125ms is a correctness control, not measured performance. Fresh package, installed-image checks, actual H800 routing/model execution and measurement-table acceptance remain pending. B's schema/DP findings do not block this independent prefill path.

## Merged-source and builder selection

Root merged source and resolved only three task-document add/add conflicts, preserving the primary approvals and the C owner history. All five C implementation files remained byte-identical to8c2502c before final builder selection. Combined A/C affected command added `tests/unit/test_balanced_routing.py` and `selected_producer_rows or disaggregated_roles` to the command above, with `PYTHONPATH` set to the primary worktree plus `/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages`. XML `/data/ycfeng/tmp/glm53_dsv4/root_balanced_ac_merged_20261023.xml`:26 tests,0 failures/errors/skips. Merge84c806f.

Root then selected six existing builder values: package `h800_prefill_balanced`, name `PREFILL_WARM_6144_C0_BALANCED`, current-clean-HEAD revision via the existing Git snapshot mechanism, revision-derived job/cloud names, and task identity `W128-V4-BALANCED-PREFILL`. This is exactly12 additional baseline changed lines; C totals five files/202 lines. The whole delivered feature with retained B is887/900. AST/import/current-HEAD/name/fresh-path assertions and source whitespace check passed; builder main was not invoked. The previously checked model/geometry/measurement arguments are unchanged.

## Fixed39b2bf8 verification

Working directory `/data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run`, Python3.10.6. Root ran:

```bash
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run --label candidate --output-root /data/ycfeng/tmp/glm53_dsv4/balanced_prefill_39b2bf8_20261023 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 6 --clean-cache
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py compare --output-root /data/ycfeng/tmp/glm53_dsv4/balanced_prefill_39b2bf8_20261023 --baseline-label /data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/baseline --candidate-label candidate
env PYTHONPATH=/data/ycfeng/Frontier/.worktrees/v4-balanced-prefill-run:/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_dsv4_attention_profiling.py tests/unit/test_dsv4_graph_measured.py -q -p no:cacheprovider --junitxml=/data/ycfeng/tmp/glm53_dsv4/balanced_prefill_39b2bf8_units.xml
```

Expected and observed:184 unit checksPASS, zero failures/errors/skips; strict77 complete with77/77identical and451 artifact entries, zero failed/missing/mismatched/provenance/cache findings, clean predictor cache populated. All77 raw rows have source and harness39b2bf8088dff40dc8b695b6927d6e6322c337e9, source_dirty=false and returncode0. Comparison SHA256135494258bb972dc29cc794bab50f8e6aacc064ef6cb53008cab4d7bf12ece35. Root receipt: `prefill_primary_fidelity.json`. Baseline is retainedbfa82de, not a new claim of a direct-parent capture. These are source/CPU/fidelity results, not GPU or serving qualification.

## Sealed package and locked-image readiness

Package `../task_2026-10-22_v4_h800_prefill/h800_prefill_balanced` in worktree `v4-balanced-prefill-run` binds manifest `c6a009f87220c96d19f9b89097762b2f92e3ccdf05199398eaab331510331c8c`. Root rechecked the exact529-file set and hashes after image checks. Package-owner CPU evidence verifies511 Git source files,519 tool pins,60 native source pins (retained51 plus9 routing modules), sequential6144/C0 with warm2/measured3 and configured/effective128. Historical packages remain byte-identical. Root accepted this source/CPU slice independently of B.

Root ran the retained bootstrap image verifier with only the independently derived expected counts updated51→60 and517→519. Attempt01 selected absent `python` and Docker exited127 before running the verifier. The sealed launcher uses `python3`; attempt02 changed only that entrypoint and passed. Actual interpreter `/usr/bin/python3`, Python3.12.3. All60 installed runtime-source,519 tool and20 binary hashes passed; Torch/vLLM had not been imported and CUDA was not initialized at the bootstrap launch boundary. Image digest remains `2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635`.

The old whole-package verifier is generation-specific. Root reused its existing CLI/engine-boundary child code unchanged, executing it after the real C26 bootstrap in the locked image against the actual staged balanced adapter. Only engine construction was intercepted. Added assertions verified the loaded strategy class originates at the sealed flat `balanced_routing.py`, the registry originates at installed `vllm/model_executor/layers/fused_moe/router/routing_simulator_router.py`, the effective environment selects balanced_topk and the runner receives the bound config. PASS, CUDA remained uninitialized. This closes the new adapter startup check without claiming native model/GPU execution. Unchanged archive/runtime code retains its previous verification; the184 current tests cover row/consumer behavior.

Exact Docker argv, executed check scripts, source-template hash, receipts and manifest are archived in `prefill_ready/`; `prefill_root_readiness.json` hashes this evidence. Original attempt01 logs remain `/data/ycfeng/tmp/glm53_dsv4/balanced_prefill_image_39b2bf8_pxh8hgn_/`; successful source/binary check `/data/ycfeng/tmp/glm53_dsv4/balanced_prefill_image02_39b2bf8_sy4l1y0j/`; successful real adapter check `/data/ycfeng/tmp/glm53_dsv4/balanced_prefill_adapter_image_39b2bf8_8dx100k9/`. Commands used CPU-only Docker with read-only package mount, network disabled,4GB memory,2CPUs and128PIDs. No image or sealed input was changed.

Readiness permits only the approved1H800/codesign/nonpreemptible900s (0.25GPUh maximum) supplement. Actual GPU, balanced model/router/backend,6+6 measurement rows, trace and terminal evidence remain required. Submission uses the unchanged sealed launcher after fresh quota/full-pagination personal census and exclusive `prefill_root_submission.json`; launcher log `/data/ycfeng/tmp/glm53_dsv4/v4_balanced_prefill_39b2bf8_launch.log`.
