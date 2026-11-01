# Q1 CPU verification and evidence reuse

Working directory for every executed command: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Interpreter `/data/ycfeng/envs/frontier-py310/bin/python`, Python3.10.6. No dependency install, GPU operation, RJob/platform/cloud action, Git mutation, formal replay, threshold/reference/fold/table change or production-source edit was performed.

The named audits, requirements/plan/progress/summaries, tickets05/09/11/17/18/19, Spec5.5/10.4, root AGENTS.md, existing analysis code and exact B29/two-fix/native run roots were inspected. No applicable nested AGENTS.md/CLAUDE.md was found in the assigned target paths. Existing `.cortexkit/`, `.pi/`, `remaining_measurements/` and pre-existing `codex_assignment_Q1_p2_input_qualification.md` were preserved. A pre-existing/concurrent modification of task `progress.md` was encountered and left intact.

## Retained input/attention/hardware audit

Executed (final version, same command also used during development):

```bash
(ulimit -v 31457280; env PYTHONPATH=$PWD:$PWD/tests/analysis PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 600s /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/audit_inputs.py --output task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/evidence/audit.json) > task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/evidence/audit.log 2>&1
```

Expected: original forward aggregation exactly reproduced; both formal runs'7table hashes/counts, trace/input hashes and request populations agree with B29; all omitted measured rows present; production exact lookup/independent windows compared without interpolation; native original node/rank sources unchanged and measured-capacity identities hold.

Observed: **PASS, exit0**, `PASS_EVIDENCE_CHECKS_NOT_FORMAL_ACCEPTANCE`.6,923heldout rows exactly reaggregated;1,373complete/48omitted/24partial/24fully unevaluable; all182missed layer rows retain positive measured windows. Seven omitted exact geometries queried at both scales. Each run has128requests with1024outputs and one prefill per request; forward-query censuses equal B29. Both independent passes have7,105unique rows with geometry/hardware/model/layout identity checked against the original and all positive finite measured windows.240omitted layer-group queries use production exact lookup; interpolation is an error. Maximum proposed cross-pass full-forward error3.663151%; eager prefill0.971587%.108native node/rank files rehashed;96exact memory identities hold; original measured pool109934/112914blocks remains bound. [Raw results](evidence/audit.json), [stdout/warnings](evidence/audit.log).

This closes the1373complete forward-workload subset under the existing rule. It does **not** qualify the48omissions; their supplemental cross-pass criterion awaits YC. Per-scope CSA failure is diagnostic under the already approved forward rule. Compute/table provenance and original R4 remain reused evidence, not a new GPU qualification or full native acceptance.

## NIXL, collectives, memory comparisons and warm readings

Executed:

```bash
(ulimit -v 31457280; env PYTHONPATH=$PWD:$PWD/tests/analysis PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp timeout 600s /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/verify_retained.py --output task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/evidence/retained_checks.json) > task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/evidence/retained_checks.log 2>&1
```

Expected: reuse the existing checkers against pinned original inputs; rebuild NIXL CSV exactly; reproduce exact collective qualification; hash actual F4/B28 comparison files; warm numerical predicates reproduce original values. Observed: **PASS, exit0**, `PASS_RETAINED_CPU_CHECKS_NOT_FORMAL_ACCEPTANCE`.

- NIXL integrity: all3download hashes/sizes,80cases x3passes=240records,32,400reader trials,214,272expected/received notifications, all recorded data checks, reader ranks and exact raw-to-CSV reconstruction PASS. Prompt6144absent. Observed full-batch completion tails up to38.814150ms; no normalization by inflight into a purported bound. H200 fit negative-intercept FAIL is **inspected/reused**, not refit or changed.
- Existing `nixl_window_check.check_run`:128KVwindows each;510/507starts checked; minimum next-step gap18.724965326/24.380075963ms. Structural/wire prices2.570035277/2.987238193ms; zero hits, zero gaps below new wire price, `NO_COMPLETION_CAN_MOVE` both. This closes the observed old/new price-change effect only.
- Existing `v4_collective_exact_repeatability.check_replay`: exact JSON result equals the retained decision-rule output.44/38priced points,6,310,733/9,003,672A8calls, zero interpolation/violations, no single-pass point; weighted maximum deviations5.462298771%/0.783290838% <=10%, PASS both. These are the retained F2 diagnostic A8 census counts; B29/two-fix billing counters6,310,644/9,002,176 differ and must not be relabeled as the same count. The table is the unchanged formal measured input. The existing repeatability rule receipt is reused; Q1 does not claim to have generated a new B29 A8 census.
- Original native hardware/reference receipts: six manifest/R4/terminal source hashes rechecked; each scale's eight R4 checks PASS;32/64H200,4/8distinct nodes, exact pinned image, all containers exit0/restart0, job identities agree. Historical clean-reference/basic scope only; no future promotion or new platform query.
- F4 measured-capacity isolation:5actual metric artifacts per scale rehashed byte-identical to B28, PASS. Planner mismatch+5.548783816%/+6.827319907% remains. This supports input-isolation reuse, not autonomous planner accuracy or independent phase peaks.
- Warm readings: B29/two-fix warm metric rows and event opener sets equal exactly,16predicates per scale (four pooled and four per each of three repeats),32/32PASS. Maximum5.675932671%.0/5event opener pairs excluded;384/379pairs remain. Original P2-64 repeat01 all-request TTFT12.254940665% FAIL is retained. Native and Frontier warm populations use the same event rule. The original numerical harness/gates are **reused**, not rerun or accepted anew.

[Result](evidence/retained_checks.json), [stdout](evidence/retained_checks.log). Original checker implementations and input hashes appear in the result and ledger.

## Targeted NIXL GPU package CPU checks

Executed after the final builder/resource changes:

```bash
env PYTHONPATH=$PWD:$PWD/tests/analysis PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 120s /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/build_nixl_package.py > task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/evidence/package_build.log 2>&1
```

The builder executes the actual producer CLI via `subprocess.run` with this interpreter: `python -m frontier.profiling.collectives.nixl_kv_read_benchmark --config <absolute Q1 directory>/packages/nixl6144/nixl_leaf.json --list-cases`. It calls the existing B30b `dry_run` planner, actual native layout-source/log verifier and existing runtime `worker_command` generator. It compiles included Python sources in memory; no bytecode/cache is written into source directories.

Expected: unique complete grid, valid physical placements, native layout/source binding, exact46/51page geometry, sufficient scratch pool, budget below timeouts, producer count matches planner, included Python syntax/source hashes agree. Observed: **PASS, exit0**, `PASS_CPU_PACKAGE_CHECKS`;30cases x3passes=90measurements;12,150reader trials and80,352notifications expected;16H200; wall463.376234939s;2.059449933containerGPUh, timeout8.0GPUh. Planned **whole-job preemptible=true**. Native source SHA and12native startup log excerpts/registration signatures are checked by the retained planner. Exact source list/digests are in [cpu_check.json](packages/nixl6144/cpu_check.json). [stdout](evidence/package_build.log), [producer CLI output](packages/nixl6144/list_cases.txt).

Committed-source submission lock: **NOT_RUN / NOT_BUILT**, because existing `build_lock` requires committed config/source and this assignment prohibits commits. A review-only command references the future lock, explicitly marked unavailable. Claude must review/commit and use the existing lock builder; no commit rule was bypassed. Platform/resource/runtime/GPU checks and all GPU outputs: **NOT_RUN**. Native payload observation producer/package: **NOT_IMPLEMENTED / NOT_BUILT**, prerequisite described in [payload_observation_contract.md](payload_observation_contract.md).

## Focused tests and ledger integrity

Executed:

```bash
(ulimit -v 31457280; env PYTHONPATH=$PWD:$PWD/tests/analysis PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=/data/ycfeng/tmp/codex-implement/65c06d69-e1a5-43c1-836f-440fe3bd1713 timeout 600s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_p2_input_qualification.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/codex-implement/65c06d69-e1a5-43c1-836f-440fe3bd1713/p2_qualification_pytest) > task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/evidence/unit.log 2>&1
```

Expected: full128x1024census and tail6143..7166reconstruction; order-independent paired request/context keys; rejection of dropped output, impure dummy and repeated prefill; production mean reducer over duplicate features. Observed: **PASS,6passed in1.64s**, exit0. [Log](evidence/unit.log). These are analysis-logic tests; no broad production suite is needed for this report-only assignment.

Executed:

```bash
env PYTHONPATH=$PWD:$PWD/tests/analysis PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/write_ledger.py
```

Observed: **PASS**:27input/disposition rows;19QUALIFIED,6OPEN,2NEEDS_DECISION;48individual omissions;7directly queried;priorQ1-Q14mapped. [Ledger](ledger.json) separates actual consumed formal inputs from prospective candidates, inactive configured values and future receipt obligations. Final artifact/source-hash/link checks executed:

```bash
env PYTHONPATH=$PWD:$PWD/tests/analysis PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/check_handoff.py > task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/evidence/handoff_check.log 2>&1
```

Observed **PASS**, including all local Markdown links, valid JSON evidence pointers, all27statuses/48omissions/Q1-Q14mapping, updated production cross-pass binding and17package-source digests. These checks are separate from qualification and do not promote a table or criterion. Result: [handoff_check.json](evidence/handoff_check.json).

## Preserved failed attempts and limits

- Initial audit compared raw profiling path strings and failed: B29's historical checkout prefix differs from the current checkout. The check was corrected to compare filename, SHA256, row count and consumer identity, then rehash actual current contents. No profiling input changed. The final check passes; the initial transient stdout was overwritten during development, so no raw failure log is claimed.
- A preliminary cross-pass calculation used raw scope row medians, which do not implement the production mean reducer over duplicate feature coordinates. It was replaced before the final ledger by the real production model-key admission/exact reducer/scope composition. Only the corrected3.663151%maximum is evidence for the proposal; preliminary values are not qualification evidence.
- First `verify_retained.py` attempt failed on CSV boolean `True` being treated as numeric because Python bool subclasses int. [Original failure](evidence/retained_checks_initial_failure.log) preserved. The auxiliary CSV comparator now distinguishes exact bool/string from numeric values; all240CSVrows reconstruct. Raw data/producer unchanged.
- A shell `python` lookup was unavailable during initial inspection; the declared absolute interpreter was used for every executed check listed here. Two guessed evidence paths during read-only inspection did not exist; the actual retained native/source_v2 paths were found and cited. No unavailable evidence was treated as PASS.
- The requested replay scratch `/data/ycfeng/tmp/glm53_dsv4/input_qualification_20261101/` could not be created (`Permission denied`). No replay was necessary. Config-construction/test scratch used the explicitly allowed private runtime directory, not another host path. No scratch permission workaround or retained-run write occurred.
- No full CPU replay was repeated: retained bit-identical B29/two-fix outputs and the previous completed independent-pass diagnostics were reused. Reported replay costs for future choices are planning estimates (2.7h/3.3h each scale), not newly executed checks. No attempt to repair the three unrelated baseline unit failures/errors recorded by the two-fix report was made; they are outside this analysis assignment.
