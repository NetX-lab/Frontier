# Test report: F1-F4 comparison variants of the A10 inputs (decision 46)

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Completed F2 P2-64 with repeat1 TTFT FAIL, exact A8 coverage, and the full eight-reading F comparison; formal qualification remains open. |
| 2026-11-01 | Verified F2 P2-32 all-repeat numerical PASS and A8 exact coverage; only F2 P2-64 remains running. |
| 2026-11-01 | Verified six completed readings/96 metric entries for the final comparison collector while both original F2 replays continue. |
| 2026-11-01 | Completed F1a P2-64 with all-repeat latency FAIL; retained exact references, six compute hashes and active F2 replays. |
| 2026-10-31 | Codex update: F4/B28 five-artifact identity on both scales, B29 per-repeat results, F1a P2-32 final FAIL and fresh P2-64 replay. |
| 2026-10-31 | Created: F1 per-call screen and astra defect, F4 planner terms; F4 replays, B29, B30a and B30b pending. |

## Completed diagnostic comparison — 2026-11-01

Decision 46's F1–F4 experiment series has terminal outcomes. The unchanged collector reports **PASS_EVIDENCE_CHECKS, 8 readings, 128 metric entries, 3 unavailable results**. This means the retained evidence is consistent; it does not mean the variants or formal qualification pass. No simulator, native run, normalization or gate was repeated in this closeout.

F2 K18 P2-64 fails repeat01 TTFT: native 779.888727400 ms, predicted 680.123765687 ms, absolute error 99.764961713 ms, relative error 12.792204607%. Its other 11 per-repeat metrics pass. Pooled TTFT 8.242207% does not override the failed repeat. F2 P2-32 passes all 12 per-repeat metrics. The original P2-64 launcher and watcher both exited0; no comparison replay or watcher remains active.

Each cell below is **prediction / worst error across the three individual repeats**. Latencies are ms and THPT is tok/s. The verdict includes every metric and every repeat. All native and predicted values, signed differences, absolute errors and relative errors for every repeat are in `h3o_operator_costs/f_comparisons_20261101/complete/metrics.csv`; `comparison.json` pins the source revisions, inputs, reference hashes and readings.

| Variant | Case | TTFT | TPOT | E2E | THPT | Numerical verdict |
| --- | --- | --- | --- | --- | --- | --- |
| B29 | p2_32 | 344.878991 / 5.607176% | 19.926786 / 0.192044% | 20729.981427 / 0.269220% | 6225.735862 / 0.134548% | PASS |
| B29 | p2_64 | 684.313827 / 12.254941% | 25.295337 / 3.656983% | 26561.443186 / 3.710082% | 5975.251570 / 0.457663% | FAIL |
| F4 | p2_32 | 323.178991 / 11.546431% | 19.926786 / 0.192044% | 20708.281427 / 0.373618% | 6226.652709 / 0.119841% | FAIL |
| F4 | p2_64 | 662.613827 / 15.037389% | 25.295337 / 3.656983% | 26539.743186 / 3.788748% | 5976.096120 / 0.471862% | FAIL |
| F1A | p2_32 | 376.104429 / 3.903495% | 14.265762 / 28.533481% | 14969.979091 / 27.980269% | 6468.179037 / 3.844430% | FAIL |
| F1A | p2_64 | 463.897052 / 40.517533% | 15.565031 / 40.717055% | 16386.923646 / 40.594510% | 6404.273989 / 7.670513% | FAIL |
| F2 | p2_32 | 345.032252 / 5.565229% | 19.929652 / 0.206450% | 20733.065841 / 0.254381% | 6224.402906 / 0.155930% | PASS |
| F2 | p2_64 | 680.123766 / 12.792205% | 25.292822 / 3.666561% | 26554.680541 / 3.734598% | 5971.825583 / 0.400064% | FAIL |

F1a specification links fail latency metrics at both scales. F1b's measured link fit fails its independent G5h validation (EP graph-forward predictions 11.449–19.300 times measured); both replays additionally stop with compute COVERAGE_MISS. Their complete E2E metrics remain null. F3's complete H200 NIXL measurements yield a negative affine intercept −46.676920 us and p90 residual 29.175907%; no admissible input or F3 replay is available. These three unavailable results are explicitly recorded by the collector.

F4's five metric artifacts equal B28 byte for byte at both scales. F4 and B28 use zero serving delays; B29 uses 10.3/11.4 ms. The F4/B29 difference cannot be attributed to the memory planner. The planner changes its pool by +5.55%/+6.83%, but the retained workload's request metrics are unchanged against the matched B28 baseline.

F2 changes only K19 to18. At P2-64 this lowers predicted TTFT by 4.190060917 ms and increases repeat01's TTFT error from 12.254940665% to 12.792204607%. It does not resolve the failing formal reading. The one-GPU single-capacity explanation remains AMBIGUOUS; K18 is only the retained local-interpolation diagnostic. Under D-NO-INTERRUPT, keep measured CC, formal K19, ingress 10.3 ms, handoff 11.4 ms, all three native references and every threshold unchanged. No F variant is promoted.

## F2 P2-64 terminal verification

Source `662a427ed32bc1bc2f29b177e7c073cbd460c0b4`, clean at launch. Initialization 73.574429 s, simulation 8979.857535 s. The output contains 128 complete unique formal requests and 917,504 tokens. All three native CSV hashes and all six compute-table hashes equal B29; the pre-run comparison pins all seven profiling tables and records K as the only behavior change. No dummy mode or whole-step table is bound.

The completed gzip contains 39,680 forward records, equal to the receipt and stage-end count. Both observer-file SHA256 values reproduce. Zero stale stage ends. A8 returns 38 distinct collective requests, 9,003,672/9,003,672 exact calls, zero interpolated calls and zero violations. Its actual stdout is retained; exit 0 alone was not used as the acceptance condition. Memory admission retains the existing DIFFERS status (2,940 modeled versus 2,944 native blocks per request).

The original watcher executed the gate once from 2026-11-01 03:26:49 to 03:26:50 UTC+8. It reused the three clean references. Supplementary pooled values and every individual repeat follow; all actual values are positive, so no undefined relative errors occur.

| Reference | Metric | Native | F2 K18 | Absolute error | Relative error | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| pooled | ttft_ms | 741.216351 | 680.123766 | 61.092586 | 8.242207% | PASS |
| pooled | tpot_ms | 25.870531 | 25.292822 | 0.577709 | 2.233078% | PASS |
| pooled | request_e2e_time_ms | 27206.769532 | 26554.680541 | 652.088991 | 2.396789% | PASS |
| pooled | token_throughput_tps | 5963.206318 | 5971.825583 | 8.619265 | 0.144541% | PASS |
| repeat01 | ttft_ms | 779.888727 | 680.123766 | 99.764962 | 12.792205% | FAIL |
| repeat01 | tpot_ms | 25.429869 | 25.292822 | 0.137047 | 0.538921% | PASS |
| repeat01 | request_e2e_time_ms | 26794.644444 | 26554.680541 | 239.963903 | 0.895567% | PASS |
| repeat01 | token_throughput_tps | 5948.029629 | 5971.825583 | 23.795953 | 0.400064% | PASS |
| repeat02 | ttft_ms | 725.492280 | 680.123766 | 45.368514 | 6.253480% | PASS |
| repeat02 | tpot_ms | 26.255496 | 25.292822 | 0.962674 | 3.666561% | PASS |
| repeat02 | request_e2e_time_ms | 27584.864259 | 26554.680541 | 1030.183719 | 3.734598% | PASS |
| repeat02 | token_throughput_tps | 5976.592892 | 5971.825583 | 4.767309 | 0.079766% | PASS |
| repeat03 | ttft_ms | 718.268047 | 680.123766 | 38.144281 | 5.310591% | PASS |
| repeat03 | tpot_ms | 25.926229 | 25.292822 | 0.633407 | 2.443112% | PASS |
| repeat03 | request_e2e_time_ms | 27240.799892 | 26554.680541 | 686.119351 | 2.518720% | PASS |
| repeat03 | token_throughput_tps | 5965.065720 | 5971.825583 | 6.759863 | 0.113324% | PASS |

Commands ran in the current worktree with Python 3.10.6 at `/data/ycfeng/envs/frontier-py310/bin/python`. The closeout command inherits a 30 GiB address-space limit, uses `/data/ycfeng/tmp`, sets `PYTHONDONTWRITEBYTECODE=1`, and gives each existing checker a 60 s timeout. `finish_comparison_command.py` preserves the exact orchestration; `complete_command.json` records times, environment and stdout. The script waited for the existing watcher and never invoked a gate.

```bash
/data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/glm53_dsv4/finish_f_comparison_20261101.py
```

The existing commands it executes once are:

```bash
/data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a8_collectives.py /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/runs/f2_p2_64_k18_codex_20261101
/data/ycfeng/envs/frontier-py310/bin/python tests/analysis/v4_f_variant_comparison.py --task-root task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800 --stage-root /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4 --output task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/f_comparisons_20261101/complete
```

Observed: the closeout command and both existing checkers exit 0; A8 `violations 0`; collector `PASS_EVIDENCE_CHECKS readings=8 metrics=128 unavailable=3`. F2 accuracy remains FAIL as stated above. Evidence: `f2_p2_64/reading.json`, its normalized rows, `f2_p2_64_{receipt,verification,collectives,a8}.json`, `f2_p2_64_a8_stdout.txt`, and `complete/` in the comparison evidence directory.

Report values were checked against all 32 variant/case/metric cells; the document diff check passed. The staged default whitespace check then flagged the collector CSV’s 129 CRLF line endings. All 129 records use CRLF consistently and have no trailing spaces or tabs; the original collector bytes are preserved. `git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --cached --check` passes while recognizing those CSV line endings. No simulator or profiling source changed in this step.

The next qualification work belongs to tickets 05/09: a supported uniform measurement correction for the failed A6 attention/collective conditions, plus the retained hardware/NIXL/memory requirements. The completed 128-case repeatability result does not establish a sufficient repeat count. The scratch 16-reconstruction draft is not selected or authorized; its estimated 14.850153 GPUh is above the remaining approximate 2.160175 GPUh. No new GPU operation, table generation, source behavior or resource increase occurred in this closeout. Tickets 05/09/11/19 remain open.

## Historical F1a checkpoint — before F2 P2-64 completion

F1a is complete at both scales. P2-32 has pooled TPOT28.38101% and E2E27.82468% errors (FAIL). P2-64 completed128 formal requests and917,504 tokens: TTFT/TPOT/E2E FAIL in every reference repetition; total THPT PASS. Specification-link parameters substantially underpredict the observed latencies on these two cases. The original incomplete P2-64 attempt remains retained; these results come only from the fresh completed run.

B29 remains the formal measured-CC input: P2-32 numerical PASS in all three repetitions; P2-64 repeat1 TTFT12.254940665% FAIL (779.888727400ms native versus684.313826604ms Frontier). Pooled TTFT7.67691% does not override it. F4 five metric artifacts remain byte-identical to B28 at both scales; its zero ingress/handoff differs from B29. F1b's prescribed positive fit fails G5h held-out validation and both replays terminate COVERAGE_MISS; no complete four-metric F1b result exists. F3's completed NIXL data yields an invalid affine fit and no replacement/replay. F2K18 P2-32 is now complete and numerical PASS in all three references; F2 P2-64 remains running and watcher23683 owns its gate.

F1a P2-64 uses source `9f5f8b614f1e278b02ad39015f0230de562c82ba`, clean, in `f1a-astra-diagnostic`. Both roles use ASTRA Switch links3,600/400Gbit/s and1/1µs latency, with K19 and B29 ingress10.3ms/handoff11.4ms. No dummy or step-cost table is bound. The six compute-table hashes match B29; all three normalized native reference CSVs are byte-identical to B29's retained CSVs. There are128 unique complete formal request rows, no token/request/metric gaps,53,248 forward and stage-end records, and zero stale stage ends. `collectives.calls=0` is the measured-backend census field on an ASTRA run; it is not evidence that no communication was simulated. Memory admission remains DIFFERS (2,940 modeled versus2,944 native blocks per request), as in the existing qualification record.

Execution: initialization72.455524s, simulation12,215.365581s; launcher session50972 exit0 and workload log ends `EXIT 0`. Existing startup warnings are retained in the selected log tail. The watcher executed the existing gate once at2026-10-31T18:22:43.811894Z; exit0 means the gate ran successfully, its accuracy verdict is FAIL. Evidence: `h3o_operator_costs/f_comparisons_20261101/f1a_p2_64/`, `f1a_p2_64_receipt.json` and `f1a_p2_64_verification.json`.

Commands:

```bash
bash /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/run_f1a.sh p2_64 f1a_p2_64_codex_20261031a
/data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_gate.py p2_64 /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/runs/f1a_p2_64_codex_20261031a /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/f_comparisons_20261101/f1a_p2_64
/data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/f_comparisons_20261101/verify_f1a_p2_64.py
```

The wrapper and exact environment are retained as `f_comparisons_20261101/run_f1a.sh`: Python3.10.6, the isolated checkout's PYTHONPATH, one BLAS/OpenMP thread, `/data/ycfeng/tmp` scratch and60GiB RLIMIT_AS. Verification initially looked for old normalized CSVs beside the archived reading; `commands.json` points to the original scratch output instead. Correcting that lookup produced `PASS_EVIDENCE_FAIL_ACCURACY`:128 rows, three unchanged references and six unchanged compute hashes. No normalization or gate was rerun.

Pooled reference values are supplementary; each repeat below still decides its own gate. Relative errors use absolute reference denominators, all positive; zero would be undefined.

| Metric | Native pooled | B29 measured-CC prediction | F1a prediction | F1a absolute error | F1a relative error | F1a verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TTFT (ms) | 741.216351 | 684.313827 | 463.897052 | 277.319300 | 37.414083% | FAIL |
| TPOT (ms) | 25.870531 | 25.295337 | 15.565031 | 10.305500 | 39.834900% | FAIL |
| E2E (ms) | 27206.769532 | 26561.443186 | 16386.923646 | 10819.845886 | 39.768947% | FAIL |
| THPT (tok/s) | 5963.206318 | 5975.251570 | 6404.273989 | 441.067671 | 7.396485% | PASS |

| Reference repeat | Metric | Native | F1a | Absolute error | Relative error | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| repeat01 | TTFT (ms) | 779.888727 | 463.897052 | 315.991676 | 40.517533% | FAIL |
| repeat01 | TPOT (ms) | 25.429869 | 15.565031 | 9.864838 | 38.792327% | FAIL |
| repeat01 | E2E (ms) | 26794.644444 | 16386.923646 | 10407.720798 | 38.842541% | FAIL |
| repeat01 | THPT (tok/s) | 5948.029629 | 6404.273989 | 456.244359 | 7.670513% | PASS |
| repeat02 | TTFT (ms) | 725.492280 | 463.897052 | 261.595228 | 36.057617% | FAIL |
| repeat02 | TPOT (ms) | 26.255496 | 15.565031 | 10.690465 | 40.717055% | FAIL |
| repeat02 | E2E (ms) | 27584.864259 | 16386.923646 | 11197.940614 | 40.594510% | FAIL |
| repeat02 | THPT (tok/s) | 5976.592892 | 6404.273989 | 427.681097 | 7.155935% | PASS |
| repeat03 | TTFT (ms) | 718.268047 | 463.897052 | 254.370995 | 35.414494% | FAIL |
| repeat03 | TPOT (ms) | 25.926229 | 15.565031 | 10.361198 | 39.964153% | FAIL |
| repeat03 | E2E (ms) | 27240.799892 | 16386.923646 | 10853.876246 | 39.844191% | FAIL |
| repeat03 | THPT (tok/s) | 5965.065720 | 6404.273989 | 439.208269 | 7.363008% | PASS |

Decision under D-NO-INTERRUPT: this completes the F1a comparison requested by decision46. Preserve the failing readings; do not tune specification links against these native values or promote B30a/the diagnostic backend. Continue the two existing F2 replays, then assemble the F1–F4 comparison. A6 repeatability is independently complete as a diagnostic with no qualification change; see [report](test_report_2026-11-01_a6_repeatability.md). Formal tables, K19, metric endpoints, thresholds and three references remain unchanged.

The initial screen, predictions and then-pending statuses below are historical. The measured backend stays formal; no diagnostic variant or B30a default change is promoted.

## Historical F2 P2-32 completion — before P2-64 termination

The existing launcher86884 exited0 and the watcher ran the canonical gate once at02:57:46–02:57:47 UTC+8. The reading is numerical PASS against every reference repeat. Source662a427e is clean, initialization71.904930s and simulation7225.211242s. The128 complete requests contain917,504tokens; allthree native CSVs and six compute-table hashes equal B29. The pre-run input comparison verifies allseven profiling hashes and effective behavior differs only by K19→18. Both serving delays remain10.3/11.4ms. There are23,040 forwards and matching stage ends, zero stale ends. A8 reports6,310,733/6,310,733 exact collective calls,44 distinct requests, zero violations.

The following values are read from the retained gate, without normalization or gate reruns. Pooled results are supplementary. Latencies are ms; throughput is tok/s.

| Reference | Metric | Native | F2 K18 | Absolute error | Relative error | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| pooled | ttft_ms | 364.029650 | 345.032252 | 18.997398 | 5.218640% | PASS |
| pooled | tpot_ms | 19.918966 | 19.929652 | 0.010686 | 0.053645% | PASS |
| pooled | request_e2e_time_ms | 20741.131945 | 20733.065841 | 8.066104 | 0.038889% | PASS |
| pooled | token_throughput_tps | 6231.415496 | 6224.402906 | 7.012590 | 0.112536% | PASS |
| repeat01 | ttft_ms | 361.974764 | 345.032252 | 16.942512 | 4.680578% | PASS |
| repeat01 | tpot_ms | 19.906845 | 19.929652 | 0.022807 | 0.114569% | PASS |
| repeat01 | request_e2e_time_ms | 20726.676790 | 20733.065841 | 6.389051 | 0.030825% | PASS |
| repeat01 | token_throughput_tps | 6234.123762 | 6224.402906 | 9.720856 | 0.155930% | PASS |
| repeat02 | ttft_ms | 364.748496 | 345.032252 | 19.716244 | 5.405435% | PASS |
| repeat02 | tpot_ms | 19.888592 | 19.929652 | 0.041060 | 0.206450% | PASS |
| repeat02 | request_e2e_time_ms | 20710.777696 | 20733.065841 | 22.288145 | 0.107616% | PASS |
| repeat02 | token_throughput_tps | 6231.404806 | 6224.402906 | 7.001900 | 0.112365% | PASS |
| repeat03 | ttft_ms | 365.365689 | 345.032252 | 20.333437 | 5.565229% | PASS |
| repeat03 | tpot_ms | 19.961462 | 19.929652 | 0.031810 | 0.159359% | PASS |
| repeat03 | request_e2e_time_ms | 20785.941348 | 20733.065841 | 52.875508 | 0.254381% | PASS |
| repeat03 | token_throughput_tps | 6228.720263 | 6224.402906 | 4.317357 | 0.069314% | PASS |

Evidence: `h3o_operator_costs/f_comparisons_20261101/f2_p2_32/reading.json`, `f2_p2_32_receipt.json`, `f2_p2_32_verification.json`, `f2_p2_32_collectives.json` and `f2_p2_32_a8.json`. The existing `a8_collectives.py RUN_ROOT` returned0 with zero violations. `v4_f_variant_comparison.verify_reading` and `verify_f2` returned successfully on these actual outputs. Its sixteen metric entries pass the collector checks. No source behavior or input was changed after launch.

Decision under D-NO-INTERRUPT: retain this diagnostic result and formalK19. A close E2E match does not resolve the single-capacity AMBIGUOUS finding or establish that the synthetic one-GPU launch queue represents native concurrent streams and collectives. P2-64 remains running in its original session87208; its gate and final comparison collection remain pending. A6/memory/NIXL qualification is unchanged.

## Historical comparison collector precheck — before P2-64 termination

`tests/analysis/v4_f_variant_comparison.py` reads retained canonical outputs and does not execute a normalizer, gate or simulator. It checks the eight expected B29/F1a/F2/F4 readings against the corresponding B29 reference CSV bytes, six compute-table hashes, fixed request/token population, all four metrics, every repeat and the unchanged10% threshold. Its output uses a nonnegative absolute error and retains a separate signed delta. F1b coverage failures and F3 invalid-fit results have null metrics. F4 retains its pre-B29 zero-delay configuration.

Before F2 termination, the existing six readings were exercised directly through the collector: **PASS, six readings,96 pooled/per-repeat metric entries, three identical reference files per reading, six identical compute inputs**. This is a collector check on existing evidence; F2 is still running and no complete comparison result is written. Evidence: `h3o_operator_costs/f_comparisons_20261101/comparison_precheck.json` and `precheck_command.py`. Initial checks exposed three assumptions about old records: the canonical E2E name is `request_e2e_time_ms`; F4 predates the delay fields; and F4 predates the receipt's `cc_backend_types` field. The collector now reads the inspected canonical name and actual configuration. No input, old result or gate changed.

Command, current worktree, Python3.10.6 with30GiB RLIMIT_AS:

```bash
env PYTHONDONTWRITEBYTECODE=1 timeout 60s /data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/tmp/glm53_dsv4/f_comparison_precheck_20261101.py
```

The exact command script is retained as `precheck_command.py`. Final exit0; AST parsing and `git diff --check` PASS. The complete collector command now waits only for F2 P2-64 and the existing watcher's once-run reading:

```bash
env PYTHONDONTWRITEBYTECODE=1 timeout 60s /data/ycfeng/envs/frontier-py310/bin/python tests/analysis/v4_f_variant_comparison.py --task-root task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800 --stage-root /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4 --output task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/f_comparisons_20261101/complete
```

## Historical initial scope

YC ruled on the ex-situ audit (decision 46): measured communication data is an allowed input, so the measured backend stays the formal input. The F1-F4 repair proposals of the audit ([test_report_2026-10-31_ex_situ_input_audit.md](test_report_2026-10-31_ex_situ_input_audit.md), section 5) run as diagnostic A10 variants, and their errors are compared with the B28 baseline. A variant replaces a formal input only after a new decision by YC.

| Variant | Input replaced | Replacement | Status |
| --- | --- | --- | --- |
| F1a | measured collective table | `astra_sim_analytical` with H200 specification link parameters | CPU per-call screen done (section 2); A10 replay waits for B29 and B30a |
| F1b | measured collective table | `astra_sim_analytical` with link parameters fitted from one node and a two-node one-rank-per-node test | probes and fit tools in B30b; GPU job not submitted |
| F2 | K = 19 from the native P2-64 run | K predicted from a one-GPU launch-depth measurement | probe in B30b; GPU job not submitted |
| F3 | H800 NIXL fit (126.2 Gbit/s, 64.7 us) | H200 NIXL fit with the V4 KV layout | probe in B30b; GPU job not submitted |
| F4 | native explicit `num_blocks` | Frontier's memory planner (`memory_planner_profiled`) | planner terms done (section 3); replays running |

Baseline (B28, main `ee29d548`, warm, gate `stage4/a10_gate/p2_{32,64}_warm_k19b28_post/gate_stdout.txt`):

| Case | TTFT | TPOT | E2E | Throughput | Gate |
| --- | --- | --- | --- | --- | --- |
| P2-32 | −11.22% (323.18 against 364.03 ms) | +0.04% | +0.16% | +0.08% | FAIL (TTFT) |
| P2-64 | −10.60% | +2.22% | +2.45% | +0.22% | FAIL (TTFT) |

## 1. Method of the F1 per-call screen

`h3o_operator_costs/f_variants/f1_astra_per_call.py` (commit `b0e568f5`; wrapper `run_f1_per_call.sh`) builds the DECODE-cluster CC backend of each A10 case through `Cluster`, so the backend gets the same materialized layout as in a replay (P2-32: 2 servers x 8 GPUs, TP8 DP2 EP16; P2-64: 4 x 8, TP8 DP4 EP32). It prices every TP and EP row of the formal H200 table (`data/profiling/network/h200_dgx/summary.csv`, the table the A10 inputs resolve) at the row's payload, the largest rank's input bytes, which is the payload rule of the measured backend. Rows of one call are pooled over passes by their median. The output reports the predicted/measured ratio per curve and per CUDA Graph decode forward (calls per forward x per-call time).

This screen checks only the collective prices. It does not run the simulator; the effect on TPOT and TTFT below is an inference.

## 2. F1 per-call results

Per CUDA Graph decode forward, predicted/measured ratio (graph token counts 1-32 per lane):

| Backend parameters | EP16 (P2-32) | EP32 (P2-64) | TP8 |
| --- | --- | --- | --- |
| `astra_sim_analytical` defaults (FullyConnected, 600 / 100 Gbit/s, 1 / 1 us) | 0.159-0.183 | 0.141-0.198 | 0.846-1.096 |
| Specification links (Switch, 3,600 / 400 Gbit/s, 1 / 1 us) | 0.239-0.299 | 0.204-0.337 | 1.567-1.911 |
| FullyConnected, 3,600 / 400 Gbit/s, inter latency 5, 6, 7 or 8 us | 0.124-0.152 | 0.105-0.170 | 0.799-0.964 |

Measured decode forward (graph_t1-t32): EP16 8.69-11.19 ms, EP32 15.97-27.11 ms, TP8 0.66-0.82 ms.

Per-curve median ratios with the specification links: `all_gatherv` EP16 0.215, EP32 0.440; `reduce_scatterv` EP16 0.485, EP32 0.169; `all_reduce` TP8 2.75; `all_gather` TP8 0.559; SP `all_gather` TP8 1.75; SP `reduce_scatter` TP8 1.46.

Observations:

1. **astra defect: a flat ring never uses the inter-server link.** `_rank_to_physical_address` returns `(local index, server index)` and `_get_first_differing_dimension` returns the first component that differs (`frontier/cc_backend/backends/astra_sim_analytical_cc_backend.py:435-450`). Ring neighbours on different servers, such as 7 -> 8, differ first in the local index, so `_send_ms` prices them on the intra-server link. Direct check on the P2-32 and P2-64 backends with inter latency 50 us and intra latency 1 us: `send(0 -> 1)` = 0.001 ms, `send(7 -> 8)` = 0.001 ms, `send(0 -> 8)` = 0.05 ms. The inter latency sweep 5-8 us gave byte-identical `per_call.csv` files. No unit test covers multi-server rings. Fix: assignment B30a.
2. **Payload geometry of the dispatch.** Native V4 dispatch is three grouped `all_gatherv` calls (hidden states, top-k weights, top-k ids); Frontier's analytical path prices one all-gather over the summed 8,264 B, so it carries the per-step latency of one call. The measured per ring step cost is about 9.9 us for `all_gatherv` and 4.5 us for `reduce_scatterv` at EP16 (8.3 us dispatch per step at EP32), so one per-link latency cannot fit both operations. A single latency could fit their per-layer sum.
3. **Size of the gap.** With the specification links the EP decode collectives come out at 0.20-0.34 of measured, and TP8 at 1.6-1.9.

Inference (not a replay): EP collectives are about 8.9 of the 19.88 ms P2-32 decode forward and 16.05 of the 25.30 ms P2-64 forward (audit `comm_share.txt`). At the specification-link ratios, the decode forward would shrink to about 14.1 ms (P2-32) and 15.1 ms (P2-64), so TPOT would move by about −29% and −40%. F1a with specification parameters is therefore expected to fail TPOT clearly, even after the B30a fix moves the cross-server steps onto the inter-server link.

## 3. F4 planner terms

`h3o_operator_costs/f_variants/f4_planner_blocks.py` (commit `c5094806`) initializes the A10 configuration with `vllm_v1_scheduler_config_num_blocks_mode=memory_planner_profiled` instead of the native explicit count. Outputs `stage4/f_variants/f4/p2_{32,64}_init/planner_blocks.json`:

| Case | Native blocks | Planner blocks per decode lane | Difference |
| --- | --- | --- | --- |
| P2-32 | 109,934 | 116,034 | +5.55% |
| P2-64 | 112,914 | 120,623 | +6.83% |

Planner terms: requested memory 128,687,957,606 B (141 GiB x 0.85; torch reports 139.81 GiB for the device, about 1.01 GiB less at 0.85), parameter memory per device 12,393,714,140 B (P2-32) and 7,794,659,804 B (P2-64) from `ParamCounter` (about 1.13 and 2.63 GiB below the native weight memory), non-KV overhead 0 (native about 3.54 GiB), block size 1,002,240 B. A driver fix was needed first: `kv_admission` derived the native per-request block range from the simulated pool, so a planner pool failed the driver after initialization; `6a7ff2a4` reads both native values from the case.

Full replays `stage4/f_variants/runs/f4_p2_{32,64}_planner` (detached worktree `a10-f-variants` at `6a7ff2a4`, warm cache copied from B28) started at about 09:10 UTC. Results: pending.

## 4. Commands and environment

- Python `/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6, `PYTHONPATH` the repository root, `ulimit -v 31457280` (screens) or `62914560` (replays).
- F1 screen: `bash task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/f_variants/run_f1_per_call.sh <name> [field=value ...]`; outputs `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/f1/<name>/{per_call.csv,summary.json}`. Runs: `astra_default`, `astra_spec`, `astra_fc_lat{5,6,7,8}`; `astra_default_r2` and `_r3` reproduce `astra_default` byte for byte after the script cleanup.
- Dimension check: `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/scripts/check_astra_dims.py`.
- F4: `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/run_f4_init.sh <case> <name>` and `run_f_variant.sh <case> <name> [field=value ...]`.

## Limits

- The F1 screen prices the measured rows only; it does not include the effect of overlap or the eager prefill paths on the request metrics.
- The TPOT effect of F1a is an inference from the forward composition, not a replay.
