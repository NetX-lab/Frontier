# Test report: Codex resume, B29 gates/A8 and F comparisons

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-31 | Captured the five existing canonical readings, reran B29 A8, compared F4 with B28, and synchronized current execution records. |

## Scope and passing conditions

Decision 46 keeps measured communication formal and requests F1–F4 diagnostic comparisons plus ingress A. The current request resumes the task directly in Codex CLI. Native references use vLLM 0.28.0 at `2cf0a6915ce544dc493a0990f2ea38d81601128a`; the P2 reference R4 records are PASS under their original admission. No admission or promotion receipt is backfilled.

Numerical PASS requires mean client TTFT, mean request TPOT, mean request E2E and full-window token THPT each within 10% for **every** admitted clean repetition. Each has 128 fixed 6144/1024-token formal requests, 917504 tokens and separate warmup/window boundaries. Pooled values are supplementary. Empty identity/token/client gaps and routing MATCH are required. Full H3o/table qualification is separate from these numerical results.

## Commands and environment

CPU Python: `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6. Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Stdlib artifact inspection uses `/usr/bin/python3`. No Conda activation is used.

Gate command shape, with exact commands/timestamps/exit codes in `h3o_operator_costs/codex_resume_20261031/commands.json`:

```bash
/data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a10_gate.py CASE RUN_ROOT OUTPUT_ROOT
```

The five commands were executed at 2026-10-31T14:45:48–14:45:49Z and returned 0. The tool may return 0 for a FAIL reading; `reading.json` decides the verdict. This continuation retained those complete outputs and command logs without rerunning their simulations.

Fresh A8 commands, both exit 0, with exact start/end and source identities in `a8_p2_32_command.json` and `a8_p2_64_command.json`:

```bash
/data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a8_collectives.py /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_32_warm_k19b29
/data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/a10/a8_collectives.py /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/a10/p2_64_warm_k19b29
```

## Source and run identities

| Run | Source commit | Run state | Numerical verdict |
| --- | --- | --- | --- |
| p2_32_warm_k19b29 | `a69a52132257ca854a8759249bca65b32ffa06bf` | FINISHED, clean source | PASS |
| p2_64_warm_k19b29 | `a69a52132257ca854a8759249bca65b32ffa06bf` | FINISHED, clean source | FAIL |
| f4_p2_32_planner | `6a7ff2a458b18cb6359021f7ac227646e1b292c8` | FINISHED, clean source | FAIL |
| f4_p2_64_planner | `6a7ff2a458b18cb6359021f7ac227646e1b292c8` | FINISHED, clean source | FAIL |
| f1a_p2_32_spec | `9f5f8b614f1e278b02ad39015f0230de562c82ba` | FINISHED, clean source | FAIL |

B29 selects 10.3 ms ingress and 11.4 ms handoff, measured CC, K=19. F1a includes B29 delays and uses the approved diagnostic B30a checkout with specification links (Switch, 3600/400 Gbit/s, 1/1 us). F4 uses B28's zero-delay configuration and the memory planner. Each receipt pins its actual profiling inputs and native case.

## Pooled numerical values (supplementary)

Errors below use absolute magnitude; `Delta` is signed Frontier minus native. Units are ms for latencies and tok/s for THPT. The serialized gate's `absolute_error` field contains a signed delta; it is not used as an absolute magnitude here.

| Run | Metric | Native | Frontier | Delta | Absolute error | Relative error | Metric status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| p2_32_warm_k19b29 | TTFT ms | 364.029649531 | 344.878991339 | -19.150658192 | 19.150658192 | 5.260741321% | PASS |
| p2_32_warm_k19b29 | TPOT ms | 19.918966075 | 19.926786349 | +0.007820274 | 0.007820274 | 0.039260442% | PASS |
| p2_32_warm_k19b29 | E2E ms | 20741.131944572 | 20729.981426836 | -11.150517736 | 11.150517736 | 0.053760411% | PASS |
| p2_32_warm_k19b29 | THPT tok/s | 6231.415496061 | 6225.735861750 | -5.679634311 | 5.679634311 | 0.091145171% | PASS |
| p2_64_warm_k19b29 | TTFT ms | 741.216351415 | 684.313826604 | -56.902524811 | 56.902524811 | 7.676911701% | PASS |
| p2_64_warm_k19b29 | TPOT ms | 25.870530968 | 25.295336618 | -0.575194350 | 0.575194350 | 2.223357345% | PASS |
| p2_64_warm_k19b29 | E2E ms | 27206.769531712 | 26561.443186463 | -645.326345249 | 645.326345249 | 2.371933002% | PASS |
| p2_64_warm_k19b29 | THPT tok/s | 5963.206317891 | 5975.251569855 | +12.045251964 | 12.045251964 | 0.201992876% | PASS |
| f4_p2_32_planner | TTFT ms | 364.029649531 | 323.178991339 | -40.850658192 | 40.850658192 | 11.221794226% | FAIL |
| f4_p2_32_planner | TPOT ms | 19.918966075 | 19.926786349 | +0.007820274 | 0.007820274 | 0.039260442% | PASS |
| f4_p2_32_planner | E2E ms | 20741.131944572 | 20708.281426837 | -32.850517736 | 32.850517736 | 0.158383437% | PASS |
| f4_p2_32_planner | THPT tok/s | 6231.415496061 | 6226.652709244 | -4.762786817 | 4.762786817 | 0.076431861% | PASS |
| f4_p2_64_planner | TTFT ms | 741.216351415 | 662.613826604 | -78.602524811 | 78.602524811 | 10.604531951% | FAIL |
| f4_p2_64_planner | TPOT ms | 25.870530968 | 25.295336618 | -0.575194350 | 0.575194350 | 2.223357345% | PASS |
| f4_p2_64_planner | E2E ms | 27206.769531712 | 26539.743186463 | -667.026345249 | 667.026345249 | 2.451692563% | PASS |
| f4_p2_64_planner | THPT tok/s | 5963.206317891 | 5976.096120186 | +12.889802295 | 12.889802295 | 0.216155565% | PASS |
| f1a_p2_32_spec | TTFT ms | 364.029649531 | 376.104429448 | +12.074779916 | 12.074779916 | 3.316977046% | PASS |
| f1a_p2_32_spec | TPOT ms | 19.918966075 | 14.265762133 | -5.653203942 | 5.653203942 | 28.381010948% | FAIL |
| f1a_p2_32_spec | E2E ms | 20741.131944572 | 14969.979091326 | -5771.152853246 | 5771.152853246 | 27.824676438% | FAIL |
| f1a_p2_32_spec | THPT tok/s | 6231.415496061 | 6468.179036603 | +236.763540542 | 236.763540542 | 3.799514584% | PASS |

## Per-repetition TTFT and verdict

The retained complete `reading.json` files and `metrics.json` contain all four metrics for each repetition. This table highlights TTFT, the B29/F4 failure boundary.

| Run | Repeat | Native TTFT ms | Frontier TTFT ms | Absolute error ms | Relative error | All-four verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| p2_32_warm_k19b29 | repeat01 | 361.974763982 | 344.878991339 | 17.095772644 | 4.722918376% | PASS |
| p2_32_warm_k19b29 | repeat02 | 364.748495980 | 344.878991339 | 19.869504641 | 5.447453481% | PASS |
| p2_32_warm_k19b29 | repeat03 | 365.365688631 | 344.878991339 | 20.486697292 | 5.607176024% | PASS |
| p2_64_warm_k19b29 | repeat01 | 779.888727400 | 684.313826604 | 95.574900796 | 12.254940665% | FAIL |
| p2_64_warm_k19b29 | repeat02 | 725.492279951 | 684.313826604 | 41.178453347 | 5.675932671% | PASS |
| p2_64_warm_k19b29 | repeat03 | 718.268046894 | 684.313826604 | 33.954220290 | 4.727235248% | PASS |
| f4_p2_32_planner | repeat01 | 361.974763982 | 323.178991339 | 38.795772643 | 10.717811434% | FAIL |
| f4_p2_32_planner | repeat02 | 364.748495980 | 323.178991339 | 41.569504641 | 11.396758342% | FAIL |
| f4_p2_32_planner | repeat03 | 365.365688631 | 323.178991339 | 42.186697292 | 11.546431043% | FAIL |
| f4_p2_64_planner | repeat01 | 779.888727400 | 662.613826604 | 117.274900796 | 15.037388883% | FAIL |
| f4_p2_64_planner | repeat02 | 725.492279951 | 662.613826604 | 62.878453347 | 8.667005161% | PASS |
| f4_p2_64_planner | repeat03 | 718.268046894 | 662.613826604 | 55.654220290 | 7.748391500% | PASS |
| f1a_p2_32_spec | repeat01 | 361.974763982 | 376.104429448 | 14.129665465 | 3.903494628% | FAIL |
| f1a_p2_32_spec | repeat02 | 364.748495980 | 376.104429448 | 11.355933467 | 3.113359916% | FAIL |
| f1a_p2_32_spec | repeat03 | 365.365688631 | 376.104429448 | 10.738740817 | 2.939176050% | FAIL |

Observation: B29 P2-32 passes all four metrics for all three references. B29 P2-64 repeat 1 TTFT remains FAIL (12.254940665%); repeats 2/3 pass. Pooled 7.676911701% cannot override it. F1a P2-32 FAILs TPOT/E2E in each repeat. No reference or request was excluded.

## B29 A8 census

Expected: finished runs, census calls equal observer calls, all recorded bounds valid, forwards equal stage ends and no stale ends. Actual:

| Case | Distinct requests | Calls | Exact calls | Forward rows | Stage ends | Stale ends | Violations | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| p2_32 | 44 | 6,310,644 | 6,310,644 | 23,040 | 23,040 | 0 | 0 | PASS |
| p2_64 | 38 | 9,002,176 | 9,002,176 | 39,680 | 39,680 | 0 | 0 | PASS |

Uneven prefill dispatch/combine counts are 16512 each for P2-32 and 27520 each for P2-64, all exact. Selected logs: `a8_p2_32.log`, `a8_p2_64.log`. Exact calls avoid interpolation in these runs; they do not qualify failing held-out interpolation curves.

## F4 comparison with B28

Direct Python byte comparisons inspected the two finished receipts, five metric artifacts and flattened actual configurations. At each scale, `request_metrics.csv`, `system_metrics.json`, `metrics_ground_truth.jsonl`, `frontier_transfer_ledger.jsonl` and `op_precision_metadata.csv` are byte-identical: **5/5 PASS for P2-32 and 5/5 PASS for P2-64**. Config differs only in `num_blocks_mode`, `num_blocks` and three output/cache paths. Exact differences are in `f4_p2_32_b28_comparison.json` and `f4_p2_64_b28_comparison.json`.

Observation: using planner blocks did not change these workload's output metrics. The planner predicts 116034 vs native 109934 (+5.55%) and 120623 vs 112914 (+6.83%). Both runs remain TTFT FAIL under zero ingress/handoff. Their difference from B29 must not be attributed to the planner because B29 adds 21.7 ms of serving-path delay.

## Running work and limits

Fresh F1a P2-64 command:

```bash
bash /data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/run_f1a.sh p2_64 f1a_p2_64_codex_20261031a
```

Run root `/data/ycfeng/tmp/glm53_dsv4/h3o_b/stage4/f_variants/runs/f1a_p2_64_codex_20261031a`, session 50972, source 9f5f8b61 clean. Initialization 72.4555 s; run still executing at this checkpoint. The earlier attempt has no terminal evidence or known initiating cause; it is retained as incomplete. The wrapper's final shell status alone is insufficient; read the workload EXIT marker and FINISHED receipt.

B30b preparations/probes/fits and F1b/F2/F3 replays remain pending. The latest full A6 after attention supplementation has not been verified; the last archived A6 has three attention p90 failures (12.207%, 11.518%, 17.476%) and failing uniform collective held-out curves. Current NIXL uses the H800 fit; native V4 packed transfer is 1 region, 1002240 B/block, with a documented difference from the simulator payload. Memory terms and valid formal promotion remain open. This report establishes numerical comparison/artifact inspection depth, not full family/operator/hardware qualification.

The read-only quota snapshot at 2026-10-31T14:59:48–14:59:50Z is retained as `quota_codex_20261031a.json`; this task had zero GPU reservations. It is not a reservation or guarantee of placement. New jobs are submitted only after refreshed personal/quota checks and the required cost report. No GPU job or external publication was performed during this documentation/evidence step.

## Document verification

Current execution entries in 17 maintained documents and 26 current links/anchors pass direct inspection; `git diff --check` passes. The first document-only validator asserted Codex must appear in the plan title, an unsupported requirement; its assertion was corrected and the actual document/link checks passed. No metric, runtime or task semantics changed for that correction. Receipt: `h3o_operator_costs/codex_resume_20261031/document_verification.json`.
