# Current warm TTFT residual: WARM-RESIDUAL

## Modification History
| Date | Change |
| --- | --- |
| 2026-11-01 | Joined all six current replays, canonical warm references, wave64 and lifecycle32 spans; retained uncertainty and FIFO alignment separately. |

## 1. Result and scope

The remaining underprediction is **36.33 / 20.06 / 20.91 ms on P2-64** and
**19.15 / 20.34 / 20.41 ms on P2-32** (B29 / pass1_all / pass2_all).
The three P2-64 values do not establish different underlying accuracy: their
prefill prices differ by <0.44 ms, but their prefill queue waits differ by
15.66 ms. The FIFO model reproduces each current queue mean within 0.103 ms.

No GPU, simulator rerun, code change in `frontier/` or `tests/comparison/`,
table promotion, commit or external operation was performed. Analysis scripts
and outputs are confined to this directory. Native identity remains vLLM
0.28.0, `2cf0a6915ce544dc493a0990f2ea38d81601128a`.

D-WARM-TARGET excludes the five previously evidenced first-contact requests
from P2-64 repeat01, not slow requests generally. Clean warm native means are
718.057862 / 725.492280 / 718.268047 ms; the 379-row pool is 720.639680 ms.
P2-32 keeps all 384 rows, mean 364.029650 ms. The requested replay numbers use
all 128 replay requests. Applying the same repeat01 exclusions to Frontier
gives pooled P2-64 **684.545783 / 701.662204 / 700.865508 ms**, rather than
684.313827 / 700.581587 / 699.733815 ms. This is a population correction,
not a simulation change; no new official gate reading is claimed here.

## 2. Span tables

Units: ms. Parentheses are **Frontier minus native**. Native clean runs observe
only client dispatch, proxy-to-P-response, P-response-to-first-D-byte and
client egress. Fine native allocations below are diagnostic estimates, not
additional clean measurements:

- P2-64 ingress/response/READ/first-D spans: warm wave64, n=125. Nonqueued
  step 289.6 ms: six wave-run requests, A10 section 5.5. Queue is the clean
  P-side remainder after subtracting these inputs. READ starts at D enqueue
  and ends at `finished_recving`, including engine-loop wait and polling.
- P2-32 lifecycle32, n=384, does not record enqueue. Its P ingress stops at
  engine add; the 0.22 ms queue is add-to-schedule only. Step 222.63 ms is
  inferred from the **clean** P side minus lifecycle ingress/response/queue,
  not the slower lifecycle step of 244.65 ms. D ingress 9.08 ms is borrowed
  from wave64; its remaining ingress plus READ is kept in the KV column.
- A nonzero clean–probe remainder is **unallocated**, never charged to egress
  or host time to force a cause. All fine P2-32 differences are provisional.
- Frontier 10.3 ms ingress and 11.4 ms handoff are split by their approved
  arithmetic into dispatch 0.6 + ingress 9.7 and response 2.3 + D ingress 9.1.
  These are one field each, not separately simulated clocks.

### P2-64

| Span | Native warm allocation | B29 (delta) | pass1 (delta) | pass2 (delta) | Current status |
| --- | ---: | ---: | ---: | ---: | --- |
| Client dispatch | 0.63 | 0.60 (-0.03) | 0.60 (-0.03) | 0.60 (-0.03) | Modeled in 10.3 ms ingress |
| Prefill ingress, proxy → enqueue | 9.70 | 9.70 (0.00) | 9.70 (0.00) | 9.70 (0.00) | Modeled; wave64 input |
| Prefill queue, enqueue → nonqueued step | 250.38* | 240.23 (-10.16) | 255.89 (+5.51) | 255.47 (+5.08) | K19 modeled; alignment dominates cross-table change |
| Nonqueued prefill step | 289.60* | 279.87 (-9.73) | 279.43 (-10.17) | 279.77 (-9.83) | Operators and kernel gaps modeled; host explicitly zero |
| Prefill response | 2.32 | 2.30 (-0.02) | 2.30 (-0.02) | 2.30 (-0.02) | Modeled in handoff field |
| Decode ingress, response → enqueue | 9.08 | 9.10 (+0.02) | 9.10 (+0.02) | 9.10 (+0.02) | Modeled |
| KV lifecycle, enqueue → receive report | 97.18 | 87.29 (-9.89) | 88.40 (-8.78) | 87.59 (-9.59) | Post/find/report and depth2 modeled; late handles not modeled |
| Receive report → first D output | 55.18 | 55.23 (+0.04) | 55.16 (-0.02) | 55.21 (+0.03) | Depth2 modeled; initial replay request masks ~3.4 ms ordinary gap |
| Egress, D update → client | 0.93* | 0 (-0.93) | 0 (-0.93) | 0 (-0.93) | Not modeled; only client-byte part observed clean |
| Unallocated clean–probe handoff difference | +5.63 | 0 (-5.63) | 0 (-5.63) | 0 (-5.63) | Unexplained |
| **TTFT** | **720.64** | **684.31 (-36.33)** | **700.58 (-20.06)** | **699.73 (-20.91)** | Clean total observed |

### P2-32

| Span | Native warm allocation | B29 (delta) | pass1 (delta) | pass2 (delta) | Current status |
| --- | ---: | ---: | ---: | ---: | --- |
| Client dispatch | 0.46 | 0.60 (+0.14) | 0.60 (+0.14) | 0.60 (+0.14) | Modeled, P2-64 input |
| Prefill ingress, proxy → engine add* | 7.81* | 9.70 (+1.89) | 9.70 (+1.89) | 9.70 (+1.89) | Endpoint mismatch; P2-32 enqueue unobserved |
| Prefill queue, add → schedule* | 0.22* | 0 (-0.22) | 0 (-0.22) | 0 (-0.22) | GPU queue not separately observed; Frontier queue zero |
| Prefill step inferred from clean P side* | 222.63* | 209.95 (-12.68) | 209.52 (-13.12) | 209.85 (-12.78) | Cause of this P2-32 step deficit not isolated |
| Prefill response* | 4.83* | 2.30 (-2.53) | 2.30 (-2.53) | 2.30 (-2.53) | Borrowed P2-64 input; P2-32 clean value unknown |
| Decode ingress* | 9.08* | 9.10 (+0.02) | 9.10 (+0.02) | 9.10 (+0.02) | Borrowed wave64 split, not independently observed |
| KV lifecycle* | 83.00* | 69.96 (-13.04) | 69.28 (-13.72) | 68.81 (-14.19) | Provisional; clean–probe mismatch below prevents attribution |
| Receive report → first D output* | 42.76* | 43.26 (+0.50) | 43.20 (+0.43) | 43.25 (+0.49) | Initial request masks ~2.6 ms ordinary gap |
| Egress* | 4.30* | 0 (-4.30) | 0 (-4.30) | 0 (-4.30) | Diagnostic host crossing; clean client-byte span only 0.15 ms |
| Unallocated clean–probe handoff difference | **-11.06** | 0 (+11.06) | 0 (+11.06) | 0 (+11.06) | Lifecycle handoff is slower; cannot assign it to clean KV |
| **TTFT** | **364.03** | **344.88 (-19.15)** | **343.69 (-20.34)** | **343.62 (-20.41)** | Clean total observed |

The stable, directly observed coarse budgets are:

| Case | Native dispatch + P response | Frontier ingress + P step/queue + response (B29/pass1/pass2) | P-side shortfall | D-side shortfall through client |
| --- | ---: | ---: | ---: | ---: |
| P2-64 | 552.64 | 532.69 / 547.92 / 547.83 | 19.94 / 4.72 / 4.80 | 16.38 / 15.34 / 16.10 |
| P2-32 | 235.96 | 222.55 / 222.12 / 222.45 | 13.40 / 13.84 / 13.50 | 5.75 / 6.50 / 6.91 |

### Alignment-adjusted diagnostic, not a new gate

The scratch's ~689 ms "phase average" is reproduced by averaging the FIFO
wait over dummy price **24.20–24.50 ms in 0.001 ms increments**, not selecting
a favorable table. It is a **price-neighborhood phase proxy**: price changes
alter the sync/wave pattern, so it is not literally an initial-phase average
of one unchanged simulator. These limits come from the supplied scratch
analysis, not a native-TTFT fit.

| P2-64 diagnostic | B29 | pass1 | pass2 |
| --- | ---: | ---: | ---: |
| Recorded queue | 240.23 | 255.89 | 255.47 |
| Price-neighborhood average queue | 244.80 | 244.83 | 244.73 |
| Phase-proxy TTFT | **688.88** | **689.52** | **689.00** |
| Native minus phase-proxy TTFT | 31.76 (4.41%) | 31.12 (4.32%) | 31.64 (4.39%) |
| Fixed-price long-window mean queue (8192 requests, first128 omitted) | 247.98 | 255.79 | 255.37 |
| Fixed-price long-window TTFT estimate | 692.07 | 700.48 | 699.63 |

Across the 903 price/step combinations the queue range is 232.37–257.98 ms.
Even the narrower 24.33–24.39 ms neighborhood spans ~234.5–258.0 ms. Therefore
one replay's alignment is not a validated accuracy improvement. P2-32 has
zero simulated prefill queue under every tested price and step perturbation.

## 3. Ranked causes and evidence class

Contributions below use **native minus Frontier**; percentages use native
TTFT, not the residual denominator. Rows marked overlapping are not additive.

| Rank | Cause / contribution | Evidence | Classification / decision |
| --- | --- | --- | --- |
| 1, P2-64 | Nonqueued P step **9.7–10.2 ms (1.35–1.42%)**. Within it, ~7.5 ms host and ~2.2–2.7 ms device; these are not additional terms. | Step 289.6 vs current 279.4–279.9 observed; host/device split inferred across wave and profile runs. A10:404–428, 465–476, 530. | Host: `simulation_logic` explicit zero; device: candidate `measurement_input` kernel gaps/composition. Warm recurring cost, consistent with D-WARM-TARGET; host guard/behavior changes need YC under D-NO-NEW-MODELING. |
| 1, P2-32 | Clean P-side shortfall **13.4–13.8 ms (3.68–3.80%)**; provisional nonqueued step part 12.7–13.1 ms. | Clean coarse span observed; fine attribution inferred. Lifecycle step is 22 ms above the canonical inferred step; its profile forward is peer-delayed, not an absolute clean step witness. | **Unexplained** split between missing host, device inputs and response. Explicit host-zero is known, but no measured P2-32 7.5 ms attribution. Warm-target compatible; no correction by fitting a step time. |
| 2, P2-64 | READ lifecycle **8.8–9.9 ms (1.22–1.37%)** short relative to wave64. | Enqueue→report observed 97.18 vs current 87.29–88.40. One transfer time per request in NIXL code; native waits for all TP handles. A10:277–279; design.md:39–50, 114–122. | Candidate `simulation_logic` absent rank-handle spread, plus `measurement_input` H800 wire fit/payload mismatch. Actual progress/contention/late-post cause **unexplained**. New mechanism needs YC; warm steady READ only, no handshake modeling. |
| 2, P2-32 | Clean post-P shortfall **5.8–6.9 ms (1.58–1.90%)**, not an established 13–14 ms KV deficit. | Clean observed. Diagnostic KV deficit 13–14 ms, egress4.3 ms and first-step agreement are offset by **-11.06 ms clean–probe handoff**. | **Unexplained** clean allocation. Do not import the slower lifecycle component totals as measurement inputs. |
| 3, P2-64 | Queue shortfall after price-neighborhood averaging **5.6 ms (~0.78%)**. Recorded alignment contributes **-4.6 / +11.1 / +10.7 ms** to Frontier TTFT relative to that average. | Current forwards and FIFO check observed; native queue partition and smoothing interpretation inferred. summary.json:51–69, 128–146, 205–223; sweep_summary.json:38–85. | Queue mechanism already modeled. Residual candidate `measurement_input` prefill/dummy/host prices, not proof of a K19 logic defect. No arbitrary K or dummy-price tuning. |
| 4, P2-64 | **5.63 ms (0.78%)** clean–wave handoff difference remains unallocated. | Canonical handoff vs sum of warm wave spans, observed arithmetic; cause not observed. Canonical handoff repeat means 160.62/172.35/170.19 ms. | **Unexplained** run/reference variation, not an extra model delay. |
| 5 | Egress P2-64 ~**0.93 ms (0.13%)**, P2-32 diagnostic4.30 ms (1.18%), clean value unknown beyond0.15 ms. | Native update→byte crosses hosts; client-byte spans observed clean. | `simulation_logic` boundary omitted, amount only partly measured. New field or mechanism needs YC; do not enlarge existing ingress fields to absorb it. |

**First decode residual is masked, not demonstrably fixed.** Current P2-64
receive→first token =55.16–55.23 ms, but excluding request0 *for sensitivity
only* gives51.75–51.81 ms vs native55.18 ms (~3.4 ms). P2-32 similarly gives
40.58–40.64 vs42.76 ms (~2.1 ms). Request0 alone queues 19 dummy forwards on
the initially idle D group. It raises the full-run mean by3.4/2.6 ms compared
with the other requests' mean. It is queue initialization, not an evidenced
compile/handshake request, so this report does **not** exclude it from the
comparison under D-WARM-TARGET. Neither adding another decode queue mechanism
nor calling the all-request agreement a fixed first-step model is justified.

**Old residuals already repaired:** K19 launch rule; engine-step READ and
DECODE depth2/output placeholders; eager EP aligned-call/exact uneven rows
(the old -7.08 ms EP32 combine gap is not an open correction); dispatch,
P ingress, P response and D external ingress via B29's two fields. The V4
host-zero contract, late handles, egress and unallocated reference differences
remain. Graph dummy price is still24.34–24.38 vs measured paced23.81 ms on
wave64; its overprice opposes TTFT underprediction. Correcting it is not a way
to close the TTFT gap, and could also move TPOT/E2E.

## 4. Smallest next corrections and independent validation

No correction below is authorized by this analysis. They exclude warmup costs.

| Action | Smallest complete proposal | TTFT estimate, P2-64 / P2-32 | Cost and independent evidence |
| --- | --- | --- | --- |
| Host omission | If YC reopens explicit-zero decision, port the existing V4 CPU probe/table path, preserve host/device overlap, and remove the V4 host-zero guard only with admitted inputs. Do not add a TTFT constant. | For a **measured +7.5 ms P-step**: price-neighborhood FIFO gives **+9.2–9.5 ms / +7.5 ms**. P2-32 value is hypothetical until its own host spans are measured. The exact128-request FIFO effect is **+19.0/-1.5/-1.5 ms** on P2-64's three prices; alignment can reverse one replay's mean change. | Retained-trace CPU accounting first; zero GPUs. New probe/data and held-out native validation require approval. Clean/probe pair on fresh32/64-GPU allocations, ~10/20 GPUh for one formal repeat; additional reference repeats budgeted separately. Validate span prediction and ordinary clean client TTFT, not only exit status. |
| Small P-device gap | Recheck existing per-segment kernel-gap inputs and overlap composition against matched native kernels. Correct only a witnessed count/input/overlap defect; no 282.1 ms step-table substitution. | A real +2.7 ms P-step gives **+3.9–4.3 ms** smoothed P2-64; **+2.7 ms P2-32 only if the same measured defect exists**. Combined +9.7 ms step gives +12.9–13.0 ms /+9.7 ms. | CPU zero GPU for retained matched traces. New kernel-gap measurements need GPU approval. Hold out different DP lane/forward/shape and clean runs not used to choose the change. Current dp0 trace is only one absolute device witness. |
| READ residual | First replay the post/find/report rule on retained native step timestamps and reconcile eight native handle bytes with Frontier's grouped payload. If an actual structural payload or lifecycle defect is proved, fix that existing path; otherwise propose warm multi-handle progress measurement, not a delay factor. | At fixed schedule the observed budget suggests **up to +8.8–9.9 ms P2-64**. **P2-32 clean amount cannot be inferred** from the13–14 ms diagnostic deficit; total clean post-P budget is only5.8–6.9 ms and includes other terms. Step quantization can move actual effects by whole steps. | CPU first, zero GPU. A two-node eight-handle sender/receiver probe needs ~16 GPUs; ten minutes is2.7 GPUh, plus approved held-out clean32/64 runs. Hold out another load level/handle size and a new native run; existing lifecycle64 can check structure but not the canonical absolute P step. |
| P2-32 response / input portability | Measure its own enqueue and response endpoints before changing the existing two driver inputs. Do not choose10.3/11.4 from client TTFT. | If lifecycle response4.83 is representative clean, response correction alone **+2.53 ms P2-32 / ~0.02 ms P2-64**. P2-32 ingress correction might offset part of it; missing enqueue evidence prevents a signed net estimate. | CPU raw log check first. Missing enqueue requires an approved native diagnostic; independently validate on a later same-build clean/probe pair. |
| Egress | Only after a warm clean host-clock/endpoint measurement, decide whether the existing request-output boundary should price a recurring response path. Separate egress from ingress. | Arithmetic **~+0.93 ms P2-64**; P2-32 **unknown** (diagnostic+4.30 ms is not a clean estimate). | CPU endpoint study first; GPU only if the native observation is missing. New config/mechanism needs YC; validate on different requests/repeats, not the chosen mean. |

Recommendation: retain current formal inputs and K19. Finish the **CPU native
READ/payload check and per-segment P-step accounting** first. Then ask YC about
one concrete, evidenced correction. Do not chase a best attention table,
reopen PREFILL depth2 for a one-request/no-chunk workload, fit an intercept to
the rejected F3 transfer data, or model lazy handshakes. P2-32's missing absolute
step/clean handoff evidence is the main limitation on a joint correction.

## 5. Commands, outputs and validation limits

Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.
Python `/data/ycfeng/envs/frontier-py310/bin/python`,3.10.6;
`PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp`, `ulimit -v 31457280`.

Executed commands:

```bash
bash task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/warm_residual_20261101/run.sh
# At that first execution run.sh contained the native32 join and analyze.py;
# sweep.py was added afterwards and executed once separately:
ulimit -v 31457280
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp
OUT=task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/warm_residual_20261101
/data/ycfeng/envs/frontier-py310/bin/python "$OUT/sweep.py" "$OUT" > "$OUT/sweep_stdout.json"
/data/ycfeng/envs/frontier-py310/bin/python "$OUT/validate.py" "$OUT" > "$OUT/validation.txt"
```

The updated `run.sh` is the combined reproducer. Source inspection used read,
grep and bounded stdlib/gzip parsing; no Torch import or GPU access. The one
initial config inspection used the wrong root path and returned no file;
discovery found `metrics/.../config.json`, and all six actual configurations
were read and asserted in analyze.py. No run input changed.

Artifacts: `summary.json` (six replay spans, exact warm clean coarse spans,
fixed-price FIFO checks), six `*_rows.json`, two `*_clean_rows.json`,
`native32_remainder.{json,txt}`, `sweep_summary.json`, `span_tables.csv`,
`validation.{json,txt}` (PASS and script hashes), `checkpoint.md`, timestamp,
Python and Git identity text files. New scripts: `analyze.py`, `fifo_model.py`,
`sweep.py`, `validate.py`, `run.sh`.

Passing conditions: native populations379/384; six complete128-request joins;
span sum and first-D queue/compute conservation; FIFO mean wait error<0.11 ms;
price-neighborhood TTFT688.8–689.6 ms for all three P2-64 prices. These verify
the analysis arithmetic, not a corrected Frontier prediction. No fine native
span is claimed to be measured in the canonical run; no held-out experiment
or new native gate was executed. Auxiliary estimates are not independent
validation. Clock offsets, instrumented-run slowdown and P2-64's six-request
nonqueued anchor limit exact causal attribution.

Task ID: **WARM-RESIDUAL**; session `ses_e68fcce2effep415tOL36M7v4g`.
Native join/current-replay analysis executed12:11:17–12:11:50Z; the separate
price-neighborhood sweep ended12:13:23Z. All analysis commands exited0.
Validation: **9/9 case/generation checks PASS**; no core/comparison diff.
Primary owns task-wide warm-gate integration and
the final approval decision; all requested read-only deliverables are complete.
