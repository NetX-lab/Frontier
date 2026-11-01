H200 NIXL READ timing proposal for P2 (Q2)

**1. Assignment and outcome.** Analysis completed on 2026-11-02 Asia/Hong_Kong. Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. All new files are under `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/`; `scratch/` contains trial copies and initial logs. Outcome: **implemented as a proposal and evidence handoff; formal adoption remains with YC/Claude**. No Frontier source, profiling table, formal input, threshold, fold, reference, task record, receipt or retained measurement was changed. No Git command, commit, dependency installation, GPU, platform, cloud, deployment or messaging operation was performed. The pre-existing `.cortexkit/`, `.pi/`, and `remaining_measurements/` are outside this task and were not edited. This directory was absent before the assignment.

The controlling requirements are D-P2-INPUT-QUALIFICATION and D-P2-Q1-CHOICES (timing route B), D-WARM-TARGET, D-WARM-TWO-FIX and D-COLLECTIVE-EXACT-REPEATABILITY in [requirements.md](../../../requirements.md), together with tickets 05/09/11/19. The collective exact-point rule does **not** automatically authorize a NIXL rule. Ticket 19 forbids the current H800 substitution. The packed payload remains the committed selected-page count × 1,002,240 B; native request descriptor-byte observation remains a separate OPEN edge. The requested `20261101` directory and historical report dates are retained; the raw run spans 2026-11-01 UTC / 2026-11-02 Hong Kong time.

**Recommended single choice for YC: approve `AFFINE_MEDIAN_P2_SINGLE_READ` in [proposed_input.json](proposed_input.json)**, with its exact P2 domain and proposed median rule. Use `network_bandwidth_gbps=390.393499282548` and `network_latency_ms=0.023652184281886` on the existing analytical transfer predictor. This is an H200 fit of all 90 full-batch medians, with a positive intercept; it prices one 46,103,040 B READ at **0.968402341951 ms**. Every held-out case/pass median meets the proposed 5% rule (maximum **0.919518%**). Restrict adoption to the retained P2 payload, TP8, one READ per pair and eight active pairs, unchanged trace and `engine_step`. A bytes-only configuration cannot enforce these restrictions itself; the external qualification census must reject other workloads. This choice qualifies a **median timing input**, not a universal tail model or full serving-fabric fidelity.

The proposed fit and exact point cannot move any READ release in either scale. This was proved over all four B29/two-fix event traces, so the conditional fresh CPU replays were **NOT_RUN**. The empirical samples at the actual coordinate also lie below every relevant boundary. A universal guarantee about future native serving tails is **NOT_ESTABLISHED**. If YC instead requires the proposed per-trial median 5% / p90 10% fidelity rule on the full grid, reject both deterministic median candidates: the affine rule fails 8/90 case-passes and the point rule fails 7/90. None of those failures was removed.

**2. Measured behavior and timing unit.** Inputs: [package README](../packages/nixl6144/README.md), stage/config/plan; [run verification](../nixl6144_run/verification.json), [case summary](../nixl6144_run/case_summary.csv), [pass spread](../nixl6144_run/case_pass_spread.csv), [equal-byte summary](../nixl6144_run/paired_6143_6144.csv), [run notes](../nixl6144_run/observation_notes.md), [terminal record](../nixl6144_run/platform_terminal.json), and [lead report](../../../test_report_2026-11-01_nixl6144_h200.md). Raw local input: `/data/ycfeng/tmp/glm53_dsv4/nixl6144_q1_20261102/download/C_nixl/nixl_reads.jsonl`; SHA256 `c2739c0af5a4c12fb57376b252f16bbf493402e05d87b73ce4f32a0cc825eabb`.

Rebuilt integrity: **23/23 downloaded file hashes and sizes match; 30 cases × passes 0/1/2 = 90 unique records; 12,150 reader trials; 2,700 critical-path repeat trials; 80,352/80,352 notifications; all reader data checks true and completion times positive; 90 producer CSV rows exactly reconstructed.** The existing 11/11 verification PASS was inspected and its matrix/geometry/trial/hash/CSV arithmetic was independently rechecked. Stage hardware identity and terminal status are retained records; no platform query was rerun. The source verifier is bound by hashes in [results.json](results.json). The measurement used two H200 nodes (`gpu-h200-0457`, `gpu-h200-0604`), eight TP P/D pairs and four UCX threads. Direct container exit codes remain unavailable in the retained report (HTTP 403); phase exits and replica success are recorded. Historical container cost was about 0.346 GPUh; this task adds 0 GPUh.

`complete_us` is measured from before descriptor preparation until **all requests in that reader's batch** are DONE, including preparation and posting. For each timed repetition, critical path is the maximum across active readers. `summarize_case` uses the median and sorted index `floor(0.9*N)` for p90 (index 27 of 30). Reader columns below summarize all reader-batch trials within that case/pass; [reader_summary.csv](reader_summary.csv) preserves all 405 individual reader/pass summaries and [scratch/reader_trials.csv](scratch/reader_trials.csv) all 12,150 trials with explicit millisecond column names. At inflight >1 the producer does **not** record each request's DONE time, nor an actual cross-node shared posting clock. The full-batch critical path is the slowest reader's elapsed completion, as defined by the producer. Do not divide a full-batch maximum by inflight and call it a per-request upper bound. These data cannot supply individual-request completion distributions at inflight >1.

At T6144, n=1, the pooled critical-path median is **0.964047 ms** for one pair and **0.971057 ms** for eight pairs, versus **2.987238193 ms** from the H800 pair at the committed wire bytes. Eight pairs scarcely increase median completion; concurrency mainly changes the total bytes per reader. The table below pools the critical-path trials of all three passes for each exact case; it does not select a pass.


| prompt | inflight per pair | bytes/reader batch | 1 pair median ms | 8 pairs median ms | 8-pair increase |
| --- | --- | --- | --- | --- | --- |
| 6143 | 1 | 46,103,040 | 0.963841 | 0.972165 | 0.8636% |
| 6143 | 2 | 92,206,080 | 1.909181 | 1.916816 | 0.3999% |
| 6143 | 4 | 184,412,160 | 3.796045 | 3.807770 | 0.3089% |
| 6143 | 8 | 368,824,320 | 7.574467 | 7.589901 | 0.2038% |
| 6143 | 16 | 737,648,640 | 15.124254 | 15.158731 | 0.2280% |
| 6144 | 1 | 46,103,040 | 0.964047 | 0.971057 | 0.7271% |
| 6144 | 2 | 92,206,080 | 1.911394 | 1.916522 | 0.2683% |
| 6144 | 4 | 184,412,160 | 3.795331 | 3.806405 | 0.2918% |
| 6144 | 8 | 368,824,320 | 7.573301 | 7.590653 | 0.2291% |
| 6144 | 16 | 737,648,640 | 15.125498 | 15.159331 | 0.2237% |
| 6145 | 1 | 51,114,240 | 1.065707 | 1.074791 | 0.8524% |
| 6145 | 2 | 102,228,480 | 2.113573 | 2.121797 | 0.3891% |
| 6145 | 4 | 204,456,960 | 4.205745 | 4.217019 | 0.2681% |
| 6145 | 8 | 408,913,920 | 8.391466 | 8.411231 | 0.2355% |
| 6145 | 16 | 817,827,840 | 16.761063 | 16.795158 | 0.2034% |



Same-case pass medians have median spread **0.090399%**, maximum **0.882168%** (`t6143_pairs8_inflight4`). Spread means `(max pass median - min pass median) / median of pass medians`. Every case's three values and deviation from its pooled median are listed below and in [repeatability.csv](repeatability.csv). This is repeatability on one source/hardware generation, not independence of fabric conditions or unseen geometry.


| case | pass 0 median ms | pass 1 median ms | pass 2 median ms | spread % |
| --- | --- | --- | --- | --- |
| t6143_pairs1_inflight1 | 0.965949 | 0.963639 | 0.963302 | 0.274669 |
| t6143_pairs1_inflight2 | 1.912383 | 1.909496 | 1.906587 | 0.303564 |
| t6143_pairs1_inflight4 | 3.798151 | 3.794326 | 3.795827 | 0.100792 |
| t6143_pairs1_inflight8 | 7.582201 | 7.573092 | 7.571120 | 0.146319 |
| t6143_pairs1_inflight16 | 15.128389 | 15.124768 | 15.120845 | 0.049877 |
| t6143_pairs8_inflight1 | 0.975454 | 0.972020 | 0.970116 | 0.549201 |
| t6143_pairs8_inflight2 | 1.918089 | 1.914155 | 1.916148 | 0.205303 |
| t6143_pairs8_inflight4 | 3.804237 | 3.837828 | 3.807770 | 0.882168 |
| t6143_pairs8_inflight8 | 7.590864 | 7.589765 | 7.589301 | 0.020590 |
| t6143_pairs8_inflight16 | 15.160657 | 15.152441 | 15.158569 | 0.054201 |
| t6144_pairs1_inflight1 | 0.965916 | 0.964448 | 0.963194 | 0.282164 |
| t6144_pairs1_inflight2 | 1.914194 | 1.909027 | 1.908677 | 0.289004 |
| t6144_pairs1_inflight4 | 3.794493 | 3.798235 | 3.795678 | 0.098587 |
| t6144_pairs1_inflight8 | 7.586900 | 7.572832 | 7.570038 | 0.222671 |
| t6144_pairs1_inflight16 | 15.129108 | 15.124038 | 15.122419 | 0.044226 |
| t6144_pairs8_inflight1 | 0.971612 | 0.970917 | 0.970868 | 0.076546 |
| t6144_pairs8_inflight2 | 1.914585 | 1.917183 | 1.917649 | 0.159820 |
| t6144_pairs8_inflight4 | 3.809392 | 3.805738 | 3.806241 | 0.096014 |
| t6144_pairs8_inflight8 | 7.590871 | 7.590791 | 7.589862 | 0.013300 |
| t6144_pairs8_inflight16 | 15.161661 | 15.151473 | 15.158143 | 0.067216 |
| t6145_pairs1_inflight1 | 1.065146 | 1.066942 | 1.066186 | 0.168413 |
| t6145_pairs1_inflight2 | 2.114462 | 2.113461 | 2.112670 | 0.084783 |
| t6145_pairs1_inflight4 | 4.206654 | 4.205853 | 4.205031 | 0.038574 |
| t6145_pairs1_inflight8 | 8.395636 | 8.390838 | 8.390633 | 0.059625 |
| t6145_pairs1_inflight16 | 16.771363 | 16.760232 | 16.758271 | 0.078117 |
| t6145_pairs8_inflight1 | 1.075650 | 1.074819 | 1.072941 | 0.251976 |
| t6145_pairs8_inflight2 | 2.122747 | 2.120985 | 2.122186 | 0.083031 |
| t6145_pairs8_inflight4 | 4.216738 | 4.216209 | 4.218290 | 0.049341 |
| t6145_pairs8_inflight8 | 8.413067 | 8.411402 | 8.410454 | 0.031068 |
| t6145_pairs8_inflight16 | 16.797956 | 16.795909 | 16.793266 | 0.027924 |



T6143 and T6144 select the same 46 pages / 46,103,040 B; T6145 selects 51 / 51,114,240 B (byte ratio **1.108695652**). Equal-byte pooled medians agree as in the retained summary (-0.0014 to +0.0022 ms). The stronger matched-pass comparison below retains all 30 pairs of case/pass medians: maximum difference **0.836141%**. The largest difference is the slow T6143 eight-pair n=4 pass, not a discarded trial. T6145 batch medians scale close to the 51/46 byte ratio, with a small fixed preparation/posting term. [equal_byte_passes.csv](equal_byte_passes.csv) also retains both p90 and maximum values; agreement of medians does **not** mean equality of tails.


| pairs | inflight | pass | 6143 median ms | 6144 median ms | difference % |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 0.965949 | 0.965916 | 0.003471 |
| 1 | 1 | 1 | 0.963639 | 0.964448 | 0.083889 |
| 1 | 1 | 2 | 0.963302 | 0.963194 | 0.011215 |
| 1 | 2 | 0 | 1.912383 | 1.914194 | 0.094672 |
| 1 | 2 | 1 | 1.909496 | 1.909027 | 0.024582 |
| 1 | 2 | 2 | 1.906587 | 1.908677 | 0.109614 |
| 1 | 4 | 0 | 3.798151 | 3.794493 | 0.096316 |
| 1 | 4 | 1 | 3.794326 | 3.798235 | 0.103041 |
| 1 | 4 | 2 | 3.795827 | 3.795678 | 0.003926 |
| 1 | 8 | 0 | 7.582201 | 7.586900 | 0.061980 |
| 1 | 8 | 1 | 7.573092 | 7.572832 | 0.003443 |
| 1 | 8 | 2 | 7.571120 | 7.570038 | 0.014294 |
| 1 | 16 | 0 | 15.128389 | 15.129108 | 0.004753 |
| 1 | 16 | 1 | 15.124768 | 15.124038 | 0.004828 |
| 1 | 16 | 2 | 15.120845 | 15.122419 | 0.010409 |
| 8 | 1 | 0 | 0.975454 | 0.971612 | 0.393933 |
| 8 | 1 | 1 | 0.972020 | 0.970917 | 0.113443 |
| 8 | 1 | 2 | 0.970116 | 0.970868 | 0.077569 |
| 8 | 2 | 0 | 1.918089 | 1.914585 | 0.182663 |
| 8 | 2 | 1 | 1.914155 | 1.917183 | 0.158224 |
| 8 | 2 | 2 | 1.916148 | 1.917649 | 0.078349 |
| 8 | 4 | 0 | 3.804237 | 3.809392 | 0.135528 |
| 8 | 4 | 1 | 3.837828 | 3.805738 | 0.836141 |
| 8 | 4 | 2 | 3.807770 | 3.806241 | 0.040161 |
| 8 | 8 | 0 | 7.590864 | 7.590871 | 0.000098 |
| 8 | 8 | 1 | 7.589765 | 7.590791 | 0.013522 |
| 8 | 8 | 2 | 7.589301 | 7.589862 | 0.007387 |
| 8 | 16 | 0 | 15.160657 | 15.161661 | 0.006622 |
| 8 | 16 | 1 | 15.152441 | 15.151473 | 0.006392 |
| 8 | 16 | 2 | 15.158569 | 15.158143 | 0.002814 |



All measured case/passes and their tails follow. C = critical path across readers; R = all reader-batch trials pooled within that pass. Every maximum is retained. Twelve case-passes have a trial above 3× the case median. The largest trial is **27.497038 ms** at `t6145_pairs8_inflight16`, pass 1. The heaviest p90 is **14.979299 ms** against median **8.411402 ms** at `t6145_pairs8_inflight8`, pass 1. At the actual P2 coordinate, `t6144_pairs8_inflight1`, pass maxima are **3.958579 / 12.554511 / 0.980675 ms**. Slow single-pair trials exist too: T6144 n=1 pass 1 reaches **12.938846 ms**. Tails are mainly an eight-pair phenomenon, not exclusively one. No UCX, fabric, host scheduling or warmup cause is established here; the timed slow trials remain in the population.


| case | pass | C median ms | C p90 ms | C max ms | R median ms | R p90 ms | R max ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| t6143_pairs1_inflight1 | 0 | 0.965949 | 0.971649 | 0.972956 | 0.965949 | 0.971649 | 0.972956 |
| t6143_pairs1_inflight1 | 1 | 0.963639 | 0.966337 | 0.969328 | 0.963639 | 0.966337 | 0.969328 |
| t6143_pairs1_inflight1 | 2 | 0.963302 | 0.964578 | 0.966799 | 0.963302 | 0.964578 | 0.966799 |
| t6143_pairs1_inflight2 | 0 | 1.912383 | 1.915298 | 1.918577 | 1.912383 | 1.915298 | 1.918577 |
| t6143_pairs1_inflight2 | 1 | 1.909496 | 1.912650 | 1.914836 | 1.909496 | 1.912650 | 1.914836 |
| t6143_pairs1_inflight2 | 2 | 1.906587 | 1.908712 | 1.911297 | 1.906587 | 1.908712 | 1.911297 |
| t6143_pairs1_inflight4 | 0 | 3.798151 | 3.802631 | 3.829494 | 3.798151 | 3.802631 | 3.829494 |
| t6143_pairs1_inflight4 | 1 | 3.794326 | 3.799271 | 15.706357 | 3.794326 | 3.799271 | 15.706357 |
| t6143_pairs1_inflight4 | 2 | 3.795827 | 4.073840 | 4.084382 | 3.795827 | 4.073840 | 4.084382 |
| t6143_pairs1_inflight8 | 0 | 7.582201 | 7.600576 | 7.646777 | 7.582201 | 7.600576 | 7.646777 |
| t6143_pairs1_inflight8 | 1 | 7.573092 | 7.579021 | 16.151853 | 7.573092 | 7.579021 | 16.151853 |
| t6143_pairs1_inflight8 | 2 | 7.571120 | 7.576503 | 16.932592 | 7.571120 | 7.576503 | 16.932592 |
| t6143_pairs1_inflight16 | 0 | 15.128389 | 15.159734 | 15.177038 | 15.128389 | 15.159734 | 15.177038 |
| t6143_pairs1_inflight16 | 1 | 15.124768 | 15.150528 | 15.203755 | 15.124768 | 15.150528 | 15.203755 |
| t6143_pairs1_inflight16 | 2 | 15.120845 | 15.124500 | 15.127737 | 15.120845 | 15.124500 | 15.127737 |
| t6143_pairs8_inflight1 | 0 | 0.975454 | 1.178671 | 2.124738 | 0.966711 | 0.986207 | 2.124738 |
| t6143_pairs8_inflight1 | 1 | 0.972020 | 1.029227 | 1.064062 | 0.966201 | 0.970978 | 1.064062 |
| t6143_pairs8_inflight1 | 2 | 0.970116 | 0.978053 | 1.002323 | 0.965783 | 0.969701 | 1.002323 |
| t6143_pairs8_inflight2 | 0 | 1.918089 | 1.926009 | 1.930874 | 1.910202 | 1.917463 | 1.930874 |
| t6143_pairs8_inflight2 | 1 | 1.914155 | 1.930717 | 2.038673 | 1.910094 | 1.913909 | 2.038673 |
| t6143_pairs8_inflight2 | 2 | 1.916148 | 1.949716 | 10.977920 | 1.911325 | 1.916680 | 10.977920 |
| t6143_pairs8_inflight4 | 0 | 3.804237 | 3.824197 | 3.922895 | 3.799805 | 3.804069 | 3.922895 |
| t6143_pairs8_inflight4 | 1 | 3.837828 | 4.249938 | 4.415151 | 3.798485 | 3.822319 | 4.415151 |
| t6143_pairs8_inflight4 | 2 | 3.807770 | 3.818993 | 11.796240 | 3.800435 | 3.807481 | 11.796240 |
| t6143_pairs8_inflight8 | 0 | 7.590864 | 7.613126 | 7.656772 | 7.579938 | 7.590652 | 7.656772 |
| t6143_pairs8_inflight8 | 1 | 7.589765 | 7.954702 | 23.836222 | 7.576540 | 7.589854 | 23.836222 |
| t6143_pairs8_inflight8 | 2 | 7.589301 | 7.656310 | 7.969752 | 7.580619 | 7.589251 | 7.969752 |
| t6143_pairs8_inflight16 | 0 | 15.160657 | 15.203483 | 26.162885 | 15.146740 | 15.161246 | 26.162885 |
| t6143_pairs8_inflight16 | 1 | 15.152441 | 16.029578 | 24.118129 | 15.130375 | 15.153632 | 24.118129 |
| t6143_pairs8_inflight16 | 2 | 15.158569 | 15.495818 | 26.478983 | 15.129875 | 15.162040 | 26.478983 |
| t6144_pairs1_inflight1 | 0 | 0.965916 | 1.124252 | 1.138613 | 0.965916 | 1.124252 | 1.138613 |
| t6144_pairs1_inflight1 | 1 | 0.964448 | 0.970341 | 12.938846 | 0.964448 | 0.970341 | 12.938846 |
| t6144_pairs1_inflight1 | 2 | 0.963194 | 0.964858 | 0.969738 | 0.963194 | 0.964858 | 0.969738 |
| t6144_pairs1_inflight2 | 0 | 1.914194 | 1.918599 | 1.920637 | 1.914194 | 1.918599 | 1.920637 |
| t6144_pairs1_inflight2 | 1 | 1.909027 | 1.913168 | 1.918443 | 1.909027 | 1.913168 | 1.918443 |
| t6144_pairs1_inflight2 | 2 | 1.908677 | 1.913168 | 1.918182 | 1.908677 | 1.913168 | 1.918182 |
| t6144_pairs1_inflight4 | 0 | 3.794493 | 3.795799 | 3.797274 | 3.794493 | 3.795799 | 3.797274 |
| t6144_pairs1_inflight4 | 1 | 3.798235 | 3.803298 | 3.805686 | 3.798235 | 3.803298 | 3.805686 |
| t6144_pairs1_inflight4 | 2 | 3.795678 | 3.799129 | 3.802214 | 3.795678 | 3.799129 | 3.802214 |
| t6144_pairs1_inflight8 | 0 | 7.586900 | 7.602155 | 7.616673 | 7.586900 | 7.602155 | 7.616673 |
| t6144_pairs1_inflight8 | 1 | 7.572832 | 7.577572 | 7.579114 | 7.572832 | 7.577572 | 7.579114 |
| t6144_pairs1_inflight8 | 2 | 7.570038 | 7.573284 | 7.575653 | 7.570038 | 7.573284 | 7.575653 |
| t6144_pairs1_inflight16 | 0 | 15.129108 | 15.151594 | 15.173547 | 15.129108 | 15.151594 | 15.173547 |
| t6144_pairs1_inflight16 | 1 | 15.124038 | 15.148230 | 16.282812 | 15.124038 | 15.148230 | 16.282812 |
| t6144_pairs1_inflight16 | 2 | 15.122419 | 15.126154 | 15.132580 | 15.122419 | 15.126154 | 15.132580 |
| t6144_pairs8_inflight1 | 0 | 0.971612 | 0.992671 | 3.958579 | 0.966439 | 0.971213 | 3.958579 |
| t6144_pairs8_inflight1 | 1 | 0.970917 | 0.974223 | 12.554511 | 0.966642 | 0.970993 | 12.554511 |
| t6144_pairs8_inflight1 | 2 | 0.970868 | 0.975192 | 0.980675 | 0.965737 | 0.970755 | 0.980675 |
| t6144_pairs8_inflight2 | 0 | 1.914585 | 1.921635 | 1.954477 | 1.910692 | 1.914669 | 1.954477 |
| t6144_pairs8_inflight2 | 1 | 1.917183 | 1.930222 | 1.937464 | 1.909971 | 1.916703 | 1.937464 |
| t6144_pairs8_inflight2 | 2 | 1.917649 | 1.940649 | 19.939218 | 1.910996 | 1.917001 | 19.939218 |
| t6144_pairs8_inflight4 | 0 | 3.809392 | 4.228037 | 8.245718 | 3.800863 | 3.809363 | 8.245718 |
| t6144_pairs8_inflight4 | 1 | 3.805738 | 3.819667 | 15.088618 | 3.798153 | 3.805138 | 15.088618 |
| t6144_pairs8_inflight4 | 2 | 3.806241 | 3.865458 | 14.517773 | 3.798943 | 3.805518 | 14.517773 |
| t6144_pairs8_inflight8 | 0 | 7.590871 | 7.609390 | 16.282447 | 7.580766 | 7.590637 | 16.282447 |
| t6144_pairs8_inflight8 | 1 | 7.590791 | 7.600177 | 7.612221 | 7.584127 | 7.591132 | 7.612221 |
| t6144_pairs8_inflight8 | 2 | 7.589862 | 7.597771 | 10.414653 | 7.577805 | 7.589512 | 10.414653 |
| t6144_pairs8_inflight16 | 0 | 15.161661 | 16.513385 | 24.354961 | 15.147254 | 15.160248 | 24.354961 |
| t6144_pairs8_inflight16 | 1 | 15.151473 | 18.048193 | 24.891950 | 15.126070 | 15.147232 | 24.891950 |
| t6144_pairs8_inflight16 | 2 | 15.158143 | 15.178308 | 15.428025 | 15.146479 | 15.158258 | 15.428025 |
| t6145_pairs1_inflight1 | 0 | 1.065146 | 1.067705 | 1.070734 | 1.065146 | 1.067705 | 1.070734 |
| t6145_pairs1_inflight1 | 1 | 1.066942 | 1.069274 | 1.072388 | 1.066942 | 1.069274 | 1.072388 |
| t6145_pairs1_inflight1 | 2 | 1.066186 | 1.069173 | 1.071915 | 1.066186 | 1.069173 | 1.071915 |
| t6145_pairs1_inflight2 | 0 | 2.114462 | 2.118241 | 2.120204 | 2.114462 | 2.118241 | 2.120204 |
| t6145_pairs1_inflight2 | 1 | 2.113461 | 2.117138 | 2.121188 | 2.113461 | 2.117138 | 2.121188 |
| t6145_pairs1_inflight2 | 2 | 2.112670 | 2.116356 | 2.117116 | 2.112670 | 2.116356 | 2.117116 |
| t6145_pairs1_inflight4 | 0 | 4.206654 | 4.211191 | 4.213206 | 4.206654 | 4.211191 | 4.213206 |
| t6145_pairs1_inflight4 | 1 | 4.205853 | 4.209835 | 4.222557 | 4.205853 | 4.209835 | 4.222557 |
| t6145_pairs1_inflight4 | 2 | 4.205031 | 4.207175 | 4.210509 | 4.205031 | 4.207175 | 4.210509 |
| t6145_pairs1_inflight8 | 0 | 8.395636 | 8.405227 | 8.408900 | 8.395636 | 8.405227 | 8.408900 |
| t6145_pairs1_inflight8 | 1 | 8.390838 | 8.398514 | 8.402385 | 8.390838 | 8.398514 | 8.402385 |
| t6145_pairs1_inflight8 | 2 | 8.390633 | 8.393127 | 8.395180 | 8.390633 | 8.393127 | 8.395180 |
| t6145_pairs1_inflight16 | 0 | 16.771363 | 16.785864 | 16.810648 | 16.771363 | 16.785864 | 16.810648 |
| t6145_pairs1_inflight16 | 1 | 16.760232 | 16.768161 | 20.084813 | 16.760232 | 16.768161 | 20.084813 |
| t6145_pairs1_inflight16 | 2 | 16.758271 | 16.763151 | 16.766943 | 16.758271 | 16.763151 | 16.766943 |
| t6145_pairs8_inflight1 | 0 | 1.075650 | 1.090683 | 1.095742 | 1.070501 | 1.075841 | 1.095742 |
| t6145_pairs8_inflight1 | 1 | 1.074819 | 1.233287 | 1.987644 | 1.067594 | 1.077365 | 1.987644 |
| t6145_pairs8_inflight1 | 2 | 1.072941 | 1.110662 | 15.452307 | 1.068298 | 1.072925 | 15.452307 |
| t6145_pairs8_inflight2 | 0 | 2.122747 | 2.193477 | 13.339035 | 2.114540 | 2.121035 | 13.339035 |
| t6145_pairs8_inflight2 | 1 | 2.120985 | 2.129454 | 2.349131 | 2.114635 | 2.120547 | 2.349131 |
| t6145_pairs8_inflight2 | 2 | 2.122186 | 2.203085 | 2.259277 | 2.115786 | 2.121530 | 2.259277 |
| t6145_pairs8_inflight4 | 0 | 4.216738 | 4.218899 | 4.220895 | 4.207714 | 4.215214 | 4.220895 |
| t6145_pairs8_inflight4 | 1 | 4.216209 | 4.225660 | 4.896764 | 4.209692 | 4.216157 | 4.896764 |
| t6145_pairs8_inflight4 | 2 | 4.218290 | 4.224334 | 4.238926 | 4.211012 | 4.218280 | 4.238926 |
| t6145_pairs8_inflight8 | 0 | 8.413067 | 8.429229 | 8.454278 | 8.402256 | 8.412745 | 8.454278 |
| t6145_pairs8_inflight8 | 1 | 8.411402 | 14.979299 | 16.659163 | 8.398274 | 8.411046 | 16.659163 |
| t6145_pairs8_inflight8 | 2 | 8.410454 | 8.422647 | 8.431204 | 8.399947 | 8.409824 | 8.431204 |
| t6145_pairs8_inflight16 | 0 | 16.797956 | 16.829431 | 21.550577 | 16.784908 | 16.798023 | 21.550577 |
| t6145_pairs8_inflight16 | 1 | 16.795909 | 22.143532 | 27.497038 | 16.765710 | 16.796593 | 27.497038 |
| t6145_pairs8_inflight16 | 2 | 16.793266 | 17.419014 | 20.254578 | 16.781839 | 16.793083 | 20.254578 |



**3. Replay concurrency and coverage.** [transfer_windows.csv](transfer_windows.csv) retains all 512 transfer windows from B29 and two-fix at both scales. [results.json](results.json), `replay_windows`, cites each absolute run root, receipt/source commit and input hash. Each transfer row has one request, a bound target lane and completed status. A half-open interval sweep across **all target DP lanes**, using the full `[start_ts_s,end_ts_s)` transfer window, yields a maximum of one logical request everywhere. A physical READ's in-flight time is a positive subinterval of that window, so its maximum concurrency is also one. This is stronger than counting only priced wire intervals and does not need a guessed posting time.

The physical-pair mapping follows [AGENTS.md](../../../../../AGENTS.md) “vLLM Parallel Semantics and Frontier Mapping”: a Replica owns the full pod; a DP lane owns a request; TP is implicit. [nixl_kv_read_benchmark.py](../../../../../frontier/profiling/collectives/nixl_kv_read_benchmark.py), module documentation and `KVReadPeer`, says same-TP P/D ranks form one pair per TP rank. Thus **one logical request causes eight physical READs, one on each of eight pairs**. EP16/EP32 and DP2/DP4 do not mean 16/32 pairs actually read at once. The larger available domains do not invalidate coverage when no other request's window overlaps. The benchmark's n=16 is not the configured `max_forwards_in_flight=19`; forward queue depth is a different count.


| replay | DP / TP / EP | requests by DP lane | peak requests per pod / per pair | pairs reading at once | longest ledger window ms | min gap between ledger windows ms |
| --- | --- | --- | --- | --- | --- | --- |
| p2_32_b29 | 2/8/16 | {'0': 64, '1': 64} | 1 / 1 | 8 | 79.587854 | 920.346099 |
| p2_32_two_fix | 2/8/16 | {'0': 64, '1': 64} | 1 / 1 | 8 | 79.587854 | 920.346099 |
| p2_64_b29 | 4/8/32 | {'0': 32, '1': 32, '2': 32, '3': 32} | 1 / 1 | 8 | 100.898908 | 433.376783 |
| p2_64_two_fix | 4/8/32 | {'0': 32, '1': 32, '2': 32, '3': 32} | 1 / 1 | 8 | 100.898908 | 433.376783 |



**Coverage answer: YES for the retained Frontier replay READ concurrency at both scales**, at the exact measured point `t6144_pairs8_inflight1`. This is a source-mapping inference supported by all 128 full windows per replay, not a native per-rank in-flight observer result. The probe does not reproduce overlapping compute/collective traffic or observe native request bytes. Those are limitations of transferring standalone timing to serving, even though the READ-count grid covers these replays. The qualification restriction and separate payload OPEN status must remain visible.

**4. Candidate definitions, proposed rules and actual results.** All rules below are **proposals**. No NIXL threshold has been adopted. Timing definitions/fit targets were fixed from the standalone data before the event-window scan; no E2E metric outcome entered fitting or selection. The analysis reads arrival rows and scheduling/transfer times for coverage/proof, not E2E values. The machine-readable registry is `candidate_definitions` in [proposed_input.json](proposed_input.json).

**A — empirical bounded timing.** Use every measured completion at the admitted exact byte/pair/inflight coordinate as a positive sample, without division by n. Proposed test: each full-batch maximum must be strictly below the earliest later step start for every possible post in every P2 transfer window. At the **actual** P2 point, every trial in all three passes passes both scales; maximum 12.554511 ms leaves 6.170454 / 11.825565 ms to the respective minimum gaps. Even all n=1 controls (three prompts, both pair counts) pass: maximum 15.452307 ms. This proves the finite observed samples are compatible with the no-motion interval. It does **not** prove a hard upper bound on future serving tails. If the domain is the full grid, the numerical test FAILs 11/90 case-passes at P2-32 and 4/90 at P2-64; the table below records every pass. In particular, T6144 pairs8 n=2 pass2 =19.939218 ms already violates the P2-32 gap, and T6144 pairs8 n=16 pass1 =24.891950 ms violates both. Do not treat the T6145 global maximum as the only counterexample.

Outside the admitted measured concurrency/bytes/hardware and these retained windows: **NOT_ESTABLISHED / fail qualification**, with no extrapolation. No Frontier interface change is needed for a case-metric independence exception, but A does not turn the H800 price into an H200 timing input. Cost: 0 new GPUh and a CPU event scan; a universal bound would need separately authorized serving-tail evidence and a statistical/physical guarantee rule. A alone is therefore not the recommended way to discharge ticket 19's own-hardware price requirement.

**B — exact concurrency-indexed median.** Key the point by `(bytes per request, active pairs, requests in flight)`; merge the equal-byte T6143/T6144 samples at the same coordinate, rather than making prompt number a timing feature. Use the pooled median of the full-batch critical-path trials at each of the **20 unique byte/pair/inflight coordinates**. For P2 it is 

**0.971499830484 ms** (the raw T6144-only median is 0.971056521 ms). Proposed validation: leave each pass out, build each point from the other two passes at the same coordinate, compare separately to every original held-out prompt/case/pass median, and require <=5% error. Result: **90/90 PASS**, maximum **0.808380%**. This is repeatability at retained coordinates, not unseen-coordinate accuracy. Every trial remains in the point median and the tail table.

For the separately proposed stronger **trial** rule, compute `abs(prediction-measured)/measured` per critical-path trial on that held-out pass; require median<=5% and p90<=10% on each case/pass. Result: **83/90 PASS, 7 FAIL**; the exact failures and residuals are below. Median fidelity is not a tail claim. Outside the 20 coordinates: reject; no interpolation. At n>1 the point is a wait-all batch time and cannot be assigned to every request as an individually observed service time. A general Frontier implementation needs new concurrency/cohort context and table admission, or a separately designed way to apply a batch-completion estimate; no such source/interface change is implemented. A single exact-point-only configuration can be encoded by an affine pair matching that byte coordinate, but its decomposition is underdetermined and has no independent affine meaning. This is why B is not recommended over the physical fit already available. Cost: 0 new GPUh; separate interface/source approval for generalized use, plus fresh CPU replay only if implementation/binding can change events (about 2.7 h + 3.3 h on the stated environment).

**C — positive physical affine batch model, recommended for P2 n=1 only.** Fit equal-weight OLS of **all 90 case-pass full-batch medians**, in ms, on total bytes per reader (decimal MB):

`batch_complete_ms = 0.023652184281886 + 0.020492144502155 × (n × bytes_per_request / 1,000,000)`.

The intercept is **23.652184282 µs**, strictly positive without clamping; slope and bandwidth are positive. This physical form has one startup term and a data term proportional to the total moved bytes. At n=1 it is exactly the existing analytical configuration. At n>1 it describes full-batch completion; a per-request caller must not use per-request bytes and claim it models concurrent batch tails. The fit ignores pair count in the median; independent all-case/pass residuals test whether that simplification is supported. All three held-out fits are also physical:


| held-out pass | training medians | intercept µs | bandwidth Gbit/s |
| --- | --- | --- | --- |
| 0 | 60 | 23.598438490 | 390.448619150 |
| 1 | 60 | 23.154926099 | 390.353986037 |
| 2 | 60 | 24.203188256 | 390.377905065 |



Proposed rule C: (i) every two-pass training fit has positive finite intercept/slope; (ii) each of the 90 held-out case/pass medians has <=5% absolute relative error; (iii) every retained P2-coordinate timing trial passes the empirical window test, with a separate deterministic no-motion proof for the proposed configuration. Results: **all physical fits PASS; 90/90 median checks PASS**, maximum **0.919518%**; in-sample median maximum **0.916528%**; every P2 trial/window PASS. Rule (iii) is finite evidence, not a probability bound. Accepting C requires YC to choose median-input qualification for these P2 cases. The stronger trial median5/p90 10% rule instead gives **82/90 PASS, 8 FAIL**, maximum p90 residual **43.899069%**; do not label it a qualified p90 latency model.

For every candidate, the per-case/pass result follows. The median-error columns use **held-out pass predictions**; trial p90 is p90 of absolute relative errors, not relative error of the p90 time. Bound columns compare the full-batch maximum to the scale minimum gap, even for coordinates the P2 replay never uses. Numeric columns retain failures instead of censoring them. [case_pass_results.csv](case_pass_results.csv) additionally preserves predictions, median trial errors and all Boolean verdicts at full precision.


| case | pass | A32 / A64 | B median err % | B trial p90 err % | B trial rule | C median err % | C trial p90 err % | C trial rule |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| t6143_pairs1_inflight1 | 0 | PASS/PASS | 0.2493 | 0.8345 | PASS | 0.2346 | 0.5292 | PASS |
| t6143_pairs1_inflight1 | 1 | PASS/PASS | 0.0371 | 0.2498 | PASS | 0.4526 | 0.5525 | PASS |
| t6143_pairs1_inflight1 | 2 | PASS/PASS | 0.1226 | 0.2395 | PASS | 0.5905 | 0.7013 | PASS |
| t6143_pairs1_inflight2 | 0 | PASS/PASS | 0.2417 | 0.3936 | PASS | 0.0235 | 0.2014 | PASS |
| t6143_pairs1_inflight2 | 1 | PASS/PASS | 0.0817 | 0.2555 | PASS | 0.1755 | 0.3494 | PASS |
| t6143_pairs1_inflight2 | 2 | PASS/PASS | 0.2539 | 0.3416 | PASS | 0.3772 | 0.4651 | PASS |
| t6143_pairs1_inflight4 | 0 | PASS/PASS | 0.0702 | 0.1879 | PASS | 0.1031 | 0.1929 | PASS |
| t6143_pairs1_inflight4 | 1 | PASS/PASS | 0.0373 | 0.0929 | PASS | 0.2164 | 0.2636 | PASS |
| t6143_pairs1_inflight4 | 2 | PASS/PASS | 0.0127 | 6.8362 | PASS | 0.1983 | 6.6396 | PASS |
| t6143_pairs1_inflight8 | 0 | PASS/PASS | 0.1407 | 0.3821 | PASS | 0.0220 | 0.2637 | PASS |
| t6143_pairs1_inflight8 | 1 | PASS/PASS | 0.0252 | 0.0702 | PASS | 0.1166 | 0.1541 | PASS |
| t6143_pairs1_inflight8 | 2 | PASS/PASS | 0.0750 | 0.1076 | PASS | 0.1504 | 0.1789 | PASS |
| t6143_pairs1_inflight16 | 0 | PASS/PASS | 0.0364 | 0.2431 | PASS | 0.0600 | 0.1469 | PASS |
| t6143_pairs1_inflight16 | 1 | PASS/PASS | 0.0015 | 0.1685 | PASS | 0.1053 | 0.1277 | PASS |
| t6143_pairs1_inflight16 | 2 | PASS/PASS | 0.0398 | 0.0501 | PASS | 0.1320 | 0.1424 | PASS |
| t6143_pairs8_inflight1 | 0 | PASS/PASS | 0.4592 | 17.6212 | FAIL | 0.7421 | 17.8553 | FAIL |
| t6143_pairs8_inflight1 | 1 | PASS/PASS | 0.0535 | 5.6088 | PASS | 0.4134 | 5.9488 | PASS |
| t6143_pairs8_inflight1 | 2 | PASS/PASS | 0.2788 | 0.6395 | PASS | 0.1159 | 0.9265 | PASS |
| t6143_pairs8_inflight2 | 0 | PASS/PASS | 0.0559 | 0.4669 | PASS | 0.2741 | 0.6841 | PASS |
| t6143_pairs8_inflight2 | 1 | PASS/PASS | 0.1276 | 0.7314 | PASS | 0.0683 | 0.9256 | PASS |
| t6143_pairs8_inflight2 | 2 | PASS/PASS | 0.0148 | 1.7072 | PASS | 0.1236 | 1.8432 | PASS |
| t6143_pairs8_inflight4 | 0 | PASS/PASS | 0.0929 | 0.4295 | PASS | 0.0571 | 0.5787 | PASS |
| t6143_pairs8_inflight4 | 1 | PASS/PASS | 0.8084 | 10.4268 | FAIL | 0.9195 | 10.5272 | FAIL |
| t6143_pairs8_inflight4 | 2 | PASS/PASS | 0.0173 | 0.3111 | PASS | 0.1160 | 0.4095 | PASS |
| t6143_pairs8_inflight8 | 0 | PASS/PASS | 0.0132 | 0.3056 | PASS | 0.1361 | 0.4281 | PASS |
| t6143_pairs8_inflight8 | 1 | FAIL/PASS | 0.0086 | 4.5795 | PASS | 0.1033 | 4.6863 | PASS |
| t6143_pairs8_inflight8 | 2 | PASS/PASS | 0.0194 | 0.8560 | PASS | 0.0895 | 0.9640 | PASS |
| t6143_pairs8_inflight16 | 0 | FAIL/FAIL | 0.0216 | 0.3032 | PASS | 0.1530 | 0.4342 | PASS |
| t6143_pairs8_inflight16 | 1 | FAIL/PASS | 0.0530 | 5.4219 | PASS | 0.0776 | 5.5453 | PASS |
| t6143_pairs8_inflight16 | 2 | FAIL/FAIL | 0.0043 | 2.1722 | PASS | 0.1172 | 2.2910 | PASS |
| t6144_pairs1_inflight1 | 0 | PASS/PASS | 0.2459 | 14.2949 | FAIL | 0.2381 | 13.8791 | FAIL |
| t6144_pairs1_inflight1 | 1 | PASS/PASS | 0.0467 | 0.6538 | PASS | 0.3684 | 0.6577 | PASS |
| t6144_pairs1_inflight1 | 2 | PASS/PASS | 0.1338 | 0.2461 | PASS | 0.6018 | 0.7103 | PASS |
| t6144_pairs1_inflight2 | 0 | PASS/PASS | 0.3361 | 0.5649 | PASS | 0.0711 | 0.3006 | PASS |
| t6144_pairs1_inflight2 | 1 | PASS/PASS | 0.1064 | 0.2690 | PASS | 0.2001 | 0.3627 | PASS |
| t6144_pairs1_inflight2 | 2 | PASS/PASS | 0.1441 | 0.3061 | PASS | 0.2673 | 0.3907 | PASS |
| t6144_pairs1_inflight4 | 0 | PASS/PASS | 0.0262 | 0.0691 | PASS | 0.1996 | 0.2426 | PASS |
| t6144_pairs1_inflight4 | 1 | PASS/PASS | 0.0657 | 0.1987 | PASS | 0.1133 | 0.2405 | PASS |
| t6144_pairs1_inflight4 | 2 | PASS/PASS | 0.0088 | 0.0996 | PASS | 0.2023 | 0.2510 | PASS |
| t6144_pairs1_inflight8 | 0 | PASS/PASS | 0.2025 | 0.4028 | PASS | 0.0839 | 0.2844 | PASS |
| t6144_pairs1_inflight8 | 1 | PASS/PASS | 0.0287 | 0.0608 | PASS | 0.1200 | 0.1522 | PASS |
| t6144_pairs1_inflight8 | 2 | PASS/PASS | 0.0893 | 0.1101 | PASS | 0.1647 | 0.1855 | PASS |
| t6144_pairs1_inflight16 | 0 | PASS/PASS | 0.0411 | 0.1895 | PASS | 0.0553 | 0.0932 | PASS |
| t6144_pairs1_inflight16 | 1 | PASS/PASS | 0.0064 | 0.1533 | PASS | 0.1101 | 0.1397 | PASS |
| t6144_pairs1_inflight16 | 2 | PASS/PASS | 0.0294 | 0.0496 | PASS | 0.1216 | 0.1419 | PASS |
| t6144_pairs8_inflight1 | 0 | PASS/PASS | 0.0656 | 2.1856 | PASS | 0.3496 | 2.4636 | PASS |
| t6144_pairs8_inflight1 | 1 | PASS/PASS | 0.0600 | 0.3985 | PASS | 0.3003 | 0.6387 | PASS |
| t6144_pairs8_inflight1 | 2 | PASS/PASS | 0.2011 | 0.5326 | PASS | 0.1934 | 0.6358 | PASS |
| t6144_pairs8_inflight2 | 0 | PASS/PASS | 0.1270 | 0.2791 | PASS | 0.0916 | 0.4581 | PASS |
| t6144_pairs8_inflight2 | 1 | PASS/PASS | 0.0306 | 0.7059 | PASS | 0.2262 | 0.9002 | PASS |
| t6144_pairs8_inflight2 | 2 | FAIL/PASS | 0.0635 | 1.2479 | PASS | 0.2018 | 1.3846 | PASS |
| t6144_pairs8_inflight4 | 0 | PASS/PASS | 0.0426 | 9.9400 | PASS | 0.1923 | 10.0749 | FAIL |
| t6144_pairs8_inflight4 | 1 | PASS/PASS | 0.0280 | 0.3368 | PASS | 0.0841 | 0.4484 | PASS |
| t6144_pairs8_inflight4 | 2 | PASS/PASS | 0.0229 | 1.5094 | PASS | 0.0758 | 1.6066 | PASS |
| t6144_pairs8_inflight8 | 0 | PASS/PASS | 0.0133 | 0.2566 | PASS | 0.1362 | 0.3792 | PASS |
| t6144_pairs8_inflight8 | 1 | PASS/PASS | 0.0049 | 0.1284 | PASS | 0.1169 | 0.2402 | PASS |
| t6144_pairs8_inflight8 | 2 | PASS/PASS | 0.0120 | 0.0921 | PASS | 0.0969 | 0.2009 | PASS |
| t6144_pairs8_inflight16 | 0 | FAIL/PASS | 0.0282 | 8.2115 | PASS | 0.1596 | 8.3321 | PASS |
| t6144_pairs8_inflight16 | 1 | FAIL/FAIL | 0.0594 | 16.0000 | FAIL | 0.0712 | 16.1097 | FAIL |
| t6144_pairs8_inflight16 | 2 | PASS/PASS | 0.0071 | 0.1257 | PASS | 0.1144 | 0.2471 | PASS |
| t6145_pairs1_inflight1 | 0 | PASS/PASS | 0.1016 | 0.2555 | PASS | 0.5393 | 0.6887 | PASS |
| t6145_pairs1_inflight1 | 1 | PASS/PASS | 0.1320 | 0.3498 | PASS | 0.3524 | 0.6060 | PASS |
| t6145_pairs1_inflight1 | 2 | PASS/PASS | 0.0816 | 0.3608 | PASS | 0.5158 | 0.6579 | PASS |
| t6145_pairs1_inflight2 | 0 | PASS/PASS | 0.0707 | 0.2490 | PASS | 0.1760 | 0.2835 | PASS |
| t6145_pairs1_inflight2 | 1 | PASS/PASS | 0.0062 | 0.1675 | PASS | 0.2265 | 0.3558 | PASS |
| t6145_pairs1_inflight2 | 2 | PASS/PASS | 0.0543 | 0.1558 | PASS | 0.3076 | 0.3976 | PASS |
| t6145_pairs1_inflight4 | 0 | PASS/PASS | 0.0295 | 0.1372 | PASS | 0.1454 | 0.1975 | PASS |
| t6145_pairs1_inflight4 | 1 | PASS/PASS | 0.0055 | 0.1001 | PASS | 0.1780 | 0.2271 | PASS |
| t6145_pairs1_inflight4 | 2 | PASS/PASS | 0.0287 | 0.0886 | PASS | 0.2164 | 0.2647 | PASS |
| t6145_pairs1_inflight8 | 0 | PASS/PASS | 0.0594 | 0.1735 | PASS | 0.0751 | 0.1471 | PASS |
| t6145_pairs1_inflight8 | 1 | PASS/PASS | 0.0116 | 0.0798 | PASS | 0.1512 | 0.1701 | PASS |
| t6145_pairs1_inflight8 | 2 | PASS/PASS | 0.0234 | 0.0382 | PASS | 0.1600 | 0.1749 | PASS |
| t6145_pairs1_inflight16 | 0 | PASS/PASS | 0.0717 | 0.1580 | PASS | 0.0532 | 0.1088 | PASS |
| t6145_pairs1_inflight16 | 1 | FAIL/PASS | 0.0157 | 0.0386 | PASS | 0.1412 | 0.1614 | PASS |
| t6145_pairs1_inflight16 | 2 | PASS/PASS | 0.0330 | 0.0456 | PASS | 0.1530 | 0.1656 | PASS |
| t6145_pairs8_inflight1 | 0 | PASS/PASS | 0.1976 | 1.5732 | PASS | 0.4424 | 1.8147 | PASS |
| t6145_pairs8_inflight1 | 1 | PASS/PASS | 0.0026 | 12.8515 | FAIL | 0.3831 | 13.1831 | FAIL |
| t6145_pairs8_inflight1 | 2 | PASS/PASS | 0.2363 | 3.1680 | PASS | 0.1171 | 3.5093 | PASS |
| t6145_pairs8_inflight2 | 0 | PASS/PASS | 0.0562 | 3.2790 | PASS | 0.2150 | 3.4326 | PASS |
| t6145_pairs8_inflight2 | 1 | PASS/PASS | 0.0711 | 0.3269 | PASS | 0.1290 | 0.5263 | PASS |
| t6145_pairs8_inflight2 | 2 | PASS/PASS | 0.0333 | 3.7041 | PASS | 0.1422 | 3.8091 | PASS |
| t6145_pairs8_inflight4 | 0 | PASS/PASS | 0.0142 | 0.1462 | PASS | 0.0941 | 0.1453 | PASS |
| t6145_pairs8_inflight4 | 1 | PASS/PASS | 0.0201 | 0.2036 | PASS | 0.0680 | 0.2915 | PASS |
| t6145_pairs8_inflight4 | 2 | PASS/PASS | 0.0432 | 0.1862 | PASS | 0.0986 | 0.2415 | PASS |
| t6145_pairs8_inflight8 | 0 | PASS/PASS | 0.0268 | 0.2185 | PASS | 0.1323 | 0.3238 | PASS |
| t6145_pairs8_inflight8 | 1 | PASS/PASS | 0.0020 | 43.8476 | FAIL | 0.0936 | 43.8991 | FAIL |
| t6145_pairs8_inflight8 | 2 | PASS/PASS | 0.0227 | 0.1221 | PASS | 0.0760 | 0.2207 | PASS |
| t6145_pairs8_inflight16 | 0 | FAIL/PASS | 0.0230 | 0.2100 | PASS | 0.1052 | 0.2921 | PASS |
| t6145_pairs8_inflight16 | 1 | FAIL/FAIL | 0.0049 | 24.1536 | FAIL | 0.0715 | 24.2041 | FAIL |
| t6145_pairs8_inflight16 | 2 | FAIL/PASS | 0.0245 | 3.5687 | PASS | 0.0557 | 3.6460 | PASS |



C outside-domain policy: **fail qualification** for any other bytes, simultaneous requests, pair count, source/layout, hardware/fabric, compression setting or trace/window generation. Do not advertise general interpolation or a runtime rejection already implemented in Frontier. The proposed external census restricts this adoption; the current bytes-only predictor itself extrapolates if a caller uses it elsewhere. General n>1/concurrency admission needs an approved interface/source change. Under the adopted restriction, C needs **no Frontier interface change**, no table edits, 0 new GPUh and no fresh replay under the proof below. A future wider-domain requirement needs new design/measurements, not the original H800 substitution.

**Original NIXL02 FAIL preserved:** the earlier integrity-valid H200 run lacks T6144 and has an unconstrained intercept **−46.6769198402 µs**, bandwidth **305.707821 Gbit/s**, and p90 relative residual **29.175907%**. It remains rejected in [fit_diagnosis.json](../../b30b/f3_nixl02_20261101/fit_diagnosis.json) and [earlier report](../../../test_report_2026-11-01_f3_h200_nixl.md). The new positive fit does not reverse or replace that verdict; it qualifies a different source/data generation under a newly proposed rule. All full-grid tail FAILs above also remain visible. Existing native/all-request/attention/collective FAIL records were not rewritten.

**5. Event-window proof and conditional replay decision.** [possible_post_windows.csv](possible_post_windows.csv) enumerates **2,034** possible post-start windows (510+510+507+507) across all four retained traces. For each of the 128 transfers per trace, scan every DECODE `formed_s` from ledger start through ledger end and find the first strictly later start on that lane, even if it lies beyond the ledger end. This avoids claiming that the ledger directly observes READ posting. The candidate is tested against every possible posting step, a superset of the actual one.

Source: [NixlReadCompletion.on_engine_step_start](../../../../../frontier/scheduler/replica_scheduler/vllm_v1_nixl_reads.py), lines 39–68, checks posted wire ends first, then posts new READs; `_report_nixl_reads_of_batch`, `_report_nixl_reads_of_dummy_forward` and `consume_found_nixl_reads` report them with the owning output. [vllm_dp_busy_loop.py](../../../../../frontier/scheduler/replica_scheduler/vllm_dp_busy_loop.py), `on_engine_step_start`, `_schedule_iteration` and `_issue_dummy_pass` (lines 137–143, 210–217, 309–333), starts run-ahead connector steps at host formation. `next_pass_start` (line 355 onwards) can wake at a wire end only for a zero-duration idle-forward group. The retained traces contain **zero zero-duration DECODE forwards** (14,848 per P2-32 trace; 24,320 per P2-64 trace), so a shorter price cannot introduce that wake here.


| trace | possible starts | min next-start gap ms | old/current price ms | C price ms | P2 sample max ms | crossings |
| --- | --- | --- | --- | --- | --- | --- |
| p2_32_b29 | 510 | 18.724965326230 | 2.570035277338 | 0.968402341951 | 12.554511427879 | 0 |
| p2_32_two_fix | 510 | 18.724965326230 | 2.987238193344 | 0.968402341951 | 12.554511427879 | 0 |
| p2_64_b29 | 507 | 24.380075962759 | 2.570035277338 | 0.968402341951 | 12.554511427879 | 0 |
| p2_64_two_fix | 507 | 24.380075962759 | 2.987238193344 | 0.968402341951 | 12.554511427879 | 0 |



Both the previous H800 prices (2.570035277 ms on B29 structural bytes; 2.987238193 ms on two-fix wire bytes) and C's positive 0.968402342 ms lie strictly inside `(post, first later start)` for **every possible post**. B's point also lies inside. The posting step cannot find its own new READ, because checks precede posting and the price stays positive. The same later step therefore finds each READ, and its same output reports it. No ledger windows overlap; candidate prices finish well before another transfer can post. With deterministic DES scheduling and no idle zero-cost wake, induction preserves the next schedule and every later event. There are **zero crossings**. This proves READ release and downstream event/metric invariance for changing only this input on the retained configuration; it does not imply every metadata field stays equal (the configured transfer price and configuration fields change).

The two-fix window checker only compared 2.570035277 with 2.987238193. Q2 extends the proof to the shorter H200 candidate and every observed n=1 tail, with the explicit `(0,next step)` test. No new all-request or warm E2E metric comparison was calculated and no new replay was executed. Expected values would equal the corresponding B29/two-fix outputs by the event proof; this is an inference, **not a fabricated replay result**. A binding-only fresh replay remains Claude's optional formal process choice; if another interface/lifecycle/concurrency change is made, this proof no longer discharges it.

**6. Verification and exact commands.** Interpreter: `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6; cwd as above; `PYTHONPATH=$PWD:$PWD/tests/analysis`, bytecode disabled, OpenBLAS/OMP/MKL threads each 1, `TMPDIR` in the authorized `scratch/`. Every heavy inspection is streamed and bounded by `timeout 120s`; no full replay or new dependency is needed. Run from the working root:


```bash
env PYTHONPATH="$PWD:$PWD/tests/analysis" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$PWD/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/scratch" timeout 120s /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/analyze.py > task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/analysis.log 2>&1
env PYTHONPATH="$PWD:$PWD/tests/analysis" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$PWD/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/scratch" timeout 120s /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/checks.py > task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/checks.log 2>&1
env PYTHONPATH="$PWD:$PWD/tests/analysis" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$PWD/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/scratch" timeout 120s /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/verify.py
env PYTHONPATH="$PWD:$PWD/tests/analysis" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$PWD/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/scratch" timeout 120s /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2/render_report.py
```



`analyze.py`: expected no missing cases/duplicates, raw/CSV equality, physical fits, all four concurrency sweeps and every no-motion assertion; observed **PASS**, exit 0. See [analysis.log](analysis.log) and [results.json](results.json), `checks`, `environment`, `evidence_hashes`. The reported median/p90 rule failures are candidate outcomes, not hidden analysis-command failures.

`checks.py`: 12 focused checks of half-open occupancy, overlapping lanes/pairs, rejection of invalid windows and unphysical unclamped fits, exact p90 convention, all rows/trials, independently recomputed closed-form OLS for the full fit and three holdouts, all four window proofs, and unchanged retained-input hashes; expected all PASS; observed **12/12 PASS**, exit 0 ([checks.log](checks.log)). The checks use a different closed-form covariance calculation to verify the OLS coefficients. No tests that require GPU execution were added or run.

`verify.py` executed three commands, all exit 0: the retained lead checker independently rebuilt 23 hashes/90 rows/12,150 trials/80,352 notifications; `python -m pytest tests/unit/test_nixl_read_completion.py -q -p no:cacheprovider --basetemp <authorized scratch>/pytest` reported **9 passed in 0.79 s**; the production `AnalyticalKVCacheTransferConfig` and `AnalyticalKVCacheTransferPredictor` calculation returned **0.9684023419505275 ms**, exactly matching the proposed formula (difference 0.0 ms). All commands use the assigned interpreter and `timeout 120s`. Exact executable arguments, including the price program and full paths, expected values and observed stdout are in [verification.json](verification.json); see [lead_retained_check.log](lead_retained_check.log), [lifecycle_regression.log](lifecycle_regression.log), [production_price.log](production_price.log). Fresh P2 CPU replays: **NOT_RUN — no completion can move**. Platform/GPU/native-byte checks: **NOT_RUN in this task**; retained identity evidence was inspected. The initial host `python` lookup failed because it is not on PATH; all actual analysis/check commands used the assigned interpreter. A small exploratory inspection initially used the nonexistent top-level `cluster_config_decode` key; the saved config stores these fields under `cluster_config`. It was corrected by inspecting actual config keys before implementation; no input was edited. Initial analysis/check runs passed; their logs are preserved under `scratch/`.

**7. Changes, remaining decision and review handoff.** All files are new in `nixl_timing_q2/`: `analyze.py` rebuilds observations and candidates, sweeps concurrency and proves event windows; `checks.py` guards arithmetic and verifies the evidence; `render_report.py` renders this report; `verify.py` runs and records the retained lead check, nine existing lifecycle tests and production price comparison; `proposed_input.json` carries the precise optional input, domain and rule; `results.json` carries all case/pass results, run proofs, environment and input hashes; `case_pass_results.csv`, `repeatability.csv`, `equal_byte_passes.csv`, `reader_summary.csv`, `transfer_windows.csv`, `possible_post_windows.csv` preserve the small reviewable tables; `analysis.log`, `checks.log`, `lead_retained_check.log`, `lifecycle_regression.log`, `production_price.log`, and `verification.json` record actual checks; `report.md` is this report. `scratch/reader_trials.csv` is a full reader-trial copy and `scratch/*initial.log` retains exploratory successful runs; scratch is not intended for commit. No reusable `tests/analysis/` or `tests/unit/` file was needed.

Claude/YC can adopt C with **one choice**: approve the displayed pair and `AFFINE_MEDIAN_P2_SINGLE_READ` rule/domain, retain full-grid trial-rule FAILs and NIXL02 FAIL, keep native payload OPEN, and record the price change plus no-motion proof. The choice changes the formal timing input and its qualification rule; no formal replacement was made here. If YC requires a universal tail guarantee or the full-grid per-trial p90 rule instead, keep timing OPEN and authorize a specific serving-contention/tail investigation or a stochastic/cohort completion design with its own acceptance rule. It would be incorrect to adopt a deterministic median model under the failed trial rule. No additional permission was requested because the assigned proposal/CPU analysis was fully executable.

Review the difference between full-batch completion and individual request service, the 20 byte-keyed exact points, held-out-pass fit targets, strict positive intercept, the complete failure rows, TP/DP mapping and ledger envelopes, the zero-duration-forward guard and the domain policy in `proposed_input.json`. Rerun the four commands above and the listed existing lifecycle check if desired. Costs outside the fixed payload/concurrency, per-request n>1 completion times, full serving contention, a future-tail probability guarantee, and observed native request bytes remain limitations. Claude owns acceptance, adoption, any formal binding replay, records and commits.

