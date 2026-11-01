"""Render the self-contained Q2 decision report from the analyzed evidence."""
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / 'results.json').read_text())
C = D['candidates']
R = C['all_case_results']
W = D['replay_windows']
SPREAD = list(csv.DictReader((HERE / 'repeatability.csv').open()))
PAIRED = list(csv.DictReader((HERE / 'equal_byte_passes.csv').open()))
P = 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/nixl_timing_q2'
ROOT = str(HERE.parents[4])
MODEL = C['model']
PRICE = W[0]['proposed_price_ms']
POINT = W[0]['exact_point_ms']


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def pf(value):
    return 'PASS' if value else 'FAIL'


def maxval(key):
    return max(r[key] for r in R)


def command(script, args=''):
    return f'env PYTHONPATH="$PWD:$PWD/tests/analysis" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR="$PWD/{P}/scratch" timeout 120s /data/ycfeng/envs/frontier-py310/bin/python {P}/{script}{args}'


parts = [f'''H200 NIXL READ timing proposal for P2 (Q2)

**1. Assignment and outcome.** Analysis completed on 2026-11-02 Asia/Hong_Kong. Working root: `{ROOT}`. All new files are under `{P}/`; `scratch/` contains trial copies and initial logs. Outcome: **implemented as a proposal and evidence handoff; formal adoption remains with YC/Claude**. No Frontier source, profiling table, formal input, threshold, fold, reference, task record, receipt or retained measurement was changed. No Git command, commit, dependency installation, GPU, platform, cloud, deployment or messaging operation was performed. The pre-existing `.cortexkit/`, `.pi/`, and `remaining_measurements/` are outside this task and were not edited. This directory was absent before the assignment.

The controlling requirements are D-P2-INPUT-QUALIFICATION and D-P2-Q1-CHOICES (timing route B), D-WARM-TARGET, D-WARM-TWO-FIX and D-COLLECTIVE-EXACT-REPEATABILITY in [requirements.md](../../../requirements.md), together with tickets 05/09/11/19. The collective exact-point rule does **not** automatically authorize a NIXL rule. Ticket 19 forbids the current H800 substitution. The packed payload remains the committed selected-page count × 1,002,240 B; native request descriptor-byte observation remains a separate OPEN edge. The requested `20261101` directory and historical report dates are retained; the raw run spans 2026-11-01 UTC / 2026-11-02 Hong Kong time.

**Recommended single choice for YC: approve `AFFINE_MEDIAN_P2_SINGLE_READ` in [proposed_input.json](proposed_input.json)**, with its exact P2 domain and proposed median rule. Use `network_bandwidth_gbps={MODEL['bandwidth_gbps']:.12f}` and `network_latency_ms={MODEL['intercept_ms']:.15f}` on the existing analytical transfer predictor. This is an H200 fit of all 90 full-batch medians, with a positive intercept; it prices one 46,103,040 B READ at **{PRICE:.12f} ms**. Every held-out case/pass median meets the proposed 5% rule (maximum **{maxval('affine_holdout_median_error_pct'):.6f}%**). Restrict adoption to the retained P2 payload, TP8, one READ per pair and eight active pairs, unchanged trace and `engine_step`. A bytes-only configuration cannot enforce these restrictions itself; the external qualification census must reject other workloads. This choice qualifies a **median timing input**, not a universal tail model or full serving-fabric fidelity.

The proposed fit and exact point cannot move any READ release in either scale. This was proved over all four B29/two-fix event traces, so the conditional fresh CPU replays were **NOT_RUN**. The empirical samples at the actual coordinate also lie below every relevant boundary. A universal guarantee about future native serving tails is **NOT_ESTABLISHED**. If YC instead requires the proposed per-trial median 5% / p90 10% fidelity rule on the full grid, reject both deterministic median candidates: the affine rule fails 8/90 case-passes and the point rule fails 7/90. None of those failures was removed.

**2. Measured behavior and timing unit.** Inputs: [package README](../packages/nixl6144/README.md), stage/config/plan; [run verification](../nixl6144_run/verification.json), [case summary](../nixl6144_run/case_summary.csv), [pass spread](../nixl6144_run/case_pass_spread.csv), [equal-byte summary](../nixl6144_run/paired_6143_6144.csv), [run notes](../nixl6144_run/observation_notes.md), [terminal record](../nixl6144_run/platform_terminal.json), and [lead report](../../../test_report_2026-11-01_nixl6144_h200.md). Raw local input: `/data/ycfeng/tmp/glm53_dsv4/nixl6144_q1_20261102/download/C_nixl/nixl_reads.jsonl`; SHA256 `{D['integrity']['raw_sha256']}`.

Rebuilt integrity: **23/23 downloaded file hashes and sizes match; 30 cases × passes 0/1/2 = 90 unique records; 12,150 reader trials; 2,700 critical-path repeat trials; 80,352/80,352 notifications; all reader data checks true and completion times positive; 90 producer CSV rows exactly reconstructed.** The existing 11/11 verification PASS was inspected and its matrix/geometry/trial/hash/CSV arithmetic was independently rechecked. Stage hardware identity and terminal status are retained records; no platform query was rerun. The source verifier is bound by hashes in [results.json](results.json). The measurement used two H200 nodes (`gpu-h200-0457`, `gpu-h200-0604`), eight TP P/D pairs and four UCX threads. Direct container exit codes remain unavailable in the retained report (HTTP 403); phase exits and replica success are recorded. Historical container cost was about 0.346 GPUh; this task adds 0 GPUh.

`complete_us` is measured from before descriptor preparation until **all requests in that reader's batch** are DONE, including preparation and posting. For each timed repetition, critical path is the maximum across active readers. `summarize_case` uses the median and sorted index `floor(0.9*N)` for p90 (index 27 of 30). Reader columns below summarize all reader-batch trials within that case/pass; [reader_summary.csv](reader_summary.csv) preserves all 405 individual reader/pass summaries and [scratch/reader_trials.csv](scratch/reader_trials.csv) all 12,150 trials with explicit millisecond column names. At inflight >1 the producer does **not** record each request's DONE time, nor an actual cross-node shared posting clock. The full-batch critical path is the slowest reader's elapsed completion, as defined by the producer. Do not divide a full-batch maximum by inflight and call it a per-request upper bound. These data cannot supply individual-request completion distributions at inflight >1.

At T6144, n=1, the pooled critical-path median is **0.964047 ms** for one pair and **0.971057 ms** for eight pairs, versus **2.987238193 ms** from the H800 pair at the committed wire bytes. Eight pairs scarcely increase median completion; concurrency mainly changes the total bytes per reader. The table below pools the critical-path trials of all three passes for each exact case; it does not select a pass.
''']

pooled_rows = []
for t in (6143, 6144, 6145):
    for n in (1, 2, 4, 8, 16):
        rows = [next(r for r in R if r['prompt_tokens'] == t and r['pairs'] == p and r['inflight'] == n) for p in (1, 8)]
        ss = [next(s for s in SPREAD if s['case'] == r['case']) for r in rows]
        m1, m8 = [float(s['pooled_median_ms']) for s in ss]
        pooled_rows.append([t, n, f"{rows[0]['bytes_per_reader']:,}", f'{m1:.6f}', f'{m8:.6f}', f'{(m8/m1-1)*100:.4f}%'])
parts.append(table(['prompt', 'inflight per pair', 'bytes/reader batch', '1 pair median ms', '8 pairs median ms', '8-pair increase'], pooled_rows))
parts.append(f'''

Same-case pass medians have median spread **{D['integrity']['median_pass_spread_pct']:.6f}%**, maximum **{D['integrity']['max_pass_spread_pct']:.6f}%** (`t6143_pairs8_inflight4`). Spread means `(max pass median - min pass median) / median of pass medians`. Every case's three values and deviation from its pooled median are listed below and in [repeatability.csv](repeatability.csv). This is repeatability on one source/hardware generation, not independence of fabric conditions or unseen geometry.
''')
parts.append(table(['case', 'pass 0 median ms', 'pass 1 median ms', 'pass 2 median ms', 'spread %'],
                   [[s['case'], *[f"{float(s[f'pass{p}_median_ms']):.6f}" for p in (0, 1, 2)], f"{float(s['spread_pct']):.6f}"] for s in SPREAD]))
parts.append(f'''

T6143 and T6144 select the same 46 pages / 46,103,040 B; T6145 selects 51 / 51,114,240 B (byte ratio **{51/46:.9f}**). Equal-byte pooled medians agree as in the retained summary (-0.0014 to +0.0022 ms). The stronger matched-pass comparison below retains all 30 pairs of case/pass medians: maximum difference **{D['integrity']['max_equal_byte_pass_difference_pct']:.6f}%**. The largest difference is the slow T6143 eight-pair n=4 pass, not a discarded trial. T6145 batch medians scale close to the 51/46 byte ratio, with a small fixed preparation/posting term. [equal_byte_passes.csv](equal_byte_passes.csv) also retains both p90 and maximum values; agreement of medians does **not** mean equality of tails.
''')
parts.append(table(['pairs', 'inflight', 'pass', '6143 median ms', '6144 median ms', 'difference %'],
                   [[s['pairs'], s['inflight'], s['pass_index'], f"{float(s['median_6143_ms']):.6f}", f"{float(s['median_6144_ms']):.6f}", f"{float(s['median_difference_pct']):.6f}"] for s in PAIRED]))
parts.append('''

All measured case/passes and their tails follow. C = critical path across readers; R = all reader-batch trials pooled within that pass. Every maximum is retained. Twelve case-passes have a trial above 3× the case median. The largest trial is **27.497038 ms** at `t6145_pairs8_inflight16`, pass 1. The heaviest p90 is **14.979299 ms** against median **8.411402 ms** at `t6145_pairs8_inflight8`, pass 1. At the actual P2 coordinate, `t6144_pairs8_inflight1`, pass maxima are **3.958579 / 12.554511 / 0.980675 ms**. Slow single-pair trials exist too: T6144 n=1 pass 1 reaches **12.938846 ms**. Tails are mainly an eight-pair phenomenon, not exclusively one. No UCX, fabric, host scheduling or warmup cause is established here; the timed slow trials remain in the population.
''')
parts.append(table(['case', 'pass', 'C median ms', 'C p90 ms', 'C max ms', 'R median ms', 'R p90 ms', 'R max ms'],
                   [[r['case'], r['pass_index'], *[f"{r[k]:.6f}" for k in ['critical_median_ms','critical_p90_ms','critical_max_ms','reader_median_ms','reader_p90_ms','reader_max_ms']]] for r in R]))
parts.append('''

**3. Replay concurrency and coverage.** [transfer_windows.csv](transfer_windows.csv) retains all 512 transfer windows from B29 and two-fix at both scales. [results.json](results.json), `replay_windows`, cites each absolute run root, receipt/source commit and input hash. Each transfer row has one request, a bound target lane and completed status. A half-open interval sweep across **all target DP lanes**, using the full `[start_ts_s,end_ts_s)` transfer window, yields a maximum of one logical request everywhere. A physical READ's in-flight time is a positive subinterval of that window, so its maximum concurrency is also one. This is stronger than counting only priced wire intervals and does not need a guessed posting time.

The physical-pair mapping follows [AGENTS.md](../../../../../AGENTS.md) “vLLM Parallel Semantics and Frontier Mapping”: a Replica owns the full pod; a DP lane owns a request; TP is implicit. [nixl_kv_read_benchmark.py](../../../../../frontier/profiling/collectives/nixl_kv_read_benchmark.py), module documentation and `KVReadPeer`, says same-TP P/D ranks form one pair per TP rank. Thus **one logical request causes eight physical READs, one on each of eight pairs**. EP16/EP32 and DP2/DP4 do not mean 16/32 pairs actually read at once. The larger available domains do not invalidate coverage when no other request's window overlaps. The benchmark's n=16 is not the configured `max_forwards_in_flight=19`; forward queue depth is a different count.
''')
parts.append(table(['replay', 'DP / TP / EP', 'requests by DP lane', 'peak requests per pod / per pair', 'pairs reading at once', 'longest ledger window ms', 'min gap between ledger windows ms'],
                   [[w['name'], f"{w['dp']}/{w['tp']}/{w['ep']}", str(w['transfers_per_lane']), '1 / 1', 8, f"{w['max_transfer_window_ms']:.6f}", f"{w['min_separation_between_transfer_windows_ms']:.6f}"] for w in W]))
parts.append('''

**Coverage answer: YES for the retained Frontier replay READ concurrency at both scales**, at the exact measured point `t6144_pairs8_inflight1`. This is a source-mapping inference supported by all 128 full windows per replay, not a native per-rank in-flight observer result. The probe does not reproduce overlapping compute/collective traffic or observe native request bytes. Those are limitations of transferring standalone timing to serving, even though the READ-count grid covers these replays. The qualification restriction and separate payload OPEN status must remain visible.

**4. Candidate definitions, proposed rules and actual results.** All rules below are **proposals**. No NIXL threshold has been adopted. Timing definitions/fit targets were fixed from the standalone data before the event-window scan; no E2E metric outcome entered fitting or selection. The analysis reads arrival rows and scheduling/transfer times for coverage/proof, not E2E values. The machine-readable registry is `candidate_definitions` in [proposed_input.json](proposed_input.json).

**A — empirical bounded timing.** Use every measured completion at the admitted exact byte/pair/inflight coordinate as a positive sample, without division by n. Proposed test: each full-batch maximum must be strictly below the earliest later step start for every possible post in every P2 transfer window. At the **actual** P2 point, every trial in all three passes passes both scales; maximum 12.554511 ms leaves 6.170454 / 11.825565 ms to the respective minimum gaps. Even all n=1 controls (three prompts, both pair counts) pass: maximum 15.452307 ms. This proves the finite observed samples are compatible with the no-motion interval. It does **not** prove a hard upper bound on future serving tails. If the domain is the full grid, the numerical test FAILs 11/90 case-passes at P2-32 and 4/90 at P2-64; the table below records every pass. In particular, T6144 pairs8 n=2 pass2 =19.939218 ms already violates the P2-32 gap, and T6144 pairs8 n=16 pass1 =24.891950 ms violates both. Do not treat the T6145 global maximum as the only counterexample.

Outside the admitted measured concurrency/bytes/hardware and these retained windows: **NOT_ESTABLISHED / fail qualification**, with no extrapolation. No Frontier interface change is needed for a case-metric independence exception, but A does not turn the H800 price into an H200 timing input. Cost: 0 new GPUh and a CPU event scan; a universal bound would need separately authorized serving-tail evidence and a statistical/physical guarantee rule. A alone is therefore not the recommended way to discharge ticket 19's own-hardware price requirement.

**B — exact concurrency-indexed median.** Key the point by `(bytes per request, active pairs, requests in flight)`; merge the equal-byte T6143/T6144 samples at the same coordinate, rather than making prompt number a timing feature. Use the pooled median of the full-batch critical-path trials at each of the **20 unique byte/pair/inflight coordinates**. For P2 it is ''')
parts.append(f'''**{POINT:.12f} ms** (the raw T6144-only median is 0.971056521 ms). Proposed validation: leave each pass out, build each point from the other two passes at the same coordinate, compare separately to every original held-out prompt/case/pass median, and require <=5% error. Result: **90/90 PASS**, maximum **{maxval('point_holdout_median_error_pct'):.6f}%**. This is repeatability at retained coordinates, not unseen-coordinate accuracy. Every trial remains in the point median and the tail table.

For the separately proposed stronger **trial** rule, compute `abs(prediction-measured)/measured` per critical-path trial on that held-out pass; require median<=5% and p90<=10% on each case/pass. Result: **83/90 PASS, 7 FAIL**; the exact failures and residuals are below. Median fidelity is not a tail claim. Outside the 20 coordinates: reject; no interpolation. At n>1 the point is a wait-all batch time and cannot be assigned to every request as an individually observed service time. A general Frontier implementation needs new concurrency/cohort context and table admission, or a separately designed way to apply a batch-completion estimate; no such source/interface change is implemented. A single exact-point-only configuration can be encoded by an affine pair matching that byte coordinate, but its decomposition is underdetermined and has no independent affine meaning. This is why B is not recommended over the physical fit already available. Cost: 0 new GPUh; separate interface/source approval for generalized use, plus fresh CPU replay only if implementation/binding can change events (about 2.7 h + 3.3 h on the stated environment).

**C — positive physical affine batch model, recommended for P2 n=1 only.** Fit equal-weight OLS of **all 90 case-pass full-batch medians**, in ms, on total bytes per reader (decimal MB):

`batch_complete_ms = {MODEL['intercept_ms']:.15f} + {MODEL['slope_ms_per_MB']:.15f} × (n × bytes_per_request / 1,000,000)`.

The intercept is **{MODEL['intercept_ms']*1000:.9f} µs**, strictly positive without clamping; slope and bandwidth are positive. This physical form has one startup term and a data term proportional to the total moved bytes. At n=1 it is exactly the existing analytical configuration. At n>1 it describes full-batch completion; a per-request caller must not use per-request bytes and claim it models concurrent batch tails. The fit ignores pair count in the median; independent all-case/pass residuals test whether that simplification is supported. All three held-out fits are also physical:
''')
parts.append(table(['held-out pass', 'training medians', 'intercept µs', 'bandwidth Gbit/s'],
                   [[p, 60, f"{m['intercept_ms']*1000:.9f}", f"{m['bandwidth_gbps']:.9f}"] for p,m in C['leave_one_pass_out_models'].items()]))
parts.append(f'''

Proposed rule C: (i) every two-pass training fit has positive finite intercept/slope; (ii) each of the 90 held-out case/pass medians has <=5% absolute relative error; (iii) every retained P2-coordinate timing trial passes the empirical window test, with a separate deterministic no-motion proof for the proposed configuration. Results: **all physical fits PASS; 90/90 median checks PASS**, maximum **{maxval('affine_holdout_median_error_pct'):.6f}%**; in-sample median maximum **{maxval('affine_in_sample_median_error_pct'):.6f}%**; every P2 trial/window PASS. Rule (iii) is finite evidence, not a probability bound. Accepting C requires YC to choose median-input qualification for these P2 cases. The stronger trial median5/p90 10% rule instead gives **82/90 PASS, 8 FAIL**, maximum p90 residual **{maxval('affine_holdout_trial_p90_error_pct'):.6f}%**; do not label it a qualified p90 latency model.

For every candidate, the per-case/pass result follows. The median-error columns use **held-out pass predictions**; trial p90 is p90 of absolute relative errors, not relative error of the p90 time. Bound columns compare the full-batch maximum to the scale minimum gap, even for coordinates the P2 replay never uses. Numeric columns retain failures instead of censoring them. [case_pass_results.csv](case_pass_results.csv) additionally preserves predictions, median trial errors and all Boolean verdicts at full precision.
''')
parts.append(table(['case', 'pass', 'A32 / A64', 'B median err %', 'B trial p90 err %', 'B trial rule', 'C median err %', 'C trial p90 err %', 'C trial rule'],
                   [[r['case'],r['pass_index'],f"{pf(r['bound32_tail_pass'])}/{pf(r['bound64_tail_pass'])}",
                     f"{r['point_holdout_median_error_pct']:.4f}",f"{r['point_holdout_trial_p90_error_pct']:.4f}",pf(r['point_trial_rule_pass']),
                     f"{r['affine_holdout_median_error_pct']:.4f}",f"{r['affine_holdout_trial_p90_error_pct']:.4f}",pf(r['affine_trial_rule_pass'])] for r in R]))
parts.append('''

C outside-domain policy: **fail qualification** for any other bytes, simultaneous requests, pair count, source/layout, hardware/fabric, compression setting or trace/window generation. Do not advertise general interpolation or a runtime rejection already implemented in Frontier. The proposed external census restricts this adoption; the current bytes-only predictor itself extrapolates if a caller uses it elsewhere. General n>1/concurrency admission needs an approved interface/source change. Under the adopted restriction, C needs **no Frontier interface change**, no table edits, 0 new GPUh and no fresh replay under the proof below. A future wider-domain requirement needs new design/measurements, not the original H800 substitution.

**Original NIXL02 FAIL preserved:** the earlier integrity-valid H200 run lacks T6144 and has an unconstrained intercept **−46.6769198402 µs**, bandwidth **305.707821 Gbit/s**, and p90 relative residual **29.175907%**. It remains rejected in [fit_diagnosis.json](../../b30b/f3_nixl02_20261101/fit_diagnosis.json) and [earlier report](../../../test_report_2026-11-01_f3_h200_nixl.md). The new positive fit does not reverse or replace that verdict; it qualifies a different source/data generation under a newly proposed rule. All full-grid tail FAILs above also remain visible. Existing native/all-request/attention/collective FAIL records were not rewritten.

**5. Event-window proof and conditional replay decision.** [possible_post_windows.csv](possible_post_windows.csv) enumerates **2,034** possible post-start windows (510+510+507+507) across all four retained traces. For each of the 128 transfers per trace, scan every DECODE `formed_s` from ledger start through ledger end and find the first strictly later start on that lane, even if it lies beyond the ledger end. This avoids claiming that the ledger directly observes READ posting. The candidate is tested against every possible posting step, a superset of the actual one.

Source: [NixlReadCompletion.on_engine_step_start](../../../../../frontier/scheduler/replica_scheduler/vllm_v1_nixl_reads.py), lines 39–68, checks posted wire ends first, then posts new READs; `_report_nixl_reads_of_batch`, `_report_nixl_reads_of_dummy_forward` and `consume_found_nixl_reads` report them with the owning output. [vllm_dp_busy_loop.py](../../../../../frontier/scheduler/replica_scheduler/vllm_dp_busy_loop.py), `on_engine_step_start`, `_schedule_iteration` and `_issue_dummy_pass` (lines 137–143, 210–217, 309–333), starts run-ahead connector steps at host formation. `next_pass_start` (line 355 onwards) can wake at a wire end only for a zero-duration idle-forward group. The retained traces contain **zero zero-duration DECODE forwards** (14,848 per P2-32 trace; 24,320 per P2-64 trace), so a shorter price cannot introduce that wake here.
''')
parts.append(table(['trace', 'possible starts', 'min next-start gap ms', 'old/current price ms', 'C price ms', 'P2 sample max ms', 'crossings'],
                   [[w['name'],w['possible_posts_checked'],f"{w['min_subsequent_step_gap_ms']:.12f}",f"{w['current_price_ms']:.12f}",f"{w['proposed_price_ms']:.12f}",f"{w['observed_p2_tail_max_ms']:.12f}",0] for w in W]))
parts.append('''

Both the previous H800 prices (2.570035277 ms on B29 structural bytes; 2.987238193 ms on two-fix wire bytes) and C's positive 0.968402342 ms lie strictly inside `(post, first later start)` for **every possible post**. B's point also lies inside. The posting step cannot find its own new READ, because checks precede posting and the price stays positive. The same later step therefore finds each READ, and its same output reports it. No ledger windows overlap; candidate prices finish well before another transfer can post. With deterministic DES scheduling and no idle zero-cost wake, induction preserves the next schedule and every later event. There are **zero crossings**. This proves READ release and downstream event/metric invariance for changing only this input on the retained configuration; it does not imply every metadata field stays equal (the configured transfer price and configuration fields change).

The two-fix window checker only compared 2.570035277 with 2.987238193. Q2 extends the proof to the shorter H200 candidate and every observed n=1 tail, with the explicit `(0,next step)` test. No new all-request or warm E2E metric comparison was calculated and no new replay was executed. Expected values would equal the corresponding B29/two-fix outputs by the event proof; this is an inference, **not a fabricated replay result**. A binding-only fresh replay remains Claude's optional formal process choice; if another interface/lifecycle/concurrency change is made, this proof no longer discharges it.

**6. Verification and exact commands.** Interpreter: `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6; cwd as above; `PYTHONPATH=$PWD:$PWD/tests/analysis`, bytecode disabled, OpenBLAS/OMP/MKL threads each 1, `TMPDIR` in the authorized `scratch/`. Every heavy inspection is streamed and bounded by `timeout 120s`; no full replay or new dependency is needed. Run from the working root:
''')
parts.append('```bash\n'+command('analyze.py')+f' > {P}/analysis.log 2>&1\n'+command('checks.py')+f' > {P}/checks.log 2>&1\n'+command('verify.py')+'\n'+command('render_report.py')+'\n```')
parts.append('''

`analyze.py`: expected no missing cases/duplicates, raw/CSV equality, physical fits, all four concurrency sweeps and every no-motion assertion; observed **PASS**, exit 0. See [analysis.log](analysis.log) and [results.json](results.json), `checks`, `environment`, `evidence_hashes`. The reported median/p90 rule failures are candidate outcomes, not hidden analysis-command failures.

`checks.py`: 12 focused checks of half-open occupancy, overlapping lanes/pairs, rejection of invalid windows and unphysical unclamped fits, exact p90 convention, all rows/trials, independently recomputed closed-form OLS for the full fit and three holdouts, all four window proofs, and unchanged retained-input hashes; expected all PASS; observed **12/12 PASS**, exit 0 ([checks.log](checks.log)). The checks use a different closed-form covariance calculation to verify the OLS coefficients. No tests that require GPU execution were added or run.

`verify.py` executed three commands, all exit 0: the retained lead checker independently rebuilt 23 hashes/90 rows/12,150 trials/80,352 notifications; `python -m pytest tests/unit/test_nixl_read_completion.py -q -p no:cacheprovider --basetemp <authorized scratch>/pytest` reported **9 passed in 0.79 s**; the production `AnalyticalKVCacheTransferConfig` and `AnalyticalKVCacheTransferPredictor` calculation returned **0.9684023419505275 ms**, exactly matching the proposed formula (difference 0.0 ms). All commands use the assigned interpreter and `timeout 120s`. Exact executable arguments, including the price program and full paths, expected values and observed stdout are in [verification.json](verification.json); see [lead_retained_check.log](lead_retained_check.log), [lifecycle_regression.log](lifecycle_regression.log), [production_price.log](production_price.log). Fresh P2 CPU replays: **NOT_RUN — no completion can move**. Platform/GPU/native-byte checks: **NOT_RUN in this task**; retained identity evidence was inspected. The initial host `python` lookup failed because it is not on PATH; all actual analysis/check commands used the assigned interpreter. A small exploratory inspection initially used the nonexistent top-level `cluster_config_decode` key; the saved config stores these fields under `cluster_config`. It was corrected by inspecting actual config keys before implementation; no input was edited. Initial analysis/check runs passed; their logs are preserved under `scratch/`.

**7. Changes, remaining decision and review handoff.** All files are new in `nixl_timing_q2/`: `analyze.py` rebuilds observations and candidates, sweeps concurrency and proves event windows; `checks.py` guards arithmetic and verifies the evidence; `render_report.py` renders this report; `verify.py` runs and records the retained lead check, nine existing lifecycle tests and production price comparison; `proposed_input.json` carries the precise optional input, domain and rule; `results.json` carries all case/pass results, run proofs, environment and input hashes; `case_pass_results.csv`, `repeatability.csv`, `equal_byte_passes.csv`, `reader_summary.csv`, `transfer_windows.csv`, `possible_post_windows.csv` preserve the small reviewable tables; `analysis.log`, `checks.log`, `lead_retained_check.log`, `lifecycle_regression.log`, `production_price.log`, and `verification.json` record actual checks; `report.md` is this report. `scratch/reader_trials.csv` is a full reader-trial copy and `scratch/*initial.log` retains exploratory successful runs; scratch is not intended for commit. No reusable `tests/analysis/` or `tests/unit/` file was needed.

Claude/YC can adopt C with **one choice**: approve the displayed pair and `AFFINE_MEDIAN_P2_SINGLE_READ` rule/domain, retain full-grid trial-rule FAILs and NIXL02 FAIL, keep native payload OPEN, and record the price change plus no-motion proof. The choice changes the formal timing input and its qualification rule; no formal replacement was made here. If YC requires a universal tail guarantee or the full-grid per-trial p90 rule instead, keep timing OPEN and authorize a specific serving-contention/tail investigation or a stochastic/cohort completion design with its own acceptance rule. It would be incorrect to adopt a deterministic median model under the failed trial rule. No additional permission was requested because the assigned proposal/CPU analysis was fully executable.

Review the difference between full-batch completion and individual request service, the 20 byte-keyed exact points, held-out-pass fit targets, strict positive intercept, the complete failure rows, TP/DP mapping and ledger envelopes, the zero-duration-forward guard and the domain policy in `proposed_input.json`. Rerun the four commands above and the listed existing lifecycle check if desired. Costs outside the fixed payload/concurrency, per-request n>1 completion times, full serving contention, a future-tail probability guarantee, and observed native request bytes remain limitations. Claude owns acceptance, adoption, any formal binding replay, records and commits.
''')
(HERE / 'report.md').write_text('\n\n'.join(parts) + '\n')
print('Rendered', HERE / 'report.md')
