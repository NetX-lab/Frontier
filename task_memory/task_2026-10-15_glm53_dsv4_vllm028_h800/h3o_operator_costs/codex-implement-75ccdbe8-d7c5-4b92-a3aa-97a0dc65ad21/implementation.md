The assigned assessment is complete. P2-32 is **PASS within the qualified retained-coordinate scope**. P2-64 is **BLOCKED_ON_DECISION**: its numerical and input checks pass, but the retained evidence does not establish the required promotion receipt before native submission. These are executor findings; Claude retains review, receipt issuance and final acceptance.

The complete results for items 1–8, including all 27 ledger rows, consumed paths and hashes, native/Frontier values, exclusion evidence, window endpoints, 16 milestone mappings and 35 GPU jobs, are in [assessment/report.md](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/formal_p2_20261102/assessment/report.md). Its final SHA-256 is `50e3f2805e4b75231134749d151e136b8f6e508373e84965dbd789245587bf64`.

Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Let T denote `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800`, H denote `T/h3o_operator_costs`, and A denote `H/formal_p2_20261102/assessment`. All project writes were confined to A and its scratch directory. The Q1 ledger, tables, native evidence, timing input, replay records, replay roots and formal readings were preserved. No Frontier/native source change, Git operation, commit, dependency installation, GPU/platform action, replay modification or delegation was performed. Existing work from this same executor session was continued; unrelated `.cortexkit/`, `.pi/` and `remaining_measurements/` were untouched.

The assessment applies ticket 19 and the specified P2 qualification, attention, timing, payload, warm-reading, two-fix and EP-defer decisions. It preserves the adopted independent attention check, the H200 timing pair, source-derived payload qualification, the per-repeat 10% gate and the separate 30% promotion predicate. No criterion was changed.

**Results and evidence**

Both formal command records are terminal FINISHED, including gates and readings:

| Case | Started UTC, November 2, 2026 | Finished UTC | Full runner elapsed seconds |
| --- | --- | --- | --- |
| P2-32 | 03:09:44.327780 | 05:59:27.683187 | 10183.355138233863 |
| P2-64 | 03:09:44.327470 | 06:37:22.883681 | 12458.555925155059 |

1. **Current inputs:** 22 QUALIFIED, 4 OPEN and 1 FAIL. Whole hardware/operator coverage, ledger row 26, is PASS under its recorded criterion because attention, H200 timing and payload are now resolved. OPEN rows 18, 23 and 24 concern the retired H800 pair, independent memory peaks and autonomous planning; their retained limitations do not make them prerequisites for these measured-pool isolation cases. Row 27 remains OPEN and is required for P2-64 promotion. Rejected NIXL02 row 19 remains FAIL and is unconsumed. Each OPEN row retains a closure proposal and cost; numerical success was not used to close it.

2. **Binding:** seven profiling tables per case, the trace and exact request population, resolved configuration, source, driver/dependencies and 879 non-lock generated cache files per case bind by path and SHA-256. The formal source is `abd1e14198b2b258921ccd669ac87c91f499e6dd`; Frontier/data trees and driver/dependency bytes match the two-fix generation. The driver SHA-256 is `3172d0d0a326382c59e3c9637fc1f1b756059064087f70e555b14700c2a8b527`. Five explicit finding groups per case disclose authorized config/root changes, the post-ledger timing successor, source-generation changes, the new formal runner and auxiliary inputs bound through qualified source/runtime owners. No unexpected consumed cost change, omission or fallback was found.

3. **Comparison:** the original `compare_b29.py` control and the expanded complete-record control passed on two-fix versus B29 before formal comparison. Formal versus two-fix then passed with zero complete forward differences. Request metrics, scheduler KV start/completion/duration, collective data, gate rows and complete reading documents are identical after separately recording administrative provenance. Forward counts are 23040 and 39680. Wire-end movement is derived rather than exposed as an observed ledger timestamp. The adopted pair is 390.39349928254796 Gbit/s and 0.0236521842818856 ms, giving 0.9684023419505275 ms for the 46103040-byte bound. Minimum possible next-step gaps are 18.72496532622958 and 24.380075962758763 ms; no crossing occurs.

4. **Readings:** all 24 required warm checks PASS independently at 10%. Absolute relative errors, expressed as percentages, are:

| Case/repeat | Mean client TTFT | Mean request TPOT | THPT | Mean request E2E |
| --- | --- | --- | --- | --- |
| P2-32/01 | 4.722918 | 0.100175 | 0.134548 | 0.015944 |
| P2-32/02 | 5.447453 | 0.192044 | 0.090974 | 0.092723 |
| P2-32/03 | 5.607176 | 0.173713 | 0.047914 | 0.269220 |
| P2-64/01 | 4.599811 | 0.552985 | 0.457663 | 0.661659 |
| P2-64/02 | 5.675933 | 3.656983 | 0.022443 | 3.710082 |
| P2-64/03 | 4.727235 | 2.433412 | 0.170758 | 2.493894 |

The unchanged all-request P2-64 repeat01 TTFT FAIL, 12.254940665008353%, remains beside its warm result. Only that repeat excludes the paired requests r006/r017/r027/r035/r036, supported by eight-TP-worker cohorts and source/event ownership evidence. Its retained population is 123 requests and 881664 tokens, versus the original 128 and 917504. Other repeats retain all 128. The excluded requests are interior, so current endpoints remain unchanged; this is an observation rather than a general fixed-window rule. Exact endpoints, full IDs, native/Frontier values and errors are retained in the outputs. Pooled values and 144 p50/p95 rows are diagnostics only.

5. **Milestones:** all 16 checkbox items in the four assigned ticket sections are mapped to ticket lines, evidence paths and case-specific labels. Runtime checks establish non-dummy timing, measured communication, feature flags, effective `max_model_len=7168`, `max_num_batched_tokens=8192`, measured memory budgets and 128 exact 6144/1024 completions per case. The GPU-accounting checkbox is FAIL/INSUFFICIENT_EVIDENCE because exact actual spend is unknown. This reporting limitation is disclosed separately from the ticket's explicit numerical/input/promotion Acceptance predicates. Whole-ticket or missing H800-case closure is not claimed.

6. **Promotion:** every required P2-32 error is finite, has a nonzero native denominator and is at most 0.30 in every repeat; reference and coverage checks pass. The numerical/reference/coverage promotion predicate is true. `promotion.json` contains draft receipt contents only: `draft_only=true`, `issued_receipt=false`, timestamp null.

The existing P2-64 native job is `exp-1025-155441-668927`. Its manifest was written on October 25, 2026 at 15:54:37.229732Z, submission occurred at 15:54:41Z, success was recorded at 16:19:44Z and R4 was written at 17:58:23Z. The bounded retained-record search found no qualifying pre-launch receipt. This is not a global absence claim. The historical non-preemptible launch had explicit D-H200-TEMP-MAINLINE authorization and is not relabelled as unauthorized.

7. **GPU-hours:** 35 distinct H200 jobs are inventoried with purpose, GPU count/resource class, source and clock limitations. The 32 later P2 campaign jobs have 135.0033333333333 witnessed container GPU-hours plus 19 incomplete intervals. Three earlier shared DSV4 prerequisites add 6.407222222222222 GPU-hours. The combined witnessed subtotal is 141.41055555555553 GPU-hours; actual P2 and combined totals are UNKNOWN. Unknowns are not zero, and queue/phase envelopes are not summed as actual spend.

8. **Verdicts:** P2-32 has no blocking reason within this assigned retained scope. P2-64 is blocked on the pre-launch/reference-reuse decision. Unknown spend is a separately reported limitation. Both verdicts retain restrictions on attention generalization, source-derived rather than observed payload arrays, finite NIXL tails and runtime domain, host-zero/ingress/handoff/K19 inputs, deferred EP dispatch, measured-pool isolation and future native warm/idle proof.

**Changes and artifact inventory**

Every new or changed file is under A. The requested files and their responsibilities are:

| File | Contents |
| --- | --- |
| `report.md` | Complete assessment results 1–8 and review handoff |
| `input_status.json` | All 27 current dispositions, decisions, evidence and OPEN-row requirements/costs |
| `binding.json` | Consumed path/hash bindings, cache inventory and explicit findings |
| `formal_vs_two_fix.json` | Controlled equality checks and provenance differences |
| `gate_readings.csv` | All 24 warm/all-request metric pairs, populations and endpoints |
| `milestones.json` | All 16 ticket checkbox mappings |
| `promotion.json` | Exact predicate, draft receipt, chronology and costed options |
| `gpu_hours.csv` | Deduplicated 35-job inventory and unknowns |
| `verdict.json` | Case verdicts, blocking reasons and limitations |

Supplemental files are `diagnostics.csv`, `reading_details.json`, `runtime.json`, `command_receipts.json` and `changed_files.csv`. Assessment helpers are `assess_inputs.py`, `git_objects.py`, `compare_formal.py`, `read_formal.py`, `audit_runtime.py`, `account_gpu.py`, `promotion_assessment.py`, `render_report.py`, `finish_checks.py` and `watch_replays.py`. Their symbols and line references are listed in the report. Scratch files preserve controls, protected-input snapshots, polls, command logs and accounting detail. The stale render preview is explicitly CONTROL ONLY.

The final refresh changed the renderer/report, appended actual refresh receipts and refreshed output hashes and inventory. [changed_files.csv](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/formal_p2_20261102/assessment/changed_files.csv) enumerates all 66 assessment files other than itself by absolute path, SHA-256 and size. No retained evidence or external task record was edited.

**Verification**

Interpreter: `/data/ycfeng/envs/frontier-py310/bin/python`, Python 3.10.6. Working directory was the project root. Environment: `PYTHONPATH=$PWD:$PWD/tests/analysis`, `PYTHONDONTWRITEBYTECODE=1`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`; child commands used `TMPDIR=A/scratch`. Heavy reads ran under timeout.

The completed final wrapper command was:

```bash
env PYTHONPATH="$PWD:$PWD/tests/analysis" PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 timeout 1200s /data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/formal_p2_20261102/assessment/finish_checks.py
```

It executed all seven child checks successfully. Each used `timeout <limit>s /data/ycfeng/envs/frontier-py310/bin/python A/<script>` with the following limits and observed results:

| Script / timeout | Expected and observed | Result |
| --- | --- | --- |
| `assess_inputs.py` / 120s | 27 dispositions; coverage PASS; both bindings complete | PASS, exit 0 |
| `compare_formal.py` / 300s | Zero full-forward differences; exact metric/gate/reading identity | PASS, exit 0 |
| `read_formal.py` / 120s | Official arithmetic reproduced; 24 warm PASS, one retained all-request FAIL, 144 diagnostics | PASS, exit 0 |
| `audit_runtime.py` / 180s | Correct flags/limits, 128 completions each, qualified domain and no crossings | PASS, exit 0 |
| `account_gpu.py` / 120s | 35 deduplicated jobs; estimates separated from 19 unknown intervals | PASS, exit 0 |
| `promotion_assessment.py` / 120s | Exact finite per-repeat predicate; draft only; chronology and options | PASS, exit 0 |
| `render_report.py` / 120s | Complete report, verdicts and 16 milestones | PASS, exit 0 |

The original B29 comparator and expanded control were executed successfully before formal use. The final template refresh reran only the renderer, followed by `finish_checks.validate_outputs()`, protected-evidence comparison, Python syntax and report-content checks. Both refresh commands passed and were appended without discarding the seven existing receipts. A final read-only check verified all nine deliverable hashes, all nine command-log hashes and the complete 66-file inventory. Exact argv, including the integrity-check Python code, environment, expected/observed output, exit status and log hashes are in [command_receipts.json](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/formal_p2_20261102/assessment/command_receipts.json).

NOT_RUN: another simulation, native/GPU campaign, platform query, deployment or commit. Existing shared/fidelity/collective evidence was inspected and reused; it is not represented as newly rerun production validation. No formal replay failure occurred.

**Remaining decisions and review handoff**

YC has two evidenced P2-64 options:

- **A — explicit exception:** waive launch order for reuse of the existing native reference while retaining its historical resource authorization. Incremental GPU cost: 0 GPU-hours. This requires an explicit exception record; it cannot establish retrospective receipt existence.
- **B — preserve the ticket:** Claude reviews and issues the qualified P2-32 receipt, then obtains/uses authorization for a new three-repeat formal P2-64 native campaign. Resource: eight nodes × eight H200, `step_main`, wholly preemptible. The original campaign provides a 24.322222222222223 container GPU-hour planning proxy; its 3600-second timeout gives a 64 GPU-hour execution ceiling excluding queue/startup. Stronger kernel/graph/lazy-pair warm and idle proof may increase cost. Normalize and gate the new reference; the unchanged simulator inputs do not by themselves require another CPU replay.

Option B is recommended because it preserves the existing ticket. Neither option was executed. For exact GPU accounting, Claude should seek authoritative allocation/billing records for the 19 incomplete-clock jobs; no GPU rerun is needed for that reporting gap.

Claude should review row 26 against its unchanged criterion, each disclosed binding difference, comparator control and full identity evidence, all warm repeats beside the original all-request FAIL, promotion chronology/resource authorization, and GPU unknown-clock labels. The small assessment checks can be rerun if review requires it; independent official gate/warm checks should use new scratch outputs. Missing active H800 results, paused GLM work and broader autonomous/generalization claims remain outside this assessment. No Claude acceptance, issued promotion receipt or full ticket closure is claimed.
