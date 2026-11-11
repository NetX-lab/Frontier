# Independent review: H200 DSV4 delivery at b7fddec3

## 1. Review context

**Request.** Determine whether DeepSeek-V4-Flash simulation support and accuracy calibration on H200 are complete, correct, and accurately recorded. Overall verdict: **changes_required** — **0 P0, 0 P1, 1 P2, 3 P3 findings**, covering record and test defects. No numerical acceptance blocker was established for either active H200 case.

**Working root (`W`).** `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. All commands ran from this worktree. In this report, `T` means `W/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800`; `O` means `/data/ycfeng/tmp/codex-review/7053879f-03a9-4e51-ae74-c6bbab2891dd/output`. Paths beginning `T/` or `O/` resolve through those absolute definitions.

**Resolved Git scope.** Branch `feat/glm53-dsv4-vllm028`, HEAD `b7fddec361b0189942391428ea657336948920e9`. The newest delivery range is `9a45a135..b7fddec3`: heartbeat port `efbe09a9`, issue records `7fbed2ae`, and ticket reconciliation `b7fddec3`. The broader history context uses the last revision before October 15, 2026 in the supplied Asia/Hong_Kong timezone, `9a650fba342a0da835e162d6f030f056e7bc9211`. That range contains **693 reachable commits / 520 first-parent commits**, rather than the request's 672; no particular 672-commit base was supplied. Review followed acceptance-relevant sources, callers, tests and artifacts across this range; it was not a line-by-line audit of every commit or all 192 changed `frontier/` paths.

No staged or unstaged changes were present. Excluded untracked `.cortexkit/`, `.pi/`, `launcher.pid.txt`, and `remaining_measurements/` were retained. No sources, configurations, Git state, or task records were changed; review output and test temporaries were confined to `O`.

**Authority and rules.** Read the worktree `AGENTS.md`, including sequential PDD, parallel/Replica mapping, modular predictors, numerical fidelity, profiling identity, and regression requirements. The parent `Frontier/AGENTS.md` is byte-identical; no additional applicable nested `AGENTS.md` or `CLAUDE.md` was found. Reviewed the task's `requirements.md`, `plan.md`, tickets 17–19, progress/issues/review records, central formal/heartbeat reports, and their linked qualification and implementation evidence. Current authority includes D-WARM-TARGET (`requirements.md:435–451`), D-NO-STEP-COST-SHORTCUT (`:660–666`), D-P2-Q1-CHOICES (`:344–354`), the subsequent H200 NIXL/payload decisions, and the explicit November 11 closures (`:172–209`). The acceptance entry is ticket 19's four independent 0.10 per-repeat gates and pre-launch 0.30 promotion predicate.

**Concrete retained inputs.** H200-32 native endpoints were read from `T/h200_mainline/diagnostic/inputs/p2_32r/run/client_requests.jsonl`; H200-64 from `T/h3o_operator_costs/formal_p2_64_native_20261102/generation08_run2/inputs/p2_64r/run/client_requests.jsonl`. Frontier raw completion-event files were resolved through the bound `run_root` in each `formal_p2_20261102/p2_32_command.json` / `p2_64_command.json`. The original generation08 local download is `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/formal_p2_64_20261107_08_ssh/dsv4-h200-p2-64-formal-20261107-08/artifacts/`, with its inventory beside it. No remote retrieval was performed.

**Reviewer/environment.** Caller-supplied model/effort: `gpt-6.1-sol-sys-paygo` / `ultra`. Every Python check used `/data/ycfeng/envs/frontier-py310/bin/python` (3.10.6). Review date: November 11, 2026. This was CPU and local-artifact work only: no GPU, platform, cloud-storage, StepMind, RJob, software installation, publication, or merge operation.

**Scope decisions respected.** GPU-hour accounting is `WAIVED_BY_USER`; H800 is `CLOSED_NOT_PURSUED`; GLM remains paused; integration into other worktrees is separately owned. I-018, the three upstream README contract failures, and the specified environment-limited native tests are not reopened as task defects. Measured-capacity isolation and retained-shape qualification are accepted bounds, not claims of autonomous planner accuracy or unseen-shape generalization.

## 2. Assessment

| Question | Assessment |
| --- | --- |
| Is the active H200 acceptance scope complete? | **Numerical acceptance is supported within its stated bounds; the full delivery needs corrections.** Both formal cases, three native repeats per case, four metrics, own-hardware inputs, published results, and pre-launch promotion evidence are present. No additional active numerical criterion was found missing. The explicit accounting waiver and H800 closure resolve those former scope items by decision, not by fabricated measurements. Focused CPU verification and record reconciliation remain defective (F001, F002, F004). |
| Is it correct? | **The inspected formal numerical results and calibration method are supported.** Independent raw-timestamp arithmetic reproduces both PASS results; no forbidden whole-step input or arbitrary prediction multiplier was found on the inspected formal paths. The delivered test surface has the collection defect F002 and stale H200 table assertion F004. Full native/GPU behavior remains outside this review's executable validation. |
| Do records agree with artifacts? | **Not completely.** The major PASS values and historical verdict distinctions agree, but current acceptance text remains internally contradictory (F001), and the older qualification auditor reports a false Git provenance limitation (F003). |

### Requirement and claim disposition

| Material requirement / claim | Disposition and evidence |
| --- | --- |
| P2-H200-32 formal PASS, worst 5.6072% at repeat03 TTFT | **Satisfied.** Raw Frontier completion events and native client timestamps reproduce `0.05607176023817675`. Bindings: `T/h3o_operator_costs/formal_p2_20261102/p2_32_command.json`, its replay receipt, gate rows, and `readings/p2_32_warm_reading.json`. |
| P2-H200-64 generation08 formal PASS, worst 3.1623% at repeat02 TTFT | **Satisfied.** Same independent calculation gives `0.03162330523324778`. `T/h3o_operator_costs/formal_p2_64_native_20261102/generation08_run2/` retains gate rows, raw inputs, warm reading and checker command/stdout. |
| Three complete repeats; 432/432 native requests; no repeat dropped | **Satisfied for both cases.** Each native file has 432 distinct transport IDs: 3 × (16 warmup + 128 formal). Every record completed with HTTP 200, `finish_reason=length`, 6144 prompt / 1024 output tokens, stream completion and no validation error. Each case retains 384/384 formal request-repeat pairs. The deterministic Frontier side is one 128-completion replay compared separately against each native repeat. |
| Four metrics <= 0.10 in every repeat, warm and unchanged all-request readings | **Satisfied for the two admitted results.** Both have empty opener sets; the two populations are identical. Independently recomputed values match warm and all-request per-repeat entries, without subtracting delays or selecting latency outliers. Warmup requests remain outside the formal population by definition. |
| Generation08 terminal/resources | **Satisfied in retained local evidence.** Terminal receipt has eight Succeeded replicas, eight exit-0 containers, zero restarts, 8 × 8 GPUs and preemptible label `yes`. All 130 original downloaded files / 75,585,300 bytes match their inventory hashes. This verifies local retention, not a fresh cloud read. |
| 32→64 promotion before build/submission | **Satisfied, with independent Git corroboration.** Receipt issued November 2 at 07:38:49.523696Z; byte-identical receipt committed in `1a7950ed5` at 07:39:13Z; first receipt-bound build at 08:42:19.175887Z; retained platform creation at 08:45:36Z. Digest `d505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596` matches the first build and generation08 bindings. |
| Qualified operator method; no shortcut / widened gate | **Supported on inspected paths.** A10 rejects dummy inputs, adjacent whole-step files and a bound step predictor (`a10_driver.py:161–181`). Both replay receipts pass these checks. Seven consumed profiling tables and 879 cache files per case rehash correctly. `dsv4_composition.py:88–108,158–187` composes measured operator costs/gaps on the source stream program. Approved measured ingress/handoff and K19 inputs were retained; they are disclosed inputs, not evidence of arbitrary residual fitting. The independent audit used 0.10 directly. |
| Input qualification and shared fidelity | **Supported by artifact recheck, with scope limits.** Reran the retained qualification auditor: 906/906 checks PASS, 96 omitted-workload independent attention comparisons (maximum 3.663151%), raw NIXL fit/repeatability, and six retained 77-case fidelity comparisons. Five matrices are identical; B29's differences are the disclosed ingress/handoff config fields, with no unexpected artifact or cache-name difference. These are rechecks of retained evidence, not newly executed 77-case simulations. |
| Heartbeat port changes no simulator numeric path | **Satisfied by diff and artifact binding.** `efbe09a9` changes only the worker and its test under `tests/`. From formal source `abd1e141` to HEAD, `frontier/` and `data/` differ only in six comment/docstring lines in `model_architectures.py`; no executable simulator change was found. Current formal replay bytes remain bound to their original receipts. |
| I-017 cause and correct fix | **Supported.** A fresh 3.10 stdlib probe reproduces stub visibility `[True, False]` after the original restore pattern and `[True, True]` with the module-scope import. Worker/gate tests pass; all 43 worker cases also pass reversed. The import stabilizes module identity without weakening assertions or extending deadlines. Fresh stopped/live-peer timings match the claimed mechanism and leave no thread behind. |
| Retained 15-module baseline comparison / release sanity / accuracy gate | **Supported at the stated evidence tiers.** Reparsed retained logs: both FAILED/ERROR sets have exactly the same 93 IDs. Independently reran release sanity: the same three accepted README failures / 17 passes. Worker plus gate: 83 passes, including the 40 gate tests. The 15-module suite was not independently rerun in full. Separate fresh model/table checks expose F002 and F004; they are outside the claimed heartbeat comparison set. Environment-limited cases remain unknown, not passing. |
| November 11 reconciliation | **Substantively justified, textually incomplete.** The 15 newly resolved ticket-19 boxes can be mapped to the H200 evidence, H800 closure, or accounting waiver. Ticket 18's corrected formal-reference Status and review 12.5's appended closures are supported. F001 identifies remaining contradictory prose; no evidence was found that the old 64-GPU reference was retroactively marked PASS. |

### Independent numerical results

Metrics use raw client-visible endpoints: TTFT = first token − dispatch; TPOT = (last token − first token) / 1023; E2E = last token − dispatch. THPT = 917,504 prompt-plus-output tokens divided by last formal completion − first formal dispatch, not the 127-second arrival span. Pooled throughput uses summed token totals / summed repeat spans.

| Case | Frontier mean TTFT ms | TPOT ms | E2E ms | THPT tokens/s |
| --- | ---: | ---: | ---: | ---: |
| H200-32 | 344.878991 | 19.926786 | 20,729.981427 | 6,225.735862 |
| H200-64 | 684.313827 | 25.295337 | 26,561.443186 | 5,975.251570 |

Every entry below is an independently reconstructed relative error in percent. It applies to both the warm and all-request readings of the admitted result.

| Case / repeat | TTFT | TPOT | E2E | THPT | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| H200-32 / 01 | 4.722918 | 0.100175 | 0.015944 | 0.134548 | PASS |
| H200-32 / 02 | 5.447453 | 0.192044 | 0.092723 | 0.090974 | PASS |
| H200-32 / 03 | **5.607176** | 0.173713 | 0.269220 | 0.047914 | PASS |
| H200-64 / 01 | 2.746657 | 2.184558 | 2.051245 | 0.513588 | PASS |
| H200-64 / 02 | **3.162331** | 2.264673 | 2.117232 | 0.497900 | PASS |
| H200-64 / 03 | 2.974286 | 2.140537 | 2.002004 | 0.281770 | PASS |

The November 2 H200-64 assessment used the earlier October 25 native reference, retained its 12.254941% repeat01 all-request TTFT failure and `BLOCKED_ON_DECISION` promotion verdict, and is historical. Generation08 is a separate, properly promoted native reference. These results must not be conflated.

## 3. Findings

### F001 — P2 medium — Current acceptance records still contradict the resolved scope and retained population

**Confidence: high. Classification: record-keeping defect, not a numerical defect.**

**Locations and evidence:**

| Exact location | Actual current wording / state | Contradicting authority or artifact |
| --- | --- | --- |
| `T/tickets/19-comparison-and-acceptance.md:224`, `final_acceptance` | Checked `WAIVED_BY_USER` item also says “This checkbox remains open until all actual clocks are added…” and ends with “no new GPU run is needed to close this box.” | D-GPU-HOURS-NOT-REQUIRED (`requirements.md:182–192`) closes it by waiver and specifically orders no reconciliation. The preserved figures/gaps are valid; the present-tense open status is not. |
| Ticket 19 `:183`, `comparison_and_calibration` | “In the current P2 readings the five excluded requests are interior…” | Generation08 `warm_reading.json:636–644` retains 384/384 with empty openers. H200-32 also has zero exclusions. Five exclusions belong to the old repeat01 reference, not either admitted current result. The formal report `test_report_2026-11-07_p2_64_generation08_formal.md:111–120` explicitly states zero exclusions. |
| Ticket 19 `:178–182` | Says no cause-localization round was required because no multiplier/threshold change/benchmark patch was used. | Requirements D-WARM-TARGET (`:435–449`) documents localization of five lazy NIXL handshakes; D-WARM-TWO-FIX / D-WARM-EP-DEFER (`:406–433`) documents subsequent geometry/count investigation and a source-backed correction. `plan.md:233–243` retains that evidence. “No further round after final qualification” could be correct; the unqualified claim about the delivered work is not. |
| `T/tickets/17-dsv4-canonical-case.md:35–44`, **Status** | Graph/measured consumption and decode profiling “now prioritize the H80032 P0 case,” followed by H800 being closed. | D-H800-CLOSED (`requirements.md:194–201`) and the same Status's closure sentence. This is not a historical heading. |
| `T/plan.md:1`; `T/summary.md:1` | Current titles still say “H800 follows.” | H800 has been closed, not queued for continuation. |
| Ticket 17 `:88–110`, **Milestones**, especially `:101` | All-case boxes remain unchecked and “Full-model effective evidence remains pending” has no current per-instance disposition. | Ticket 17's own `:39–44` and ticket 19's resolved groups establish H200 acceptance and H800 closure. The remaining boxes need the same explicit per-instance explanation, rather than looking like unfinished active H200 prerequisites. |

**Violated contract.** Current status must agree with the acceptance authority and retained artifacts; D-H800-CLOSED requires closed rather than pending H800 rows, and D-WARM-TARGET requires accurate population/exclusion reporting. Ticket 19's acceptance requires values, evidence and limitations to remain explicit.

**Reachable trigger and impact.** A next reader uses the current Status/Milestones or the newly checked ticket-19 acceptance list to determine remaining work or reproduce the result. They can infer that accounting reconciliation or H800 profiling is still required, or that the accepted H200-64 PASS depends on removing five slow requests. Those inferences conflict with the actual closure and complete admitted population. The November 11 reconciliation therefore does not fully deliver its record-consistency claim.

**Focused correction.** Make each current milestone's H200/H800/waiver disposition explicit. Move the old open-accounting sentence and five-exclusion observation into clearly dated historical context. Clarify the localization sentence as applying only to any further round after final qualification, while preserving performed investigations and their original verdicts. Correct the current H800 titles/Status. Preserve all accounting figures, known gaps and historical failures; do not check closed H800 scope as evidence-satisfied.

**Verification.** Cross-check current Status/Milestones against `requirements.md:172–209`, both admitted warm readings and the two formal reports. Current admitted populations must say zero exclusions / 384 retained; accounting must say waived, and H800 must say closed. Historical five-exclusion/FAIL records should still exist under their original dates.

### F002 — P3 low — Two attention test modules depend on another module's collection-time path mutation

**Confidence: high. Classification: executable verification defect.**

**Locations.** `W/tests/unit/test_v4_attention_coverage.py:20` imports `v4_attention_coverage` by a bare name. `W/tests/unit/test_v4_attention_phase_heldout.py:16–18` similarly imports `v4_attention_phase_heldout` and `v4_attention_heldout`. These implementations live under `W/tests/analysis/`, outside the prescribed root-only `PYTHONPATH`. `test_v4_attention_multipass_table.py:8–12` inserts that directory during collection, masking both defects when collected first.

**Violated contract.** The declared worktree-root CPU test workflow must collect relevant model/predictor verification independently; collection of one test module must not rely on unrelated test modules mutating `sys.path` first. The defect is separate from the accepted missing `torch`/`aiohttp`/engine limitations.

**Trigger / expected / actual.** Run either affected module without the multipass module preceding it, with the required interpreter and `PYTHONPATH="$PWD"`. Expected: collect its CPU checks and report unavailable retained-data cases explicitly. Actual: coverage alone aborts at line 20 with `ModuleNotFoundError: No module named 'v4_attention_coverage'`; phase-heldout alone and the phase-heldout/H200-tables/placement selection abort at phase-heldout line 16 with `ModuleNotFoundError: No module named 'v4_attention_phase_heldout'`. Both exit 2 without executing tests. The initial 29-module selection also aborts when coverage precedes multipass. Moving multipass first permits collection, demonstrating the dependency.

**Evidence.** `O/coverage-single.log` reports `1 error in 0.51s`; `O/remaining-tests.log` reports the phase-heldout collection error in 0.87s; a fresh standalone invocation reproduces it in 0.89s (`O/phase-single.log`). `O/dsv4-tests.log` retains the initial broader collection error. The ordered runs are recorded below. No source correction was applied.

**Impact.** A focused attention-coverage rerun, or a change in collection order, breaks the CPU verification entry point and can prevent otherwise collectible tests from running. This does not establish an attention prediction error or invalidate the retained formal values.

**Focused correction.** Make both modules and their transitive analysis imports independently resolvable through package-qualified imports or explicit repository-relative file loading. Do not depend on another test's global path mutation.

**Verification.** Run each affected module alone, then both before and after multipass under the pinned interpreter and root-only `PYTHONPATH`; every invocation must collect. Retained-data skips, if any, must remain explicit.

### F003 — P3 low — Qualification audit falsely states that Git provenance for the promotion receipt is unavailable

**Confidence: high. Classification: evidence-record defect, no numerical consequence.**

**Location.** `T/review_20261108_h200_mainline/audit_qualification.py:111` hard-codes `git_history: "task_memory is ignored; no Git creation timestamp proof is available"`; the retained `qualification_audit.json:4768` repeats it.

**Violated contract.** Verification limits and provenance claims must follow actual artifacts/Git state, rather than an assumption based on ignore rules.

**Trigger / expected / actual.** A reader or rerun uses the qualification summary to assess whether the receipt preceded the first formal build. Expected: report the tracked receipt's introduction and byte identity. Actual: the auditor reports that evidence unavailable even though `git ls-files` includes the receipt and `git log --follow` identifies `1a7950ed5ba9b766a5322313eb7e0f50fa76a2e4`, committed November 2 at 07:39:13Z. `git show` of that commit's file hashes to the same `d505eece…7596` digest embedded in the first 08:42:19Z build. This commit is an ancestor of the reviewed HEAD.

**Evidence / impact.** Independently checked the Git commands and file digest; see `O/additional-artifact-checks.json`. Rerunning a scratch copy of the retained auditor reproduces the incorrect hard-coded limitation despite 906/906 numerical/evidence assertions passing. The statement unnecessarily weakens, and misdescribes, available pre-launch provenance; it does not make the actual receipt retrospective.

**Focused correction.** Derive tracked-file/history availability with Git rather than directory ignore status. Add a dated correction to the retained review summary, preserving the original audit as historical if immutability is intended. Ordinary Git timestamps and retained platform receipts remain local evidence, not an external trusted timestamp service.

**Verification.** Report the introducing commit, its UTC timestamp, and the matching file SHA. Ensure the corrected summary distinguishes “ignored directory” from “this file is tracked,” and agrees with the first build/submission chronology.

### F004 — P3 low — H200 promotion test still expects the superseded 4,945-row attention table

**Confidence: high. Classification: executable verification defect, not a data or numerical defect.**

**Locations.** `W/tests/unit/test_dsv4_h200_tables.py:62–71`, specifically the parameter at line 63. The promoted table is `W/data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv`; its current receipt is `T/h3o_operator_costs/h200_tables/manifest.json:72–74`.

**Violated contract.** The CPU promotion invariant must validate the delivered, qualified table generation. Its expected row count must match the accepted supplement rather than an earlier generation.

**Trigger / expected / actual.** Run the H200 table checks under the pinned interpreter. Expected: the matching table and manifest pass the hash/count invariant. Actual: the hash assertion passes, and both actual CSV and manifest give **7,105**, but the final chained comparison requires **4,945**, failing at line 71. This is independent of missing native dependencies or temporary-directory restrictions.

**Evidence.** `O/remaining-ordered.log`: `1 failed, 54 passed in 75.66s`; the sole failure is `test_output_hashes_and_row_counts_equal_manifest[attention_graph_replay_kernel_only.csv-4945]`. A focused repeat with `-x` gives the same failure after 26 passes in 3.11s (`O/remaining-first-failure.log`). Independent CSV counting/hashing (`O/table-count-audit.json`) gives 7,105 rows and SHA256 `01a9d51b4389db15c0d61eaa7347482e6a99df421540929abc7f0c55052365eb`, exactly the manifest. Commit `fdb6e032` promoted the G1h supplement on October 29; qualification source `input_qualification_20261101/audit_inputs.py:133–140` also binds 7,105 original and independent-pass rows. The test parameter was left at the earlier generation's count.

**Impact.** The current H200 input-validation module is red on consistent accepted artifacts. It gives a false regression signal and prevents a clean focused verification result. The inspected table hashes and formal values remain valid; the failure does not support changing the measurements or loosening a numerical gate.

**Focused correction.** Update the expected count to the accepted supplemented generation, retaining the independent file SHA and manifest-count assertions. Preserve the earlier generation in historical evidence rather than reverting the delivered table.

**Verification.** Run `tests/unit/test_dsv4_h200_tables.py` alone with the pinned interpreter; all promotion invariants should pass against the unchanged 7,105-row CSV and its matching receipt.

## 4. Verification

### Environment and exact commands

Commands below use these definitions, all from `W`:

```bash
FRONTIER_REVIEW_PY=/data/ycfeng/envs/frontier-py310/bin/python
FRONTIER_REVIEW_OUT=/data/ycfeng/tmp/codex-review/7053879f-03a9-4e51-ae74-c6bbab2891dd/output
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$FRONTIER_REVIEW_OUT" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export FRONTIER_TMP_ROOT="$FRONTIER_REVIEW_OUT"
```

| Check | Exact command / retained command artifact | Expected vs observed | Result |
| --- | --- | --- | --- |
| Working state / revisions | `git status --short`; `git rev-parse HEAD`; `git log -1 --before='2026-10-15T00:00:00+08:00' --format='%H %cI %s' HEAD`; `git rev-list --count 9a650fba..HEAD`; same with `--first-parent` | Clean tracked tree, stated HEAD; observed base/counts as recorded in context. | PASS |
| Heartbeat/simulator diff | `git show --format=fuller --stat efbe09a9`; `git diff --stat abd1e141..HEAD -- frontier data`; `git diff abd1e141..HEAD -- frontier/model_architectures.py` | Only two test files in the port; subsequent simulator changes limited to comments/docstring. Observed exactly that. | PASS |
| Independent raw metric/hash audit | `"$FRONTIER_REVIEW_PY" "$FRONTIER_REVIEW_OUT/audit_artifacts.py"` | Both numerical results reproduce; all consumed/retained hashes match. Observed 886 bindings per case, 130 download files, 13 promotion path/hash pairs, 432 native records per case, and identical 93-ID historical regression sets. `artifact-audit.log/json`. | PASS |
| Warm population, terminal and first-commit binding | `"$FRONTIER_REVIEW_PY" "$FRONTIER_REVIEW_OUT/additional_artifact_checks.py"` | 12/12 errors per case match raw arithmetic, 384 retained each; eight exit-0 replicas; byte-identical receipt in introducing commit. Observed all assertions passing. Initial scratch-only overly strict relative tolerance differed by 1.76e-16 for one near-zero E2E error; corrected to a 1e-12 absolute error tolerance, without changing inputs. | PASS |
| Qualification arithmetic | `"$FRONTIER_REVIEW_PY" "$FRONTIER_REVIEW_OUT/audit_qualification_recheck.py"` | Scratch copy of the retained auditor, changing only its output destination. Observed **906 PASS / 0 FAIL**. Raw NIXL: 90 records, 12,150 reader trials, 80,352 notifications; held-out median maximum error 0.919518%; intercept 0.023652184 ms and bandwidth 390.393499 Gbit/s reproduced. `qualification-recheck.log`, `qualification_audit.json`. Hard-coded prose limitation F003 is outside those assertions. | PASS assertions; F003 prose defect |
| Worker and accuracy gate | `"$FRONTIER_REVIEW_PY" -m pytest tests/unit/test_native_pd_worker.py tests/unit/test_e2e_gate.py -q -p no:cacheprovider --basetemp="$FRONTIER_REVIEW_OUT/pytest-core"` | All 43 worker and 40 gate cases pass. Observed **83 passed in 9.64s**; `core-tests.log`. | PASS |
| Reverse worker order | Collect with `"$FRONTIER_REVIEW_PY" -m pytest tests/unit/test_native_pd_worker.py --collect-only -q -p no:cacheprovider`; execute all 43 node IDs in reverse, exact argv retained in `O/worker-reverse-command.json` | Order-independent success. Observed **43 passed in 6.47s**, exit 0; `worker-reverse.log`. | PASS |
| Module identity probe | `"$FRONTIER_REVIEW_PY" "$FRONTIER_REVIEW_OUT/probe_module_identity.py" original`; same with `fixed` | Original restore drops entry / second stub not visible; fixed preserves identity. Observed `[True, False]` versus `[True, True]`. No HTTP operation performed. | PASS |
| Heartbeat stopped/live timing | `"$FRONTIER_REVIEW_PY" "$FRONTIER_REVIEW_OUT/heartbeat_timing_recheck.py"`. Recorded source calls the existing CPU `exercise()` fixture for stopped with staleness timeout `10**9`, stopped with timeout 1, and live with timeout 1; only the disabled scenario overrides the config value. | Deadline failure / earlier stale-peer failure / clean live completion, no remaining heartbeat threads. Observed **10.020844 s / 1.006931 s / 3.021342 s**, exits 1/1/0 and expected exceptions. `heartbeat-timing-recheck.log/json`. First equivalent inline probe also passed (10.099105 / 1.088617 / 3.035723 s), retained as `heartbeat-timing.log/json`. | PASS |
| Release sanity | `"$FRONTIER_REVIEW_PY" -m pytest tests/unit/test_pdd_public_surface_docs.py tests/unit/test_examples_pdd_scripts.py tests/e2e/test_colocation_examples_layout.py -q -p no:cacheprovider --basetemp="$FRONTIER_REVIEW_OUT/pytest-release"` | Three accepted baseline README failures, no new ID. Observed **3 failed, 17 passed in 0.22s**; `release-tests.log`. | FAIL, accepted baseline |
| Standalone attention coverage | `"$FRONTIER_REVIEW_PY" -m pytest tests/unit/test_v4_attention_coverage.py -q -p no:cacheprovider --basetemp="$FRONTIER_REVIEW_OUT/pytest-coverage-single"` | Collect CPU checks. Observed exit 2, **1 collection error**, helper missing; `coverage-single.log`. | FAIL — F002 |
| Broader DSV4/scheduler checks | 29-module selection listed immediately below; original order and multipass-first order logged separately. | Original: collection abort at coverage, no tests ran. Ordered run: **1 failed, 920 passed, 1 skipped, 7 warnings in 776.93s**, interrupted during an expensive attention-case enumeration (`v4_attention_cases.py:76`). The failure was a reviewer setup error: this initial run omitted `FRONTIER_TMP_ROOT`, so the canonical builder attempted its default read-only temporary root. The focused rerun with the supported override passed all four canonical tests. | INCOMPLETE; initial environment failure resolved |
| Canonical manifest rerun | `"$FRONTIER_REVIEW_PY" -m pytest tests/unit/test_canonical_fixed_manifest.py -q -p no:cacheprovider --basetemp="$FRONTIER_REVIEW_OUT/pytest-canonical-rerun"`, with `FRONTIER_TMP_ROOT` set | Frozen canonical bytes reproduce. Observed **4 passed in 1.67s**, `canonical-rerun.log`; initial broad-run temporary-root failure was reviewer setup, not a source defect. | PASS |
| Standalone phase-heldout collection | `"$FRONTIER_REVIEW_PY" -m pytest tests/unit/test_v4_attention_phase_heldout.py -q -p no:cacheprovider --basetemp="$FRONTIER_REVIEW_OUT/pytest-phase-single"` | Expected collection; observed exit 2, **1 collection error in 0.89s**, `v4_attention_phase_heldout` missing. No tests ran. `phase-single.log`; the earlier three-module selection also aborted in 0.87s (`remaining-tests.log`). | FAIL — F002 |
| Remaining modules with collection workaround | `"$FRONTIER_REVIEW_PY" -m pytest tests/unit/test_v4_attention_multipass_table.py tests/unit/test_v4_attention_phase_heldout.py tests/unit/test_dsv4_h200_tables.py tests/integration/test_vllm_dp_placement_runtime.py -q -p no:cacheprovider --basetemp="$FRONTIER_REVIEW_OUT/pytest-remaining-ordered"` | Expected passing CPU checks after resolving imports by order. Observed **1 failed, 54 passed in 75.66s**, `remaining-ordered.log`; only failure is the stale 4,945-row assertion. This order is a review workaround, not a fix for F002. | FAIL — F004 |
| Focused table failure reproduction | `"$FRONTIER_REVIEW_PY" -m pytest tests/unit/test_v4_attention_multipass_table.py tests/unit/test_v4_attention_phase_heldout.py tests/unit/test_dsv4_h200_tables.py -x -q -p no:cacheprovider --basetemp="$FRONTIER_REVIEW_OUT/pytest-remaining-first-failure"` | Observed **1 failed, 26 passed in 3.11s**, same count mismatch, `remaining-first-failure.log`. File SHA and receipt agree. | FAIL — F004 |
| Retained 15-module baseline/candidate comparison | Parser in `audit_artifacts.py` compares `FAILED`/`ERROR` node-ID sets in `T/h200_mainline/i017_heartbeat_merge_20261111/mainline_native_group_baseline.log` and `mainline_native_group2.log` | Retained totals 32 failed / 442 passed / 62 skipped / 61 errors versus 32 failed / 444 passed / 62 skipped / 61 errors. Exactly 93 IDs on each side, equal sets. These totals were inspected, not freshly executed. | PASS set comparison; full rerun NOT_RUN |
| Full native R5 and new formal simulator executions | NOT_RUN | Retained R5 command used a separate 3.12 interpreter, exit 0, stdout PASS with no failed checks. This review was restricted to 3.10.6 and source-preserving CPU checks. No multi-hour full formal replay or GPU/native run was performed. | NOT_RUN |

The three observed release-sanity IDs are `test_top_level_docs_advertise_supported_pdd_without_upcoming_claims`, `test_public_docs_require_sequential_pdd_release_mode`, and `test_public_docs_state_internal_parallel_pdd_boundary_and_current_evidence`, all in `test_pdd_public_surface_docs.py`. They belong to the accepted release-documentation limitation.

The broader command was `"$FRONTIER_REVIEW_PY" -m pytest` followed by these exact module paths, `-q -p no:cacheprovider --basetemp="$FRONTIER_REVIEW_OUT/pytest-dsv4-ordered"`; multipass was placed first in the second attempt to expose the collection side effect rather than silently changing `PYTHONPATH`:

```text
tests/unit/test_v4_attention_multipass_table.py
tests/unit/test_dsv4_model_config.py
tests/unit/test_dsv4_attention_workload.py
tests/unit/test_dsv4_structural_pd.py
tests/unit/test_dsv4_structural_execution.py
tests/unit/test_dsv4_parameter_memory.py
tests/unit/test_dsv4_precision_metadata.py
tests/unit/test_dsv4_cache_layout.py
tests/unit/test_dsv4_kv_cache_groups.py
tests/unit/test_dsv4_features.py
tests/unit/test_dsv4_predictor.py
tests/unit/test_dsv4_graph_dispatch.py
tests/unit/test_dsv4_attention_runtime.py
tests/unit/test_dsv4_dp_sync.py
tests/unit/test_dsv4_composition.py
tests/unit/test_dsv4_operator_consumer.py
tests/unit/test_dsv4_operator_admission.py
tests/unit/test_dsv4_runner_consumer.py
tests/unit/test_dsv4_runner_tables.py
tests/unit/test_dsv4_kv_block_table_writes.py
tests/unit/test_measured_cc_backend.py
tests/unit/test_v4_measured_collective_consumer.py
tests/unit/test_nixl_read_completion.py
tests/unit/test_native_pair_warmup.py
tests/unit/test_canonical_fixed_manifest.py
tests/unit/test_v4_attention_coverage.py
tests/unit/test_v4_attention_phase_heldout.py
tests/unit/test_dsv4_h200_tables.py
tests/integration/test_vllm_dp_placement_runtime.py
```

In the original attempted order, coverage preceded multipass; all other selected paths were the same. Logs: `O/dsv4-tests.log` and `O/dsv4-ordered.log`. This initial broader run did not export `FRONTIER_TMP_ROOT`; all supplemental runs did. The interruption leaves the expensive coverage enumeration incomplete; the 920 passes are partial coverage, not a full-suite PASS.

## 5. Open questions and next actions

**Confirmed corrections, in dependency order:**

1. Correct current record scope and population statements (F001), using the already verified evidence and decisions. This requires no new measurement or accounting work.
2. Make both attention test modules and their helper imports independent of collection order, then verify each alone and in both orders with multipass (F002).
3. Update the stale table-generation assertion and rerun H200 promotion checks (F004), leaving qualified measurements unchanged.
4. Correct the qualification auditor's Git-provenance statement and its dated summary (F003), without altering the actual promotion receipt or historical verdicts.

**Unverified behavior and evidence needed.** The nine environment-limited native modules and the retained 93 failing/error cases remain unknown under this release interpreter. Resolving them would require the appropriate already provisioned native dependencies/runtime or a separate owner-authorized environment; their historical baseline equality is not proof of correctness. The old `101 passed` suite remains closed as not reproducible, not newly certified here. Full R5 was inspected through retained command/stdout and bindings rather than rerun under another interpreter.

The accepted predictions do not establish unseen attention geometry, pressure/autonomous planner agreement, independent phase memory peaks, arbitrary NIXL concurrency, or device-level graph timing. The qualification ledger explicitly separates these limitations from measured-capacity acceptance (`input_qualification_20261101/ledger.json:318–369`), and the formal reports disclose them. No new requirement is inferred from those limits.

No confirmed active H200 numerical acceptance item remains missing in the inspected evidence. The remaining supported defects affect delivery records and focused CPU verification; the supplemented attention-table test is a confirmed failure, and the broad coverage enumeration remains incompletely executed. No source corrections or external actions were performed in this review.
