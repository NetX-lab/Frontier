# Review context

This review assesses acceptance of the h200_32 host-probe evidence and its DP2 KV-reset host profile, with the five questions in the caller's request as the focus. Review date: October 27, 2026. Caller-provided reviewer execution metadata: model `gpt-6.1-sol-sys-paygo`, reasoning effort `xhigh`.

The working root is `/data/ycfeng/Frontier/.worktrees/step5-preview`. The resolved change scope is **`5e47d47c..0eaf7aad`**, comprising `cdcba47b` and `0eaf7aad`. HEAD is `0eaf7aad5b54c3bedc4cbfd1754ce12f4c691c7b`. The ten changed files contain the replay/estimator acceptance changes, their tests, and the task documentation and profile evidence. Relevant unchanged probe code and Frontier consumers were inspected to assess the changed claims.

The existing tracked modifications under `task_memory/task_2026-10-22_step5_position_routing/` and untracked `.cortexkit/`, `.pi/`, and `tests/integration/step5_pd/attention_census.py` are excluded. The h200_64 result and later changes are also excluded. Git status remained consistent with the initial inspection. Review scripts and generated outputs were confined to the supplied scratch directory; no source, configuration, Git state, or task record was edited.

Path abbreviations used below:

| Name | Absolute path |
| --- | --- |
| `T` | `/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800` |
| `S` | `/data/ycfeng/tmp/codex-review/223fee59-461a-45f2-9eae-3d6cadf74153` |
| `STAGE` | `/data/ycfeng/tmp/step5_d31_workload_20261027/h200_32_hostprobe03/stage` |
| `V` | `/data/ycfeng/tmp/d29_published_sources01/source/vllm` |
| `CLEAN` | `/data/ycfeng/tmp/step5_d31_workload_20261027/h200_32_sweep03/stage/evidence/replay/canonical_summary.json` |
| `EXTERNAL_REPORT` | `/data/ycfeng/tmp/step5_d31_profiles_20261026/host06/kv_block_zeroing_host_h200_32_report.json` |

Applicable ancestor/root `AGENTS.md` instructions were inspected; no applicable `CLAUDE.md` or nested instruction file was found. Relevant contracts include the root guide's vLLM mapping: TP ranks within one DP engine execute the same request/batch work. Task records inspected include current `requirements.md`, `plan.md` item 3, `design.md` §§2–5, 8–9, the requested test report and progress passage, and relevant handoff, host-scheduling, predictor and evidence tickets. D31 supersedes older qualification/process requirements. D26 authorizes canonical P6128/D16 work while preserving client6144/1024. D6 requires idle role boundaries between repetitions. D31.10 prohibits predictive whole-step time lookup tables; a measured component host-time term is permitted.

The staged probe is identical to the repository probe, with SHA-256 `8b4ee896d5f164bc039609bec4aef6193a09f4df1ca4a2636478f42a025de143`. The native run used source commit `02bd453aa31b8bce7218433411e066be9623d546`, before the reviewed acceptance changes; the estimator therefore retrospectively rechecks its saved boundaries. The committed estimator report is byte-identical to `EXTERNAL_REPORT`.

# Assessment

**Verdict: changes_required.** The useful-work evidence in this particular capture is stronger than the estimator's new completeness proof. The captured numerical profile is reproducible, but the generic acceptance guards and the quantitative perturbation acceptance need correction.

| Requested requirement/question | Assessment |
| --- | --- |
| Count only client-completed requests while preserving strict idle boundaries | **Violated in the general guard, F002.** A current busy snapshot is rejected. An ambiguous failed request can still have outstanding role work while the last exported snapshot is idle. The saved hostprobe03 boundaries followed long waits; this review did not observe such a race in that run. |
| Prove useful-step completeness without `end` | **Violated by the implemented proof, F001.** Gap-free prefixes plus equal TP0 token sums do not cover other TP ranks or missing dummy peers needed for useful joins. Independent all-rank accounting supports useful-work completeness of the captured files. |
| Bound perturbation and accept the profile for E | **The stated bound is unsupported, F003.** The `PERTURBED` label is correct. Exploratory E sensitivity work using the labelled profile is defensible; qualified acceptance based on the stated upper bounds is not established. |
| Apply 37.44 ms to canonical prefill steps | **Satisfied for the measured interval and this workload.** Unchunked P6128 work and the 8192-token batch limit give one full request per useful prefill step. No duplicate CPU preparation charge was found. Full prefill host preparation remains unverified because work before the first hook is excluded. |
| Use component costs rather than whole-step lookup tables | **Satisfied.** The profile represents the directly measured host interval; device costs continue through operator predictors. |
| Report accurate numbers and evidence explanations | **Numerical reproduction passes; one explanation fails, F004.** The four unmatched prefill ordinals are inside the final formal window's idle portion, rather than after that window. |
| Relevant regression verification | **98 focused tests passed.** A new native run, controlled perturbation comparison, and complete fidelity matrix were not run by this reviewer. |

The estimator reproduced prefill `37.439762921875` ms from 384 resetting samples and decode `5.110489559035633` ms from 14,144 resetting samples. The independent artifact audit found all 32 rank files gap-free and error-free. Every prefill rank in each DP engine records 392,192 non-dummy tokens per window; every decode rank records 66,496. Role totals agree with the workload: `128 × 6128 = 784384` for P and `128 × (16 + 1023) = 132992` for D. This supports retaining the capture while correcting its acceptance argument.

For prefill, [vllm_v1_kv_allocation.py](/data/ycfeng/Frontier/.worktrees/step5-preview/frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py:32), lines 32–56 and 518–523, selects and schedules the P target. Two 6128-token requests cannot fit in an 8192-token unchunked step. [vllm_v1_engine_replica_scheduler.py](/data/ycfeng/Frontier/.worktrees/step5-preview/frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:453), lines 453–473, attaches and drains newly allocated block zeroing. The predictor charges only batches carrying that metadata at `base_execution_time_predictor.py:132–136`, and the DP agreement uses `max_i(a_i + h_i)` at [forward_start_agreement.py](/data/ycfeng/Frontier/.worktrees/step5-preview/frontier/scheduler/utils/forward_start_agreement.py:112), lines 112–120.

The loader requires CPU-overhead modeling to be disabled at [kv_block_zeroing_host_time.py](/data/ycfeng/Frontier/.worktrees/step5-preview/frontier/execution_time_predictor/kv_block_zeroing_host_time.py:43), lines 43–47. That path returns zero existing CPU overhead at `sklearn_execution_time_predictor.py:7010–7011`; stage events then add the host delay at `replica_stage_schedule_event.py:269–272` and `565–568`. No double counting of input preparation was established. However, `T/test_report_2026-10-27_host_probe_runs.md:129–130` correctly acknowledges omitted host work before the first `.tolist()` return. Thus 37.44 ms is the measured component, not proof of complete prefill host-time modeling.

# Findings

## F001 — P2 medium — TP0 token equality does not prove completeness of the ranks used to estimate h

**Confidence: high.**

**Location and contract:** [kv_zeroing_host_time.py](/data/ycfeng/Frontier/.worktrees/step5-preview/tests/integration/step5_pd/kv_zeroing_host_time.py:138), lines 138–161, accepts any gap-free step prefix and checks token sums only at TP rank 0. Lines 168–175 intersect all ranks' recorded ordinals; missing participants are excluded. Lines 184–193 validate only the retained joins. This contradicts the completeness claim in `T/design.md:211–216` and `T/test_report_2026-10-27_host_probe_runs.md:59–61` that a lost request-bearing step necessarily breaks equality.

**Reachable trigger:** A force-killed nonzero TP-rank worker loses a tail containing useful formal work while TP0 retains its records. A missing dummy tail on one lane can also remove a useful resetting ordinal on another lane from the intersection.

**Expected versus actual:** Missing formal work, or a missing participant required to derive its collective wait, should prevent profile acceptance. The current rules preserve TP0 equality and silently omit the affected ordinal/sample.

**Reproduction:** `S/completeness_repro.py` loads the actual capture and truncates only decode DP0/TP7 immediately before its last request-bearing step, preserving the original sequence numbers. All three acceptance functions return normally:

| Quantity | Original | Truncated TP7 |
| --- | --- | --- |
| TP0 window tokens | `[132992, 132992, 132992]` | Same; accepted |
| Resetting samples | 14,144 | 14,143 |
| Joined/unjoined ordinals | 19,936 / 0 | 19,916 / 20 |
| Decode mean h | 5.110489559035633 ms | 5.110494561196352 ms |
| Minimum 100 samples | Pass | Pass |

This particular deletion changes the mean only slightly, but it demonstrates acceptance after useful evidence is lost. Larger or selective loss can bias the estimate. Missing lines do **not** shift subsequent joins merely because a record disappeared: the explicit collective ordinal is independent of record position. They exclude whole ordinals instead. Interior sequence gaps are caught; truncated suffixes are not. Token sums also cannot distinguish lost work compensated by extra/recomputed work, nor prove request identity.

**Correction:** Validate all TP ranks against the known workload accounting and each other. Require complete participants for every ordinal containing formal useful work, including dummy peers needed to estimate its collective wait. Permit a documented trailing dummy/no-work exemption only after checking that no useful sample depends on those missing participants. If recomputation is allowed, reconcile it explicitly rather than relying on equality of aggregate sums.

**Verification of correction:** Add tests for nonzero-TP useful-tail truncation, a lost dummy peer of a useful reset, and compensated token totals. The first two must reject; a trailing all-dummy tail after useful work may be explicitly exempted. Recheck the original capture with all-rank accounting and confirm the retained profile values remain unchanged.

## F002 — P2 medium — Ambiguous client failures can pass a boundary using an earlier idle snapshot

**Confidence: medium-high.** The acceptance failure is reproduced synthetically; native reachability is inferred from inspected source, not observed in hostprobe03.

**Location and contract:** [canonical_replay.py](/data/ycfeng/Frontier/.worktrees/step5-preview/tests/integration/step5_pd/canonical_replay.py:178), lines 178–192 and 339–343, lowers both role floors by every client failure. `observe_idle`, lines 205–224, fetches an HTTP metrics page but has no post-failure scheduler freshness or drain witness. The estimator repeats the rule at `kv_zeroing_host_time.py:95–102`. D6 and `T/design.md:217–228` require a boundary with no outstanding role work.

**Reachable trigger:** The last client request fails after transmission, while native admission/execution or abort processing is still outstanding and Prometheus still exports the preceding idle scheduler snapshot. Earlier successful requests already satisfy the reduced completion floor.

**Expected versus actual:** A failed request may legitimately require no completion on a role it demonstrably never reached. An ambiguous transport failure must nevertheless be terminal and drained before accepting the boundary. A fresh HTTP response containing an old idle snapshot currently suffices.

**Evidence:** The native Prometheus logger updates running/waiting/KV gauges upon `record(scheduler_stats)` at `V/v1/metrics/loggers.py:1096–1123`. `V/v1/engine/async_llm.py:886–897` records those stats from received engine outputs. The asynchronous engine can queue execution and await its first result at `V/v1/engine/core.py:719–768`; output-associated stats are produced at `V/v1/core/sched/scheduler.py:2510–2524`. The pre-step DP count updates at `core.py:2296–2311` go to the coordinator, through the separate routing at `core.py:1972–1977`; they do not make the scraped gauges a live engine-state query.

`S/idle_boundary_repro.py` serves the previous idle metrics while its fixture models outstanding decode work. For 16 records with one transport failure, P completion16/D15:

- New floor15: `verified=true`.
- Old floor16: rejected.
- A current busy decode snapshot: rejected.

The report's classification of the actual failure also exceeds its evidence. `T/test_report_2026-10-27_host_probe_runs.md:34–36` says the reset happened before the proxy sent to decode. But `STAGE/evidence/replay/proxy_requests.jsonl`, decoded line **8405**, contains `connection_reuseconn`, then **`request_headers_sent`**, then `request_exception`. This proves an ambiguous send/response failure, not server non-admission. The added test at `tests/unit/test_step5_canonical_native.py:159–161` models an immediate server-side502 without counting work; it does not cover delayed admission or stale metrics.

**Impact:** The general replay or retrospective estimator can place an edge while failed-request work remains outstanding, contaminating phase separation. The saved hostprobe03 observations followed 120-second waits, and this review found no evidence that its particular edges cut such work.

**Correction:** Preserve the reduced floor for proven non-admission. For ambiguous failures, require role terminal/abort-and-drain evidence or a scheduler observation causally after the failed request's disposition. Correct the report's unsupported “before sent” classification. Repeated scrapes alone do not establish freshness.

**Verification of correction:** Cover failed requests still visible as busy, stale idle gauges while first execution is pending, delayed admission after client failure, and proven unsent requests. Ambiguous outstanding cases must reject; proven unsent cases must not wait for impossible completions.

## F003 — P2 medium — The observed TPOT shift does not establish the claimed perturbation upper bounds

**Confidence: high.**

**Location and contract:** [test_report_2026-10-27_host_probe_runs.md](/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/test_report_2026-10-27_host_probe_runs.md:111), lines 111–122; `T/plan.md:316–319`; `T/progress.md:1327–1331`. `T/design.md:299–300` requires perturbation within the clean repeat spread; section9's Q6 at line377 says live perturbation is settled. The report instead observes a failed TPOT check and accepts the profile using asserted maximum bias.

**Trigger:** The PERTURBED profile is used as an accepted E input, with the decision justified by “at most”0.53 ms per reset and 0.36 ms E TPOT error.

**Expected versus actual:** A stated upper bound must separate probe bias from job variation and use the correct mapping from local host delays to client TPOT. The evidence supplies neither separation. It establishes a difference between two jobs:

| TPOT quantity | Value |
| --- | --- |
| Probe mean | 53.65910376140362 ms |
| Clean mean | 53.30255372590031 ms |
| Difference | 0.3565500355033109 ms |
| Clean repeat range | 0.20657103583931047 ms |

The report itself acknowledges the unknown between-job offset at lines117–119. A favorable offset could hide probe overhead larger than the observed difference. Therefore `0.357 / 0.676` is a conditional allocation scenario, not a general upper bound.

The denominator is also a local engine-step fraction, not the sensitivity of client TPOT. The captured local resetting fraction is0.6763, but any-lane resetting affects0.8573 of joined DP groups. Weighting by request steps gives0.7920 locally and0.9569 for exposure to either resetting lane. These are descriptive exposure measures, not a causal sensitivity estimate, but they show why the raw local-step fraction cannot identify E's TPOT bias.

**Probe placement:** [host_step_probe.py](/data/ycfeng/Frontier/.worktrees/step5-preview/tests/integration/step5_pd/host_probe_bootstrap/host_step_probe.py:138), lines138–139, resolves/writes completed records before the current execute/reset. On resetting decode steps, the recorded100% previous-device-pending share supports overlap of this work with the subsequent device wait. Hook and wrapper work after `first_sync` sits inside h. However, lines228–232 stamp `launch` **before** creating/recording the CUDA event at lines55–60. That added event work lies outside measured h and still delays native launch. Sampling/end-event instrumentation at lines165–171 adds further work outside the current h interval. The cited local wrapper benchmark does not measure those CUDA events or native contention.

**Impact:** The claims of ≤0.53 ms h bias and ≤0.36 ms/0.7% E TPOT bias are unsupported. The clean comparison and the wrapper benchmark cannot establish that the accepted predictive result changes by no more than those amounts.

**Correction:** Retain the labelled profile for diagnosis or an explicitly exploratory sensitivity rerun, but remove the upper-bound and settled-acceptance claims. Reduce or sparsely sample event instrumentation, or separate the CUDA-event cross-check from the host-stamp measurement. Obtain matched clean/probe repetitions with comparable placement and enough evidence to distinguish instrumentation effects from job variation before treating perturbation as accepted. Do not apply an arbitrary subtraction factor.

**Verification of correction:** Demonstrate accepted perturbation against the stated spread criterion, or document a revised quantitative criterion supported by controlled measurements. Recompute h and reassess E against the clean reference. The current evidence is insufficient to choose a numerical correction.

## F004 — P3 low — The four unmatched prefill ordinals are inside the final window

**Confidence: high.**

**Location and contract:** `T/test_report_2026-10-27_host_probe_runs.md:57–61` and table row86; [design.md](/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/design.md:183), lines183–185. These say all lost tails are after the last formal window and mark an “all joined in windows” expectation PASS despite four unjoined ordinals.

**Evidence and expected versus actual:** The committed JSON's third window is `[1793079023.0581493, 1793079324.1098738]`. Unmatched prefill ordinals **13644–13647** occur on DP1 at wall times approximately `1793079154.9473–1793079154.9597`, inside that window and about169 seconds before its end. They are absent on all DP0 TP ranks.

Concrete evidence: `STAGE/evidence/nodes/gpu-h200-0107.lgcm.sh.istep.fun/host_probe/prefill_gpu-h200-0107.lgcm.sh.istep.fun_2290.jsonl.gz`, decoded lines **32421, 32423, 32425, 32427**. The retained records are all dummy, non-resetting steps. Their all-reduces finish in approximately0.27–0.40 ms, so these are not collectives that complete after the window end.

**Impact:** The numerical join counts are correct, but the stated reason for accepting them is false. These four losses appear harmless to the retained h samples because they follow the last request-bearing prefill work, not because they lie outside the window. The distinction matters when applying the same exemption to future captures.

**Correction:** State “after the last request-bearing work, in the idle portion of the final window.” Change the join criterion to complete evidence for useful ordinals and all required peers, with explicit classification of any dummy-tail exemptions. Do not present four missing ordinals as satisfying an unqualified “all joined” expectation.

**Verification of correction:** Report each unmatched ordinal's timestamp, participant set and dummy/useful status. Confirm no omitted ordinal can remove a useful resetting sample. Update the table and design explanation accordingly; these documentation corrections alone need not change the current numerical profile.

# Verification

All checks used local files and existing CPU environments. No dependency was installed or upgraded. The following listing abbreviates repeated absolute paths using the context definitions above; the environment and arguments match the executed checks.

```bash
D29=/data/ycfeng/envs/frontier-runtime-py310-d29-20261025/bin/python
HTTP=/data/ycfeng/envs/asyncopd-py312/bin/python
```

The pytest commands used `PYTHONDONTWRITEBYTECODE=1`, `PYTHONPATH=$PWD`, `TMPDIR=$S`, and, where applicable, `MPLCONFIGDIR=$S/mpl`. The consumer command used `PYTHONPATH=$PWD:$PWD/tests/unit`.

```bash
$D29 -m pytest tests/unit/test_kv_zeroing_host_time_estimator.py \
  tests/unit/test_step5_host_step_probe.py \
  tests/unit/test_kv_block_zeroing_host_time.py \
  -q -p no:cacheprovider --basetemp=$S/pytest_component

$HTTP -m pytest tests/unit/test_step5_canonical_native.py \
  -q -p no:cacheprovider --basetemp=$S/pytest_http

$D29 -m pytest tests/unit/test_step5_mrv2_forward.py \
  -k kv_block_zeroing_host_time -q -p no:cacheprovider \
  --basetemp=$S/pytest_consumers

$D29 tests/integration/step5_pd/kv_zeroing_host_time.py \
  --case h200_32 --stage-dir $STAGE --clean-summary $CLEAN \
  --job-id exp-1027-124813-737818 --min-samples 100 \
  --output $S/review_profile.json --report $S/review_estimator_report.json

$D29 $S/completeness_repro.py
$HTTP $S/idle_boundary_repro.py
python3 $S/artifact_audit.py

git diff --check 5e47d47c..0eaf7aad
```

| Check | Expected | Observed | Status |
| --- | --- | --- | --- |
| Estimator/probe/profile unit suites | Passing component checks | 42 passed in8.81 s | PASS |
| Canonical HTTP replay suite | Passing replay checks | 46 passed in103.54 s | PASS |
| Focused actual forward consumers | Passing host-delay composition checks | 10 passed,75 deselected in1.57 s | PASS |
| Estimator on captured stage | Reproduce committed profile/report | Exit0; both role values reproduced; recomputed report JSON exactly equal to committed/external report | PASS |
| Nonzero-TP useful-tail fault | Reject incomplete evidence | Accepted; one resetting sample omitted | FAIL, F001 |
| Failed-request/stale-idle fixture | Reject outstanding work | New rule returned `verified=true`; current busy snapshot rejected | FAIL, F002; synthetic fixture |
| Captured useful-work accounting | Consistent all-rank expected totals | 32 files, no errors/gaps; all-rank P/D totals match workload | PASS for this capture |
| Unmatched-tail explanation | Losses outside final window | Four dummy ordinals inside final window | FAIL, F004 |
| Diff whitespace check | No errors | Exit0, no output | PASS |
| New native clean/probe comparison | Establish causal perturbation/bias | Not performed | NOT_RUN |
| Complete fidelity matrix and E/G accuracy comparison | Establish broader fidelity/accuracy | Existing records inspected where relevant; not independently rerun | NOT_RUN |

An initial combined pytest command used the D29 environment with all four files `test_kv_zeroing_host_time_estimator.py`, `test_step5_canonical_native.py`, `test_step5_host_step_probe.py`, and `test_kv_block_zeroing_host_time.py`, with `--basetemp=$S/pytest`. It exited2 during collection: `ModuleNotFoundError: No module named 'aiohttp'` from `test_step5_canonical_native.py:10`. The HTTP suite was subsequently run successfully in the already installed asyncopd environment. This was an environment limitation, not a product test failure.

The effective saved E configuration was inspected at `/data/ycfeng/tmp/d31_e05_run/h200_32_m04_host/metrics/stepfun_ai_step_5_preview/online_serving/h200_32/config.json`: chunked prefill false, `max_tokens_in_batch=8192`, CPU-overhead modeling skipped, dummy mode false, and the committed host profile selected.

# Open questions and next actions

The confirmed findings above do not prove useful data was lost or an edge cut native work in hostprobe03. The independent all-rank totals support useful-work coverage, and its saved observations follow long waits. F001 and F002 concern acceptance rules that can admit invalid evidence in other reachable cases.

The following remain unverified:

| Question | Evidence needed |
| --- | --- |
| How much of the probe/clean TPOT difference is instrumentation rather than job variation? | Matched clean/probe measurements and instrumentation variants isolating host stamps from CUDA-event work. |
| How large is omitted prefill work before the first hook? | Low-overhead timestamps from a preceding native preparation boundary to the first sync, with device-wait overlap distinguished. |
| Is the3.188-second edge clearance larger than the actual server/client clock offset? | A measured offset bound or an equivalent same-clock phase witness. Positive clearance alone is not a measured clock bound. |
| Does using mean local h reproduce the DP maximum's aggregate effect in E/G? | Completed comparison against the clean reference, with perturbation uncertainty kept separate. The design intentionally leaves this to reruns. |

Recommended correction order:

1. Strengthen all-rank/useful-ordinal completeness and failure-aware boundary evidence, with the negative checks in **F001** and **F002**. Re-audit the retained capture; a new GPU run is not automatically necessary for these two corrections.
2. Resolve or explicitly defer quantitative perturbation acceptance as described in **F003**. Treat current E use as exploratory until the accepted bias is supported.
3. Regenerate or retain the profile according to those results, then evaluate E and subsequently G against their clean references.
4. Correct the failure classification, perturbation assertions and dummy-tail/window explanation in the report, design, plan and progress records, referring to **F002–F004**.

Remaining limits: this was a local source/artifact review with CPU checks. It did not submit GPU jobs, observe a live native failure race, measure cross-host clocks, establish a clean h baseline, or independently certify final predictive accuracy.
