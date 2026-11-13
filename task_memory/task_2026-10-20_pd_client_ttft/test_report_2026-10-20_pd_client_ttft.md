# D5 T1 PD client TTFT verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-20 | Completed Step/GLM/DeepSeek comparisons, corrected the MLA fixture, matched the 132 known failing IDs and committed source as 6db2cd9. |
| 2026-10-20 | Recorded semantics, environment and running verification commands. |

## Scope and expected results

CPU-only code change on `feat/step5-pd-structural`, based on `d65a4b9`; no GPU job. D5 native evidence is the parent H800 R1W report's "Execution result on H800" section. Its suggested symbol `decode_first_token_completed_at` is not the first PD output in the inspected code: that timestamp is set at two emitted outputs. The correct existing timestamp is `first_decode_token_completed_at` (one emitted output).

| Metric | Before | After |
| --- | --- | --- |
| `Request.ttft`, CSV `ttft`, JSON `ttft_statistics`, ground-truth `ttft_s`/`ttft_ms` | Arrival to prefill completion | Unchanged, including PD |
| `Request.client_ttft` (seconds) | Not available | Arrival to existing first-output completion; zero until first output |
| PD CSV `client_ttft` (milliseconds) | Not available | D first output minus arrival, including existing transfer, D wait and first D step |
| PD ground-truth `client_ttft_s`/`client_ttft_ms` | Not available | Same interval, explicit seconds/milliseconds |
| Non-PD CSV/ground-truth schema and values | Existing contract | Unchanged; no new fields |

No new timestamp, scheduler branch, event guard, T2 work pricing or client/proxy transport estimate. The first output follows transfer completion because the actual KV-end event owns D queue admission. A test asserts that causality; no metric clamp makes an impossible ordering appear valid.

## Environment

- Python `/data/ycfeng/envs/frontier-py310/bin/python`, 3.10.6.
- Scratch `P=/data/ycfeng/tmp/d5_pd_client_ttft_20261020`.
- Common environment: `PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 TMPDIR=/data/ycfeng/tmp`.
- Long commands: `sudo -n systemd-run --scope -q -p MemoryMax=30G sudo -n -u brainpp env <common environment> <Python> -B ...`.
- Detached baseline worktrees `/data/ycfeng/Frontier/.worktrees/d5-ttft-baseline-8585748` and `d5-ttft-placement-baseline-d65a4b9`. No existing cache is deleted; only the fresh baseline uses `--clean-cache`.
- Simulator source remained unedited while structural drivers compared before/after identities. Candidate records bind `d65a4b9` plus the recorded tracked patch; the new unit test is separately SHA-256-pinned. The only later source edit supplied required `sys_arch` in an existing unit fixture. They are not formal clean-native gate evidence.

## Commands and results

1. `python -B -m pytest tests/unit/test_pd_client_ttft.py -q -p no:cacheprovider`: **9 passed / 3.59 s**. Includes one/three output tokens, offline/online, pre-first state, co-location, preserved historical latency, PD-only CSV/trace units, idempotence, and real Step NIXL reservation/admission ordering.
   Red verification from `d5-ttft-baseline-8585748`: `PYTHONPATH=$PWD python -B -m pytest --import-mode=importlib /data/ycfeng/Frontier/.worktrees/step5-pd-structural/tests/unit/test_pd_client_ttft.py -q -p no:cacheprovider`: **9 failed / 5.48 s**, expected missing `Request.client_ttft`, missing CSV field or `CLIENT_TTFT` enum; `$P/new_tests_red_baseline.log`. This is an intentional negative check, not a failed candidate or an environment/tool failure.
2. `python -B -m pytest tests/unit/test_pd_client_ttft.py tests/unit/test_step5_*.py tests/unit/test_glm53_structural_pd.py tests/unit/test_transfer_metrics_contract.py tests/unit/test_pd_transfer_entities.py tests/unit/test_op_trace_utils.py tests/unit/test_pd_transfer_types_and_configs.py tests/unit/test_e2e_gate.py tests/integration/test_vllm_v1_decode_preemption_runtime.py tests/integration/test_sglang_thinking_round_readmission.py -q -p no:cacheprovider`: **751 passed / 8 skipped in 164.30 s**; PASS, log `$P/affected_unit.log`.
   Final corrected command adds `tests/unit/test_mla_core_native_op_tracing.py` immediately after `tests/unit/test_pd_client_ttft.py` and `-rs`: **764 passed / 8 skipped in 157.60 s**, `$P/affected_unit_fixed.log`. All eight skips explicitly identify unavailable Torch in `test_step5_moe_profiling.py`.
3. From the candidate and `d65a4b9` worktrees: `python -B -u tests/integration/run_step5_structural_pd.py --output-root $P/step_{candidate,baseline}`: **4/4 cases and 26/26 requests PASS**; respective `.log` files. All original request CSV cells match exactly after removing only `client_ttft`; all five other single-case artifacts (system, transfer, stage, EP wave and op precision) and the three boundary-case artifacts are byte-identical by SHA-256. Existing path-normalized configs and validation business values are identical. Receipt `$P/step_original_outputs_comparison.json`. The `d65a4b9` baseline isolates this change from ticket 20 placement. Driver before/after identities pass in all cases.
4. Same two worktrees: `python -B -u tests/integration/run_glm53_structural_pd.py --output-root $P/glm_{candidate,baseline}`: **2/2 cases, 10/10 requests PASS**, respective `.log` files. Old CSV cells are exact, only `client_ttft` is added; stage, transfer, EP wave, system, precision and stage-summary files are byte-identical; path-normalized configs and validation business values match. Receipt `$P/glm_original_outputs_comparison.json`. The raw op trace differs at its first record's pre-existing `meta.timestamp` host wall-clock field; that raw header difference is retained in the receipt, not normalized by any gate change. The remaining **1,587,630 simulated op records per mode** are byte-identical by streaming SHA-256. The initial auxiliary raw-hash assertion exposed this header difference; source inspection and first-line comparison diagnosed it without editing simulator behavior. All ten client intervals independently match first D stage end minus arrival and follow transfer end.
5. `python -B -u tests/e2e/refactor_fidelity/run_matrix.py run --repo-root <baseline/candidate worktree> --label <baseline/candidate> --output-root $P/legacy --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 [--clean-cache only for fresh baseline]`: **77/77 executed on each side**, `$P/legacy_{baseline,candidate}.log`. Baseline is `8585748`.
6. From baseline `8585748` and the candidate: `timeout 3000 python -B -m pytest tests/unit -q -p no:cacheprovider --continue-on-collection-errors -rfE`: baseline **103 failed / 5,552 passed / 70 skipped / 29 errors in 305.63 s**; candidate iteration 1 **114 failed / 5,570 passed / 70 skipped / 29 errors in 339.80 s**. Logs `$P/unit_full_{baseline,candidate}.log`; exact ID comparison `$P/unit_full_comparison.json`. Eleven new failures all came from existing MLA fixture `_build_metrics_store`, whose `SimpleNamespace` omitted required `SimulationConfig.sys_arch`. After all identity-checked drivers finished, supplied `sys_arch="co-location"` there; no production fallback. The final exact same command reports **5,581 passed / 70 skipped / 103 failed / 29 collection errors in 304.20 s**, `$P/unit_full_candidate_fixed.log`. Its sorted **132** failing/error IDs exactly equal the parent's `scratch_evidence/step5_consolidated_20261020/unit_0d2ffda.norm.ids`, the reference named in `test_report_2026-10-20_stream_consolidation.md`, with zero new/missing IDs. Receipt `$P/unit_full_comparison_fixed.json`; old failures are preserved. The ID parser preserves spaces inside parameter values; an initial `\S+` extraction truncated three IDs and was corrected against the raw logs, not the test outcomes.
7. Additional supported DeepSeek-v3 PD baseline `8585748`/candidate runs: exact CLI arrays in `$P/deepseek_legacy_{baseline,candidate}/{offline,online}/command.json`; `python -m frontier.main` on the existing legacy `none` connector, analytical 100 Gbps / 0.1 ms transfer, TP8/DP4/EP32 per role, all 61 model layers, dummy 0.01 ms, eight prompt/three output tokens and a 125 ms online arrival. Generated commands reuse `run_glm53_structural_pd.command` with explicit model/run-ID/connector changes and op tracing disabled; no repository driver edit. **2/2 modes / 2 requests per source PASS**, all original CSV cells and other normalized artifacts exact; new client intervals equal the first D stage end minus arrival. Historical prefill TTFT is **10.35 ms**, client TTFT **20.84497408 ms** (online arithmetic differs only at floating representation). Receipts `$P/deepseek_legacy_runs.json` and `$P/deepseek_original_outputs_comparison.json`. An initial NIXL case is rejected identically by both sources (`NIXL requires an attention family with a structural transfer payload`); logs retained under `$P/deepseek_{baseline,candidate}/offline/`. The supported legacy connector is established by inspected config/event code, not a new fallback. No latent-MLA NIXL or DeepSeek-V4 PD support is claimed.

### Legacy baseline comparison

`python -B tests/e2e/refactor_fidelity/run_matrix.py compare --output-root $P/legacy --baseline-label baseline --candidate-label candidate` retains the unchanged comparator and returns 1, correctly reporting **60 identical / 17 mismatched**, zero failed/missing/uncompared cases. All seventeen differences are PD CSVs with one extra column. The comparator cannot see through a column-count change, so an additional Python check compares every original CSV cell and header after removing only the explicitly approved `client_ttft` column from candidate PD rows. Every non-PD CSV is byte-identical. For all other files, reuse `compare_artifact_directories(..., ignored_artifacts=("request_metrics.csv",))` with the harness's original path substitutions, with **zero differences across all 77 cases**. No original value, tolerance, baseline or gate rule is changed.

Counts: **451 baseline + 451 candidate artifacts (902 total)**; **77/77 original outputs identical**, consisting of 60 fully unchanged cases and 17 additive-only PD cases. Receipt `$P/legacy_original_outputs_comparison.json`; raw report `$P/legacy/comparison.json` and `$P/legacy_raw_comparison.log`. Cache manifests contain **519/519 names, zero differences**. The candidate cache was not deleted/rebuilt, and candidate source was intentionally uncommitted during testing; therefore the aggregate clean-cache/provenance gate is not claimed as PASS. This is exact original-output fidelity evidence, not a clean native accuracy gate.

### Independent client-interval cross-check

The candidate legacy run completed **77/77** cases. Read every CSV with Python
`csv.DictReader` and assert that `client_ttft` is present exactly when
`config.json["sys_arch"] == "pd-disaggregation"`. For each PD row compare it
with independently reconstructed `request_e2e_time - tpot * (outputs - 1)`;
for one output use request E2E directly. Require relative tolerance 1e-12,
absolute tolerance 1e-9 ms, and `ttft <= client_ttft <= request_e2e_time`.
Result: **17 PD cases / 179 requests PASS**, **60 other schemas unchanged** (including PD-AF),
zero errors. Receipt: `$P/legacy_candidate_semantics.json`. This checks interval
endpoints and units; it is not a native accuracy estimate.

Step structural interval receipt `$P/step_candidate_semantics.json` independently
checks all **26/26** requests against E2E/TPOT reconstruction and transfer-end
ordering (online boundary arrivals are the driver's documented request ID ×
0.01 seconds). For each single-request case, the first D stage ledger row ends
at **4,126.395036479868 ms**, exactly equal to `client_ttft`; historical `ttft`
remains **3,703.9999999998804 ms**. The endpoint gap is **422.395036479988 ms**.
These are dummy-timing structural observations, not predicted/actual native
errors. Boundary cases include queue wait and have positive endpoint gaps as
expected. The equality test uses relative 1e-12 / absolute 1e-9 ms tolerance
only for independently reconstructed arithmetic, never for original CSV-cell
fidelity or unchanged artifact hashes.

## Acceptance limits

T1's simulator endpoint code/test line is complete, committed and verified. This advances but does not close the entire aligned-client-metric acceptance of ticket 14 or 17: ticket 14 still needs native metric normalization/pairing and the same-gate client mapping; ticket 17 still needs native lifetime, wire/content qualification and T2. T2 requires ticket 10 measured attention/MoE profiles for the 16–31-token recomputed tail and representative ticket 24 cells. CPU dummy runs prove timing-path structure, not native predictive accuracy, omitted HTTP/CPU terms or CPU performance-budget closure. The inherited known 132 failing IDs remain open. No GPU operation or publication.

## Commit and changed functions

Code commit **6db2cd9**, `fix: expose PD client TTFT at the first D output`, on `feat/step5-pd-structural`; no push. Only these six owned files are in the code commit:

| File | Functions / change |
| --- | --- |
| `frontier/entities/request.py` | `Request.client_ttft`; historical `Request.ttft` docstring clarified; 1,996 lines. |
| `frontier/metrics/constants.py` | `RequestMetricsTimeDistributions.CLIENT_TTFT`. |
| `frontier/metrics/metrics_store.py` | `MetricsStore.__init__` and `_on_request_end`; reuse existing DataSeries/export unit conversion. |
| `tests/unit/test_pd_client_ttft.py` | Four test functions, nine parameterized cases: first output, co-location, exports, real handoff ordering. |
| `tests/unit/test_mla_core_native_op_tracing.py` | `_build_metrics_store` supplies the real config's required architecture. |
| `docs/general/module-size-exceptions.md` | Existing exporter boundary and exact metric semantics/limits. |

`pre_commit_source_checks.json` and `post_commit_source_checks.json` bind all six committed files to tested SHA-256 values and confirm no scheduler, cluster-scheduler or ticket-20 driver edit. AST and whitespace checks PASS; no simulator source changed after the structural/legacy runs. Selected receipts, logs and known/actual failing-ID lists are retained in this task's `evidence/`; large traces remain at their exact `$P` paths. Parent `progress.md` and ticket 14/17 status notes are updated in the requested preview task directory, which is ignored by Git and is not staged as unrelated branch content.

Evidence retention contains **55 selected files / 1,794,413 bytes**, plus
`evidence/retained_evidence_manifest.json`. The snapshot includes source pins,
post-commit checks, complete final/initial unit logs, exact known/final ID lists,
both matrix manifests/results and raw/qualified comparison receipts. All copied
file hashes are verified before the documentation archive is committed.

The historical failing-ID list and the Python-produced list use different sort
orders (`[blocks]` versus `[None]`). An initial auxiliary byte-equality assertion
therefore failed; inspection established the same 132 exact strings and equal
file sizes. Comparing both under the same Python sort passes without rewriting
either retained list. Final verification passes all 55 file hashes, the 132-ID
comparison and seven final PASS receipts.
