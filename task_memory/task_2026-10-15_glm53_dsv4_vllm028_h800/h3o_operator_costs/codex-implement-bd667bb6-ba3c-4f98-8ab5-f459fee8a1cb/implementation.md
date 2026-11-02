**1. Assignment and outcome**

Completed the CPU-only P2 NIXL payload qualification under the unchanged D-P2-PAYLOAD-C criterion. The assigned outcome and checks are finished; final acceptance remains with Claude.

Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.

All writes are under `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/payload_q3/`, including its scratch directory. Frontier, native vLLM, retained evidence, formal inputs, ledger and task records were read only. No replay, observer implementation, GPU/platform action, installation, Git operation, commit or remote write was performed. Pre-existing work was preserved.

The detailed deliverable is [payload_q3/report.md](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/payload_q3/report.md). It includes source citations, the per-group derivation, all six measured case/pass summaries, exact verification commands and proposed ledger wording. Machine-readable results are beside it.

The claim covers retained P2-32/P2-64 on H200: 128 requests per case, 6144 prompt tokens, 1024 decode tokens, homogeneous TP8/PP1, compression/prefix caching/speculation off, engine-step READ completion, explicit measured capacity and operator tracing off. D-P2-PAYLOAD-C permits source-derived, metric-invariant provenance for these cases; it does not establish observed native descriptor arrays or H800 qualification.

| Check | Expected | Actual | Verdict |
| --- | --- | --- | --- |
| Native upper bound | Derive B_max using only unconditional P2 reductions | 46 selected pages × 1,002,240 B = 46,103,040 B per TP rank pair | PASS |
| Production model | Price B_max below every possible posting-step gap | 0.9684023419505275 ms; 2,034/2,034 windows below their gaps; zero crossings | PASS |
| Measurements | Bound within eight-pair grid; every selected trial below both gaps | Grid maximum 817,827,840 B/reader; qualifying coordinate 46,103,040 B; all 1,440 trials below both gaps | PASS |
| Byte sinks | Wire bytes affect compared metrics/events only through READ time | One active pricing sink; optional wire-byte trace sink disabled in all four configs; structural/layout bytes fixed | PASS within retained P2 scope |

**2. Bound and source evidence**

Native source pin: `2cf0a6915ce544dc493a0990f2ea38d81601128a`, rooted at `/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69`. Retained method hashes match the prior native rules. Relevant line-numbered excerpts are preserved in `source_excerpts.md`.

Both retained P2 inputs use the same canonical 128-request trace. Every row has 6144 prompt tokens and 1024 output tokens. Canonical and packaged copies match SHA256 `2d557d7d923b71fbf62bc7e552d1fb547a8ec88c1e54660c0c76cc2de3c49ec2`. Prefix caching is disabled. The native proxy sends the original prompt to P with max_tokens=1, then forwards the unchanged original prompt plus KV transfer parameters to D; it does not append P's output.

The bound follows from D's allocation and selection, without requiring an additional assumption about P pruning time:

- `nixl/pull_scheduler.py:61–66` requests the whole prompt as external KV. V4 states use SlidingWindowMLASpec, so the Mamba N−1 rule does not apply.
- `vllm/v1/core/sched/scheduler.py:918–921,1008–1043` schedules zero new forward tokens during async load and suppresses lookahead. D adds no output-token pages before this READ.
- `single_type_kv_cache_manager.py:250–279,306–323,1095–1098` creates null skipped-prefix entries and allocates the remaining external pages. For window W, block size b and extra retention zero, live pages equal `ceil(6144/b) − floor(max(0,6144−W+1)/b)`.
- `kv_cache_manager.py:99–109` excludes nulls. `nixl/pull_scheduler.py:149–179` applies the exchange clipper. `nixl/base_scheduler.py:128–140,240–284` limits each window group to ceil(W/b)+1 entries.
- `nixl/base_worker.py:2437–2446` unconditionally trims each non-Mamba P group to the corresponding D count. `nixl/pull_worker.py:287–329` passes the resulting descriptors to make_prepped_xfer.

HMA is enabled on the retained path: server commands do not disable it, CUDA and NIXL support it, and the retained model has no attention_chunk_size override. The native default-resolution rules are documented at `vllm/config/vllm.py:1717–1789`.

| Group | Block tokens | Window tokens | Initial P pages | P exchange including nulls | Maximum READ pages | Structural B | Bound wire B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compressed | 256 | Full history | 24 | 24 | 24 | 24,053,760 | 24,053,760 |
| swa_0 | 64 | 128 | 96 | 3 | 2 | 1,647,360 | 2,004,480 |
| swa_1 | 64 | 128 | 96 | 3 | 2 | 1,572,480 | 2,004,480 |
| csa_state_0 | 4 | 8 | 1,536 | 3 | 2 | 1,741,824 | 2,004,480 |
| hca_state_0 | 8 | 128 | 768 | 17 | 16 | 10,506,240 | 16,035,840 |
| Total | | | 2,520 | 50 | 46 | 39,521,664 | 46,103,040 |

Window arithmetic is `96−floor(6017/64)=2` for each SWA group, `1536−floor(6137/4)=2` for CSA, and `768−floor(6017/8)=16` for HCA.

All 96 retained native registrations were rechecked: one packed region, stride 1,002,240 B, capacities 109,934 / 112,914 blocks. `nixl/base_worker.py:960–1018,1395–1455` establishes one full-stride descriptor per packed block. Kernel expansion stays at one on the retained supported block sizes. TP8→TP8 selects the corresponding source rank (`tp_mapping.py:75–114`). The descriptor builder concatenates every group (`base_worker.py:99–130`). No alias deduplication, padding discount or TP division is claimed.

Thus B_max equals the existing source-derived 46-page estimate. It exceeds structural bytes by 6,581,376 B because wire descriptors move the packed stride. Counting initial allocated positions would give 2,525,644,800 B; clipping alone gives 50,112,000 B. The unconditional D null filtering and P suffix matching justify the tighter bound. Executed native-method fixtures use host block objects; they are source evidence, not observed serving arrays.

**3. Timing and measured checks**

The production AnalyticalKVCacheTransferPredictor formula, compression off, is:

`T(B) = 0.0236521842818856 + B × 8 / (390.39349928254796 × 1,000,000)` milliseconds.

The actual production call and independent formula both return 0.9684023419505275 ms for B_max, with difference 0.0 ms.

| Run | Posting windows | Minimum gap ms | Minimum model margin ms | Crossings |
| --- | ---: | ---: | ---: | ---: |
| P2-32 B29 | 510 | 18.72496532622958 | 17.756562984279054 | 0 |
| P2-32 two-fix | 510 | 18.72496532622958 | 17.756562984279054 | 0 |
| P2-64 B29 | 507 | 24.380075962758763 | 23.411673620808237 | 0 |
| P2-64 two-fix | 507 | 24.380075962758763 | 23.411673620808237 | 0 |

`model_windows.csv` contains every one of the 2,034 comparisons.

The raw nixl6144 download matches its manifest. All 90 case/pass records match the retained summary coordinates, and all 12,150 reader trial completions match the Q2 flat CSV exactly. The largest measured eight-pair coordinate is 817,827,840 B/reader. The smallest coordinate at or above B_max is exactly 46,103,040 B/reader.

Both tied cases, t6143_pairs8_inflight1 and t6144_pairs8_inflight1, were included: three passes × eight readers × 30 trials each, totaling 1,440 trials. `selected_reader_trials.csv` reports every trial, including prep/post/completion values and both gap comparisons. None were removed.

| Case | Pass | Trials | Maximum completion ms | Failures against either gap |
| --- | ---: | ---: | ---: | ---: |
| t6143_pairs8_inflight1 | 0 | 240 | 2.124737948179245 | 0 |
| t6143_pairs8_inflight1 | 1 | 240 | 1.0640621185302734 | 0 |
| t6143_pairs8_inflight1 | 2 | 240 | 1.0023228824138641 | 0 |
| t6144_pairs8_inflight1 | 0 | 240 | 3.9585791528224945 | 0 |
| t6144_pairs8_inflight1 | 1 | 240 | 12.554511427879333 | 0 |
| t6144_pairs8_inflight1 | 2 | 240 | 0.980675220489502 | 0 |

The slowest trial is pass 1, reader rank 14, trial 0 of t6144_pairs8_inflight1. Its margins are 6.170453898350246 ms to P2-32 and 11.825564534879430 ms to P2-64. Finite measurements satisfy the recorded criterion; they do not establish a future tail guarantee.

**4. Byte dependencies and invariance**

`frontier/kv_cache_transfer/analytical_kv_cache_transfer_predictor.py:139–149` sends wire bytes only to get_transfer_time and returns structural bytes separately. Target selection/reservation occurs before pricing in `kv_cache_transfer_start_event.py:165–215`; engine-step mode registers the READ instead of scheduling completion at wire end.

`vllm_v1_nixl_reads.py:39–87` finds older READs before posting new ones, so the posting step cannot find its own READ. The later finding step holds the result until its output is consumed. `kv_cache_transfer_end_event.py:12–44,68–136` completes at that output, validates structural ownership and performs D arrival/P release.

The wire-function census has two call sites: predictor line 148 and optional operator metadata at `metrics/op_trace_utils.py:894`. The latter reports wire_bytes, but its guard is false in all four retained configurations. A tracing-enabled claim would fail the only-transfer-time condition and is not made. The optional trace's native_wire_bytes_known=True label is not an observed native witness.

Structural transfer ledgers and total_data_transferred remain 39,521,664 B per request. All four ledgers have 128 completed transfers and peak overlap one. The physical stride also serves memory planning/cache zeroing, but it is a fixed, observed layout input and is not changed by varying the READ wire operand. Substituting B_max for structural kv_cache_size_bytes would alter reporting and fail ownership checks; this qualification preserves that contract.

The only direct wire-end wake is the zero-cost idle DP-forward branch in `vllm_dp_busy_loop.py:353–380`. It is absent: each P2-32 trace has 14,848 DECODE forwards and each P2-64 trace has 24,320, with zero zero-duration forwards in every run.

Every bounded READ therefore finishes before the same first later step. The same step/output reports completion, preserving D readiness, P release, event sequence and request/output timestamps. TTFT, TPOT, E2E and THPT remain invariant by this source/retained-event proof. No new replay comparison was executed.

**5. Changes, verification and handoff**

All 16 new files are beneath the Q3 directory:

| Files | Purpose |
| --- | --- |
| analyze.py | Native-method bound, registration/workload checks, production pricing, all windows/trials, active sink audit and source capture |
| verify.py | Independent arithmetic, input hashes and complete raw/exported trial checks |
| report.md; proposed_ledger_row.json | Detailed report and proposed ledger disposition |
| results.json; registrations.json; source_excerpts.md | Four verdicts, values/environment/hashes, all 96 registrations and cited source excerpts |
| per_group_bound.csv; model_windows.csv | Five-group derivation and all 2,034 timing comparisons |
| selected_reader_trials.csv; selected_pass_summary.csv | All 1,440 selected trials and six pass summaries |
| verification.json; analysis.log; lifecycle_tests.log; verification.log | Exact commands, observed results and successful logs |
| scratch/analysis_initial_failed.log | Preserved initial Q3 schema-error log |

Verification used Python 3.10.6 at `/data/ycfeng/envs/frontier-py310/bin/python`, cwd the working root, PYTHONPATH `$PWD:$PWD/tests/analysis`, PYTHONDONTWRITEBYTECODE=1, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1 and TMPDIR set to Q3 scratch. Each executable check used timeout 120s. Exact full commands and redirects are recorded in verification.json and report.md.

Executed command arguments after that environment prefix:

- `/data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/payload_q3/analyze.py`: expected four fixed checks to pass; observed four PASS, exit 0.
- `/data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_nixl_read_completion.py -q -p no:cacheprovider --basetemp "$PWD/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/payload_q3/scratch/pytest"`: expected READ lifecycle regressions to pass; observed 9 passed in 0.78s, exit 0.
- `/data/ycfeng/envs/frontier-py310/bin/python task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/payload_q3/verify.py`: expected independent arithmetic, immutable inputs and complete exports; observed four audits PASS, exit 0. All 104 hashed inputs/sources remained unchanged; all 2,034 window identities and all 1,440 selected trial identities/timing fields matched.

Final artifact inspection also passed: both scripts parse, JSON results agree, CSV counts are 5/2,034/1,440/6, all 89 report links and cited starting lines resolve, all 16 output files are listed, and the report has no trailing whitespace.

The initial analysis failed with KeyError: byte_count because the retained ledger uses the key bytes. Only the Q3 lookup was corrected; the initial failure log was preserved. Subsequent full checks passed. Native serving/GPU execution, P2 replay, route B observer and H800 qualification were NOT_RUN as scoped.

Proposed ledger disposition: **QUALIFIED for retained P2-32/P2-64 under D-P2-PAYLOAD-C, provenance source-derived, metric-invariant, not observed.** Unconditional allocation/null filtering, clipping and suffix matching bound each rank-pair READ to 46,103,040 B. The adopted H200 price and every selected measured trial finish within both step gaps. Wire bytes reach only READ timing in the retained configurations; structural bytes/layout remain fixed. Actual final descriptor IDs/ranges/aliases remain unobserved. H800, tracing-enabled and other configurations are outside this disposition.

No implementation blocker remains. Claude should inspect the D null-filter/async-no-lookahead/P-suffix proof, both tied measured cases with every slow trial, and the distinction between wire pricing and fixed structural/layout bytes. Rerun analyze.py, verify.py and the nine lifecycle tests if needed for acceptance. Claude owns the ledger/task-record update and commit; this report does not close overall formal P2 qualification or other ticket edges. Route C required 0 GPUh.
