# Review context

This is a review of calibration item 3’s design for the host cost of Step-5 KV side-storage reset. It addresses the native mechanism, Frontier timing composition, allocation trigger, and measurement/analysis proposal. It does not review an implementation patch or authorize execution of the proposed GPU jobs.

Review date: October 27, 2026. Caller-reported reviewer model and effort: `gpt-6.1-sol-sys-paygo`, `xhigh`. No delegation or model override was used.

## Resolved scope and identities

The following prefixes identify exact filesystem roots throughout this report.

| Prefix | Root / identity | Scope |
| --- | --- | --- |
| `F` | `/data/ycfeng/Frontier/.worktrees/step5-preview`, HEAD `4e02c21f57b8c02d62ebd12ebcf2a895e3cf3888` | Current Frontier sources, relevant callers, contracts, and tests |
| `T` | `F/task_memory/task_2026-10-15_step5_preview_h800` | On-disk `design.md`, `requirements.md`, `plan.md`, `test_report_2026-10-26_decode_step_slope.md`, and relevant tickets 19, 22, 28 |
| `N` | `/data/ycfeng/tmp/d29_published_sources01/source/vllm` | Native source snapshot described by the design; package files are directly beneath this root |
| `H` | `/data/ycfeng/Frontier/.worktrees/step5-d31-native-320c208`, HEAD `320c20823797986b3fba26fdb349b67a871d0235` | Reference staging/probe harness and relevant run-kind admission callers |
| `S` | `/data/ycfeng/tmp/codex-review/c85d179e-a5f7-4554-bec4-5fa6066f55db` | Private review scripts only |

The reviewed document is [design.md](</data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/design.md:1>), 146 lines, SHA256 `cabb2d1770953da3f61733b7262bc6fd4907eb4075e904145f11bb786feeafde`. It is ignored by Git. The scope is its actual on-disk contents checked against the named artifacts, **not a base-to-head diff**. No comparison base was requested. The native snapshot has no Git metadata; it was treated as the explicitly named source authority, without claiming a live-image verification.

There were no staged changes. Excluded from the review were the eight unrelated unstaged files beneath `task_memory/task_2026-10-22_step5_position_routing/` (`handoff_prompt.md`, `plan.md`, `progress.md`, `requirements.md`, `summary.md`, and three `resumed_execution/` records), and untracked `.cortexkit/`, `.pi/`, and `tests/integration/step5_pd/attention_census.py`. The reference harness worktree was clean. Source files, task records, configurations, and Git state were left untouched; review scripts were written only under `S`.

## Applicable requirements and assumptions

The applicable `AGENTS.md` files at `/data/ycfeng/Frontier/`, `F`, and `H` have identical contents. Their relevant rules include preserved default simulator fidelity, appropriate regression checks, documented features, and the cleanup/split analysis gate for critical Python modules exceeding 2,000 lines (`F/AGENTS.md:781–800`). No subordinate instruction files were found along the reviewed live source/harness paths. Archived task snapshots were not applied as instructions.

Current D31 requirements supersede conflicting historical task decisions (`T/requirements.md:103–172`; `T/plan.md:135–139`). In particular:

- D31.10 prohibits predictive use of whole-step times measured in a run and looked up by batch composition (`T/requirements.md:161–172`). The caller explicitly permits measured **component host-time terms**. A request-count lookup for a properly identified component is not, by itself, a violation.
- Item 3 requires opt-in measurement/modeling and unchanged default results, with unit checks, dummy smoke, and the 77-case fidelity comparison (`T/plan.md:276–292`).
- Timing inputs require compatible hardware, scale, runtime, role, and graph identities, with costs attributed once (`T/tickets/22-predictor-bundle-cost-replay.md:40,59–61`; `T/tickets/28-h200-measurement-prediction.md:47,56,64`).
- Current H20032 uses P16/D16, TP8/DP2/EP16/PP1; H20064 uses P32/D32, TP8/DP4/EP32/PP1 (`T/plan.md:146–148`).

Native conclusions below are scoped to the requested ordinary Step-5 PDD workload: MTP off, PP1, async scheduling, DP coordination enabled, and the V2 GPU runner. Startup/capture phases, no-forward calls, and process restarts must be distinguished from workload forwards. Other architectures and general DeepSeek-V4 calibration are outside scope; its existing zeroing consumer was inspected to answer the default-timing question.

# Assessment

**Verdict: changes_required.** The mechanism motivating a component cost is supported, but the design cannot yet produce or compose that cost faithfully. The most consequential issues are the late Frontier DP barrier, the missing real-forward end hook, and the unsupported attribution of an inter-forward gap difference to host time.

| Requested requirement | Assessment | Evidence / disposition |
| --- | --- | --- |
| Sparse-summary reset can break async host/device overlap | Satisfied with a stream-scope qualification | The first CUDA `.tolist()` requires its copy to complete behind preceding main-stream work; retained traces also show synchronization inside indexed writes. This is not a global completion boundary for every CUDA stream. |
| Gloo DP coordination propagates a delayed lane before forward | Satisfied for the scoped runtime | Blocking CPU-group all-reduce precedes input preparation/model execution. |
| Accurate sparse reset multiplicity | Violated | 23 sparse-layer hooks produce 46 indexed writes, not 46 hooks; F005. |
| Add preparation once and remove it from final tail overhead | Satisfied at the local timing-algebra level | Existing overhead/stage ownership and both final-timing helpers support this; actual new-field integration remains unimplemented. |
| Delay every DP lane before first attention | Violated | Existing `sync_entry` reconciles timestamps after attention completion; F001. |
| Step-5 allocation drain and NIXL exclusion | Sound for the stated grouped, canonical reservation scope; implementation unverified | Correct attach/drain seam found. The proposed receiving-reservation suppression is necessary. |
| No existing Step-5 device-zeroing charge when only metadata is enabled | Supported by current consumers | Only DeepSeek-V4 consumes `cache_block_zeroing` for execution timing. Full default regression is NOT_RUN. |
| Correct PDD role ownership, including D-side prompt tail | Appropriate design intent; implementation unverified | Select the server/cluster role, so D-tail prefill work uses the DECODE-role profile. Add an explicit check. |
| Paired events for real workload forwards | Violated | `forward_end` is absent from ordinary `sample_tokens`; F002. |
| Unbiased component host-time extraction and required coverage | Violated | Gap contains other work; reset/non-reset groups are not matched, and some controls are absent; F003. |
| Profile compatibility across E and G | Violated | DP size is recorded but not validated; proposed collection is DP2 while G is DP4; F004. |
| D31.10 compliance | Permissible model form; component attribution unverified | This is not categorically a forbidden whole-step table, but F003 prevents accepting its extracted value as the claimed host component. |
| Diagnostic probe never becomes a reference | Satisfied as a design requirement; harness implementation unverified | Existing clean-run rejection is a suitable foundation; new-kind admission must also cover replay/formal-release callers. |

## Design points confirmed with evidence

1. **Step-5 side storage requires reset without requiring full base-page zeroing.** `N/models/step5/nvidia/model.py:1243–1247` sets `kv_cache_requires_zeroing` and constructs the summary cache. `N/v1/kv_cache_interface.py:1792–1806` separately defines base-page zeroing as Mamba/mixed-KV-precision safety and includes side-storage metadata in overall zeroing need. Section 2’s base-versus-side-storage distinction is correct for the stated configuration.

2. **The native wait mechanism is reachable before DP coordination.** `N/v1/worker/gpu/model_runner.py:1757–1764` updates requests before dispatch; `:1273–1280` calls the zeroer. The zeroer makes one flattened asynchronous H2D copy of grouped IDs (`N/v1/worker/utils.py:580–589`), then passes GPU slices to each registered hook (`:591–598`). The hook calls `.tolist()` through `_ids` (`N/models/step5/nvidia/sparse_summary_cache.py:486–488,507–509`). Thus the first host conversion is ordered behind the ID upload and prior work on the relevant stream. Previous main-stream work can include sampling/postprocessing, not just the prior model forward.

3. **Python-list indexing is also a synchronization source in the retained runtime evidence.** In one real decode-only reset span, the private trace check found 46 `aten::index_put_` calls and exactly one nested `cudaStreamSynchronize` in each. This confirms that part of the design’s mechanism in the retained profiled run; it does not independently establish an unprofiled cost.

4. **The DP delay propagation is real.** `N/v1/worker/gpu/dp_utils.py:65–77` uses `get_dp_group().cpu_group` and calls blocking `dist.all_reduce`. CPU groups use gloo (`N/distributed/parallel_state.py:506–510`); DP groups span DP ranks at fixed other rank coordinates (`:2271–2286`). The runner calls dispatch at `:1810–1825`, before input preparation (`:1833–1842`) and forward (`:2022–2033`). A slow participant delays its peers’ departure from that collective.

5. **Native NIXL suppression is accurately described when scoped to external allocations in loaded groups.** `N/v1/core/sched/scheduler.py:1297–1300` obtains loaded group IDs. `N/v1/core/kv_cache_coordinator.py:336–342` applies `record_for_zeroing=False` only to those groups’ external-token allocations. The NIXL connector returns its transfer groups (`N/distributed/kv_transfer/kv_connector/v1/nixl/connector.py:170–171`). This is not a blanket exemption for all subsequent local allocations of a receiving request.

6. **The exact Frontier attach/drain location is correct.** `_create_batch` begins at `F/frontier/scheduler/replica_scheduler/vllm_v1_engine_replica_scheduler.py:429`; grouped IDs are flattened at `:453–457`, attached at `:458–461`, and cleared at `:473`. Current Step-5 capacity does not yet enable `zero_new_blocks` (`F/frontier/attention/step5_memory.py:283–287`); section 3.1 should be read as a proposed change.

7. **Enabling that flag currently changes metadata rather than pricing a Step-5 device operator.** The allocator gates returned zero-required IDs at `F/frontier/kv_cache/base_kv_cache_manager.py:374–381`. Repository-wide consumer search found the batch default, scheduler attachment, and the DeepSeek-V4 timing consumer at `F/frontier/execution_time_predictor/deepseek_v4.py:267–279`. No current Step-5 timing consumer prices those IDs. The proposed NIXL suppression is still needed because `_allocate_request` presently records grouped assignments even for its recognized read-reservation case (`F/frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py:694–739`).

8. **Queue capacity 2 does not imply overlapping execution on one stage.** `F/frontier/scheduler/replica_stage_scheduler/replica_stage_schduler.py:326–327` prevents a busy stage from popping another batch. `F/frontier/events/batch_stage_end_event.py:120–149` releases it and schedules queued work at completion. Batches can be admitted in advance; the next same-stage device execution follows the previous stage end.

9. **Local preparation/tail arithmetic supports once-only accounting.** `F/frontier/entities/time_components.py:736–756` provides preparation and simulated overhead totals. `F/frontier/entities/stage_execution_time.py:129–134,382–398` owns stage overhead once. `F/frontier/scheduler/utils/collective_timing.py:102,131–135` subtracts preparation from the final tail. The isolated check moved a 40 ms P/D completion to 49 ms with 9 ms preparation while leaving tail CPU overhead zero. F001 concerns cross-lane event ordering, not this subtraction.

10. **Collective-ordinal joining is reasonable with explicit boundaries.** Ordinary target-runner dispatches and runtime dummy forwards participate in the same DP collective sequence. Zero-token no-forward calls return before dispatch (`N/v1/worker/gpu/model_runner.py:1770–1775`); `_dummy_run` enters `execute_model` (`:902–911`), and idle runtime engines invoke dummy execution (`N/v1/engine/core.py:2348–2358`). However, a generic dispatch invocation is not always one collective: DP1/reused synchronization returns early (`N/v1/worker/gpu/dp_utils.py:290–310`), and coordination can be skipped (`:74–75`). Count actual target-runner collectives, key by role/process-group/rank coordinates and process generation, include runtime dummies, and exclude startup/error/missing-record phases. Unqualified bare call-index alignment is not established by the retained evidence.

# Findings

## F001 — P1 high — The proposed DP maximum is resolved after first attention has already run

**Confidence: high.**

**Design location:** [§3.2, lines 64–74](</data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/design.md:64>). Relevant implementation: [replica_stage_schedule_event.py, line 269](</data/ycfeng/Frontier/.worktrees/step5-preview/frontier/events/replica_stage_schedule_event.py:269>), lines 269–317 and 563–610; [sync_entry.py, line 100](</data/ycfeng/Frontier/.worktrees/step5-preview/frontier/scheduler/utils/sync_entry.py:100>), lines 100–119 and 154–162.

**Violated contract:** Native blocking DP coordination delays every participant before its forward. The design claims that placing the new term in local preparation and relying on the existing synchronization already reproduces that behavior.

**Trigger:** A DP2 Step-5 P or D forward where lane A resets and has shorter first-attention work than lane B, which does not reset. Unequal attention durations are reachable with unequal local request/context workloads; dummy participation is another relevant case.

**Expected versus actual:** Native-compatible composition starts all lanes’ first attention at the common prepared start. For a common initial time, first pre-MoE readiness should be `max(preparation_i) + max(attention_i)`. Current events instead schedule attention using each lane’s local start, then reconcile at `max(preparation_i + attention_i)`.

The relevant event expression is:

```python
DecodeSyncEvent(
    batch._forward_launch_start_time + attention_time,
```

The prefill branch uses the same expression. At the later room, `sync_time` is already the maximum attention-completion arrival (`sync_entry.py:110`). Lines 115–119 overwrite stored launch timestamps, but do not move earlier attention work. `F/frontier/scheduler/utils/ep_wave_schedule.py:101–107,174–175,188–189` starts downstream work from the supplied time; it does not repair the missing delay. Final-timing helpers likewise consume the actual sync time rather than replaying first attention.

**Reproduction:** The scratch check executes the actual `enter_layer_sync` function with lightweight scheduler/batch stand-ins and chronologically consistent arrivals. Preparation `[9, 0]` ms and first attention `[1, 5]` ms yield **10 ms** pre-MoE readiness, while the pre-forward DP barrier requires **14 ms**. Both stored launch timestamps nevertheless become 9 ms. With the other CPU terms disabled, there is no tail correction that restores the missing 4 ms.

**Impact:** Part of the intended host stall is hidden by work that native peers cannot launch yet. The model understates completion time and permits first-attention execution during another engine’s host stall. This affects both requested PDD roles and can undermine calibration even if timestamp-only assertions pass. Dummies also start their first work from the original time (`F/frontier/scheduler/utils/dp_dummy_forward.py:76–83`).

**Focused correction:** Resolve common prepared readiness before scheduling any first-attention event. The existing Step forward-start group is a suitable seam: `F/frontier/scheduler/utils/forward_start_agreement.py:83–115` already holds all admitted useful/dummy lanes and resumes them before prediction. Extend that protocol, or an equivalent pre-attention barrier, to establish shared launch readiness. Preserve original stage starts and the existing local tail subtraction.

**Verification:** Assert actual first-attention event times and pre-MoE readiness, not only `_forward_launch_start_time`. Cover the 9/0 plus 1/5 ms counterexample, useful/useful and useful/dummy lanes, PREFILL and DECODE, and unchanged zero-profile behavior.

## F002 — P1 high — The proposed forward-end hook is not called for real workload forwards

**Confidence: high.**

**Design location:** [§4, lines 103–111](</data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/design.md:103>). Relevant native locations: [model_runner.py, line 2022](</data/ycfeng/tmp/d29_published_sources01/source/vllm/v1/worker/gpu/model_runner.py:2022>), lines 2019–2033, 2115–2166, and 927; `N/v1/worker/gpu/async_utils.py:48–101`.

**Violated contract:** The probe needs one valid device start/end pair for each real forward. The design states that `StepTimingCollector.forward_end` is called at the start of sampling.

**Trigger:** An ordinary canonical P or D workload forward with MTP off.

**Expected versus actual:** `execute_model` calls `forward_start` at line 2022. Ordinary `sample_tokens` has no `forward_end` call. The only such call in the entire runner is line 927 inside `_dummy_run`, on the last PP rank. AST enumeration of the actual source returned:

```text
execute_model.forward_start: [2022]
execute_model.forward_end: []
sample_tokens.forward_end: []
_dummy_run.forward_end: [927]
```

Patching the two collector methods as proposed therefore produces real starts without real ends. Pairing them with later dummy ends would also cross forward identities.

**Impact:** The principal workload observations cannot provide the claimed forward duration or preceding gap. The proposed profile cannot be generated correctly from these hooks.

**Focused correction:** Add a passive wrapper/event at an actual real-forward completion seam, paired with saved execution identity. For model-only timing, record after model work is enqueued; if using sampling entry, explicitly define inclusion of connector completion work. Record independent probe events rather than enabling native `StepTimingCollector.collect()`: its calls are inert outside collection, and collection itself synchronizes on exit (`async_utils.py:78`). Keep real, runtime-dummy, startup/capture, failed, and no-forward records distinct.

CUDA events around normal FULL replay can bracket stream execution because replay occurs at `model_runner.py:2033` after the start hook, and `N/v1/worker/gpu/cudagraph_utils.py:550–563` calls graph replay. Their scope still needs definition: the existing start precedes `kv_connector.pre_forward` (`model_runner.py:2030–2031,2055`), and `finish_forward` runs at `:2073`. Do not introduce probe events into startup graph capture or reuse one timing pair across queued forwards.

**Verification:** Source-level hook tests must exercise ordinary `execute_model` followed by `sample_tokens`, as well as runtime dummy and no-forward calls. Then a short authorized diagnostic run should show one completed pair per real forward, graph-mode identity, no cross-pairing, and event reads only after successful `query()`. This review did not run that GPU check.

## F003 — P1 high — The gap subtraction does not isolate the claimed host component and lacks required controls

**Confidence: high for the attribution/coverage defect; the magnitude and direction of bias in an unprofiled run remain unverified.**

**Design location:** [§3.2, lines 61–63](</data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/design.md:61>), §4 lines 109–121, and §2 lines 39–42.

**Violated contract:** Item 3 permits a measured component host term. The design proposes exporting `median(reset inter-forward gap) − median(non-reset inter-forward gap)` as that component, without establishing a matched baseline or handling absent controls.

**Trigger:** Normal prefill; the reported high-request decode buckets where every observed step resets; or decode cohorts whose reset status correlates with other scheduling/device work.

**Expected versus actual:** A host component should measure exposed host work on the reset-induced critical path, with prior device wait and separately owned device/communication work identified. The proposed event interval is a broader residual:

- After a model-forward end at sampling entry, sampling occurs at `N/v1/worker/gpu/model_runner.py:2164–2166`, output copy is started at `:2210–2219`, and GPU request-state postprocessing follows at `:2239–2250`.
- Output copies run on a separate stream waiting on the main stream (`N/v1/worker/gpu/async_utils.py:137–171`); output consumption itself waits at `:173–174`. Main-stream timing events do not identify completion or stalls on that copy stream.
- Before the next forward, reset ID H2D and indexed reset writes execute (`N/v1/worker/utils.py:585–597`; `N/models/step5/nvidia/sparse_summary_cache.py:347–355`), followed by input preparation and associated copies (`model_runner.py:1833–1842`).

Consequently, `end[k-1] → start[k]` is not simply GPU idle or host reset time. It can include previous-step device work, current reset/preparation device work, connector/output effects, and idle time. Conditional subtraction does not remove work that systematically differs between reset and non-reset classes.

The design itself says every prefill step resets and all observed 40–45-request steps reset. Its estimator has no non-reset comparison group for those required entries. This is a coverage problem even if a perfectly paired forward-end hook is added.

**Evidence and inference:** The retained diagnostic analysis reports a device-busy median of 48.90 ms in the sync class versus 44.87 ms in the async class in the 20–24-request bucket. Those are coarse five-request buckets, and classification in `zeroing_split.py:35–41` uses a long-copy proxy. This is evidence that the diagnostic groups are not otherwise identical, not a quantified estimate of unprofiled bias. New requests, D-side prompt tails, graph mode/padding, context, and output-queue state are plausible confounders that the proposal does not control.

A simple counterexample demonstrates non-identification: a true added host cost of 2 ms, with reset-class baseline gaps of 9 ms and control baseline gaps of 1 ms, produces an estimate of **10 ms**. No property of conditional medians prevents this. With no controls, the mathematical statistic is undefined; the scratch check’s empty median raises `StatisticsError`.

**Impact:** A profile may absorb scheduling, sampling, transfer, or device costs into the host term, or fail to produce required prefill/high-count entries. Predictive improvements would not establish that the component was modeled correctly. The profile form is allowed by D31.10 in principle; this extraction does not establish the component attribution claimed in §3.3.

**Focused correction:** Define and instrument the synchronization boundary and subsequent exposed host intervals directly. Distinguish the first hook’s prior-stream wait from remaining reset/preparation work, and identify reset/preparation device activity and connector/output ownership. Use matched phase, request/context, graph, admission, and queue-state cohorts as diagnostics. For roles/buckets without controls, use a controlled component probe or the plan’s permitted reset microbenchmark with explicit composition, or mark them UNVERIFIED and refuse profile emission. Do not silently infer, substitute zero, or extrapolate missing counterfactuals.

**Verification:** Require known-cost synthetic cases with unequal baseline classes and absent controls; ensure the estimator does not emit a spurious component. In native evidence, retain the host split, device-boundary definitions, matched-cohort coverage, sample counts, uncertainty, and perturbation comparison. Demonstrate coverage for PREFILL and high-count DECODE before producing a predictive profile.

## F004 — P2 medium — The loader omits DP validation although the profile includes DP coordination and is proposed for both scales

**Confidence: high.**

**Design location:** [§3.3, lines 80–85](</data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/design.md:80>), §4 lines 122–123, and §5 lines 136–138.

**Violated contract:** Timing inputs must have compatible scale/topology/runtime identities (`T/tickets/22-predictor-bundle-cost-replay.md:40,59`; `T/tickets/28-h200-measurement-prediction.md:47,56`).

**Trigger:** Load the proposed H20032/DP2 profile into step G’s H20064/DP4 run. Model, device, and role all match the stated loader checks.

**Expected versus actual:** The schema records DP size, but the loader validates only model profile, device, and role. The proposed interval expressly includes gloo all-reduce and DP delay propagation. Its topology compatibility therefore cannot be inferred from GPU model and role alone. Current scales differ in DP and EP (`T/plan.md:146–148`), and native coordination explicitly depends on the DP process group (`N/v1/worker/gpu/dp_utils.py:65–77,312–325`).

**Impact:** A topology-incompatible measured residual can be accepted for G without an error. The size of the numerical error is unmeasured; the confirmed defect is the missing validation and unsupported reuse path.

**Focused correction:** Validate recorded DP size and the profile’s supported TP/PP, graph, and async runtime scope. Acquire a compatible DP4 profile for G when the exported value includes group coordination. Alternatively, define and justify a portable purely local reset component and compose DP coordination separately; do not assume portability of the current measured interval.

**Verification:** Loading a DP2 profile into DP4 must fail clearly unless explicit, evidenced compatibility is part of the schema. Add graph/runtime mismatch tests and a successful same-scope load. E and G acceptance should identify their own admitted profile identities.

## F005 — P3 low — The mechanism counts indexed writes as sparse-layer reset hooks

**Confidence: high.**

**Design location:** [§2, lines 27–33](</data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/design.md:27>). The same count appears in `T/plan.md:279–280` and `T/test_report_2026-10-26_decode_step_slope.md:324–326`.

**Violated contract:** The source explanation and probe’s host split must describe the actual component multiplicity.

**Trigger:** The named MTP-off Step-5 model and retained H20032 device-timeline run.

**Expected versus actual:** Both `F/data/config/models/stepfun-ai__Step-5-Preview.json:45,154–247,682–684` and `/data/ycfeng/tmp/step5_d31_devtl_20261026/h200_32_devtl01/reference_inputs/config.json` specify 92 backbone layers: **69 sliding-attention and 23 full/sparse-attention layers**. Native attention honors serialized `layer_types`, rather than its alternating fallback (`N/models/step5/nvidia/model.py:895–914`); DSA selection and summary construction occur at `:1018–1021,1243–1247`. Hooks are registered once per unique side-storage layer and invoked once each for nonempty owner-group IDs (`N/v1/worker/utils.py:353–364,591–597`).

Each reset performs two indexed writes:

```python
self.mean[list(blocks)] = 0
self.last_slot[list(blocks)] = -1
```

These occur at `N/models/step5/nvidia/sparse_summary_cache.py:353–355`. Thus **23 hooks / GPU `.tolist()` conversions produce 46 indexed writes**. The diagnostic script counts `aten::index_put_` (`/data/ycfeng/tmp/d31_diag/hostgap/zeroing_split.py:36`), which explains the reported 46.

**Impact:** The causal account is wrong by a factor of two for hook/conversion count and can mislead probe expectations or later cost scaling. A directly measured scalar need not inherit that factor, so this is lower priority than the composition and measurement defects.

**Focused correction:** State 23 sparse-layer hooks and 46 indexed writes for the named MTP-off model. Derive hook count from model/runtime inventory or hook records, rather than dividing or reinterpreting an operator trace count without checking the source.

**Verification:** Assert the selected model’s sparse-layer count and compare it with registered/invoked hook count. Check indexed-write count separately. Do not equate hook count, D2H conversion count, and indexed-write count.

# Verification

## Environment and execution boundaries

Checks ran from `F` with `/usr/bin/python3`, Python 3.10.6, and `PYTHONDONTWRITEBYTECODE=1`. The local environment has Torch, NumPy, pandas, and sklearn; pytest, fasteners, ddsketch, and plotly are absent. No software was installed or upgraded. No GPU workload, native server, CUDA timing experiment, simulator regression, or external publication was executed.

The scratch helper compiles selected actual source classes/functions with AST extraction and uses lightweight stand-ins. It does not import or exercise the complete simulator. Its preparation test uses the existing `schedule_time` field as a 9 ms preparation stand-in; it is not a test of the proposed, unimplemented new field.

## Commands and results

Scope/identity checks:

```bash
git rev-parse HEAD
git status --short
git diff --cached --name-only
git check-ignore task_memory/task_2026-10-15_step5_preview_h800/design.md
sha256sum task_memory/task_2026-10-15_step5_preview_h800/design.md
git -C /data/ycfeng/Frontier/.worktrees/step5-d31-native-320c208 rev-parse HEAD
git -C /data/ycfeng/Frontier/.worktrees/step5-d31-native-320c208 status --short
sha256sum AGENTS.md /data/ycfeng/Frontier/AGENTS.md /data/ycfeng/Frontier/.worktrees/step5-d31-native-320c208/AGENTS.md
```

**PASS:** Revisions and exclusions are recorded above; no staged changes; reference harness clean; the three instruction files match. Final design hash and Git status matched the initial review state.

Environment availability check:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -c 'import importlib.util, sys; print(sys.executable); print(sys.version.split()[0]); print({name: importlib.util.find_spec(name) is not None for name in ["torch", "pytest", "numpy", "pandas", "sklearn", "fasteners", "ddsketch", "plotly"]})'
```

**PASS:** Observed availability matches the environment description. This checked module discovery, not full simulator imports.

Actual-source helper checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/tmp/codex-review/c85d179e-a5f7-4554-bec4-5fa6066f55db/review_checks.py
```

The script exited 0. The following statuses assess the reviewed assumptions, rather than claiming a failing pytest suite.

| Check | Expected under design claim | Observed | Status |
| --- | --- | --- | --- |
| Real forward event pairing | Real starts and ends | Start at 2022; no real end; dummy end at 927 | FAIL — F002 |
| P/D preparation accounting | 9 ms preparation added once | 40 → 49 ms completion; tail CPU overhead remains zero | PASS — local algebra |
| DP first-attention composition | 14 ms ready for prep 9/0 and attention 1/5 | 10 ms ready; both stored launches become 9 ms | FAIL — F001 |
| Conditional-median identification | Recover true 2 ms component despite class-baseline issue, or reject unsupported estimate | Proposed subtraction gives 10 ms | FAIL — F003 |
| Empty control class | Defined handling without inventing a value | Median is undefined; caught `StatisticsError: no median for empty data` | FAIL — missing design coverage |
| Layer/hook multiplicity | Claimed 46 sparse hooks | Model has 23 sparse layers and two writes per reset | FAIL — F005 |

Retained diagnostic trace analyses were run read-only:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/tmp/d31_diag/hostgap/zeroing_split.py /data/ycfeng/tmp/step5_d31_devtl_20261026/h200_32_devtl01/stage/evidence/nodes/*/torch_trace/load_t0*_tp0_*.pt.trace.json.gz
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/tmp/d31_diag/hostgap/critical_path.py /data/ycfeng/tmp/step5_d31_devtl_20261026/h200_32_devtl01/stage/evidence/nodes/*/torch_trace/load_t0*_tp0_*.pt.trace.json.gz
```

**PASS — reproduction of retained diagnostics, not unprofiled validation.** The split reproduced 36 async / 16 sync observations in bucket 5, 23 / 33 in bucket 20, and sync-only reported observations in buckets 40 and 45. Bucket 20’s busy medians were 44.87 / 48.90 ms. These are five-request buckets and long-copy classifications. The critical-path script reproduced approximately 8.70–9.69 ms post-copy-to-first-graph-kernel intervals at the reported request counts. Its interval and the profiler’s perturbation do not establish a pure, unprofiled host component.

A focused retained-span inspection ran:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/tmp/codex-review/c85d179e-a5f7-4554-bec4-5fa6066f55db/inspect_reset_trace.py
```

**PASS:** In `execute_context_0(0)_generation_9(9)` of `.../nodes/gpu-h200-0722.lgcm.sh.istep.fun/torch_trace/load_t016_dp0_pp0_tp0_dcp0_ep0_rank0.1793053994241221858.pt.trace.json.gz`, it found 46 pre-dispatch `aten::index_put_` calls, each with one nested `cudaStreamSynchronize`; median indexed-write CPU span was 0.0573395 ms. This confirms synchronization in that profiled runtime, without assigning that number as a predictive cost.

Relevant caller/consumer searches included:

```bash
rg -n 'cache_block_zeroing' frontier
rg -n 'def schedule_layer_wave|schedule_layer_wave\(' frontier/scheduler
rg -n 'prepare_decode_final_timing|prepare_prefill_final_timing|forward_preparation_time|requires_forward_start_agreement' frontier/scheduler frontier/events frontier/execution_time_predictor
```

**PASS:** Located the drain, the sole existing timing consumer, pre-attention agreement, downstream wave scheduling, and final-timing callers cited above. Source ranges were inspected with `nl -ba`.

## Checks not run and coverage limits

| Check | Status | Limit |
| --- | --- | --- |
| Proposed host-profile loader and new-field integration tests | NOT_RUN | Design only; implementation absent |
| Full simulator unit/integration suite | NOT_RUN | Required local dependencies absent; no installation authorized |
| Dummy smoke and 77-case fidelity comparison | NOT_RUN | No implementation patch; full runtime dependencies absent |
| Unprofiled host-step probe and perturbation comparison | NOT_RUN | Requires native GPU execution outside review scope |
| Clean E/G reruns using admitted host profiles | NOT_RUN | No validated component profiles exist in the reviewed proposal |
| Live DP2/DP4 collective-ordinal and event-pair coverage | NOT_RUN | Static reasoning supports a scoped scheme; no live probe records |

# Open questions and next actions

## Unverified questions, separate from confirmed findings

- **Unprofiled cost and causality magnitude:** The reset wait mechanism is supported, but 8.7–9.7 ms comes from profiler-perturbed residual intervals. A component probe with explicit device/host boundaries and a perturbation comparison is needed before using that magnitude.
- **Collective joining across startup and lifecycle:** Actual blocking collective ordinals provide a sound identity within a process group, but the proposal’s bare utility-call counter is not proven aligned across startup/capture, skipped coordination, missing records, or restarts. Retain group/generation identity and reconcile counts, dummy participation, and gaps before joining.
- **D-tail role and trigger:** Canonical NIXL reservation supplies complete-prefix progress and zero new tokens (`F/frontier/scheduler/replica_scheduler/vllm_v1_kv_allocation.py:281–291`); later D-side local tail work can allocate new pages. Demonstrate that imported-prefix IDs are excluded, locally opened pages remain eligible, and the term uses the DECODE server profile even when the local batch phase is prefill.
- **Range and profile admission:** The design promises no extrapolation but does not fully specify missing/out-of-range table behavior or statistical admission for component samples. Define rejection/UNVERIFIED behavior and coverage before implementation. This is an open specification point, not an additional confirmed finding.

## Corrections in dependency order

1. Correct the native component inventory and documentation counts (F005), and define the host/device/connector boundaries needed for component attribution (F003).
2. Repair real-forward event pairing and distinguish workload, runtime-dummy, startup/capture, and no-forward records (F002). Use scoped actual collective identity for joining.
3. Replace or qualify the conditional-gap estimator, explicitly covering roles and request buckets without controls (F003). Emit no predictive profile until attribution and coverage are established.
4. Bind and validate profile topology/runtime compatibility, obtaining separate DP4 evidence if the exported component includes group coordination (F004).
5. Implement preparation readiness before first attention for all useful/dummy DP lanes, preserving existing once-only tail accounting (F001). Add the event-time counterexample and P/D role/trigger checks.
6. Verify unchanged default results with the required smoke/fidelity checks, then perform separately authorized diagnostic collection and clean E/G comparisons. Keep probe runs excluded from reference admission throughout staging, replay, and formal-release handling.

The supported findings do not depend on an unprofiled GPU run. The eventual cost value, profile portability, complete native probe coverage, and final predictive improvement remain unverified.
