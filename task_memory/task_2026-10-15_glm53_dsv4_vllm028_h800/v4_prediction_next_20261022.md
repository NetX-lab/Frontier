# V4 prediction: next executable integration step (2026-10-22)

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Created: read-only audit of the V4 prediction/serving guard chain, P0 cost-class coverage of existing H800 tables, and one named next step. No source, plan, ticket or GLM change. |
| 2026-10-22 | Correction after primary review: earliest non-dummy production guard is the predictor constructor family gate, not `serving_context`; `(i)` relabelled to isolated eager data availability (not serving qualification); warm-q1 planner values and native TP/`max_num_seqs` caveat added; authority distinguished (D-Q10 approved-but-unfinished implementation vs required native qualification vs unapproved GPU diagnostic method). Same file only, no new investigation. |
| 2026-10-23 | W128-V4-COVERAGE-AFTER-BALANCED-PREFILL: §3 class 4 superseded for the accepted balanced 6144/C0 case (isolated eager rows now exist); §5 recommendation replaced by §7; corrected the D-Q10 citation to `requirements.md:1063-1069`; §§1-6 kept as the 2026-10-22 historical record. Same day, same section: corrected stale status (prefill supplement already accepted, native06 in CPU package checks), separated approved-unfinished consumer wiring from new-collection authority, stated the probe's satisfiability-only limit, and removed any suggestion to change the legacy checker default. |

Task ID: `W128-V4-P0-PREDICTION-NEXT`. Branch `feat/glm53-dsv4-vllm028`, HEAD `2c150a7` (code `7c1b888`).
Everything below is CPU/static evidence. No serving run, no GPU, no guard bypass, no shimmed timing.

## 1. Earliest actual non-dummy production guard

A real predictor construction fails **before any layer prediction and before `serving_context` is
reachable**, at the attention-family execution gate inside the base constructor.

Call chain (verified by reading the code and by a bounded real constructor repro, §4):

| Order | Guard | Location |
| --- | --- | --- |
| 1 (earliest) | predictor `__init__` family loop | `frontier/execution_time_predictor/base_execution_time_predictor.py:27` class, `:39-46` loop, `:44` call |
| 2 | `family.require_enabled_for_execution(structural=predictor_config.enable_dummy_mode)` | `frontier/attention/ops.py:181-189` |
| 3 | operator-family spec gate | `frontier/operators/spec.py:349-354` (`:350` raise) |
| 4 | declaration that closes the gate for V4 | `frontier/attention/families.py:645-650` (`structural_execution_enabled=True`, `execution_enabled=False`, `disabled_reason=`) |

Observed error on the real chain (`RandomForrestExecutionTimePredictor` -> MoE base -> `__init__`):

```
NotImplementedError: Attention family deepseek_v4_attention is not enabled for execution:
DeepSeek-V4 execution requires compressed cache, compressor, indexer, mHC and mixed-precision
operator coverage.
```

Only after this gate does the sequence reach the seams reported previously, which remain valid but are
*inside* the path, not its entry: `serving_context` PD/graph check
(`frontier/execution_time_predictor/deepseek_v4.py:453,457`), co-location
(`frontier/attention/deepseek_v4.py:95-102`), attention serving admission
(`sklearn_execution_time_predictor.py:7608-7611`), FFN/routing admission
(`sklearn_moe_execution_time_predictor.py:1379-1380`), synthetic-scope price (`deepseek_v4.py:155-159`),
reader phase/graph refusal (`deepseek_v4.py:596-601`). The earlier direct `serving_context` probe was a
later seam tested in isolation; it is not evidence about the full call chain.

Consequence: no measurement, table or dataset can by itself make a non-dummy V4 prediction run today.
The blocking item is the family/scope admission decision plus its operator coverage, not row coverage.

## 2. Authority per guard (approved vs required evidence vs unapproved)

D-Q10-V4-GRAPH-MEASURED-AND-NIXL (`task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/requirements.md:982-988`)
already authorizes, on native V2 evidence and CPU-first: graph padding and measured-timing consumption,
extending existing NIXL to the V4 grouped cache, with typed rejection whenever tables or scope/phase
evidence are missing. Its final sentence excludes only *new GPU diagnostic attempts and runtime-stack
changes*. So the existence of a guard is not, by itself, evidence that user design approval is missing.

| Item | Classification |
| --- | --- |
| Measured-timing consumption and typed missing-evidence rejection for V4 scopes (`deepseek_v4.py:155-159,537,596-601`; `:7608-7611`; `:1379-1380`) | Approved (D-Q10), **implementation unfinished** |
| Graph padding / replay-admitted cost path under the same approval | Approved (D-Q10), **implementation unfinished**; needs native V2 phase/padding semantics pinned before enabling |
| Family `execution_enabled` for `deepseek_v4_attention` (`families.py:645-650`) | Consumer admission decision on top of the above; unfinished implementation, not a new user approval |
| Exact P0-layout prefill rows, graph/replay rows, native routing histogram for grouped GEMM (classes 4/6/8 below) | **Required native qualification**; measurement work under the existing measurement plan, not a new method |
| A new GPU diagnostic campaign, new probe/serializer, CUPTI restart, runtime-stack change | **Actually unapproved**; would need explicit approval |
| H200 tables as H800 qualification | Not authorized, and not used here |

## 3. Cost classes required for a full P0 result

P0 target (ticket 17 / native03): H800 32 GPU, P16+D16, TP8/DP2/EP16/PP1, prefill 6144 uncached and
unchunked, decode context 6143..7166, native `max_model_len=7168`,
`max_num_batched_tokens=8192`, `max_num_seqs=128`, graph on (`enforce_eager=false`),
`kv_cache_dtype=fp8` -> `fp8_ds_mla`.

Status key, deliberately weaker than "qualified": **(i)** measured rows exist in isolated eager form on
the right device but are **not phase/TP/graph serving-qualified and not bound**; **(ii)** measurement or
qualification missing; **(iii)** consumer implementation unfinished (some of it already approved).

| # | Cost class | Existing evidence | Status | Action |
| --- | --- | --- | --- | --- |
| 1 | Family/scope admission for V4 execution | guard §1; `families.py:645-650` | iii | finish D-Q10 consumption + admission chain; no new approval needed for the CPU-first part |
| 2 | PD phase/graph equivalence for all V4 scopes | `deepseek_v4.py:457`; no V4 PD/first-tail equivalence artefact in `frontier/` | ii (+iii) | pin native V2 phase semantics; P16/D16 is not `MONOLITHIC`, so a PD decision is still required |
| 3 | Role co-location assumption | `attention/deepseek_v4.py:95-102` | iii | same chain; prefill/decode role split must be resolved |
| 4 | Attention prefill 6144 at P0 layout | `w6_dsv4_profiles/attention.csv`: 7116 rows, CUDA_EVENT/eager/H800/physical TP1; prefill `total_tokens` = {255,256,512,1024,2048,2050,2051,2052,4096,8192} -> **no 6144**; `max_model_len=33024`, `max_num_batched_tokens`=256..8192, `max_num_seqs=1024`, i.e. not the P0 planner geometry | ii | native qualification: exact (6144, batch 1, P0 planner geometry) rows for all three variants and both tile schedules |
| 5 | Attention decode q1 window 6143..7166 | `v4_q1_warm_results/Q1_WARM_6143_7166/.../attention.csv`: 6144 rows, H800, decode, 1024 contexts; `plans_tile_schedule` two values are **planning/reuse roles, not native tile sizes**; the native effective `max_num_seqs` is 1 despite CSV planner 128, and physical TP is 1 (per `task_2026-10-21_v4_q1_warm_remeasure/observed/warmup_comparison.md`, not reanalysed here) | i, and additionally blocked by guard 2/3 phase semantics | bind only after phase contract exists; treats as isolated eager, not serving latency |
| 6 | Graph/compiled replay costs | every V4 table examined is `graph_mode=eager`; graph branch refuses at `deepseek_v4.py:596-601` | ii (implementation approved) | replay/graph rows under the native config that actually runs (`enforce_eager=false`) |
| 7 | Linear/SP projections and norms | `linear_op.csv`: 323 rows, physical TP1, eager, `num_tokens` 1..8192 incl. 768 and 6144; consumer returns 0.1730 ms at 6144 | i | usable as isolated eager cost only; serving binding still blocked by 1-3 |
| 8 | MoE gating/shuffling/grouped GEMM | `moe.csv`: 4059 rows, physical TP1 eager, EP 8/16/32, 256 experts, topk 6; grouped GEMM needs an exact expert-load histogram (`deepseek_v4.py:670-694`) | ii + iii (routing call site) | native routing disposition (ticket 14) then the production call site |
| 9 | Communication: SP allgather/RS, mHC allgather, embedding allreduce, EP16 dispatch/combine, logits gather, NIXL KV | H800 curves under `data/profiling/network/h800_dgx/{summary.csv,nixl_summary.csv,glm_p0_ep16_20261021a,b}`; scope names at `deepseek_v4.py:13-63` | i (curves), ii for V4 payload admission | curves are measured bandwidth, not V4 admission; NIXL wire semantics blocked by guard 2 |
| 10 | Wrapper scopes: `dsv4_embedding`, `*_allreduce`, `dsv4_mhc_*` (4), `dsv4_final_norm`, `dsv4_lm_head`, `dsv4_logits_gather`, `kv_cache_block_zero` | names at `frontier/operators/families.py:340-354,822`; priced only via `_dummy_execution_time` (`deepseek_v4.py:155-159,300-320`) | ii | no table exists on any hardware |
| 11 | Host CPU overhead | only `data/profiling/cpu_overhead/h200_dgx` | ii | H800 host table missing; H200 does not substitute |
| 12 | Native reference admission (ticket 18) | candidate03: 384/384 formal, 431/432 all, still UNVERIFIED; candidate05 unsealed | ii | owned elsewhere |
| 13 | Frozen workload binding | `data/processed_traces/canonical_fixed_6144_1024_req128_qps1/{manifest.json,request_ids.json,trace.csv,warmup_pool.csv,SHA256SUMS}`; generator reads `processed_traces` (`frontier/config/request_generator_config.py`) | i (config-level only; no Frontier P0 driver) | driver entry after 1-3 |

Distilled: nothing above is serving-qualified for P0. Classes 5, 7, 9 and 13 have measured/structural
material on the right device (i); 4, 6, 8, 10, 11, 12 need native measurement or qualification (ii);
1, 2, 3 and the grouped-GEMM call site are unfinished consumers (iii), and the CPU-first part of 1 and 6
is already approved by D-Q10.

## 4. Verification receipts (CPU, `frontier-py310`, ~1 min total)

Read-only, `PYTHONPATH=.`, no files written, no GPU.

1. Real constructor, non-dummy: `RandomForrestExecutionTimePredictor(SimpleNamespace(enable_dummy_mode=False, ...), replica, scheduler, metrics)`
   with `BaseModelConfig.create_from_name("deepseek-ai/DeepSeek-V4-Flash")` ->
   `NotImplementedError: Attention family deepseek_v4_attention is not enabled for execution: ...`.
2. Same call with `enable_dummy_mode=True` passes the family gate (structural path) and stops later on
   missing replica fields, confirming the gate is the non-dummy-specific entry.
3. `deepseek_v4.serving_context` with `KV_CONNECTOR=nixl` -> `NotImplementedError: V4 predictor PD
   phase/graph equivalence remains unqualified`; with `KV_CONNECTOR=none` it returns a context.
4. `build_deepseek_v4_attention_workload(..., cluster_type=ClusterType.PREFILL)` ->
   `NotImplementedError: V4 attention component requires co-location`.
5. `read_isolated_scope` H800/TP8/DP2/EP16, prefill 6144, layers 0/2/3 ->
   `DeepSeekV4CoverageError: V4 isolated attention layout has no exact measured rows`; decode `(1,6143)`
   against w6 and q1-warm -> `DeepSeekV4CoverageError: V4 decode/first-tail phase equivalence is unqualified`;
   `share_expert_up_proj` at 6144 -> `PASS 0.1730 ms`.

Table facts are direct CSV reads: `attention.csv` 7116 rows, `linear_op.csv` 323 rows, `moe.csv` 4059
rows, q1-warm `attention.csv` 6144 rows. The w6 layout mismatch is the 6-column diff
`max_model_len 6145->33024`, `max_num_batched_tokens 6144->{256..8192}`, `max_num_seqs 128->1024`, plus
three `swa_metadata.*` columns.

## 5. One smallest useful immediately executable step

**CPU-only, already covered by D-Q10: make the decode/first-tail phase contract consumable at
`frontier/execution_time_predictor/deepseek_v4.py:596-601`** for the one V4 scope family that has
device-correct eager rows in the exact P0 context window (q1-warm decode, 6143..7166), reading the native
V2 phase/padding semantics already pinned for D-Q10, and keeping a typed
`DeepSeekV4CoverageError` wherever the evidence is still absent.

- Why this one: it is source-level, unapproved-method-free, needs no GPU, and converts the only V4
  serving-shaped dataset from unreachable to addressable-or-explicitly-typed. It also produces the
  precise missing-evidence message that later prefill/graph requests must satisfy.
- What it does **not** do: it does not make a non-dummy V4 prediction run (guard 1 in §1 still stops
  construction), does not qualify P0, and does not relax the four-metric rules.
- If the phase-equivalence work cannot be completed from pinned native V2 evidence alone, the fallback is
  the P0 coverage preflight over the existing checker (`tests/integration/check_dsv4_measured_scopes.py:38-53`)
  parameterized to H800/TP8/DP2/EP16 + prefill 6144 + q1 decode. That preflight is a bookkeeping harness
  for the missing cells; it does not complete or qualify the performance path.

Remaining gap that no code step closes: class 4 (exact prefill-6144 P0 rows) and class 6 (graph/replay
rows) are measurement prerequisites; they need the native collection that the accepted P0 plan calls for,
not a new diagnostic method, and they stay blocked behind the §1 admission chain.

## 6. Boundaries

- No source edit, no test/framework addition, no plan/ticket/requirements change, no GLM work.
- No serving admission, no numerical reference, no accuracy claim, no CPU-to-GPU cost substitution.
- H200 tables are not H800 qualification and were not used as such.
- q1-warm counts, first-event spike and `warmup_comparison.md` findings stay with their current owner;
  this note cites them and does not redo the analysis.

## 7. 2026-10-23 override after accepted balanced prefill

Task ID: `W128-V4-COVERAGE-AFTER-BALANCED-PREFILL`. HEAD `b62c155`. Evidence:
`task_memory/task_2026-10-22_v4_balanced_routing/prefill_results/primary_acceptance.json`
(`ACCEPTED_ISOLATED_DATA`), `accepted/PREFILL_WARM_6144_C0_BALANCED/`, routing
`checkpoint_2026-10-23.md`, parent `tickets/09-predictor-bundle.md` (still OPEN).

**Class 4 changes status.** The accepted case supplies 6 prefill rows, one per layer, at
6144 prompt / 0 context, `max_model_len` 7168, `max_num_batched_tokens` 8192, configured =
effective `max_num_seqs` 128, `physical_tp` 1 / eager, TP8-local dimensions, all three variants
plus both `plans_tile_schedule` values, `source_revision` 39b2bf8, `routing_runtime_path`
`balanced_topk`, `routing_strategy_sha256` 6562ec1c..., `device_name` NVIDIA H800. So the
"no 6144 rows" statement in §3 describes the w6 tables only and is superseded here: class 4 is
now (i) isolated eager availability, still not TP8/PD/graph/serving/E2E qualified
(`serving_admission=false`, `steady_state_qualified=false`). This 1×H800 prefill supplement has
already run and been accepted (job exp-1023-023708-916112, 500 s / 0.1389 GPU-h); it is not
blocked and its budget cannot be spent again.

**New CPU probe (read-only, `/data/ycfeng/envs/frontier-py310`, no files written).** The probe
built each variant's layout from that variant's own CSV row and then drove the real
`read_isolated_scope` (`frontier/execution_time_predictor/deepseek_v4.py:579`), obtaining 50/50
scope receipts at exactly the CSV medians. Feeding one shared layout, as the H200 fixture supplies,
gives 20/50: csa passes and every hca / uncompressed scope is rejected by `_select_rows` (`:537`)
because `compress_ratio` and the `*_metadata.*` columns differ per variant.

**Limit of that probe — do not quote 50/50 as more than this.** Since the layout was read from the
measured row, the probe proves only that the selector is *satisfiable* for these rows. It does not
prove an independent contract exists, does not show any producer-side derivation, and exercises no
production integration: no shipped caller passes a layout, and prefill carries no accepted-input
provenance. The nearest existing contract,
`tests/fixtures/deepseek_v4_attention_layouts.json` (H200/6145/6144), is test-only and encodes a
different device and different limits.

**Provenance asymmetry.** `deepseek_v4.py:634-635` applies accepted-input provenance
(`source_revision`, `runtime_fingerprint`, `model_config_sha256`, `input_file_sha256`) only when
`phase == "decode"`; prefill selects on layout columns alone. `read_isolated_scope` has no
production caller at all — only `tests/integration/check_dsv4_measured_scopes.py` and unit tests —
and that checker's legacy case is H200/DP4/EP32 (`:41-46`) with the H200 fixture (`:49-50`). Leave
that legacy default behavior in place: the accepted H800 case should arrive through an explicit new
input/contract argument, not by rewriting or replacing the existing case.

**Corrected citation.** D-Q10-V4-GRAPH-MEASURED-AND-NIXL is `requirements.md:1063-1069`
(approval Q&A `:1065-1067`), not `:982-988` as §2 states.

**Replacement for §5 (same authority, earlier executable).** Make the accepted isolated prefill
segment addressable by production code: declare the layout per variant from an independent source
(the producer already derives these values from `plan.limits`, the engine step and the per-variant
step metadata in `frontier/profiling/attention/dsv4_attention_rows.py:160-180`; the accepted
package retains `raw/dsv4_raw_e0.jsonl` with 1 engine / 6 step records carrying
`{csa,hca,uncompressed}_*_metadata`), emit that declaration with the measurement, and bind the
prefill `input_file_sha256`/`source_revision` the way decode already does, through an explicit new
input rather than a changed default. Keeps `DeepSeekV4CoverageError` for graph, PD, TP8, first-tail
and absent rows. This is V4-local, CPU-only, needs no new GPU work, and does not depend on native06;
it changes class 4 from "unreachable" to "addressable or explicitly typed". Root has begun this
approved CPU isolated-contract implementation independently; that is not a claim that the full
non-dummy path opens.

**Authority for the remaining wiring.** D-Q10 (`requirements.md:1063-1069`) already covers
measured-timing consumption, graph padding and typed rejection, so the constructor family gate
(`frontier/attention/families.py:645-650`, `execution_enabled=False`), the `serving_context` PD
refusal (`:458`), co-location (`frontier/attention/deepseek_v4.py:95-102`) and the graph refusal
(`:601`) are approved-but-unfinished consumer wiring, not missing user permission. Opening them is
evidence-dependent — pinned native V2 phase/padding semantics plus the required measured rows and
phase/scope evidence — and CPU wiring alone does not complete family admission end to end.

**New collections have no authority from D-Q10.** Graph/replay rows, physical-TP8 rows, balanced
decode 6143..7166 rows and the independent balanced expert histogram are **not** covered by either
approved package: the 1×H800/900 s prefill supplement already Succeeded and was accepted (budget
spent, not reusable), and the 32×H800/60 min native06 reference is in CPU package checks on the
fresh `b62c155` worktree `v4-balanced-native-model`, not waiting on the earlier B source repairs.
Any such collection needs separate authority; native06 also does not supply operator GPU timings.

**Still open.** Coverage classes 6, 8 (independent balanced expert histogram), 9 (V4 payload
admission), 10, 11 and 12; all TP8/PD/graph/serving/E2E qualification of the accepted rows;
historical q1 decode tables stay unbalanced and cannot be relabelled.
