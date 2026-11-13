# D29-INIT-VALIDATION-SCOPE report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Correct the `family_only_loop.py` call counter (it double-counted the one-time resolution), rerun the 3 000-row loop, record the procedural timeout/scan deviation. |

Task ID: **D29-INIT-VALIDATION-SCOPE**. Read-only investigation; no source, test,
input, workload or gate was modified. Frozen source
`e41abc073500d61f224eda4fca1123e74e40544d` at
`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05`. Probes and raw
results: `/data/ycfeng/tmp/d29_init_validation_scope/`. Environment
`/data/ycfeng/envs/frontier-py310/bin/python` 3.10.6, RLIMIT_AS 30 GiB.

### Recorded procedural deviation

Three probe runs broke the assignment's stated limits; they are preserved, not
repeated:

- `full-current01` and `full-hoisted01` each parsed the whole stored `moe.csv`
  (37 044 rows, all 44 columns). The assignment prohibited new full input scans and
  capped probes at 60 s; these two used a 75 s outer timeout for margin. Both
  exceeded the intended scope even though they finished inside their cap.
- `family-only01` reported `validate_step5_config_calls_total = 2` for one actual
  call: the counted wrapper already included the one-time resolution, and the
  result line added it again. The counter was fixed and the 3 000-row loop rerun as
  `family-only03`; no cap was raised and no further full scan was performed.
- All later work stays inside the original 60 s / 30 GiB limits, uses the 3 000-row
  head slice only, and no cap was increased.

## 1. The path under investigation

| Element | Location |
| --- | --- |
| Per-row typed-contract map (MoE loader) | `frontier/execution_time_predictor/profiling_dataframe_loaders.py:718-727` (`:723` lambda) |
| Per-operator metadata validator | `frontier/operators/typed_contracts.py:405` `validate_typed_operator_contracts` |
| Per-operator metadata checks | `frontier/operators/typed_contracts.py:96` `validate_typed_operator_metadata` |
| Expensive model-derived call inside it | `frontier/operators/typed_contracts.py:137` -> `frontier/attention/model_binding.py:261` `resolve_runtime_attention_family` -> `:179` `_bind_homogeneous_attention_family` -> `frontier/attention/step5.py:149` `validate_step5_config` |
| Re-export shim (no logic) | `frontier/profiling/common/typed_contracts.py` |

`frontier/profiling/common/typed_contracts.py` only re-exports names; the `:96/405`
frames in the earlier profile are functions of `frontier/operators/typed_contracts.py`.
Neither file is memoized: no `lru_cache`/`functools.cache` exists on
`validate_step5_config`, `resolve_runtime_attention_family`, or the validators.

Call frequency of the map (position mode): `prediction_family_trainers.py:215`
`base_moe_model_names = list(operator_facts)`, and `:354/:363` loops
`_get_moe_df_for_op(model_name)`; the position branch at `:268-275` calls
`_load_moe_df` per operator with no cache, while the legacy branch 40 lines above
(`:249-258`, `:313-341`) already keeps a per-call `moe_df_cache`. main07 `run.log`
emits four MoE operator metadata warnings per cluster
(`moe_gating_linear`, `moe_gating_routing_topk`, `moe_input_quantize`,
`moe_grouped_gemm`). That log supports an *inference* of 4 loads per cluster and
8 per initialization (one warning per operator); the count itself was never
instrumented, and no probe in this task measured it.

## 2. Direct measurements (stored fixture03 moe.csv, 2 114 905 379 bytes, 37 044 rows)

| Probe | Variant | Read | Typed map | `validate_step5_config` | Payload SHA256 |
| --- | --- | --- | --- | --- | --- |
| `full-current01` | production call site | 9.232 s | **35.344 s** | 43 218 | `6fa8630f…e3359e` |
| `full-hoisted01` | profile+family+mtp hoisted | 9.303 s | **3.426 s** | 1 | `6fa8630f…e3359e` |
| `probe02` (3 000 rows) | current vs hoisted | – | 2.885 s vs 0.270 s | 3 502 vs 1 | `f4603590…` both |
| `family-only01` (3 000 rows) | **only** `architecture_family_id` hoisted | – | 0.272 s vs 2.982 s baseline | 2 total (counter bug, see below) | `f4603590…` both |
| `family-only03` (3 000 rows) | same variant, corrected counter | – | 0.273 s vs 2.838 s baseline | **1** | `f4603590…` both |
| `census01` | typed column only | 6.397 s | – | – | **3 distinct payloads** |

Details:

- Per row current cost 0.954 ms; after hoisting 0.0925 ms (10.3x). 90.3% of the
  per-row work is the repeated resolution of one frozen model configuration.
- `census01`: 37 044 rows contain exactly **3 distinct** `typed_operator_contracts`
  strings, repeated 32 928 / 2 058 / 2 058 times; payload lengths 406-1 234 bytes.
  Operator count per row 1/2/3 = 2 666/166/168 (3 000-row head).
- `probe02` negative control: corrupting `tensor_parallel_mode` in one payload still
  raises `ValueError`, including when the hoisted variant is used.
- Corrected count: with `architecture_family_id` hoisted, the 3 000-row loop calls
  `validate_step5_config` exactly **once** (`family-only03`, script SHA256
  `be8973b4778694555d5626e9e6b4f570f65cd1a61fcf9e8656fdd414592203ea`), and the
  validated contracts are byte-identical to the unhoisted baseline
  (`f460359008814440c80a40fd2e97b557e7bcbb802c72a4076c6ebe3c8a2cdd3f` on both
  sides). `family-only01`'s "2" was a reporting defect, not a second call: the
  counter wrapper already incremented on the one-time resolution and line 68 of the
  script added that same value again.
- Peak RSS 1.0 GiB for the full read; nothing cached beyond the call.

## 3. What is input-constant, what is row-dependent

Input-constant (per loader call, and in fact per model binding):

1. `resolve_runtime_attention_family(model_config)` and the full
   `validate_step5_config` sweep it triggers (92 geometry checks plus
   `parse_moe_layer_ids` and regex work): 4.975 s of 5.696 s in the earlier
   `loader01` profile, 35.34 s of 44.58 s here (79% of a whole load).
2. Registry/profile lookup `get_model_architecture_profile` and the MTP method read
   (cheap, but same category).
3. `bind_operator_query` per operator name (0.183 s / 2 589 rows profiled, ~3%).

Row-dependent and must stay:

1. `json.loads` of each row's payload plus `_copy_contract_mapping` deepcopy
   (0.260 s profiled; the residue after hoisting).
2. Schema/ownership/enum/width/equality checks against that payload, including
   operators outside the caller's selected query (the deliberate "malformed metadata
   cannot be hidden by an unrelated selector" rule at `:718-719`).
3. Downstream per-row admission in `frontier/moe_routing_runtime.py:105-131`
   (`filter_moe_operator_rows`) which re-parses each selected row and validates
   against `contract.typed_metadata_identity()`; plus `profiling_dataframe_loaders.py:803`
   `_typed_row_matches_contract`. These are untouched by any proposal here.

Existing reuse that does *not* cover this: the only file-level reuse in the MoE path
is `moe_df_cache` in the legacy branch (`prediction_family_trainers.py:249-258`).
The position branch bypasses it. No memoization exists for the attention-family or
step5-config validation.

## 4. RED/GREEN feasibility without bypassing checks

The loop can go RED/GREEN deterministically, because the defect is a repeated
*derivation*, not a repeated *decision*:

- RED (today): `validate_step5_config` calls scale as rows x operators (43 218 for
  37 044 rows, 4 operators), and each of the 8 loads repeats them from scratch.
- GREEN: at most one resolution per `_load_moe_df` call, with the same validated
  payload hash and the same per-operator selected frames.
- Not bypassed: every row still parses and validates its own JSON, every operator
  ownership/enum/width/equality rule still runs per operator, and the
  all-rows-before-filtering admission rule is preserved. The `probe02` negative
  control and the equal-payload hash are the falsifiers.
- The measured gain is a property of the fixed config, not of the fixture: a corpus
  with 37 044 distinct payloads would still lose the same 31.9 s per load, because
  the removed work is config derivation.

The residue (3.43 s per load, dominated by `json.loads` + deepcopy) could be cut
further by memoizing on the exact payload string (3 distinct values here), but the
copy-on-hit needed to keep per-row dict independence costs ~2.8 s of the 3.4 s, so
that mechanism does not earn its place. No cache is required at all.

## 5. Proposed minimal mechanism

Hoist the frozen model-derived validation input at the existing call site:

- File: `frontier/execution_time_predictor/profiling_dataframe_loaders.py`, the
  block at `:718-727`.
- Change: compute `resolve_runtime_attention_family(replica_config.model_config).family_id`
  once before the `.map` (the symbol is already imported at line 13) and pass it as
  the existing `architecture_family_id=` keyword of
  `validate_typed_operator_contracts`. Optionally also pass
  `architecture_profile=`/`mtp_method=` for the same reason; the family id alone
  already removes 90.3% and is the smaller hunk.
- Size: about 3-6 gross lines (1-2 added + 1-2 changed at the lambda). No new
  import, no new API, no cache, no lifetime or invalidation contract: the values are
  locals of one loader call and are read-only.
- Test seam: `tests/unit/test_moe_grouped_gemm_padded_features.py:324-334` already
  drives the real `_load_moe_df` with position `operator_facts` and training; add a
  counting assertion (one family resolution per load, equal validated payloads, one
  corrupted-payload negative case). Estimated 25-35 gross.
- Total repair estimate 30-45 gross, well inside the current CORE allocation.

Deliberately **not** proposed, with reasons:

- File-level reuse of the parsed+validated frame across operators (extending
  `moe_df_cache` into the position branch): saves only the 9.2 s read per extra
  operator (~55 s of the 255 s) but adds a large mutable object, a lifetime contract
  and a memory increase; poor trade.
- Payload memoization: see section 4.
- Any edit inside `frontier/operators/typed_contracts.py`,
  `frontier/attention/model_binding.py` or `frontier/attention/step5.py`: outside
  the ledger / no headroom (section 6).

Supported mutation/lifetime semantics if a later owner still wants reuse: scope it
to one `_train_ffn_models_for_cluster`/`_load_moe_df` call (same lifetime as the
existing `moe_df_cache`), key by model-config identity plus absolute file path, and
return an independent dict copy per row so the `_copy_contract_mapping` deepcopy
invariant is preserved. Nothing may outlive the call, because the inputs are
per-run artifacts validated by `dataset_fingerprint`.

## 6. Ledger position and remaining ceilings

| Path | In 111 ledger | Owner | Ceiling after amendments | Measured | Headroom |
| --- | --- | --- | --- | --- | --- |
| `frontier/execution_time_predictor/profiling_dataframe_loaders.py` | yes | core_source | 17 | 9 | **8** |
| `tests/unit/test_moe_grouped_gemm_padded_features.py` | yes | core_source | 210 | 197 | **13** |
| `tests/unit/test_moe_routing_runtime.py` | yes | core_source | 70 | 53 | **17** |
| `frontier/execution_time_predictor/prediction_family_trainers.py` | yes | core_source | 88 | 80 | 8 |
| `frontier/execution_time_predictor/layer_contract_resolution.py` | yes | core_source | 25 | 11 | 14 |
| `frontier/execution_time_predictor/moe_dataset_training.py` | yes | core_source | 164 | 162 | 2 |
| `frontier/attention/step5.py` | yes | pricing | 3 | 3 | **0** |
| `frontier/operators/typed_contracts.py` | **no** | – | – | – | 0 |
| `frontier/attention/model_binding.py` | **no** | – | – | – | 0 |
| `frontier/profiling/common/typed_contracts.py` | **no** | – | – | – | 0 |

Ceilings come from `scope.json` plus the amendment chain through
`model_group_reuse_scope_amendment.json`; measured values from
`final_parent/final_ledger_proposal03.json`.

Aggregate: core_source 4 404 ceiling, 3 873 used (531 free), less the 105-line
`D29-MODEL-GROUP-REUSE` allocation already granted to the other agent -> **426
free**. Pricing 3 956 / 3 879 -> 77. Source 9 942 / 9 102 and global 10 000 / 9 116
at delivery remeasurement -> 735 / 779 after that 105-line allocation. A 30-45 gross
repair therefore fits, but the per-path ceilings need a small zero-sum transfer
(precedent: the same amendment moved 100 lines between CORE paths).

The two paths with the same per-row pattern but outside the minimal scope:
`profiling_dataframe_loaders.py:142` (`_load_linear_op_df`, 24-row `linear_op.csv`,
negligible) and `moe_dataset_training.py:384/:610` (different trainer API, not on the
shared-manager path observed in `init01`).

## 7. Limits of this evidence

Direct: per-row and full-file map cost, call counts, payload equality, distinct
payload census, negative control, profile attribution.

Extrapolated: the 8 loads per initialization (from the preserved `run.log` operator
warnings, not from an instrumented run) and therefore the ~255 s total saving.
`loader01`'s earlier 2.2 ms/row was a cProfile-inflated figure; the unprofiled
0.954 ms/row measured here supersedes it.

Unattributed and material: main07 spent ~21.5 min between the 15:05:16 start and the
first 15:26:46 metadata warning. The preserved `init01` sample covers only 5.04 s of
that block (stopped in `cache_io.dataset_fingerprint`, 3.2 s of hashing). The typed
map lives in the later 15:26:46-15:30:57 window (~251 s of inter-op gaps), so this
repair cannot explain, and will not fix, the 21.5 min block. Fixing this seam is
worth ~4.3 min of a 26 min initialization in a run that then times out in pricing;
no timeout passing claim is made.
