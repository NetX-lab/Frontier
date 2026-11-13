# D29-EARLY-INIT-REPRO report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Initial read-only investigation of the unexplained 21.5-minute initialization window; bounded unit probes and budget reconciliation. |
| 2026-10-23 | Second pass: added the omitted Step attention training/cache-hit phase and its hash multiplicity, retracted the column-wise hashing proposal, corrected the budget to the current overlays. Earlier observations are preserved; superseded numbers are marked **REVISED**. |
| 2026-10-23 | Third pass (D29-INIT-CACHE-RECONCILIATION): cache verdict from published bundles + mtimes + real load/save code; corrected MoE read/admission multiplicity, the admission rate, and the 22/78 split. Bindings: historical claims stay on e41 (`verify-step5-d29-integrated-05`); the current merged source is not used. Superseded figures are marked **REVISED**. |

Task ID: **D29-EARLY-INIT-REPRO**. Read-only. No production source, test, input,
prior run, GPU or platform state was touched. Frozen source
`e41abc073500d61f224eda4fca1123e74e40544d` at
`/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05`. Scratch:
`/data/ycfeng/tmp/d29_early_init_repro/`. Every probe ran under a hard 60 s
outer `timeout`, `RLIMIT_AS = 30 GiB`, `OMP_NUM_THREADS=1`; no probe read the
whole 5.63 GB input. This work is disjoint from the in-flight repair to
`profiling_dataframe_loaders.py` / `tests/unit/test_moe_grouped_gemm_padded_features.py`.

## 1. The target interval, from retained evidence only

| Fact | Value |
| --- | --- |
| offline wall / cap | 2400.10 s / 2400 s, timeout |
| first log line | 15:05:16 model-architecture warning, then the printed cluster summary |
| **predictor training finished** | bundles written **15:26:43-15:26:46** (`3e462326`, `48e35a87`, `3d960f88`, `52781c91`) |
| first profiling-metadata warning | 15:26:46 `quantization_manager.py:814` op=attn_pre_proj |
| last metadata warning | 15:30:57 op=moe_grouped_gemm (cluster 2) |
| CLI `--log_level` | `warning` — nothing else is logged in this window |

**REVISED anchor.** `_publish_step5_model_bundles()` runs immediately after
`_train_all_required_models()` (`shared_prediction_model_manager.py:112-113`), and
the retained bundle `bundle_3d960f88a3aecb8fd3e5b732f251bec594838d257e3c044d6e1aa93dfadc4934.json`
has mtime 15:26:44, and the last of the four is 15:26:46 — the first metadata warning
is 15:26:46, so training ended no later than that. Therefore the
unexplained window **15:05:16 -> 15:26:46 (~1290 s; 1288 s used as the round figure
below and in §6a)** is entirely predictor-model
construction, and the 15:26:46 onward warnings belong to the *subsequent*
quantization/cluster phase, not to training. Earlier wording that placed the
window's end at "the FFN preflight" is superseded by this anchor.

Retained `init01` sample (5.0 s bounded, cProfile) confirms the entry path:
`simulator.py:157` -> `shared_prediction_model_manager.__init__` 4.934 s ->
`_train_all_required_models` 4.934 s -> `_train_attn_models_for_cluster` 3.836 s ->
`load_step5_attention_input_union` 3.832 s -> `dataset_fingerprint` 3.830 s
(3.201 s in `hashlib.update`); the faulthandler dump in `init01.log` shows the
identical live stack.

## 2. Traced call graph (corrected)

Per cluster, per measurement family, `_train_all_required_models`
(`shared_prediction_model_manager.py:498-530`) runs:

1. `_train_step5_input_layernorm_for_cluster` (`prediction_family_trainers.py:85`)
2. `_train_attn_models_for_cluster` (`:690`); for Step:
   `load_step5_attention_input_union` (`step5.py:315`) -> per source
   `load_step5_attention_rows` (`:193`: `pd.read_csv(path).drop_duplicates()` at
   `:201`, whole-frame row loop `:247-300`) -> `step5_attention_input_hash`
   (`:431`) -> `_get_model_hash` (`prediction_model_registry.py:128`) ->
   `hashlib.md5(df.to_json().encode())` (`:153`)
3. **`train_step5_attention_models` (`step5.py:461`) — the stage omitted in the
   first pass**: `for name in STEP5_OPERATOR_NAMES` with
   `STEP5_OPERATOR_NAMES == ('step5_layer_e2e',)` (measured, §3), two
   `_train_single_model` calls per name (duration, `_kernel_count`), each of which
   reaches `_fit_single_model` (`:1693`) and calls `_get_model_hash` at `:1753`
   **before** the cache lookup at `:1757`
4. `_train_ffn_models_for_cluster` -> `_validate_moe_dataset_contract`
   (`layer_contract_resolution.py:383`, full read at `:397`) -> per-operator
   `_load_moe_df` -> `_register_additional_profiling_metadata_from_files`
   (`moe_dataset_training.py:584`)

Why the attention stage hashes the whole union: `_train_single_model` takes the
`step5_operator_training_rows` branch only when `training_context` carries
`step5_generic_compute` or `operator_facts` (`prediction_family_trainers.py:1644-1653`).
The attention `training_context` built at `:722-730` carries **neither**, so
`frames = [(df, training_context)]` and every model is hashed on the **entire
union frame**. Measured directly: `diagnostic_root_rows_in_union = 0`
(`# step5_layer_e2e` appears in zero `typed_operator_contracts` values) while the
hash calls receive all 300 slice rows. The retained pickle agrees: the CUDA_EVENT
root model covers 94 852 rows = the whole file.

## 3. Measured constants and retained-cache evidence

- `STEP5_OPERATOR_NAMES` = `('step5_layer_e2e',)` — **1**.
- `STEP5_SCOPE_NAMES` = 9 attention scope names; `STEP5_FEATURE_COLUMNS` = 41.
- Retained pickles (`cache01/result.json`, read-only):

| File | Bytes | hash | target | coverage rows | exact-lookup keys | measurement |
| --- | --- | --- | --- | --- | --- | --- |
| `step5_layer_e2e_b5cab8e1.pkl` | 71 552 314 | `b5cab8e1` | `.median` | **94 852** | 94 852 | CUDA_EVENT |
| `step5_layer_e2e_kernel_count_6a1b9de5.pkl` | 71 551 759 | `6a1b9de5` | `.kernel_count` | **94 852** | 94 852 | CUDA_EVENT |
| `step5_layer_e2e_816d519f.pkl` | 326 497 | `816d519f` | `.median` | 276 | 276 | KERNEL_ONLY |
| `step5_layer_e2e_kernel_count_e422457b.pkl` | 326 084 | `e422457b` | `.kernel_count` | 276 | 276 | KERNEL_ONLY |

- `_ensure_exact_lookup_metadata` (`prediction_model_registry.py:639`) **returns
  immediately** on a cache hit because the retained models already carry
  `_frontier_exact_lookup` (94 852 keys above). Evidenced, so it is excluded from
  attribution: no measured cost on the production path.
- Pickles were written 10:06/10:15, before main07 (15:05); fixture mtime 09:59,
  unchanged. Whether main07 *hit* or *missed* these hashes is **not** determinable
  without the full union frame. Both branches are costly: a hit costs the
  serialisation; a miss additionally runs `GridSearchCV` over 94 852 rows
  (`:1851-1866`).

## 4. Measured multiplicity of the hash/serialisation work

Probe `attn_phase_probe.py` drives the **real** `_train_attn_models_for_cluster`
with a single-role cluster map (`_cluster_configs = {PREFILL: real cluster}`, so the
union has exactly one source: a 300-row head slice written to fresh scratch), the
real predictor config, real model and a fresh scratch model cache. The production
phase runs first on a fresh manager (empty union retention, as at real startup); the
content-sensitivity control runs afterwards under its own counter marks.

Exact command:

```bash
timeout 60s env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-05 TMPDIR=/data/ycfeng/tmp \
  /data/ycfeng/envs/frontier-py310/bin/python -B -u \
  /data/ycfeng/tmp/d29_early_init_repro/attn_phase_probe.py \
  --output /data/ycfeng/tmp/d29_early_init_repro/attnphase07 --rows 300
```

Result (`attnphase07/result.json`): `outcome=completed`, `phase_wall_s=1.392`,
`diagnostic_union_rows=300`, `diagnostic_root_contract_rows_in_union=0`.

| Call | Model name | Hash | Serialised bytes | Payload SHA256 |
| --- | --- | --- | --- | --- |
| 1 (union hash, `step5.py:431`) | `step5_layer_e2e` | `eb2d54ef` | 18 267 117 | identical |
| 2 (duration model, `:1753`) | `step5_layer_e2e` | **`eb2d54ef`** | 18 267 117 | identical |
| 3 (kernel-count model) | `step5_layer_e2e_kernel_count` | `d2bbb379` | 18 267 117 | identical |

Content-sensitivity control, same process: re-hashing the freshly loaded union gives
`eb2d54ef` — equal to call 1 and 2, confirming the control sees the same content and
that the hash is reproducible within a run — while the same frame with one row
removed gives `e2c35839`.

Cross-directory note (measured, corrects an earlier claim): absolute hash values are
**not** stable across scratch directories, because the hash-relevant config embeds
the resolved input paths and the union carries a `step5_attention_inputs` identity
column. Three independent runs gave `cf9c12e8`, `cd863a44`, `eb2d54ef` for the same
300 rows in different output directories, while the RED *pattern* (3 identical
serialisations, `calls[0].hash == calls[1].hash`, control equal to `calls[0]`) held in
every run. Assertions must therefore be on counts and equalities, never on a literal
hash.

**Complete actual invocation multiplicity per attention stage:**

- `to_json()` serialisations: **3**, all byte-identical (measured, identical SHA256;
  18 267 117 bytes for 300 rows = 60 890 bytes/row, projecting ~5.78 GB per pass at
  94 852 rows — projection, not measured).
- `_get_model_hash` calls: **3**; calls 1 and 2 produce the **same 8-hex hash** for
  the same model name — an exact duplicate, including the subsequent
  `_load_model_from_cache` lookup.
- Cache lookups: 2 (duration, kernel-count) — the union hash is not looked up.
- Across one initialization the signature gate
  (`prediction_family_trainers.py:744`, `attention_signature` contains
  `device/model/tp/block_size/arch_profile/family` and the union hash) lets only the
  first stage of each measurement family train: from `mult02`, that is
  PREFILL/CUDA_EVENT and PREFILL/KERNEL_ONLY. DECODE re-enters
  `load_step5_attention_input_union` (so it re-fingerprints the 5.63 GB file) but
  returns at the signature check before hashing.

## 5. RED loop and control: now available

A deterministic, seconds-scale, red-capable loop exists **for the hash phase**, on
real code and a legitimately derived frame:

- **RED** (`attnphase07`, 1.39 s; pattern reproduced in `attnphase02` and
  `attnphase04`): assert `phase_to_json_calls == 3`,
  `len(set(payload_sha)) == 1`, and `phase_model_hash_calls[0].hash ==
  phase_model_hash_calls[1].hash`. It fails on the repeated-work defect while every
  per-model cache filename still matches production.
- **GREEN** would be the same assertions with the duplicate serialisation removed
  and `phase_model_hash_calls[0].hash` unchanged.
- **Control** (same run, one row removed): `e2c35839 != eb2d54ef`, proving the loop is
  not satisfied by a constant; `attnphase03` (299-row run) shows the same at the
  whole-phase level.
- Secondary control on the same seam: within one process, the control re-hash must
  equal `calls[0].hash` (it does, `eb2d54ef`). Cross-process literal equality is not
  a valid assertion (paths enter the identity); see the cross-directory note above.

What this loop **cuts**: the 5.63 GB read, `drop_duplicates`, the `step5.py:247-300`
row loop, the two typed maps, `_validate_targets`, the union `concat`, the real
cache-hit/miss decision for the production hashes, and all downstream
`Simulator`/`Cluster`/scheduling work. It therefore cannot express the 1288 s
symptom; it expresses the mechanism and its multiplicity.

## 6. Revised accounting

Direct measurements (unchanged from the first pass unless marked):

| Unit | Direct | Source |
| --- | --- | --- |
| attention `read_csv` | 207 MB/s (0.572 s / 2 000 rows) | `slice01` |
| `drop_duplicates` | 0.160 s / 2 000 rows | `slice01` |
| `to_json` | 0.289 s / 2 000 rows (120.4 MB) | `slice01` |
| `md5(encode)` | 0.314 s / 2 000 rows | `slice01` |
| per-row validation body | 0.491 ms/row | `attnloop01` |
| `_validate_targets` | 0.321 ms/row | `trainpass02` |
| MoE position admission | **REVISED**: 3.813 s over 7 calls x 3 000 rows = 0.1816 ms per row-visit (the earlier "1.09 ms/iterated row" divided by selected rows, not visited rows), 7 operator calls | `moefilter02` |
| attention hash multiplicity | **3 serialisations, 2 identical hashes** | `attnphase02` |
| root-op contract rows | **0**, yet whole frame hashed | `attnphase02` |

Scaled to the full CUDA_EVENT union (94 852 rows; **estimates**, row-linear):

| Term | Estimate |
| --- | --- |
| fingerprint 5.63 GB | 6.0 s |
| `read_csv` | 27.1 s |
| `drop_duplicates` | 7.6 s |
| per-row validation body | 46.6 s |
| 2 typed-contract maps | 32.8 s |
| `_validate_targets` | 30.4 s |
| **3 x (`to_json` + `md5`)** | **85.9 s**, each pass serialising ~5.78 GB (was 28.6 s before the omission was found) |
| attention union subtotal | **~236 s** |
| MoE preflight (read + full admission) | ~~49.6 s~~ **REVISED, see 6a** |
| layernorm | 1.1 s |
| **explained** | ~~**~287 s of 1288 s (22 %)**~~ **REVISED: ~1035 s (80 %), see 6a** |
| **unattributed** | ~~**~1001 s (78 %)**~~ **REVISED: ~253 s (20 %)** |

The attention term grew by the two extra serialisations. The 78 % residual figure
was an artefact of counting the MoE path once instead of at its real multiplicity;
6a replaces it.

### 6a. Corrected multiplicity and accounting (D29-INIT-CACHE-RECONCILIATION)

Measured read multiplicity, `moereads01` (real `_train_ffn_models_for_cluster`, both
roles sharing one `trained_model_signatures` set, head-slice input so the *counts*
are faithful and the timings unused):

| Call | MoE `read_csv` calls | linear-op reads | Outcome |
| --- | --- | --- | --- |
| PREFILL / one family | **10** | 6 | completes: 12 models |
| DECODE / same family | **1** | 0 | returns after the facts read (signature hit) |

The 10 MoE reads decompose exactly as `frontier/execution_time_predictor/prediction_family_trainers.py:141-146`
(1 unconditional facts read, before any signature check) + `:244-247`/`layer_contract_resolution.py:397`
(2 per-group contract reads) + `:270` (7 per-operator loads through the position branch of
`_get_moe_df_for_op`, which has **no** `moe_df_cache`; the cache at `:252-345` is the legacy branch only).

Derived scaling per measurement family (37 044 rows, measured unit rates):

| Term | Count | Per unit (measured) | Total |
| --- | --- | --- | --- |
| MoE `read_csv` 2.11 GB | 11 | 9.232 s | 101.6 s |
| typed-contract map (pre-repair) | 7 | 35.344 s | 247.4 s |
| position admission | 7 | 6.73 s | 47.1 s |
| **MoE per family** | | | **~396 s** |
| **MoE, two families (CUDA_EVENT, KERNEL_ONLY)** | | | **~792 s** |
| attention CUDA union (§6, incl. 2 fingerprints) | 1 | | ~242 s |
| **explained** | | | **~1035 s of 1288 s (80 %)** |
| **residual** | | | **~253 s (20 %)** |

With the accepted loader repair (typed map 35.344 s -> 3.426 s, measured at 37 044
rows) the MoE term becomes 101.6 + 24.0 + 47.1 = ~173 s per family, ~345 s for two,
i.e. the repair is worth roughly **450 s of the 1288 s window**. That is a scaled
statement from measured unit rates times measured multiplicity, not a measurement of
the repaired run.

## 7. Hash-equivalence evidence, and the retracted proposal

Measured fact: the three `to_json()` payloads in one attention stage are
byte-identical (same SHA256), and the union hash equals the duration model hash
exactly (`cf9c12e8`, control `2c94b972`).

**Retracted (not approved).** The first-pass proposal to hash the same content
column-wise is **withdrawn**. It changes the bytes fed to `md5`, therefore changes
`_get_model_hash` output, therefore renames every `{model}_{hash}.pkl`, and breaks
the retained typed/cache identity that strict 77 depends on. Column-wise hashing is
not equivalent and must not be used.

Any future optimization must (a) reproduce the **exact** existing
`md5(f"{config_str}_{model_name}_{df_json_hash}_{precision}_{measurement}")`
bytes, and (b) preserve content sensitivity — proven by the `--rows 299` control.
The only identity-preserving shape visible from the evidence is reusing the already
computed serialised string / digest **within one stage** for the two calls that
share `(df, model_name, precision, measurement)`; that requires no new cache
namespace and changes no filename. No implementation is proposed or authorized
here, and no line count is claimed until the missing measurement below is made.

## 8. Budget (corrected to the current overlays)

Superseded: the first-pass figures (CORE 3 833 / global 9 076, "44 planned slack")
came from `final_parent/final_ledger_proposal03.json` and are **stale**.

Historical baseline of this task, recorded for context only and **not** a new
budget request; the authoritative caps are the overlay chain named below:

| Figure | CORE (core_source) | global |
| --- | --- | --- |
| cap (unchanged) | 4 404 | 10 000 |
| measured at source e41 | **3 873** | **9 116** |
| group-reuse candidate `88fe` | **3 932** | **9 175** |

Overlay chain now applied after `final_scope_reconciliation.json`:
`cli_placement_scope_amendment.json` -> `model_group_reuse_scope_amendment.json` ->
`init_validation_scope_amendment.json` (`24f970cb`, D29-INIT-FAMILY-RESOLUTION,
repair delta 45: source 6, test 39; `test_moe_grouped_gemm_padded_features.py`
210 -> 237, `test_moe_routing_runtime.py` 70 -> 53,
`layer_contract_resolution.py` 25 -> 15) -> `collective_artifact_parse_scope_amendment.json`
(pricing, delta 50: source 15, test 35). All caps are unchanged; every allocation
is zero-sum inside its owner.

Path ceilings relevant to any future work in this area: `prediction_model_registry.py`
24 free, `execution_time_predictor/step5.py` 17 free, `moe_routing_runtime.py` 2
free, `moe_dataset_training.py` 2 free, `attention/step5.py` (pricing) 0 free;
`shared_prediction_model_manager.py` is **not** in the 111 ledger.
No budget is consumed or requested by this task.

## 9. Remaining gap and the one supported next measurement

**REVISED by the third pass.** The cache-hit/miss half of this gap is now closed by
§12: every model lookup in the retained window hit the cache, so no GridSearchCV ran.
The remaining gap is the ~253 s residual of §6a, not 78 % of the window.

Still unknown, and not closable by any head slice: the composition of that ~253 s
(linear-op, residual and communication sub-stages, GDN predictors, model unpickling,
and any stage after `_publish_step5_model_bundles`).

Missing seam, unchanged in shape: an authorized phase census over one full
`_train_all_required_models` at real scale. It cannot run under this task's 60 s /
30 GiB caps, because the phase itself is 1288 s. Two bounded substitutes exist and
neither was run: (a) a no-instrumentation stage census from the existing INFO lines
(`shared_prediction_model_manager.py:445-450`, `prediction_family_trainers.py:206`,
`:331`) on a run the primary authorizes with a longer cap, and (b) further unit
probes, which have reached the point of diminishing returns because the two large
terms are already accounted for in §6a.

## 10. Artifacts

| Artifact | SHA256 |
| --- | --- |
| `slice_rate.py` | `475fdf041b74ac4ec3c0a92e78937438dc2e5e30ed6310c52335f9883957c689` |
| `attention_row_loop.py` | `3ffd898383bae9799dc291e9d09eafe756a2e1273e27b5b98aa3688e490443fe` |
| `moe_filter_probe.py` | `c2b6d0d4341ae4ee520e26a1c7457bb39c72c02b78d2c374cd37000a63b1aeb9` |
| `multiplicity_probe.py` | `3ebbe8e17090e55d14224a50753308d49987b7fa11e07f6abe0f5395758a15f4` |
| `train_passes.py` | `050431104e9735296248c2538430707c97d41aad1d38c1b7221763c39f1a0c16` |
| `cache_and_sample_probe.py` | `e358781a2176078e91c65ce5110eb2bc8f0f9b6f438df746cbf9b61e68d88ec2` |
| `attn_phase_probe.py` (final) | `843d7c730bd924a86babf5f298cc8b7478144ef4f5839b625d76db95dca69d2a` |
| `hash_phase_probe.py` (superseded, wrong context) | `42cd6eb875e62b7aa5a6b8a8ad5b19c6e482bc18b5d7ed6ec21d264275f9d0d8` |
| `slice01/result.json` | `294659b9799597d292ae411aa962993116693ff3c1474aa2fe82820f2208e096` |
| `attnloop01/result.json` | `e477986a4a9c2b98b540f6f8f097b10cac25c7781536479e87279c32501d51e9` |
| `moefilter02/result.json` | `a7716edf8bb04436bd15c38fe388db5ebffbfa7bed3ef0cd94854816cb374428` |
| `mult02/result.json` | `63703de1d831d6cb58f1b11826edba34379c261d50375d2014ee23d71a1851a6` |
| `trainpass02/result.json` | `28299e9616a08437c4ed12741adff0d979c56b39674c132c03a7787a618ae2af` |
| `cache01/result.json` | `3032c8a47126c8b5c36df60ed62ae7f5fdfa87ee0de96d672e69c6fad4f4614f` |
| `attnphase02/result.json` | `1558a4ac6e9e5d263b3505ea1cb168dcc02217046d83b8a1facf3de19c90e6e5` |
| `attnphase03/result.json` | `83765eba4a4919a172724471a95306b28c2f3f2e6fcdb1e66c6d2a681131ae19` |
| `attnphase07/result.json` (final clean run) | `40a01009dc5f4a6ab868805f23c1459f356ea4aeabdaff33ac30853924dcab1b` |

Inputs and retained artifacts unchanged: `fixture_attention_cuda_event.csv` mtime
`2026-10-23_09:59:05`, `step5_layer_e2e_b5cab8e1.pkl` mtime `2026-10-23_10:06:00`.

## 11. Unresolved limits

- **REVISED (third pass):** the residual is ~20 % (~253 s, §6a), not 78 %. The MoE
  term is now scaled from measured multiplicity (`moereads01`); the linear-op,
  residual, communication and GDN sub-stages, and model unpickling, remain estimated
  only by omission.
- **REVISED (third pass):** cache hit versus miss is **settled as hit** for every
  lookup in the retained window (§12). What remains unverified is only the strength of
  the absence-of-warning argument, not the outcome.
- `hash_phase_probe.py` is retained as a superseded negative result: it passed
  `step5_generic_compute`, which activates an operator-selection path the attention
  stage does not use, and raised. The production context has neither
  `step5_generic_compute` nor `operator_facts`, which is why the whole union is
  hashed.
- Sparsity of the contract census (12 spaced samples + head + tail, 0 rows declaring
  `step5_layer_e2e`) supports but does not prove the zero count over all 94 852 rows.

## 12. Cache reconciliation: were the production lookups hits or misses?

TaskID **D29-INIT-CACHE-RECONCILIATION**. Read-only. No source, input, cache byte or
prior run was changed; the four attention pickles and both CSV inputs keep their
original mtimes. Scratch: `/data/ycfeng/tmp/d29_init_cache_reconciliation/`.
Bundle parsing read JSON metadata only (20 files x 4.2 MB); no pickle body and no CSV.

### 12.1 Observations

1. `/data/ycfeng/tmp/d29-parent-position-fixture03/cache` holds **20 bundles, 38
   models each**, 10 `prefill` and 10 `decode` (`census.json`, `roles`).
2. The four bundles written at the end of main07's training phase are
   `3e462326` 15:26:43 prefill, `48e35a87` 15:26:43 prefill, `3d960f88` 15:26:44
   decode, `52781c91` 15:26:46 decode. **Correction to the briefing as I received
   it:** they do not all name the same attention caches. They name the same *pair per
   measurement family*, shared by both roles:

| Family | `step5_layer_e2e` | `step5_layer_e2e_kernel_count` | mtime | bytes |
| --- | --- | --- | --- | --- |
| CUDA_EVENT | `b5cab8e1` | `6a1b9de5` | 10:06:00 / 10:06:52 | 71 552 314 / 71 551 759 |
| KERNEL_ONLY | `816d519f` | `e422457b` | 10:15:17 / 10:15:17 | 326 497 / 326 084 |

   The `coverage` recorded on the CUDA pickles is 94 852 rows (the whole file), the
   KERNEL_ONLY ones 276 rows — consistent with §3.
3. **None** of the 96 `.pkl` files has an mtime inside 15:05-15:30; the window's
   buckets are empty and the newest pre-window writes are 12:38. Total pickle bytes
   are 178 MB, of which the two CUDA attention models are 143 MB.
4. main07 `run.log` (offline and online, 4 839 / 4 838 bytes, continuous from
   15:05:16 to 15:30:57) contains **0** occurrences of `MODEL CACHE MISS` and **0** of
   `CACHE MISS: Training`, and exactly 12 `Profiling metadata mismatch` lines
   (2 per op, 6 ops) starting 15:26:46.

### 12.2 Real code and config behaviour (read from e41)

- Miss path: `prediction_family_trainers.py:1841` `logger.warning(error_msg)` — the
  "MODEL CACHE MISS ERROR" block is logged at WARNING, so `--log_level warning` (the
  actual main07 argv) does **not** hide it. Then `:1849-1866` builds and fits
  `GridSearchCV`, then `:1896` `_store_model_in_cache(...)`.
- The stale comment at `:1792` ("we expect all models to be pre-trained") is
  contradicted by the live code: `:1837` says real-time training is temporarily
  enabled, and the fit is unconditional on a miss. The comment is historical; the code
  is authoritative.
- Hit path: `_load_model_from_cache` (`prediction_model_registry.py:611-630`) returns
  `None` only when `{model_name}_{model_hash}.pkl` is absent (the `expected_sha256`
  check is optional and the call site at `:1757` passes none).
- `_ensure_exact_lookup_metadata` (`:639-663`) **can** write: it falls through to
  `_store_model_in_cache` when `_frontier_exact_lookup` is missing. It does not here:
  the retained CUDA pickles already carry 94 852 exact-lookup keys and the
  KERNEL_ONLY ones 276 (`cache01`, previous task). That is also why mtime is a valid
  witness.

### 12.3 Verdict (inference, with its support and its limit)

**All model lookups in the 15:05:16-15:26:46 window hit the cache; no model fitting
or GridSearchCV ran there.** Support: (i) the miss path is a mandatory WARNING at the
run's own log level and it never appears; (ii) every pkl mtime predates the run and a
miss or a metadata migration would have rewritten the file; (iii) the hashes published
by the run's own bundles at 15:26 are exactly the files on disk; (iv) the bundles were
written into `fixture03/cache`, which proves that directory was the active
`_cache_dir` for main07, so a miss would have deposited its pickle there.

Limit: this is inference from absence-of-warning plus mtimes, not from observing
`_load_model_from_cache` return values; an actor that rewrote a pickle with identical
bytes and restored mtime would defeat it, and none is known. The run log shows no
truncation or rotation. Consequence for §6a: the "cache-hit phase" costs only
unpickling (<=178 MB), which is why it does not appear as a separate term.

### 12.4 Corrections this pass makes to earlier statements

| Earlier statement | Corrected |
| --- | --- |
| attention input read twice (prefill and decode) | read **once per measurement family**: `load_step5_attention_input_union` builds sources from `{*manager._cluster_configs, cluster_type}` (`step5.py:331-334`), so both roles share one union; the second role re-enters and pays only `dataset_fingerprint` before the retention check |
| MoE preflight ~50 s | **~792 s** for two families: 11 reads + 7 typed maps + 7 admissions per family, multiplicity measured in `moereads01` |
| admission 1.09 ms per iterated row | **0.1816 ms per row-visit**; the earlier figure divided by selected rows |
| 22 % explained / 78 % residual | **~80 % / ~20 %** (§6a) |
| hit/miss unknowable without the full frame | **hit**, per 12.3 |

### 12.5 Is a new bounded whole-manager census justified?

Not as a *new* measurement of the two large terms — those are now covered by measured
unit rates times measured multiplicity. It would only be justified for the ~253 s
residual, and it cannot be executed inside this task's 60 s / 30 GiB caps because the
phase is 1288 s. The cheapest faithful route is the existing INFO instrumentation on a
primary-authorized full run; I did not run one, and no optimization is recommended on
the strength of the residual.

### 12.6 Binding

Historical claims in this report remain bound to frozen source
`e41abc073500d61f224eda4fca1123e74e40544d` (`verify-step5-d29-integrated-05`). The
current merged source on `feat/step5-preview` (this records-only worktree is
`eece87b2`) was **not** used for any measurement here. No per-column hash change and
no other optimization is proposed.

### 12.7 Artifacts (this pass)

| Artifact | SHA256 |
| --- | --- |
| `bundle_census.py` | `0114d2ac45944c6e9721183914968b2272fa637c79f52dfc56f02103d1e8523a` |
| `moe_read_census.py` | `09be9efa27a5534af29ef365daea4489f254e577497bf8a3d944df27fd25fdd8` |
| `census.json` | `e8e4d7c3336ece972629e831afe3d5812b82f82aa6eb31fccacfdbf4ad035715` |
| `moereads01/result.json` | `1e1681a6b4fc1dcf1041f0e79a61aedc326be4460903b9653cdfdcfa55252bc1` |
