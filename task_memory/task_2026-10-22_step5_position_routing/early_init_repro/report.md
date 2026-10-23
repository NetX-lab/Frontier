# D29-EARLY-INIT-REPRO report

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Initial read-only investigation of the unexplained 21.5-minute initialization window; bounded unit probes and budget reconciliation. |
| 2026-10-23 | Second pass: added the omitted Step attention training/cache-hit phase and its hash multiplicity, retracted the column-wise hashing proposal, corrected the budget to the current overlays. Earlier observations are preserved; superseded numbers are marked **REVISED**. |

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
| **predictor training finished** | `cache/bundle_3d960f88….json` written **15:26:44** |
| first profiling-metadata warning | 15:26:46 `quantization_manager.py:814` op=attn_pre_proj |
| last metadata warning | 15:30:57 op=moe_grouped_gemm (cluster 2) |
| CLI `--log_level` | `warning` — nothing else is logged in this window |

**REVISED anchor.** `_publish_step5_model_bundles()` runs immediately after
`_train_all_required_models()` (`shared_prediction_model_manager.py:112-113`), and
the retained bundle `bundle_3d960f88a3aecb8fd3e5b732f251bec594838d257e3c044d6e1aa93dfadc4934.json`
has mtime 15:26:44 — two seconds before the first metadata warning. Therefore the
unexplained window **15:05:16 -> 15:26:44 (1288 s)** is entirely predictor-model
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
| MoE position admission | 1.09 ms/iterated row, 7 operator calls | `moefilter02` |
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
| MoE preflight (read + full admission) | 49.6 s |
| layernorm | 1.1 s |
| **explained** | **~287 s of 1288 s (22 %)** |
| **unattributed** | **~1001 s (78 %)** |

The attention term grew by the two extra serialisations, but the 78 % residual is
still the dominant unknown, and it is not a full-frame-only artefact of the
attention union: the window also contains every other predictor stage for both
clusters and both measurement families (dense/linear, MoE, residual, communication)
whose full-scale costs were not measured here.

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

Current, per the latest overlays and the primary's measurements:

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

Still unknown, and not closable by any head slice: the 78 % residual, and whether
main07's production model hashes were cache hits or misses.

Missing seam: one instrumented, authorized run of **one** attention stage on the
real CUDA_EVENT union — the same phases as §6 (read, dedup, row loop, typed maps,
`_validate_targets`, concat, and the three serialisations) with monotonic timers and
`getrusage` around each — plus the actual `_load_model_from_cache` verdict for
`b5cab8e1`/`6a1b9de5`. Bounded at the assignment's 60 s per command by splitting it
into per-phase commands over a persisted intermediate frame, or by one run of
~240 s only if the primary raises the cap; I did not do so.

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

- The 78 % residual is unattributed; the MoE/dense/communication stages for the
  second cluster and the second measurement family were not measured at full scale.
- Cache hit versus miss for the production hashes is unknown; a miss would add
  full-union `GridSearchCV` time and is not bounded here.
- `hash_phase_probe.py` is retained as a superseded negative result: it passed
  `step5_generic_compute`, which activates an operator-selection path the attention
  stage does not use, and raised. The production context has neither
  `step5_generic_compute` nor `operator_facts`, which is why the whole union is
  hashed.
- Sparsity of the contract census (12 spaced samples + head + tail, 0 rows declaring
  `step5_layer_e2e`) supports but does not prove the zero count over all 94 852 rows.
