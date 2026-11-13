## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 HKT | Close the recovered init diagnosis with source and retained receipts only. |

# Final diagnosis

**ALLOC**: recommend dataframe and full-row typed-contract preparation reuse within one selected-group call of shared `_train_ffn_models_for_cluster`. No source allocation or implementation was performed. Current startup benefit is **UNQUALIFIED**; no missing prerequisite blocks this finite source diagnosis.

## Binding and reachable repetition

Inspected candidate: `1461573de388b72450189bc0c1b88b4ac32491f0`. Records baseline/parent: `886608ea574dd2107d0b30771d63d8057d23cac6`. All thirteen source/test files in [evidence.json](evidence.json) are byte-identical to baseline886 and committed production baseline `2e8d6168fd72d13955216d59320c9f68b306eca5`; actual primary bytes also match. Their targeted886-to146 diff is empty. This does not claim that the entire886 checkout equals the ledger candidate. Primary documents advanced to `2207331e17a07c7da2186c32f7b37cc6ba2f3890` during preparation; their snapshot is separate from the production code binding.

Root facts are already bound once from a default-parser read (`prediction_family_trainers.py:141–147`); inner group signatures are checked at `:156–185`. Group preflight reads another default-parser frame and filters each operator (`layer_contract_resolution.py:383–401`). The position operator branch (`prediction_family_trainers.py:268–275`) reloads for each operator, bypassing the legacy dataframe cache. `_load_moe_df` uses Step `round_trip` (`profiling_dataframe_loaders.py:690–696`) and validates the complete typed column on every operator call (`:705–732`). Its architecture-family derivation at `:720–722` is the already accepted repair, not a new saving.

## Smallest safe seam

Prospective production paths: `prediction_family_trainers.py` and `profiling_dataframe_loaders.py`. Lazily prepare one full `round_trip` dataframe and model-bound full-row typed validation inside each inner group call, after signature/preflight checks. Keep standalone loader calls fresh and retain no frame on the manager, registry or fitted model. Share none across groups, roles, measurement families or later calls. Preserve the distinct default-parser facts/preflight path. The full frame remains resident through that group's operator fits and ends on return/exception; its added residence and peak-memory effect are unmeasured.

Keep per-operator loader/facts agreement (`profiling_dataframe_loaders.py:734–739`) and every physical admission check (`moe_routing_runtime.py:77–143`): applicable native binding and selected TP/EP, complete typed targets, opposite-frame absence, positive duration/integral count, physical=live+masked, source-row validity, selected typed identity, gating context and precision. Keep feature and measurement binding, conflicting measured counts, duration/count exact-key parity and cache identity (`prediction_family_trainers.py:1632–1685,1744–1777`). No prediction, fit, numerical output or pricing proof is memoized by this recommendation.

The independent trainer rereads on group recursion and repeats preflight filtering (`moe_dataset_training.py:375–397`). Metadata registration separately rebinds/reads per operator (`:584–597`); manager-backed initialization can invoke it (`sklearn_execution_time_predictor.py:685–686`). These observations receive no extra timing credit or cross-invocation reuse recommendation. Shared root facts receive no new saving either.

## Measured and inferred contribution

Historical e41 `moereads01/result.json`: ten PREFILL MoE reads plus one DECODE facts-only read per family; timings explicitly unused. `moefilter02/result.json`:3000-row slice, two groups with3/4 operators, read0.804030907s, facts0.585296253s, seven filters totaling3.812784798s. Its47.080266691s projection at retained37044 rows covers **all seven filters**, not one operator pass.

For this seam, source plus historical multiplicity imply eleven->six reads and seven->two typed maps per family: root, preflight and DECODE reads stay; seven operator preparations become two. `early_init_repro/report.md:241–258` retains rounded measured units9.232s/read and **post-loader**3.426s/map. Two historical families give `2*5*(9.232+3.426)=126.58s` of potentially removed work. This is a model from retained evidence, not current startup timing, an upper bound or a guaranteed saving. The already accepted roughly450s loader estimate is excluded. Admissions, facts, metadata, fit/hash/cache costs and whole-workload RSS receive zero new credit. The2400s CLI criterion remains unqualified by this diagnosis.

## Future RED design; not executed

Two existing test paths: `tests/unit/test_moe_grouped_gemm_padded_features.py` and `tests/unit/test_step5_measured_ffn.py`. Existing source covers unselected malformed typed rows and round-trip parity (`:300–331`), both trainers plus cold/warm hashes and exact keys (`:334–404`), invalid physical/opposite-frame/typed cases (`:416–427`), and both routed precisions (`test_step5_measured_ffn.py:223–249`). These test sources were inspected, not executed.

The smallest future RED assertion instruments preparations in the existing producer-valid two-group fixture: each fresh group shares one round-trip preparation among its operators; a later standalone/group call reads and validates anew. Present repeated preparation would fail that assertion. Compare exact selected frames, hashes, measurement identities and duration/count keys; retain malformed-unselected, opposite-frame-zero, source-conservation, mismatched-facts, rebound-identity and changed-input freshness cases. No probe was added/run. Two source/two existing test paths are an estimate, not allocation; gross lines remain unset until a real patch and the existing owner-cap process.

## Receipt and limits

Consumed technical controls: **0 new,0 recovered**. Recovered history:33 completed tool calls,1 permission rejection,0 preceding owned writes; analyzer availability is unchanged. Required shapes stay6/6 with0 added. No application imports, new datasets/models/traces, runtime tests, GPU/platform/network/cloud operations or delegation were used. Source/records-only verification is in [test_report_2026-10-25_init_preparation.md](test_report_2026-10-25_init_preparation.md). The records commit preserves886 as its parent; its SHA is returned externally to avoid a self-referential artifact. Primary subsequently confirmed ledger146 admission at production2ca5e571 with records2207331e. That admission does not implement this seam or qualify startup savings; whole CLI, memory and native/G3/E2E remain primary-owned gates.
