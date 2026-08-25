## Modification History

| Date       | Summary of Changes |
|------------|--------------------|
| 2026-08-25 | Added explicitly deferred post-PR17 work that is outside the A' lane contract and merge-conflict repair. |
| 2026-08-22 | Added evidence from the producer-metadata and MTP unified-interface audits and recorded the remote PR description synchronization. |
| 2026-08-22 | Added the PR #21 Option-B deferred WATCH items: true-mixed block rounding, duplicate targets, CLI derivation, strict integer APIs, and exception narrowing. |
| 2026-08-22 | Added the separately scoped MTP unified-family/interface audit requested after the focused PR #20 repair. |
| 2026-08-21 | Listed work explicitly outside the current scoped PR. |
| 2026-08-21 | Deferred an independent profiling-facing adapter and broad profiling-module relocation unless later evidence requires them. |

# Future Work

The following post-PR17 items remain outside A' and require their own evidence
and design decisions:

- Full profiling-schema redesign and canonical CSV regeneration for every new
  EP divisor; A' fixes the runtime descriptor, not measured-data publication.
- Complete MTP family/interface consolidation across registry enumeration,
  profiling plans, kernels, quantization, and runtime policy.
- A broader scheduler identity/ownership refactor beyond the exact PR #17
  stage/KV fields that A' must preserve.
- Unsupported or non-equal expert ownership topologies; the current runtime
  contract requires divisible equal-width ownership.
- The existing broad post-PR17 unit baseline failures unrelated to typed lanes.

The following items are intentionally outside this session and must not be
quietly added to the scoped PR:

- N3/N4/N5 incremental profiling database, preflight runner, and coverage-report
  publication workflow.
- Canonical profiling CSV publication, merge sidecars, compatibility aliases,
  and repair of historical provenance.
- Broad H800 data collection and full heterogeneous PDD/PD-AF release matrices.
- A complete migration of every profiling-plan/trainer/parity catalog after the
  first `bind_operator_query` slice, if that migration proves larger than this
  PR.
- An independent profiling-facing adapter or a second operator catalog, unless
  `bind_operator_query` cannot represent a concrete timing-owner mapping.
- Relocation of shared CPU/PP CSV schema and validation helpers unless a small
  extraction can be proven without duplicating producer/consumer invariants.
- Product decisions for MLA D24 identity, non-KV-cache-memory boundary ownership,
  or new timing families.
- MTP unified-family/interface audit: `mtp_fusion_proj` and `lm_head_linear`
  are separate physical `ColumnParallelLinear` operators, currently declared
  in the dedicated MTP registry rather than the generic `OperatorFamilySpec`
  registry. The focused TP-consumer repair is correct, but registry extension
  does not yet propagate to every model-name and profiling-plan enumeration
  site, and the method contract tuple can diverge from the module-level helper.
  Decide separately whether to add an MTP family/interface and how to make
  method-specific policy the single source of truth. Trace enumeration, CSV
  schema, profiling kernels, quantization defaults, runtime policy, and TP
  consumers before any cross-cutting change.
- Standard-attention Option-B is complete for this PR. Keep these review WATCH
  items separate:
  - true-mixed aggregate-token filtering versus rounded paged-block allocation;
  - duplicate `(model, TP)` CLI targets that repeat work and undercount
    `total_work`;
  - direct CLI derivation when only `--max_seq_len` is supplied;
  - strict rejection of non-integral values in programmatic helper APIs;
  - narrowing broad automatic online-grid exception catches;
  - aligning `num_tokens_list` and `extra_num_tokens` capacity semantics.
- Predictor accuracy tuning or calibration methodology.
- Producer-side optional metadata contract (`SCOPE-011`/`SCOPE-021`):
  production currently writes feature names, model hashes, and exact lookup
  rows, but no bounds/domain/constraint/identity metadata. Standalone
  `BaseTrainer` also pickles only the raw estimator. A future repair must
  define one descriptor schema, persist it through every producer/cache path,
  pass it through runtime `model_info`, and verify the consumer after a
  serialized-cache reload.
