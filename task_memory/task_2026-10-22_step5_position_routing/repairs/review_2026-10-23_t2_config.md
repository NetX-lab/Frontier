# T2 engine binding acceptance review

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Preserve independent No findings and parent full-file rebinding closure. |

TaskID: D29-T2-ENGINE-LIMITS-REVIEW.
Reviewer: native OpenCode reviewer, session ses_e9afdb7c4ffejb3hbkjZCeHFKS.
Frozen source:662849e99223c98b9ea0d47d34783c3977cb3282, baseline6ebb038.
Artifact: [t2_config/limits.diff](t2_config/limits.diff), four paths,
SHA2567b0a769a0d290dd6d65c6e5d50a7408df2c1f64310409e074f10e6ecc4925a10.

## Independent disposition: No findings

- step5_dispatch_probe.py:328–348 passes all supplied limits before EngineArgs;
  omitting the optional mapping preserves original kwargs/eager/topology/quant flow.
- test_position_cyclic_graph.py:253–295 requires controlled7168/8192/128 input,
  uses it in both constructors and observes final post-with_hf_config values.
  Drift appends a soft diagnostic; native/constructor exceptions remain native.
- Existing-owned tests cover actual entrypoint/helper calls, final objects, both
  role drifts, missing input, exception identity and old kwargs.
- Published EngineArgs/model/scheduler/with_hf_config and fused_moe/layer.py:274,361
  support the binding. No contradiction in controlled inputs, metadata or proposed
  resources. Thin graph/MRv2/backend/G3 boundaries and native four files unchanged.

The reviewer read requirements/plans, exact diff, native records, preparation
artifacts and installed interfaces. It did not run tests, rehash the supplied
diff, use platform tools or modify files. Its response arrived without the later
parent actual-image result in its inspected evidence set.

## Disclosed prerequisite and parent closure

The prepared provenance referenced3f54bc6's complete helper file, while extraction
only established selected-function AST equality. Reviewer correctly required
binding the actual complete integrated files before final-source acceptance.

Parent has now compared each complete bound script/helper against Git662849e and
the frozen checkout; updated provenance, command source path and six-entry index;
retained original prepared identity as history. Actual integrated helper SHA256:
8284363ed6c4090fbd57f5f9c00bac32f0b3183790f56e2c85e452e24bc9afc4.
All6 input checksums and bash syntax pass; cases/metadata bytes remain unchanged.
This closes source rebinding for this controlled bundle, not an allocated run.

Parent independently observed177 PASS/15 SKIP and real published-image CPU config
construction for both roles at7168/8192/128. See
[test_report_2026-10-23_t2_config.md](test_report_2026-10-23_t2_config.md).

## Acceptance boundary

Accept constructor fidelity and this controlled source/input binding. Runtime
GPU config, memory fit, backend/graph execution, full serving/G3/E2E remain
unverified.16H800/1800s is an unapproved proposal; no allocation or launch occurred.
