# D18-NATIVE plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded scoped implementation and offline qualification sequence. |

Dependency: `primary D18 + locked source -> existing-client extension -> {canonical phases, H200 configuration} -> HTTP/regression tests -> source commit -> immutable stages -> stage checks -> evidence commit`.

1. Inspect the primary D18/spec/plan and the locked Step API/configuration; compare GLM's existing canonical contracts read-only.
2. Extend the single existing HTTP client only for opt-in token-ID transport and failure retention. Build Step repetition orchestration around that client and the existing `pd_proxy.build_app`.
3. Add full-role H200 serving configuration and qualification/canonical stage policies. Keep every frozen package unchanged. Use native graph dispatch logging, not custom engine hooks.
4. Verify exact package pins/populations, actual P/D HTTP forwarding, stable prompts, counts/timestamps/metric windows, open-loop dispatch and negative controls; rerun legacy/GLM/Step harness seams. Simulator metric regression is narrow, not a simulation sweep.
5. Commit the verified source. Build distinct qualification and canonical stages with copied model metadata, probe pins, native-source bindings, preflight argv/environment, submit wrapper and offline receipt.
6. Record exact stage paths, parent-only commands, resource/runtime estimates and unverified runtime dependencies. Commit task-local evidence. No remote operations.

Parent-only next steps: fresh H200 quota check and image/source identity -> qualified 64-H200 native startup, per-lane graph dispatch/capture and same-shape streaming -> parent R2 disposition plus qualification acceptance -> three-repetition canonical collection -> existing metric admission/comparison. Stages do not automatically close these dependencies.
