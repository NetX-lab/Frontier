# D23-MRV2-INPUTS plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-22 | Define CPU implementation and verification edges. |
| 2026-10-22 | Complete steps1–5; source commit ba484b0 and final CPU evidence retained. |

1. Re-run inherited producer and affected measured suites; create a red-capable real producer→consumer regression.
2. Centralize kind label/code/scope classification and extract existing MRv2 descriptor unchanged. Validate producer kind/physical provenance and preserve query/context/padding semantics.
3. Carry the kind feature and kind-bound physical scopes through existing loader/input union/training/cache and cost lookup; retain isolated diagnostic runtime binding.
4. Repair only affected unit fixture source inconsistencies; verify NONE/FULL/PIECEWISE mixed keys, duration/count cold/warm cache, missing-kind failures, unions and bundle identities.
5. Self-check progress, document exact commands/results/limits, commit verified work and hand off clean tree.

Dependency: `1 -> 2 -> 3 -> 4 -> 5`. Baseline producer and measured commands can run independently; no delegated work. Parent owns scheduler/native binding and whole-main/fidelity revalidation. Use existing Python environments and `/data/ycfeng/tmp` for raw logs/caches; retain selected evidence here.

All five steps are complete. Source/tests commit `ba484b0d9e1534f5f8e742127addee5cafe62560`; producer467 PASS, measured263 PASS, focused49 PASS, actual13 descriptor groups/26 CPU rows PASS. No scheduler/native qualification or accuracy acceptance is claimed. Full commands, intermediate failures, final results and limitations are in the test report; completion archive is `summary.md`.
