# Preparation plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Defined source-backed preparation and root-owned launch boundaries. |

Dependency: `inspect accepted builders/inputs -> builder + explicit package plan -> fresh package -> {entry/hash/plan controls, affected CPU suites, existing matrix binding} -> scoped code commit -> records commit -> root review/image/seal/launch`.

1. Verify the actual accepted input manifests and source pins. Read the existing prefill builder and q1 launcher/bootstrap/worker/case builder. Preserve all original inputs.
2. Add only `prepare_package.py` and `package_plan.json` within 300 gross changed lines. Pin accepted source `97803f07`; snapshot existing Frontier/config files and use the accepted flat balanced registration wrapper. Generate only a fresh destination.
3. Verify all generated manifest inputs, native/C26/model/strategy pins, job resources, exact q1 window, runtime entry paths and phase commands. Derive rows/calls/steps from the current plan. Fresh-process entry checks use only CPU stand-ins and stop before native execution/platform operations.
4. Run existing q1/profiling and balanced-routing tests; bind the existing 77-case comparison to byte-identical simulator/data/examples/matrix domains. Test rejection of source/routing corruption and a second generation attempt, without changing the actual package.
5. Commit builder/config after checks, archive generated records separately and stop writing. Deliver hashes, exact commands/results, changed-line accounting and unknowns. Root must perform actual image/native checks and independently decide seal/launch readiness. No GPU/serving/admission qualification is claimed.
