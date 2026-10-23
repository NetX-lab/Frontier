# V4 private attention input boundary

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Root bounded the approved private-input implementation before core edits. |

Decision: D-V4-KERNEL-ONLY-GPU-COST; D-Q10 measured-consumer authority. This design implements the input prerequisite, while the independent audit establishes whether the measured operators support a total.

## Interface and responsibility

The existing selected attention CSV remains the only selection surface. A private `dsv4_attention.json` sidecar beside it carries per-table/per-layer flat layouts, independently derived by the existing `build_isolated_prefill_layouts()` from pinned plan and non-timing engine/step records. It contains no derived serving total. CSV/raw/summary/model/plan lineage is bound by hashes. Original accepted tables, raw files and packages are immutable; exporting creates a fresh directory.

Use a small `frontier/execution_time_predictor/dsv4_attention_input.py` reader, following the established `dsa_full_dummy_input.py` boundary. It checks a separate externally reviewed `.admission.json` binding the sidecar and referenced evidence. The admission is explicitly isolated-input only and cannot assert serving admission. A preparation tool exports a candidate sidecar; it never issues its own PASS admission. Root admits the representation only after verifying its layouts against the existing independent builder and all94 accepted isolated results. Negative-control fixtures may supply test admissions; they are not production qualification evidence.

In `read_isolated_scope()`, an omitted prefill layout may be loaded from the selected CSV's sidecar when that sidecar exists. Explicit-layout callers remain unchanged. Missing sidecar retains the existing typed missing-layout error; present-but-invalid sidecar/admission yields a typed error without fallback. The existing exact column/layout/source/routing/phase/graph checks remain authoritative. CUDA_EVENT stays readable for isolated diagnostics; KERNEL_ONLY supplies the approved operator-cost route. Neither path enters serving totals in this change.

The non-core preparation tool owns package/CLI-plan reconstruction and artifact copying; core code does not parse a GPU job command or import the CLI to recover expected values. Move/reuse the existing accepted-prefill input preparation from `check_dsv4_measured_scopes.py` rather than maintaining two copies. The checker must exercise both explicit-layout compatibility and the automatic selected-input path. Tests invoke the real writer from the current repository and validate external retained references against recorded hashes.

## Gain and limits

Gain: the actual CSV selection can carry its independently qualified contract into the existing reader without test-only caller injection. Existing path overrides and classification mechanisms already handle selection; no shared field, registry or general plugin is added. GPU timing, overlap, completeness and graph equivalence are separate evidence, so this component never substitutes for them.

## File allocation

- Implementation owner: new private reader plus `frontier/execution_time_predictor/deepseek_v4.py`; at most2core files/180changed lines.
- Same owner: one preparation tool under `tests/comparison/calibration/`, existing `tests/integration/check_dsv4_measured_scopes.py`, and existing `tests/unit/test_dsv4_graph_measured.py`; target3non-core files/350changed lines. Report exact deltas before scope adjustment.
- Audit owner writes only its analysis script/audit records; source explorer writes only native_profile_seam.md; root owns shared task requirements/design/plan/progress.

## Verification

Prepare a fresh candidate from the accepted source receipts with original hashes intact. Validate all layouts independently and root-bind an isolated admission. Selected-input discovery must reproduce44KERNEL_ONLY and50CUDA_EVENT exact medians with `serving_admission=false`; explicit-layout and legacy102/72 paths remain valid. Exercise missing/altered admission, sidecar, CSV, layout/source/routing and unsupported phase/graph cases. Run affected unit suites and required fixed-source default-matrix checks. Review source before merging. Missing full-coverage evidence continues to reject serving prediction.
