# W128-V4-ATTENTION-INPUT archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Archived verified CPU component pending fixed-source capture and root acceptance. |

## Outcome

Implemented a V4-private sidecar boundary for isolated prefill layouts. The selected CSV can discover externally admitted per-table/per-layer layouts without caller injection. Existing explicit-layout callers and missing-sidecar typed failures remain unchanged. Invalid present sidecar/admission/evidence rejects; no serving duration is introduced.

## Deliverables

- `frontier/execution_time_predictor/dsv4_attention_input.py`: reviewed isolated-only admission, sidecar/evidence/original-source bindings and selected layout retrieval.
- `frontier/execution_time_predictor/deepseek_v4.py`: omitted-layout discovery integrated with the existing exact guards.
- `tests/comparison/calibration/prepare_dsv4_attention_input.py`: validates accepted sources, reconstructs plan using local code, calls the independent layout builder and exports a fresh candidate without admission.
- `tests/integration/check_dsv4_measured_scopes.py`: reuses preparation and supports selected-input verification against original accepted inputs.
- `tests/unit/test_dsv4_graph_measured.py`: current writer/actual94 dual-path equality and typed refusal fixtures. Test admissions explicitly disclaim production/GPU qualification.
- Local records: `task_memory/task_2026-10-23_v4_remaining_measurements/input_bundle/{requirements,plan,progress,summary,test_report_2026-10-23_input_bundle}.md`, `verification_receipt.json`, `explicit_comparison.json`, `automatic_fixture_comparison.json`, `legacy_comparison.json`.
- Actual candidate: `/data/ycfeng/tmp/glm53_dsv4/v4_attention_input_20261023_w128_candidate02/` (no admission). Earliercandidate01 is an intermediate immutable export, not the final representation.

Scope:2core94changed lines;3non-core285changed lines. Original accepted data/packages and native06 R4 remain unchanged.

## Validation

Final full module:194passed/0fail/error/skip. Both entry paths:50CUDA_EVENT+44KERNEL_ONLY exact medians/layouts; serving_admission=false. Legacy:102positive/72typed negative and exact equality to SHA-pinned prior102medians. All source digests and artifacts are in `verification_receipt.json`.

## Open root gates

1. Actualcandidate02 independent layout/data review and isolated-only `.admission.json`.
2. Independent source review and final branch integration.
3. One clean fixed-source77candidate capture (next local step), then root strict comparison against retained576d278 or972equivalent evidence.

No GPU/platform/SSH/cloud/Docker/native/C26 or data-default action was performed. This input prerequisite is not coverage, overlap or fullTP8 serving acceptance.
