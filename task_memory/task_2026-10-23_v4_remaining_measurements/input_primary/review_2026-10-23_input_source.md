# V4 isolated attention input source review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Persisted the two independent read-only source reviews. |

Fixed point: `972f0cb1`; code: `4a1b53e6`; delivery: `04a72b08`.
Diff: `git diff 972f0cb1...04a72b08`; source-only artifact: owner `input_bundle/source.diff`.
Scope: two core files / 94 changed lines; three non-core files / 285 changed lines.

## Standards

Reviewer: `ses_e98edeb48ffeZHp8W52SMqoE4c`, task `W128-V4-ATTENTION-INPUT-STANDARDS`.
**No findings.** The reviewer inspected global/repository AGENTS.md, parent and component requirements/plan/design/progress, all source hunks and surrounding code, and retained verification receipts.

- `deepseek_v4.py:624–628,666–670` preserves explicit layouts and exact downstream guards; only omitted layouts trigger discovery.
- `dsv4_attention_input.py:26–88` checks external admission, hashes, original lineage and isolated-only qualification; invalid present inputs do not fall back.
- `prepare_dsv4_attention_input.py:23–100` reuses the independent builder, verifies original bytes, creates a fresh candidate and issues no admission.
- `check_dsv4_measured_scopes.py:33–65,74–95` replaces its previous preparation copy and checks independent layouts and original CSV equality.
- `test_dsv4_graph_measured.py:607–609,666–793` binds external acceptance by SHA, invokes the current repository writer and labels fixture admissions explicitly.

No actionable heuristic smell was found. The private JSON/dict representation follows the established precedent; no additional abstraction was requested. The reviewer did not rerun tests, hashes or platform operations.

## Spec

Reviewer: `ses_e98edec61ffeU00o3w8dLLNj4a`, task `W128-V4-ATTENTION-INPUT-SPEC`.
**No findings; source-only review passes.** The reviewer inspected local and parent requirements/plan/design, complete source diff, five actual files, the original `dsa_full_dummy_input.py` precedent and candidate/evidence records.

- No missing implementation requirements or scope expansion was found. Actual representation admission and strict comparison are explicitly root-owned.
- Explicit layouts bypass discovery; invalid sidecars fail without fallback; preparation only creates fresh candidates.
- The independent builder uses non-timing engine/step records. Existing checks compare both entry paths' layouts and medians.
- Candidate02 was observed without an admission file. No serving total, graph/TP8 coverage promotion or shared configuration was introduced.

The reviewer did not execute commands, tests, Git or hash checks. Source review does not supply representation, default-fidelity or native/GPU admission.

**Summary:** Standards 0 findings; Spec 0 findings. Root owns actual candidate verification, external isolated-only admission, automatic lookup, strict77 comparison and integration.
