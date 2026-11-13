# 02: Immutable native reference and text metadata

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-20 | Recorded the image version and candidate source hash difference. |
| 2026-10-18 | Documentation-only audit: reconciled evidence, execution authority, shared ownership and stage-specific prerequisites; no new implementation acceptance. |
| 2026-10-15 | Approved planning ticket created. |
| 2026-10-15 | Expressed the required dummy-weight loading behavior without an inline CLI fragment. |
| 2026-10-15 | Started after ticket01 closure under explicit continuous execution and CPU/GPU measurement authorization. |
| 2026-10-16 | Accepted nine reference/metadata criteria against the observed supplement; retained tooling performance as the open final criterion. |
| 2026-10-16 | Closed after phase review and primary acceptance of the permitted UNVERIFIED CPU disposition; no performance or model-support PASS. |

**What to build:** A reproducible S01.G0 reference-lock package that binds the official Step image and exact text-loader metadata to immutable identities without acquiring checkpoints or asserting unmeasured execution support.

**Blocked by:** `01` Baseline and shared-capability reuse audit.

**External prerequisites:** The G0 image/metadata lock is accepted. A changed reference remains conditional D-S5-REF work; the GLM/DS runtime is not the Step runtime. Authorized platform operations use the original personal local StepMind Python RJobBackend, following the current host/storage rules. No platform operation is performed in this review. Use existing shared contracts and record their sufficiency. Only a demonstrated material Frontier core refactor or materially unresolved semantic/acceptance choice requires an explicit decision; ordinary extensions do not require renewed approval.

**Status:** done

**Review note — 2026-10-20:** The image's vLLM version suffix `g8224bb82f` differs from the candidate source `jeejeelee/vllm@02e28ab2…`. The image remains the locked reference; the candidate source is used only for reading, not as image-build code.

Ticket02 is accepted as S01.G0: a locatable reference only. All ten criteria are complete. The final tooling criterion is satisfied by77exact supported-scenario comparisons, measured preregistered budgets and the primary's UNVERIFIED disposition under `performance_budgets.md` item3. Controlled results remain6PASS/2OVER_BUDGET; the residual MoE timing differences remain unexplained. This is not performance PASS, fidelity-cost acceptance or a waiver of later code-step checks. The observed lock remains an immutable pre-acceptance snapshot; this ticket is the current status authority. G1–G4 remain later qualification work. Ticket03's dependency is released, but its execution has not started.

**Source workstreams:** S01.
**Shared owners:** C01, using the existing reference-manifest workflow.

## Review disposition — 2026-10-18

DONE for G0 identity only. The observed Step package is `0.1.dev22912+g8224bb82f`; the full image build SHA remains unknown and the public source candidate is not proven to match it. `reference_lock_2026-10-16_observed.json` is authoritative for the retained observation.

## Acceptance

- [x] Deliver the S01.G0 reference-lock package with a metadata acquisition manifest, image/platform identity, source/library identity ledger, unresolved facts, and a reproducibility verdict as specified in §§1, 3, and 6.
- [x] Resolve the official Step image to an immutable target-platform digest; capture image ID, RepoDigests, architecture, entrypoint, actual package versions, relevant source/library hashes, and available build provenance.
- [x] Keep unavailable build SHA explicitly unknown; label the public source candidate SOURCE_CANDIDATE until correspondence is demonstrated. A tag, recipe, or matching name alone cannot establish image-build identity.
- [x] Freeze an exact revision-pinned file allowlist for model configuration and required text-loader metadata, recording each admitted file's purpose, content identity, and hash; admit additional small sidecars only upon demonstrated native-loader need.
- [x] Verify the acquisition inventory contains no checkpoint weights or whole Hugging Face repository snapshot; reject a request or manifest that expands beyond the metadata allowlist.
- [x] Record complete text construction, native quantization/exclusions, unchanged routing, and explicit selection of the native dummy-weight loader as mandatory later-run inputs; preserve initialization seed and parameter/buffer/scale identity fields without claiming those runtime fields have already been measured.
- [x] Keep the Step reference distinct from unchanged GLM/DS locks; reject reference mixing, mutable-only image identity, wrong metadata revisions, and candidate hashes presented as measured image provenance.
- [x] State that G0 establishes a locatable reference only: G1–G4, memory fit, Hopper dispatch, graph replay, and PD support remain unknown until their respective native measurements.
- [x] Preserve previous evidence generations; a subsequently approved reference change produces a new lock and explicit affected requalification scope rather than rewriting the old lock.
- [x] Any reference-tooling code change satisfies §10's at-least-50-scenario fidelity comparisons and preregistered wall-time/memory budgets, with measured results and over-budget dispositions before acceptance.

## Acceptance evidence

The package comprises the unchanged initial acquisition lock, `reference_lock_2026-10-16_observed.json` and its hash-pinned selected business evidence. `test_report_2026-10-16_explicit_proxy.md` records primary content/package/ledger verification and five rejection controls. `test_report_2026-10-16_bytecode_control.md` preserves every budget outcome and the accepted UNVERIFIED disposition. Reviewer `ses_ebf40acefffesyjkUcmfaqIO5R` found no conflict in this scoped acceptance; its sole P3 permission/status wording fix for03 was applied. Full review and primary acceptance are in `review.md`.
