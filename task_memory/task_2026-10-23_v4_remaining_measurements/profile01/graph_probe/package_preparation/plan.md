# Package preparation plan

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Defined immutable template transformation, safe CPU acceptance and root handoff. |

Dependency: `A authority/source/template inspection -> B builder/plan -> C fresh package -> D CPU entry/command/integrity controls + matrix binding -> E scoped commit/handoff`.

| Step | Done condition |
| --- | --- |
| A | Separate authorized worktree; source and template pins read; parser/diagnostic ordering and existing API hash gaps inspected. |
| B | Maintenance builder+plan at most220 lines; deterministic reviewed template transformations only; generated launcher at most100 lines. |
| C | Eight locked input files plus manifest; exact frozen probe copy; fresh package untracked/unsealed; old packages unchanged. |
| D | Real generated launcher help/dry-run/compile/AST/bash syntax pass; explicit SDK stand-ins capture resources/receipts; local fake-native command verifies diagnostic-before-entry and absent output, native failure/initial-tool failure preservation; no actual platform/GPU call. |
| E | Requirements re-read; exact hashes/counts/budgets and existing77 binding recorded; only preparation files committed; package remains untracked/unsealed. |

Reuse the old launcher's in-place Python `verify_and_run`/exec mechanism, adding only manifest validation and sibling diagnostic-directory creation. Retain source validation, initial `nvidia-smi -L`, `timeout600`, ResourceConfig, spawn arguments and launch receipts. The package's missing seal is intentional, not a new launcher fallback or a runtime permission.

CPU command exercises use a copied launcher in a fresh local scratch package, fully local config roots, explicit fake nvidia-smi and a stdlib-only fake native probe. They qualify launch wiring/exit propagation, not actual image/SDK/CUDA behavior. Root-only unavailable checks remain explicit gaps.
