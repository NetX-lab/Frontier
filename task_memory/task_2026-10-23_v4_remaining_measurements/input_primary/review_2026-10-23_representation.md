# External review of the V4 isolated input representation

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Root verified the actual candidate and approved its isolated-only representation. |

## Decision

**PASS — isolated_prefill_input_only.** Reviewer: OpenCode root acting under the recorded D-Q10 and D-V4-KERNEL-ONLY-GPU-COST authority. Candidate: `/data/ycfeng/tmp/glm53_dsv4/v4_attention_input_20261023_w128_candidate02/dsv4_attention.json`, SHA256 `d688f3a27e972b5ca9903498f841de0fb50311512cb2b98404c5d478208489f7`.

The representation can be supplied to the isolated prefill reader. This decision does not admit serving totals, physical TP8 profiling, graph/PD equivalence, complete kernel ownership, stream additivity, or CPU overhead.

## Observed evidence

- `representation_verification.json` records root execution on source `4a1b53e6`, all five source digests, original acceptance, all 529 sealed package inputs and candidate file digests.
- Root reconstructed the plan through the current repository and reproduced all twelve table/layer layout dictionaries using only seven engine/step records. All 282 call/timing records were removed before this reconstruction.
- Both CSVs and all copied original evidence remain byte-identical. A separate fresh export has identical parsed layout/evidence content; its serialized key ordering differs. This admission binds the actual candidate's own bytes.
- Root explicit lookup reproduced all 94 accepted medians. Legacy lookup reproduced 102 positives and 72 typed negatives; ten error strings differ only in their fresh temporary directory names.
- Before admission, both candidate timing methods rejected automatic lookup for the missing admission file.
- Root reran the affected unit module: 194 PASS, zero failures/errors/skips. Independent Standards and Spec reviews found no actionable issue (`review_2026-10-23_input_source.md`).
- Root strict comparison: 77/77 identical, no cache-name or provenance findings; all 451 candidate artifacts were rehashed. Baseline binding is recorded in the parent task.

The preparation tool did not issue this approval. Root issues a separate digest-bound admission and then verifies the real admitted automatic path. The final automatic result and integration decision are recorded separately so this review's hash remains stable.
