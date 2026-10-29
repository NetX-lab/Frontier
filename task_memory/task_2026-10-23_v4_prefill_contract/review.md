# Isolated prefill contract review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded independent dual-axis review and root verification; closed the documentation P3 and retained the matrix gate. |
| 2026-10-23 | Root closed the matrix gate and accepted this isolated prerequisite. |

Component: isolated prefill contract/provenance. Fixed comparison: `git diff b62c155abd981a71e12e62aa0c0fe3990741d2f8...e5464fe`; source commit2f87828. Three maintained files197changed lines, core55; full source artifact `source.diff`.

## Standards

Reviewer task `W128-V4-PREFILL-CONTRACT-STANDARDS`, session `ses_e9aa667f2ffeTyFKLPmJK4xNf4`: no actionable hard violations or blocking smells. Existing producer metadata readers, explicit receipt/package opt-in and neighboring external-dataset fixture convention are reused. Metadata flattening similarity did not justify a new abstraction. Reviewed source and retained evidence only, without independent Git/tests/hashes. The repository >=50-scenario matrix remained an unverified delivery gate;94 isolated scopes cannot substitute for it.

## Spec

Reviewer task `W128-V4-PREFILL-CONTRACT-SPEC`, session `ses_e9aa6bb31ffe3SeJqHhzsAbvV9`: no confirmed code defect or scope expansion. Independent non-timing layouts, hash-pinned source/runtime/model/table identity, balanced/legacy distinction and typed PD/graph/q1 rejection match the isolated prerequisite. One P3: parent approval line references drifted after current-status edits. Root replaced both references with the stable D-Q10 decision heading; original approval text is unchanged. Reviewed source/records, without independently running Git/tests or recomputing hashes.

## Root verification and pending acceptance

Root independently reran the fixed source:144 unit PASS/0failure/error/skip,94 exact accepted comparisons (50CUDA_EVENT+44KERNEL_ONLY), legacy102positive/72negative. Every receipt retains serving_admissionfalse. Exact outputs: `/data/ycfeng/tmp/glm53_dsv4/root_prefill_contract_2f87828.xml`, `_accepted.json`, `_legacy.json` and adjacent logs.

Findings: Standards0; Spec1P3, closed. No code edits required. Root subsequently ran the required clean strict77 capture/comparison at576d278:77/77 identical,451artifact hashes checked and519cache filenames equal. `primary_acceptance.json` records isolated acceptance. Full non-dummy predictor/TP8/PD/graph/serving/E2E qualification is outside this isolated acceptance.
