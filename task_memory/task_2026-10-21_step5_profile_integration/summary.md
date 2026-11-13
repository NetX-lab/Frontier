# Step profiling integration archive

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archive reviewed source integration and observed validation. |

Provisional source f0ab2eb combines b0018c4 and5fa1926. The37-file delivery provides native Step attention profiling, dense/shared BF16 linear scopes and activation/norm ownership, plus fixed-shape H200 probe packaging. Independent whole-branch review returned zero actionable findings. Parent tests:787 PASS; complete legacy fidelity77/77 identical with identical predictor cache filenames. The initial comparison's output-path-only mismatch and corrected absolute-label invocation are retained in the dated test report.

Deliverables: `frontier/profiling/step5/`, Step linear/common profiling extensions, `frontier/model_architectures.py`, associated integration/unit tests, and this task directory. Exact changed files: `git diff --name-only b0018c4 f0ab2eb`. These records and the source are ready for parent integration.

Open work: native H200 timing judgment, canonical serving/runtime/transfer qualification, full reached-shape coverage, typed admission and the unresolved CPU performance budget. The parent runs the separately sealed one-H200 probe; no GPU operation occurred from this integration worktree. Source acceptance does not close those gates.
