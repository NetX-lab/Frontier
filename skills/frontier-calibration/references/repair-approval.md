# Code-Change Approval

Every repair entry reads this file. It owns permission and order; the repair
entry owns the domain evidence and its focused checks.

## Modeling Rationale

Before proposing a repair, apply `contract.md`, Calibration Objective. The
proposal must show a representative clean-run gap, causal evidence for the
behavior, ordinary-user inputs, expected metric benefit and implementation
cost, and validation on fresh runs or cases. For an unmodeled mechanism, make
its frequency and affected configurations explicit. A gate-specific assumption
or a favorable subset of observations cannot supply this rationale.

## Required Order

```text
complete analysis with status PASS and correction_state=pending
-> proposal to the user: evidence, options, cost, recommended option first
-> the user's decision, recorded with a decision ID (contract.md, Decisions and Approval)
-> design note and approval when the change touches a core module
-> ticket and change_marker naming the decision ID and approved scope
-> one scoped Frontier commit
-> focused tests and the regression set
-> rerun the affected cells
-> fresh analysis
```

The next code change needs a new analysis and a new decision. An analysis
result, supplement result, run status, or post-change result never authorizes
itself.

## Review Record

The user is the reviewer. The record is the user's verbatim answer in the
task's `requirements.md` under its decision ID, together with
`repairs/human_review_receipt_<ticket>_<date>.json` naming the case, the
analysis artifact and its timestamp, `requesting_user`, `reviewer_identity`,
the decision ID, and the approved scope. `reviewer_identity` is the requesting
user or a delegate the user named in `requirements.md` before the review. The
repair entry writes the receipt only after the user has answered; a receipt
without a recorded answer, or one that predates the analysis it cites, returns
`PENDING_HUMAN_REVIEW`.

## Scope Limits

The code-change approval requirements in this file apply only to core modules
under `frontier/`. Within the current calibration task, edits to `tests/` and
other non-core files are directly authorized and require no additional user
approval. They do not count toward the file or changed-line limits below,
including when they accompany a core-module change.

For core modules under `frontier/`, a decision covers the scope it names. Ask
again before going past it: another core module or domain, a shared interface
or config schema, a gate-rule change, or more than 5 core files or 500 changed
lines in core files. Add a configuration field only when the analysis measured
its fidelity effect and the gain justifies its implementation and maintenance
cost for the selected engine and workload.

Direct authorization of non-core file edits does not resolve decisions that
change calibration semantics; those still follow `contract.md`, Decisions and
Approval.

## Regression Set

Run before the commit and record each result in the result receipt:

- unit tests: the failed set equals the baseline's failed set;
- `tests/e2e/refactor_fidelity/run_matrix.py` against the pre-change commit:
  identical, or every difference explained by the change;
- cells the change does not apply to (for example, `attn_dp = 1` cells for a DP
  change): byte-identical Frontier metrics;
- public examples unchanged unless the change targets them;
- the size and naming gates of the repository `AGENTS.md`.

## Result Receipt

After the rerun, write the domain result receipt with the changed files, the
commit, the `change_marker`, the commands, before and after values (absolute
and relative), the regression results, the validation status, and the decision
ID. A result other than `PASS` keeps the case open and authorizes nothing.

Repairs target Frontier. A `groundtruth_defect` is fixed in the ground truth
by its own decision, as a new ground-truth build identity and run generation; Frontier
never models it.
