# Code-Change Approval

This reference is read by every repair entry. It describes permission and order;
the domain skill supplies the domain-specific evidence and validation command.

## Required Order

```text
complete analysis with status PASS
-> human review decision PASS
   or immutable case_init.auto_recycle=true plus complete auto receipt
-> record code-change intent and change_marker
-> apply one scoped Frontier code change
-> isolated rerun
-> fresh analysis
```

The next code change needs a new analysis timestamp and a new approval receipt.
An analysis result, supplement result, run status, or post-change result never
authorizes itself.

## Human Review

The receipt names the case, exact analysis artifact and timestamp, the immutable
`requesting_user`, `reviewer_identity`, decision, review time, and approved
scope. `reviewer_identity` must equal the requesting user or a delegate named in
`case_init` before the review. The reviewer is external to the repair entry. A
missing, stale, contradictory, unbound, or self-issued receipt returns
`PENDING_HUMAN_REVIEW`.

## Automatic Exception

The only automatic path uses a case initialized with `auto_recycle: true`. The
repair entry writes a complete auto receipt before the first code change, binds
it to the immutable case initialization and analysis timestamp, and records the
automation principal and scope. A later change to the manifest does not enable
automatic repair.

## Result Receipt

After the isolated rerun, write the domain result receipt with the exact changed
files, code-change marker, commands, before/after values, validation status, and
authorizing receipt. A result with validation other than `PASS` keeps the case
open and does not authorize another change.

Repairs target Frontier. The ground-truth scheduler and inference semantics stay
unchanged unless the case explicitly declares an experiment knob.
