# D29 native-key component reviews

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Record independent Standards and Spec reviews of the committed three-gross-line candidate. |

## Component and fixed point

Task: D29-NATIVE-KEY-BYTES. Candidate:
`376983d2ecc441d1e0a8cb8b943e65a2dcb217b5`. Fixed point:
`9870597776229d84af5eb58d49bf9b1756092999`.

Both reviewers inspected the exact two-path diff, surrounding backend/test
behavior, applicable AGENTS.md, parent task requirements/plan and component
requirements/plan/design, and retained verification evidence. Neither ran
commands, tests/probes, delegated work, edited files or used GPU/network access.

## Standards

Reviewer: D29-NATIVE-KEY-STANDARDS,
session `ses_e928c1cc6ffepH81XvJ3TLbxxU`. **No findings.**

The existing byte accessor is reused with no new interface, magic value or local
reinvention. The full canonical plan key, instance-local LRU and pre-lookup
checks remain. Backend943 lines is within the soft module-size limit. Two files
change by3gross/net1, with no donor edit. No documented-standard violation or
actionable Fowler smell was found.

Decisive paths: `frontier/cc_backend/backends/collective_sim_cc_backend.py:880`
and `tests/unit/test_collective_sim_cc_backend.py:90`; AGENTS.md:798.

## Spec

Reviewer: D29-NATIVE-KEY-SPEC,
session `ses_e928bd454ffeGozXzaLlVfc9OQ`. **No findings.**

No missing component requirement, scope creep or reachable implementation
regression was found. Requirements16-19 and21-31 are met. Qualification,
scenario, model, context, counts and evidence checks remain before lookup at
backend841-878. Run-fixed bytes plus an empty LRU on a new run prevent a new
supported formatting-only cache miss; cost construction and errors remain.

One supplied receipt path, `scope_receipt.json`, did not exist. The reviewer
correctly inspected `scope_check.json` instead and identified it as precommit
accounting, not clean final integration acceptance.

## Observed evidence and limits

- RED1FAIL, GREEN1PASS; focused baseline/candidate each124PASS/1existingSKIP.
- Clean strict77:77/77 identical;451 artifacts and519 cache filenames per side.
- Clean unit:8028PASS/279FAIL/32ERROR/281SKIP, rawexit1; all8620IDs/states equal.
  Neither review describes this as an entirely passing unit suite.
- Canonical208-event wave preserves19 file comparisons and all costs/proofs/
  live calls, but does not enter the changed native-key branch.
- Early dirty-provenance and auditor failures remain; no gate waiver is inferred.

Findings: Standards0, Spec0. No component remediation is required. Independent
clean accounting and primary integration remain pending. Native hardware,
6144/1024 workload within2400s, G3 and E2E acceptance remain separate open gates.
