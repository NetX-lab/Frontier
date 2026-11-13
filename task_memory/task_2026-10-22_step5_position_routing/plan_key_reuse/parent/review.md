# D29 plan-key component review

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Archive independent Standards/Spec findings and primary acceptance. |

Scope: fixed point `087db1bf` through owner `d1e2fb8c`, source `c4e06a61`.
Inspected diff: `../evidence/implementation.diff`; changed files:
`frontier/cc_backend/collective_plan.py` and `tests/unit/test_step5_moe_pricing.py`.
Primary integration is source `30fb6a30`, with records and audit at `46fe83ac`.

## Standards

Reviewer task `D29-PLAN-KEY-STANDARDS`, session
`ses_e95d89f3affe7KKNlmjuvGg1J9`: **No findings**.

- `collective_plan.py:91–95` conforms to `AGENTS.md:791–800,838`: the existing
  domain owner publishes the immutable canonical string only after encoding
  succeeds. No new encoder, interface or numerical cache; public dictionaries
  still use declared dataclass fields. The private attribute is not a logical
  dataclass field.
- `test_step5_moe_pricing.py:60–70` extends the existing regression and preserves
  subsequent ordering, replacement and count assertions. The canonical encoder
  in the assertion is an oracle, not duplicated production behavior. Repeated
  failures verify that encoding errors are not cached.
- No actionable heuristic smells. Review used supplied files and receipts;
  it did not execute tests, hashes, Git accounting or probes.

## Spec

Reviewer task `D29-PLAN-KEY-SPEC`, session
`ses_e95d89df9ffeTaYF2NvmcNHsYE`: **No findings**.

- `../requirements.md:14` requires retention of only the successful full
  canonical string, no publication on failure, and fresh detached `to_dict()`
  values. The unchanged encoder and post-success assignment satisfy these;
  bytes/bytearray errors and public dictionary isolation remain tested.
- The prohibited API, protocol, plan-object, numerical, whole-proof and native
  backend-key changes were not introduced. Current admission/count/context
  checks remain. The diff is 16 gross / 14 retained, within 17 / 15.
- The reported 124 PASS / 1 Torch SKIP and canonical 208-event receipts are
  component evidence. The reconstructed cached manager with real P/D input
  binding is not full Simulator, CLI or native qualification. Review ran no
  commands and independently recomputed no hashes.

Standards: 0 findings. Spec: 0 findings. No remediation was required.
Primary subsequently passed the tests, full-wave parity and integrated union
guard recorded in `test_report_2026-10-24_parent.md` and `acceptance.json`.
