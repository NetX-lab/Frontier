# Final native collector source review

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-23 | Persisted both independent re-reviews of f8621a8; all five unique original findings closed. |

Reviewed repair: `d94b821a..f8621a8258ffa445abfb51f5ea73258ad271494e`.
Cumulative scope: five non-core files/880changed lines within987; collector457, test400, worker11, command8, P/D environment4. Both reviewers inspected actual source/test hunks, owner receipts and relevant pinned native callsites. Driver remains outside this phase.

## Spec

Reviewer `ses_e98ca2a05ffe2CYjLe8mE23lzM`, W128-V4-NATIVE-COLLECTOR-SPEC-R1.
**No findings. All five original issues CLOSED; no actionable new regression.**

- Lifecycle: collector36–39/331–347 installs manager class hooks independently of the load-time None instance and isolates instances/parallel/replay/facts. Test141–150 covers delayed manager creation and FULL dispatch.
- Early metadata: collector36–37 and test153–164 execute the real pinned prepare_attn body and preserve the slot-mapping-before-model-state order.
- PDL/CuTeDSL: collector47–52/276–300 and339–341 observe actual indexed launch kwargs and the locally imported head512 callable. Tests167–199 and the retained direct control match native RMSNorm, inverse-RoPE and compressor seams; actual GPU PDL is not claimed.
- Retention: collector400–419/431–441 admits the complete detached step once, drops overflow, and prevents later shared-dispatch mutation; test202–219 covers each condition.
- Error output: collector75–84/373 limits detailed formatting/retention/printing to64, then counts suppression; test254–275 and direct receipt preserve native return/exception identity.

## Standards

Reviewer `ses_e98ca2963ffeaN1Qc0kHtYPS70`, W128-V4-NATIVE-COLLECTOR-STANDARDS-R1.
**All three original hard findings CLOSED; no actionable heuristic or source regression.**

The complete-row admission and detachment resolve the retained-byte violation; the95-byte control retains/accounted0 and dropped1. Detailed error output is bounded before formatting;501failures yield64details/437suppressed with100native returns and the original exception retained. Actual indexed PDL arguments and local-import CuTeDSL ownership close the static/source omission. The manager lifecycle and early metadata fixes also align with the real native order without adding synchronization.

## Evidence limits and root disposition

Reviewers did not run tests, Git blob/hash verification or platform commands. Root independently ran the eight suites separately on the fixed revision and verified actual source/receipt digests (`source_verification.json`, `test_report_2026-10-23_final.md`). The31installed-source symbols are static evidence only.

**Source/CPU review accepted for integration.** Root owns merged checks, explicit matrix binding and the final component acceptance. Native execution, GPU stream mapping, hardware PDL, compile/breakable survival, captured-segment ownership and CUPTI restart remain UNKNOWN; emitted profile records remain UNVERIFIED.
