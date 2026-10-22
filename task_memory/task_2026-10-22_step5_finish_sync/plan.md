# D30 isolated implementation plan

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record dependency chain and focused acceptance. |

`approval/source inspection/design` → `focused RED` → `scheduler + observer`
→ `focused GREEN + legacy checks` → `code commit + record commit`
→ parent integration/review/strict77 → native scalar observation.

1. Read approval/source and existing owners; record design and human receipt.
2. Add focused tests: step1, 16, reset, active peer, dummy, legacy32 and DP1.
3. Apply family-scoped scheduler policy and existing-install scalar observer.
4. Verify boundaries and fault isolation, code size, diff scope and whitespace.
5. Commit verified source/tests separately from records; report exact paths,
   commands, logs, commits and unverified live-native evidence to the parent.

Acceptance is the complete focused matrix, not merely a successful test exit.
No full CLI or CPU-overhead suite. Parent owns strict77 and native package/pins.
