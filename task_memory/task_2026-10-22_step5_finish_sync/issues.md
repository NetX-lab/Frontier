# D30 remaining adjacent regression scope

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Separate five new fixture failures from the baseline READ-byte mismatch. |

## Observed failures and cause

Candidate extra suite: `tests/unit/test_step5_measured_pd.py`,
`test_step5_structural_pd.py`, `test_step5_pd_placement.py`:68PASS/6FAIL.
Same files with only the scheduler restored in memory from `fa5bae2`:
73PASS/1FAIL. No disk source or parent file was restored/overwritten.

Five newly failing fixtures schedule only the selected lane after each output.
That lane now correctly waits at step1 for peers that the fixture never drives:

- `test_step5_measured_pd.py:56`: first NIXL shape case, offline and online.
- `test_step5_measured_pd.py:368`: unchanged-coverage capacity retry.
- `test_step5_pd_placement.py:157`: P response/source-KV retention.
- `test_step5_pd_placement.py:198`: D capacity/READ/readiness/decode.

The Step finish boundary must not be bypassed to make these tests pass.
Adapt the tests to drive all existing peer lanes and their request-free dummy
iterations/finish releases, then preserve every existing request/transfer
assertion. Full event dispatch is another valid test method; no new production
scheduler or flag is needed.

The sixth failure (`test_step5_measured_pd.py:159`) already fails on baseline:
actual READ bytes102203904, stale expectation103051776. Parent already owns
uncommitted updates to this file, observed before isolated worktree creation.
Do not change this assertion as a D30 scheduling fix.

## Concrete expansion proposal

Current candidate:4 source/test files,327 changed lines.
Minimal additional ownership: the two existing PD test modules above, estimated
~70-100 lines for a shared peer-driving test utility and five call sites.
Result:6 files/<450 changed lines, no new production source, no native/platform
operations. Six files exceed the explicit bound; additional edits are stopped.
Parent should combine this with its D29/READ test updates or seek the specific
file-count expansion. No result is marked PASS by excluding these failures.

Evidence: `/data/ycfeng/tmp/d30_finish_pd_adjacent01.{log,xml}` and
`/data/ycfeng/tmp/d30_finish_pd_baseline01.{log,xml}`.
