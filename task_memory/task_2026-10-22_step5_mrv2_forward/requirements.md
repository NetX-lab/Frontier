# D23-MRV2-FORWARD Requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-22 | Record the delegated approved CPU core scope. |
| 2026-10-22 | Record reviewer P2 correction and exclusive forward ownership. |

## [Original Request]
Implement approved pre-first-prediction agreement using existing stage group,
owners and stage events for all useful and dummy P/D lanes. Bind
FULL_AND_PIECEWISE; NONE preserves local widths; graphs share capture padding.
Preserve local query/context, serving ownership and zero dummy request credit.
Reuse the shared MRv2 descriptor, retaining FULL raw null and PIECEWISE null
request dimension. Validate first calls, lifecycle/release/wakeup/duplicates,
producer-fit-cache-count-lookup coverage, and legacy behavior. CPU only, no
further delegation or platform/SSH/push. Parent owns publication, G3 and full
integration; do not edit actual-main runner/fixtures. Base is immutable
5c2319cd; inputs worktree remains untouched.

Authoritative approval: primary worktree
`task_memory/task_2026-10-15_step5_preview_h800/requirements.md:106-118` and
`mrv2_consumer_20261022/design.md`.

## [Original Request] Review correction P2

Work only in the existing clean `step5-mrv2-forward` worktree on
`feat/step5-mrv2-forward`, initial HEAD `d8e3e941` (source `3cad8dbe`). Own
forward agreement/policy capacity plumbing, focused tests and these records.
Do not delegate, touch the main integration tree or access/edit fresh CLI
fixtures owned by the separate `step5-mrv2-main` session. No GPU, platform,
authentication, cloud or push.

First verify source/config mapping and run a failing real first-event regression:
`batch_size_cap=3`, capture sizes `[1,2,4,8]`, three useful decode requests and
one dummy lane. Native uniform FULL captures cannot exceed `max_num_seqs=3`.
`full_and_piecewise` must dispatch PIECEWISE4; `full_decode_only` must dispatch
NONE with local widths3/1. Larger PIECEWISE prefill buckets remain legal.
Carry existing scheduler request capacity into agreement and select supported
FULL/PIECEWISE candidates before DP min/max re-dispatch. Add no flag, fallback
cost, duplicated enum/registry, model threshold or blanket capture-list rejection.
Preserve other policies, null fields, zero dummy credits and real query/context/
owner snapshots. Cover cross-lane effects, FULL<=cap, capture miss, both exact
corrections and legal wide prefill. Preserve exact RED failures in records.

Parent independently passed444 affected+measured checks on `3cad8dbe`. Re-run
affected scoped checks, not the costly unaffected producer suite. Commit the
correction, tests and records; return exact commits/commands/logs/lines and
outstanding acceptance. Parent owns re-review, merge, full CLI and strict77.
