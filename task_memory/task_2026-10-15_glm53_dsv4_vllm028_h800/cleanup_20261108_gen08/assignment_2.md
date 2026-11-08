# Task: delete the four snapshot files your first run correctly declined

## Why this is a second run

Your first run (`deletion_receipt.json`, status PARTIAL) deleted groups `generation08` and `launcher_log` and declined
group `previous_snapshots` with `Observed counts/allocation differ from assignment table`. That decline was right: the
assignment table said 300,759,552 allocated bytes and you observed 300,761,088, a difference of 1,536 bytes. The error
was in the lead's table, not in the files. The lead re-measured and confirms your observation:

| File | Size bytes | Allocated bytes | SHA256 |
| --- | ---: | ---: | --- |
| `protected_hash_before.json` | 75,188,389 | 75,190,272 | `8cb0e44aabf312785b149e59e305f05f8c4c21b8560d20c056b7db7d2faccda8` |
| `protected_hash_after.json` | 75,188,388 | 75,190,272 | `55c8d5d65e499451186a30af078a2301077b18735e35b2425baa1c73b091c1c3` |
| `protected_hash_before_final.json` | 75,188,546 | 75,190,272 | `cd50532f06fd0d027b4bee62459e7f5bf0d5b0d806c6733ed191783ba5915b7b` |
| `protected_hash_after_final.json` | 75,188,546 | 75,190,272 | `9cf0b5b8c4cf1afa8094e26b4d3f7da758e45994bad8b1eef8dfabc756c7f265` |
| **total** | **300,753,869** | **300,761,088** | — |

All four live in `<TASK>/cleanup_20261107_gen05_07/`. The corrected total of 300,761,088 allocated bytes is the figure
to check against now; treat the first assignment's 300,759,552 as withdrawn.

Why they are invalid is unchanged: they are the bulk 119,453-file inventories of the previous cleanup run, deliberately
left uncommitted. The lead verified today that the committed `lead_acceptance.json` in the same directory records each
one's exact size and SHA256 under `snapshot_files_not_committed`, and that all four recorded values equal the live files
byte for byte. The committed record is therefore the surviving evidence, and deleting the files loses nothing.

Acceptance: the four files are gone; nothing else is removed or modified; each deleted file is accounted for with its
pre-deletion size and hash; `lead_acceptance.json` and the other 24 files of `cleanup_20261107_gen05_07` are unchanged
and still match their first-run hashes.

## Approach

Reuse your own first-run scripts and record format in this directory rather than writing new machinery. Extend the
existing journal and write the second-run artifacts under distinct names so the first run's records stay readable:
`deletion_manifest_round2.json`, `deletion_journal_round2.jsonl`, `deletion_receipt_round2.json`,
`verification_round2.json`. Keep the first run's files unchanged.

Verification scope for this round is small and should stay that way: the four target paths, their parent directory
`cleanup_20261107_gen05_07` including the 25 files that must survive, and nothing else. Do not re-inventory
`generation08_run2`, the SSH copy, or the launcher-log siblings; your first run already verified those and they are not
in this round's blast radius.

Your first run's final rerun reported an intervening replacement of `<TASK>/test_report_2026-11-07_p2_64_generation08_formal.md`
at 2026-11-08T08:12:36.801866Z and correctly asked who wrote it. That was the lead, adding an independent recomputation
of the published metrics and a history row; it is unrelated to your deletions, and no restoration is wanted. Expect the
same during this round: while you work, the lead is editing task records outside your write scope and deleting the
out-of-repo duplicate copy `/data/ycfeng/tmp/glm53_dsv4/h200/downloads/formal_p2_64_20261107_08`. So keep your
protected-input baseline to this round's scope above and do not treat lead edits elsewhere as anomalies; if you do see
a change inside your scope, that is still worth reporting.

## Process prerequisite

The same lead-supplied host scan applies and is already in this directory as `cleanup_scope_processes.json`
(2026-11-08T07:59:12Z, 289 processes, `clear_for_deletion: true`, sha256
`20c553d72918382e08f7b636c9ab4d1896cd7f155c449fda5132534164340296`). It names `protected_hash_before.json` and
`protected_hash_after.json` among its scope roots but not the two `*_final` files; the lead accepts it as the
prerequisite for all four, because nothing has written to any of them since 2026-11-07 and their hashes are unchanged.
Record that limitation as you did before. Do not attempt your own host process scan: your executor runs in its own PID
namespace and cannot see host processes.

## Scope and authority

- Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, branch `feat/glm53-dsv4-vllm028`.
  `<TASK>` = `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800`.
- Record directory: `<TASK>/cleanup_20261108_gen08/`. Leave `assignment.md`, this file, and every first-run artifact
  unchanged.
- Deletion authority is exactly the four files in the table above. `cleanup_20261107_gen05_07` itself stays: it still
  holds 25 files and must not be removed even if you think it is empty.
- Do not run Git, submit or inspect any platform or RJob operation, install packages, or write outside the record
  directory and the four deletion targets. Leave `~/.pi`, `.cortexkit/`, `.pi/`, `launcher.pid.txt` and
  `remaining_measurements/` untouched. Never print credentials or proxy values.
- Environment: system `python3` (3.10.6), run with `-B` so no bytecode cache is written.

## Expected return

- Report language: English.
- Return the four deleted paths with pre-deletion sizes and hashes, the reconciled total against 300,761,088 allocated
  bytes, and the verification result for the 25 surviving files of `cleanup_20261107_gen05_07`.
- State explicitly whether `lead_acceptance.json` still matches its first-run hash.
- If any observed value still disagrees with the corrected table, do not delete that file: report the difference.
