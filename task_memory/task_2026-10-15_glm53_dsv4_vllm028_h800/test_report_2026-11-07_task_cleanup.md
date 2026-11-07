# Task artifact cleanup — 2026-11-07

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-07 | Recorded the codex-implement cleanup 86673e2f and the lead acceptance of its verification differences. |

## Scope and result

Standing rule from `requirements.md`, "prompt cleanup of invalid task artifacts through codex-implement": after each
checkpoint, remove artifacts this task produced that are established as invalid, through `/codex-implement`.

Codex request `86673e2f-3b87-4663-a948-1a9ec9eb03a4` (report `cleanup_20261107/report.md`, request record
`codex-implement-86673e2f-3b87-4663-a948-1a9ec9eb03a4/`) removed 171 exact directories under
`/data/ycfeng/tmp/glm53_dsv4` and `/data/ycfeng/tmp/ds4_runs`:

| Category | Directories | Regular files | Allocated bytes | GiB |
| --- | ---: | ---: | ---: | ---: |
| pytest temporary fixture trees | 59 | 23,122 | 496,013,312 | 0.4619 |
| regenerable sklearn estimator caches | 11 | 15,698 | 2,947,719,168 | 2.7453 |
| superseded historical CPU fidelity case outputs | 52 | 27,456 | 36,607,066,112 | 34.0930 |
| Node compile caches inside finished Pi labels | 49 | 1,363 | 126,234,624 | 0.1176 |
| Total | **171** | **67,639** | **40,177,033,216** | **37.4178** |

`df -B1 /data/ycfeng`: available 211.93 GiB at 06:57:11Z, 248.33 GiB at 07:01:40Z (+36.41 GiB).

Codex reported outcome `partial`, because its retained-state comparison found four differences on protected paths. The
lead traced all four to lead work that ran at the same time as the cleanup, so the cleanup is accepted. The 89
uncertain candidates in `deletion_manifest.json` (`uncertain_retained`) stay; they need new evidence of invalidity.

## Lead acceptance

Command: `PYTHONDONTWRITEBYTECODE=1 timeout 120 python3 cleanup_20261107/lead_acceptance.py` (system `python3` 3.10.6;
it reads the cleanup records, the worktree and Git, and writes only `cleanup_20261107/lead_acceptance.json`). Status
`ACCEPTED_DIFFERENCES_FROM_CONCURRENT_LEAD_WORK`, 11/11 checks.

| Reported difference | Lead explanation | Evidence | Result |
| --- | --- | --- | --- |
| `h200_mainline/build_package.py` hash ef8d7504… -> 980d8ab3… | the lead added the `p2_64_formal_retry07` CASES entry | 980d8ab3… equals the current file and the blob in commit 3d69016f | PASS |
| main worktree 39,591 -> 39,672 regular files | lead files created between the two protected-hash snapshots (06:55:11Z-06:59:06Z) | 81 new files, all under `p2_64_formal_retry06/terminal_20261107/` or `p2_64_formal_retry07/` | PASS |
| `p2_64_formal_retry06` 65 -> 96 files | the generation06 terminal archive | `terminal_20261107/` holds 31 files | PASS |
| inodes of `p64-formal6-submit-1107j/execution.json` and `exit` replaced | turn 2 of that label (the generation06 submission) ran after the inventory | inventory 06:42:27-40Z; `turn` = 2, written 06:43:09Z | PASS |

Other checks:

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Removal receipts | 171/171, no unlisted path missing, no errors | as expected | PASS |
| Deletion entries on the P2-64 mainline | none under `formal_p2_20261102`, `p2_64_formal*`, `downloads/`, `launch_p2_64*` | none | PASS |
| Completed Frontier replay used by the post-run command | `formal_p2_20261102/runs/p2_64` keeps `collectives.json`, `forwards.jsonl.gz`, `receipt.json` | present | PASS |
| Generation05 manifest and promotion receipt seals | 23e26da6… and d505eece… | unchanged | PASS |
| Native and harness pins | a06a72a4 / f33306d4 | unchanged (codex `source_pins_after.json`) | PASS |

`verify_cleanup.py` was not rerun: it rewrites `verification.json` and `df_after.json`, and the lead checks above
answer the open question without it.

## Committed and local-only records

Committed under `cleanup_20261107/`: the report, scripts, deletion manifest, final removal receipts, verification,
lead acceptance, snapshots of disk and pins, process scans, logs and small catalogs. Kept local only (not committed,
because of size or duplication): `inventory.sqlite3` (201 MB), `protected_hash_before.json` and
`protected_hash_after.json` (2.5 MB each, summarized in `verification.json`), `protection_plan.json`,
`tracked_paths.json`, `candidate_manifest.json`, `context.json`, the superseded `deletion_receipt.json`,
`evidence/` (104 copies of fidelity run manifests and results whose originals remain beside the deleted cases) and
`reference_20261103/` (copies of the committed November 3 scripts).

## Limits

- The codex `/proc` scans and the lead's shell see only their own process namespace, not every HOST process.
- No simulation, GPU or cloud operation was part of this cleanup; it claims no new calibration result.
