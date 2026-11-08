# Task: delete the four invalid generation08-era artifacts and record the outcome

## Goal

Reclaim about 2.38 GiB by deleting three artifact groups that are now invalid or redundant, and leave a receipt that
accounts for every file. YC authorized cleanup of this task's invalid artifacts; the three groups below are the lead's
scope for this run and the only paths you may delete.

| # | Path | Files | Allocated bytes | Why it is invalid |
| --- | --- | ---: | ---: | --- |
| 1 | `<TASK>/h3o_operator_costs/formal_p2_64_native_20261102/generation08` | 173 | 78,544,896 | The first comparison attempt ran without `PYTHONPATH` and stopped in `run_gate.py` with `ModuleNotFoundError: No module named 'tests'`. Its status is UNVERIFIED and it is superseded by `generation08_run2`, which is complete and committed. |
| 2 | `<TASK>/native_track_20261020/g0d_launch/launcher.log` | 1 | 2,114,797,568 | 4.33 GiB logical, 2.0 GiB allocated, holding only 389,322 repeated platform-poll tracebacks from 2026-10-30 to 2026-11-08. The original submission output was already destroyed by an earlier truncation that left a 2,218,037,248-byte hole at the front of the file. Its identity, counts and first/last bytes are preserved in the sibling `launcher_log_evidence.json`, and the 29 sibling record files stay. See `issues.md` I-015. |
| 3 | `<TASK>/cleanup_20261107_gen05_07/protected_hash_{before,after,before_final,after_final}.json` | 4 | 300,759,552 | Bulk 119,453-file inventories from the previous cleanup run. The lead deliberately left them uncommitted and recorded each one's size and SHA256 under `snapshot_files_not_committed` in the committed `lead_acceptance.json`, which is the surviving evidence. |

A fourth group, the duplicate platform-route copy of the same cloud run at
`/data/ycfeng/tmp/glm53_dsv4/h200/downloads/formal_p2_64_20261107_08` (135 files, 76,001,280 allocated bytes), is
outside this working root and stays with the lead. Do not touch it or its sibling `formal_p2_64_20261107_08_ssh`, which
is the copy the comparison consumed.

Acceptance: the listed paths are gone; nothing else is removed or modified; every deleted file is accounted for with
its pre-deletion size and hash; the preserved evidence named above still matches its recorded hash afterwards.

## Approach

Reuse the method of the previous accepted run in `<TASK>/cleanup_20261107_gen05_07/` (`execute_cleanup.py`,
`verify_cleanup_final.py`, `deletion_receipt_final.json`, `lead_acceptance.json`): classify first, hash every candidate
before deleting, journal each deletion, then verify independently and report counts that reconcile.

Two differences from that run, both decided by the lead:

- **The host process check is supplied, not repeated.** Your executor runs in its own PID namespace and cannot see host
  processes; that is what blocked the first attempt. The lead ran the check from the host instead:
  `/data/ycfeng/tmp/ds4_runs/cleanup-scan-1108b/scratch/cleanup_scope_processes.json`, 2026-11-08T07:59:12Z, 289
  processes scanned, `clear_for_deletion: true`, no process referencing any scope root through its cwd, exe or any open
  fd. Coverage limit, stated plainly: 127 entries had unreadable links; of 15 sampled, 14 had already exited between
  being listed and being read, and the remaining one was pid 1 `/sbin/init`. Treat this file as the prerequisite
  evidence, copy it into your record directory, and do not attempt your own host scan.
- **No repository-wide protected-input snapshot.** Hashing 119,453 files twice is disproportionate here and produced no
  usable verdict last time. Instead verify a scoped before/after inventory: the parent directory of each deleted path,
  plus the preserved neighbours named in the table (`generation08_run2`, `formal_p2_64_20261107_08_ssh`, the 29 sibling
  files of the launcher log, and the committed files of `cleanup_20261107_gen05_07`). The 4.33 GiB log is being deleted,
  so hash it once before deletion rather than twice.

The growing log that destabilized the previous run is fixed: its writer was stopped on 2026-11-08 at 07:54:55Z and the
file has not changed since (`issues.md` I-015). If you observe it changing, stop and report rather than deleting it.

## Scope and context

- Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, branch `feat/glm53-dsv4-vllm028`.
  `<TASK>` = `task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800`.
- Task records: `<TASK>/requirements.md`, `<TASK>/plan.md`, `<TASK>/progress.md`, `<TASK>/issues.md` (I-014, I-015),
  `<TASK>/tickets/18-native-ground-truth-h200.md`, `<TASK>/tickets/19-comparison-and-acceptance.md`.
- Record directory for this run: `<TASK>/cleanup_20261108_gen08/`. Write your scripts, manifest, journal, receipt and
  report there. This assignment file is already in it; leave it unchanged.
- Deletion authority is exactly the three groups in the table and the directories left empty by those deletions. Nothing
  else may be deleted, moved, truncated or edited — in particular not `generation08_run2`, not the SSH copy, not the 32
  retained scratch files under `/data/ycfeng/tmp/ds4_runs/p64-formal*/scratch`, and no committed record file.
- Do not run Git, submit or inspect any platform or RJob operation, install packages, or write outside the record
  directory and the deletion scope. Leave `~/.pi`, `.cortexkit/`, `.pi/`, `launcher.pid.txt` and
  `remaining_measurements/` untouched. Never print credentials or proxy values.
- Writable paths: `<TASK>/cleanup_20261108_gen08`, plus the four deletion targets and their parent directories.
- Existing work to preserve: the working tree has no uncommitted changes other than what this run creates; commits
  `5bc10fa5`, `da251394`, `4bedb354` and `7f114d51` hold the generation08 result and its records.
- Environment: system `python3` (3.10.6), run with `-B` so no bytecode cache is written.

## Expected return

- Report language: English.
- Return the actual deleted paths with pre-deletion sizes and hashes, the reconciled counts and reclaimed bytes, the
  verification results with expected and actual values, and any file you declined to delete with the reason.
- State explicitly whether each preserved-evidence file still matches its recorded hash.
- If a candidate's observed state disagrees with the table, do not delete it: report the difference and continue with
  the rest.
