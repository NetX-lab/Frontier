# Task: remove invalid scratch duplicates of the ended P2-64 generations 05-07

## Goal
The user's standing rule: remove task artifacts once they are invalid. Generations 05, 06 and 07 of the P2-64 native
run have ended (Failed, platform causes) and their evidence is archived with sha256 inside the task directory. Remove
the scratch copies that are now redundant and the regenerable Node caches of their finished Pi labels, and leave a
deletion manifest and receipt.

Acceptance:
- Every deleted file is either (a) byte-identical (sha256) to an archived copy recorded in one of the three archive
  manifests below, or (b) inside a `node-compile-cache` directory of one of the labels in scope; plus the two
  never-launched prompt files listed below.
- Every file in scope that is not covered by (a) or (b) remains, and is listed in the receipt as kept with the reason.
- The archive directories, their manifests and all other protected paths are unchanged (sha256 before and after).
- Receipt and manifest written under the record directory; a short `report.md` with counts, bytes and checks.

## Approach and decisions
- Labels in scope (all have an `exit` file, no live process): `/data/ycfeng/tmp/ds4_runs/p64-formal5-*`,
  `p64-formal6-*`, `p64-formal7-*`. Inside each label only the `scratch/` subtree is in scope. The Pi run records next to
  it (`output.md`, `events.jsonl`, `sessions/`, `prompt*.md`, `exit*`, logs, `execution.json` and the rest) stay.
- Archive manifests (each has `files: [{source, destination, bytes, sha256}, ...]`; `destination` is relative to the
  manifest's directory):
  - `<TASK>/h200_mainline/p2_64_formal_retry05/terminal_20261107/archive_manifest.json`
  - `<TASK>/h200_mainline/p2_64_formal_retry06/terminal_20261107/archive_manifest.json`
  - `<TASK>/h200_mainline/p2_64_formal_retry07/terminal_20261107/archive_manifest.json`
  A scratch file counts as archived only when its current sha256 equals both the manifest entry and the archived
  destination file's current sha256. Match by content, not only by the recorded source path, since some evidence was
  copied from other labels' scratch; record which archive entry covers each deletion.
- Also delete `/data/ycfeng/tmp/ds4_runs/prompts/p64-formal7-post-1107n.sh` and `.md`: that Pi label was never launched
  (generation07 did not succeed) and `p64-formal8-post-1107r.{sh,md}` replaces it.
- Before deleting, confirm no running process has a cwd, open file or argument under the paths to delete.
- Delete only files and then the directories left empty by those deletions; no `mv`, no other cleanup.

## Scope and context
- Working root: /data/ycfeng. `<TASK>` = /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800.
- Task records (read-only): `<TASK>/requirements.md`, `<TASK>/progress.md` (sections on generations 05-07 and on the
  earlier cleanup), `<TASK>/cleanup_20261107/` (the earlier cleanup's method, for reference only).
- Protected, must not change: everything under `<TASK>` except the record directory below; all `p64-formal8-*` files
  and labels; `/data/ycfeng/tmp/ds4_runs/prompts/` except the two files named above; every other ds4_runs label;
  `/data/ycfeng/tmp/glm53_dsv4/`.
- Writable paths: `/data/ycfeng/tmp/ds4_runs` (deletions in scope only) and the record directory
  `<TASK>/cleanup_20261107_gen05_07/`.
- No Git, no task-record edits outside the record directory, no GPU or platform operations, no network writes.
- Environment: system `python3` is enough.

## Expected return
- Report language: English.
- Deleted files (count, bytes) by category and label; kept scratch files with reasons; protected-path hash check
  before/after; the process check; exact paths of the manifest, receipt and report.
- Any file you could not classify: keep it and list it.
