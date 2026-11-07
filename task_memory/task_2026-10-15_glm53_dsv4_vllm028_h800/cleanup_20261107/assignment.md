# Task: remove invalid artifacts produced by the DSV4 calibration task

## Goal
Free disk space by deleting artifacts that the DSV4 H200/H800 calibration task produced and that are no longer valid. Every retained input of the current mainline must stay byte-identical. Acceptance:
- Each deleted path appears in a deletion manifest. The manifest gives the path, category, reason with evidence, regular-file count, allocated bytes, and the deletion receipt (absent afterwards, no error).
- Every protected path listed below is unchanged. Use SHA256 before/after for files, and existence plus file count for large trees. No unlisted path is deleted.
- Before/after `df` for `/data/ycfeng` is recorded.

The user explicitly authorized `rm` for this (requirements.md, "2026-11-07 — prompt cleanup of invalid task artifacts through codex-implement", and the 2026-11-03 cleanup decision). Delete only what you can show to be invalid. If an entry is uncertain, keep it and list it with your evidence.

## Approach
- Reuse the method and criteria of the 2026-11-03 cleanup. That cleanup removed 110 directories (46.86 GiB). Its record is `cleanup_20261103/` and `test_report_2026-11-03_task_cleanup.md` in the task directory. Its scripts are in `cleanup_20261103/`: validate entries, scan /proc for references, hash retained files, delete exact manifest paths once with `rm -rf -- <path>` without glob expansion or symlink traversal, and hash again. Copy and adapt them; do not edit the archived copies.
- Invalid categories:
  - regenerable sklearn estimator/prediction caches and lock files;
  - pytest basetemp and other disposable temporary trees;
  - run output copies that are superseded or failed, when their outcome and key evidence are already recorded in the task directory (manifest/results/acceptance JSON retained);
  - duplicate copies of data that exists elsewhere, with matching hashes;
  - node-compile-cache directories inside Pi run scratch.
- Read the task records to find what is still in use: the current plan's dependency line, post_run_command.json, acceptance JSONs, and the tickets. A path referenced as current input or accepted evidence is protected unless an identical copy is retained.

## Scope and context
- Working root: /data/ycfeng. Main worktree: /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028, branch feat/glm53-dsv4-vllm028.
- Task records: /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/ (requirements.md, plan.md, progress.md, summary.md, tickets/, issues.md).
- Deletion scope:
  - `/data/ycfeng/tmp/glm53_dsv4`: task-owned scratch, about 194 GiB. `h200/` is 91 GiB; the other large trees are h3o_b, m2_decode_lane, w7b_sp_collectives, t00-t02, merge_*, w7a_dsa, w7b_measured_cc, a6_pass_replays_20261101 and c06/c09.
  - `/data/ycfeng/tmp/ds4_runs`: only labels whose `cwd` file is this worktree or whose prompt references this task; other labels belong to other work.
- Protected; never delete:
  - h200_mainline/p2_64_formal_retry05/ (package_01 with manifest 23e26da6c02cc42e54134c12abfdfea298f64191e0f985a860505e374278227c, approval_receipt.json);
  - /data/ycfeng/tmp/glm53_dsv4/h200/launch_p2_64_formal_retry05.sh;
  - h200_mainline/build_package.py;
  - the monitor helpers (/data/ycfeng/tmp/glm53_dsv4/h200/monitor/, terminal_receipt.py, /data/ycfeng/tmp/ds4_runs/p64-formal2-1102n/scratch/poll_once.py);
  - /data/ycfeng/tmp/glm53_dsv4/h200/ds4_resume/;
  - native source pin a06a72a4 and harness worktree /data/ycfeng/Frontier/.worktrees/p2-64-startup-harness (f33306d4);
  - P2-32→P2-64 promotion receipt d505eece and its inputs;
  - the R5 reader;
  - the completed Frontier replay and everything that post_run_command.json references;
  - P2-32 formal PASS evidence;
  - the 2026-11-03 baseline /data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/baseline;
  - unique raw native measurements and profiling tables;
  - environments /data/ycfeng/envs/*, /data/ycfeng/tmp/stepmind-env and credentials (/data/ycfeng/tmp/brainctl-auth-*);
  - Pi labels p64-formal5-submit-1107a, p64-formal5-submit-1107d and provider-probe-1107c (issue I-010 evidence; only their node-compile-cache may go);
  - every ds4_runs label created or modified on or after 2026-11-07T05:50Z, in full. The lead may start the live generation05 submission (labels p64-formal5-*) while you run; its launcher writes into that label's scratch and into package_01;
  - every Git-tracked file, every registered worktree, and the untracked .cortexkit/, .pi/, launcher.pid.txt and remaining_measurements/ in the worktree;
  - any path referenced by a live process.
- Exclusions: no cloud access or deletion (/mnt/codesign-exp), no GPU or platform operation, no Git operation, no edits to source code or to existing task-record files. Candidates inside task_memory or in tracked files: report them only.
- Writable paths: the two deletion roots and this record directory, cleanup_20261107/.
- Environment: system python3 (3.10). Bound memory-heavy scans with `timeout` and avoid repeated full-tree walks; one inventory pass is enough.

## Expected return
- Report language: English.
- Write into cleanup_20261107/: the inventory, deletion_manifest.json, the deletion receipt, protected-hash before/after, df before/after, the scripts used, and a short report.md. The report lists categories with directories, files, bytes and GiB; the retained uncertain candidates with evidence; and checks with expected versus actual results.
- Return the same summary, the exact commands, and any blocker.
