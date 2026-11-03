# Task artifact cleanup — 2026-11-03

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-03 | Removed confirmed rebuildable task outputs and recorded independent input, evidence, launcher and disk verification. |

## Scope and result

YC explicitly requested removal of unnecessary large artifacts from the current task, including `rm`; the original request and deletion authority are in `requirements.md`, “task artifact cleanup”. This cleanup is complete. The overall DSV4 calibration task remains in progress.

Baseline branch: `feat/glm53-dsv4-vllm028`, HEAD `989dc02c8cce64ec17be4bdd88b94752140d0398`. Only exact task-owned paths under `/data/ycfeng/tmp/glm53_dsv4` were removed. No production code, measured table, native source, Git worktree or cloud data was deleted. Existing untracked `.cortexkit/`, `.pi/`, `launcher.pid.txt` and `remaining_measurements/` remain outside the cleanup scope.

| Removed category | Directories | Regular files | Allocated file bytes | GiB |
| --- | ---: | ---: | ---: | ---: |
| Regenerable sklearn estimator/prediction caches and lock files | 65 | 110,502 | 20,064,489,472 | 18.6865 |
| Completed historical CPU fidelity case outputs | 41 | 21,648 | 28,863,688,704 | 26.8814 |
| B8 pytest temporary fixture trees | 4 | 62,861 | 1,392,152,576 | 1.2965 |
| Total | **110** | **195,011** | **50,320,330,752** | **46.8645** |

The pytest trees also contained 2,535 symbolic links. These links were unlinked without traversing their targets. Three test links pointed to `/tmp`; the target directory was not deleted. No manifest entry was skipped and no deletion failed. Exact paths/reasons/counts are in `cleanup_20261103/deletion_manifest.json`; both full batch manifests and receipts are beside it.

Why these files were unnecessary:

- Missing sklearn caches are rebuilt through the existing predictor load/train path (`frontier/execution_time_predictor/sklearn_execution_time_predictor.py:3082`, `:3188`). Source CSVs, configuration and run outputs remain outside the deleted cache directories. Fold configuration files were retained.
- Each selected historical fidelity run retains its `manifest.json` and `results.jsonl`, including source/harness revisions, case scripts, arguments, environment, return codes and artifact hashes. The lead independently read all **41 × 77 = 3,157** successful case records. The authoritative baseline at `/data/ycfeng/tmp/glm53_dsv4/t00/refactor_fidelity/baseline` remains intact. Old generated case outputs would need regeneration for a future raw re-comparison; no new fidelity PASS is claimed by this cleanup.
- B8 pytest basetemp directories are disposable test fixtures. Original source/measurements and sibling test scripts, logs, XML and JSON results remain intact.

## Execution and commands

Execution used the existing `/ds4-exec` workflow through `cleanup-caches-1103c`, turns 1 and 2; both ended normally with `exit 0`. Pi used system **Python 3.10.6**, GCC 11.3.0, without package installation or a changed environment. Lead verification also used system Python 3.10.6. The original inventories are preserved in `cleanup_20261103/top_level_sizes.json`, `task_sizes.txt` and `worktree_sizes.txt`.

Exact launch commands:

```bash
bash /data/ycfeng/codex-home/skills/ds4-exec/scripts/run_ds4.sh cleanup-caches-1103c /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028 /data/ycfeng/tmp/ds4_runs/prompts/cleanup-caches-1103c.md
bash /data/ycfeng/codex-home/skills/ds4-exec/scripts/run_ds4.sh cleanup-caches-1103c /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028 /data/ycfeng/tmp/ds4_runs/prompts/cleanup-caches-1103c-replays.md
```

From `/data/ycfeng/tmp/ds4_runs/cleanup-caches-1103c/scratch`:

```bash
python3 validate_entries.py validation_pre.json
python3 scan_proc.py proc_scan_pre.json
python3 hash_retained.py hashes_pre.json
python3 delete_entries.py deletion_results.json
python3 hash_retained.py hashes_post.json
python3 replay_validate.py replay_validation_pre.json
python3 replay_scan_proc.py replay_proc_scan_pre.json
python3 replay_hash_retained.py replay_hashes_pre.json
python3 replay_delete_entries.py replay_deletion_results.json
python3 replay_hash_retained.py replay_hashes_post.json
```

Deletion scripts invoke `rm -rf -- <exact manifest entry path>` once per approved directory. They do not expand deletion globs or follow the fixture symlinks. Archived script copies are under `cleanup_20261103/`; the first-batch archive names have a `cache_` prefix. The precheck/deletion sequence validates root/ancestor identity, expected counts/bytes, mount/worktree boundaries and observed process references before removal. Deletion runtimes were **17.053 s** and **13.436 s**. Required postchecks covered absence, retained sibling files, their hashes and the active package/launcher.

Lead final verification command:

```bash
python3 /data/ycfeng/tmp/cleanup_glm53_dsv4_20261103/verify_cleanup.py
```

Its retained copy is `cleanup_20261103/verify_cleanup.py`; result: `cleanup_20261103/lead_verification.json`. No simulator replay, GPU run or test suite was rerun for cleanup.

## Independent acceptance checks

| Check | Expected | Observed | Verdict |
| --- | --- | --- | --- |
| Exact deletion scope | All selected paths absent, no unlisted deletion | 110/110 absent; zero skips/errors in both receipts | PASS |
| First-batch adjacent evidence | Same files and SHA256 | 83/83 unchanged under independent lead readback | PASS |
| Second-batch adjacent evidence | Same files and SHA256 | 114/114 unchanged under independent lead readback | PASS |
| Historical run records | Retain every selected completed run's summary and baseline | 41 manifests/results, 3,157 successful case records; baseline retained | PASS |
| Current formal inputs | All pre-cleanup protected files/roots retained | **602/602 hashes unchanged**; no missing protected root | PASS |
| Current package seal | Same manifest and all 47 sealed files | Manifest SHA256 `d75b0277a3c08592359002d256ac8115844f50fa051efe422815cc5b2d19f081`; all 47 files match | PASS |
| Worktrees | No registered worktree removed | None removed; second-batch scope checked against 175 registered roots | PASS |
| Active launcher | Preserve PID identity/cwd and keep alive | PID 81539 alive, identical command/cwd | PASS |
| GPU job | Same exact job, no cleanup-induced control | 07:30:36Z: `exp-1103-131952-259161`, UID `9d02faf7-621e-4368-ae73-4110312c6ae2`, Pending, 0 replicas, no read errors | PASS within snapshot limits |

The protected inputs include the fourth-generation package, native and harness source trees, formal P2 replay and promotion evidence, and required Python environments. Job submission/control did not occur. The exact job remains wholly preemptible in `step_main`; its queue reason is `Insufficient GPU quota`. No containers have started in the retained snapshot, so no terminal receipt or new numeric P2-64 result exists. The job snapshot is `cleanup_20261103/live_job_after.json`.

## Filesystem space

`df -B1 /data` on `/dev/rbd1`:

| Observation (UTC, 2026-11-03) | Used bytes | Available bytes | Available GiB | Use |
| --- | ---: | ---: | ---: | ---: |
| Before first deletion, 07:18:03 | 1,008,673,337,344 | 90,301,419,520 | 84.0998 | 92% |
| Lead final check, 07:31:27 | 960,539,348,992 | 138,435,407,872 | 128.9280 | 88% |

Observed available-space increase: **48,133,988,352 bytes = 44.8283 GiB**. Removed regular files account for 50,320,330,752 allocated bytes and 49,977,417,327 logical bytes. Allocated bytes are summed file blocks, not a claim of identical physical-space recovery. The precise reason for the difference was not isolated; concurrent host activity and filesystem accounting can affect the observation. The second batch had no regular files with `st_nlink > 1`. The first executor's attribution of the whole difference to concurrent writes was an inference, not verified fact; this report preserves the observation without that causal claim.

## Limits and progress correction

- Unique native/profiling traces, formal/reference inputs, current run dependencies and required historical structural acceptance outputs were retained. Cloud storage and unrelated environments/files were untouched.
- The initial temporary-directory inventory covered 441 directories (about 274.11 GiB in readable directory entries). Three roots were unreadable and retained: `/data/ycfeng/tmp/glm53_dsv4/native_capture_scope_root_3ov6bjro`, `native_capture_scope_root_cctbt1_m`, and `profile04_terminal_ybbq4f2m` under the same parent. No permission escalation was attempted.
- `/proc` visibility was incomplete for other users and protected same-user service processes. All readable cwd/fd/maps/cmdline references had no selected-directory usage, and the active launcher was independently checked. Absence of a visible reference is not a universal claim about unreadable processes.
- Two survey executors (`cleanup-h200-1103a`, `cleanup-replays-1103b`) expanded into repeated large-trace scans. The lead's check-progress identified D6 and stopped those cleanup-only labels (exit 130), after verifying their session IDs differed from the live launcher's marker. No deletion was assigned to those surveys. Existing inventory evidence was reused to make the second exact cleanup decision; no further broad trace investigation continued. `cleanup_20261103/survey_disposition.json` and `check_progress.md` record this correction.
- Overall task next actions remain: monitor the existing P2-64 job; after terminal business output, preserve clocks/receipt, copy if necessary, verify hashes/pair coverage, then R5 and all-request/warm comparisons using the retained Frontier replay. No fifth submission is authorized by cleanup.
