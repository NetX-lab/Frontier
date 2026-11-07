## Assignment and outcome

Removed established invalid DSV4 calibration artifacts under `/data/ycfeng/tmp/glm53_dsv4` and `/data/ycfeng/tmp/ds4_runs`. Working root: `/data/ycfeng`. Main worktree: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.

Deleted 171 exact artifact directories containing 67,639 regular files and 40,177,033,216 allocated bytes (37.4178 GiB). Available filesystem space increased by 36.4076 GiB. All 171 actual removal receipts PASS. Overall outcome is **partial** because the strict protected-state comparison observed concurrent changes outside this executor's write/deletion scope.

All records are under `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/`. The detailed local report is [report.md](/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/report.md).

Authorization came from requirements.md:169–173 and :209–213 and the explicit assignment. The current dependency at plan.md:212, all tickets, generation05/06 package records, promotion receipt, post-run commands and November 3 cleanup scripts/report were inspected. Generation06 appeared during inspection and was added to protections. Recent Pi labels were excluded in full; I-010 labels and their caches were conservatively retained.

This executor edited only the new cleanup record directory and deleted exact manifested tmp paths. It performed no Git commands, source edits, existing task-record edits, dependency installation, cloud access/deletion, GPU/platform operations, deployment or messages. Existing `assignment.md` was left intact.

## Changes

| Category | Exact directory entries | Directories inside entries | Regular files | Allocated bytes | GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Disposable pytest trees | 59 | 13,567 | 23,122 | 496,013,312 | 0.4619 |
| Sklearn estimator/prediction caches and locks | 11 | 11 | 15,698 | 2,947,719,168 | 2.7453 |
| Historical fidelity case outputs | 52 | 30,004 | 27,456 | 36,607,066,112 | 34.0930 |
| Old task-owned Pi node caches | 49 | 98 | 1,363 | 126,234,624 | 0.1176 |
| **Total** | **171** | **43,680** | **67,639** | **40,177,033,216** | **37.4178** |

Every removed root, category, reason, evidence, file count, allocation and successful receipt appears in `deletion_manifest.json`. Its descendants and original lstat metadata are enumerated in `inventory.sqlite3`. `deletion_receipt_final.json` contains the 171 successful removals. Each path was passed once through an argv list as `rm -rf -- <exact path>`, without shell expansion or symlink traversal.

The original `candidate_manifest.json` and `deletion_receipt.json` preserve all 172 initial candidates. One node cache, `/data/ycfeng/tmp/ds4_runs/p64-mainline-integration-1102c/scratch/node-compile-cache`, failed its immediate cutoff recheck and was retained without invoking rm. Earlier pytest cleanup in the same label updates its scratch timestamp; concurrent updates cannot be excluded. It is listed among retained candidates.

Each deleted historical fidelity tree retains 77 successful case records with source/harness revisions, scripts, arguments and artifact hashes. Original manifests/results remain beside the removed `cases/` trees; 104 byte-identical copies were added under `evidence/fidelity/<run-path-hash>/`. Cache selections contain only `.pkl`/`.file`, preserving CSVs and configuration. Pytest selections have actual test fixtures and no Git repository markers. Pi selections have ownership evidence, a recorded terminal exit and pre-cutoff timestamps.

All new record files are enumerated in `record_files.json`; untouched `assignment.md` is identified as pre-existing. Principal new scripts are `inventory.py`, `context.py`, `duplicates.py`, `select_entries.py`, `safety.py`, `protect_inputs.py`, `source_pins.py`, `prepare_deletion.py`, `delete_entries.py`, `run_cleanup.py`, `finalize_manifest.py`, `verify_cleanup.py`, and `write_report.py`. New evidence includes inventory/context/tracked-path catalogs, duplicate evidence, protection plan, before/after hashes and tree counts, source pins, process scans, validation, df snapshots, commands/logs, candidate/final manifests, receipts, verification and reports. Archived November 3 scripts were copied into `reference_20261103/` and their originals were unchanged.

## Verification

System interpreter: Python 3.10.6. Required scans/phases used `timeout`; imports used `PYTHONDONTWRITEBYTECODE=1`. No environment or dependencies changed.

| Check | Expected | Observed | Result |
| --- | --- | --- | --- |
| Actual removal receipts | Listed path absent, rc0, no stderr | 171/171 | PASS |
| Unlisted original paths | No missing path or identity change | 0 missing; 2 protected Pi-file inode replacements; 128,333 surviving nodes checked | FAIL strict identity check |
| Protected file SHA256 | Before/after equal | 9,246 equal; 1 changed; 0 snapshot errors | FAIL |
| Protected tree counts | Existence and count equal | 74 trees; 2 count differences | FAIL |
| Registered worktrees | Previously existing roots retained | 178 roots; 0 missing | PASS |
| Native/harness pins | a06a72a4 / f33306d4 unchanged | Both match before/after | PASS |
| Generation05 manifest | 23e26da6… unchanged | `23e26da6c02cc42e54134c12abfdfea298f64191e0f985a860505e374278227c` | PASS |
| Promotion receipt | d505eece… unchanged | `d505eecea90c0ea0a285a3cff76086e11b2c3914a88ad2a188dd9c94323f7596` | PASS |
| Visible process references | Zero selected-path references | Pre/immediate scans each saw 7 processes, 0 hits, 0 unreadable | PASS within namespace |
| Duplicate traces | Delete only hash-identical copies | 25 traces hashed; zero duplicates; all retained | PASS |
| Script syntax | All final scripts parse | ast.parse passed | PASS |
| Simulation, GPU runs and test suite | Cleanup-only scope | Not run | NOT_RUN |

`df -B1 /data/ycfeng` on `/dev/rbd1`:

| UTC observation | Used bytes | Available bytes | Available GiB | Use |
| --- | ---: | ---: | ---: | ---: |
| Before: 2026-11-07T06:57:11.652782Z | 871,421,964,288 | 227,552,792,576 | 211.9251 | 80% |
| After: 2026-11-07T07:01:40.493954Z | 832,329,584,640 | 266,645,172,224 | 248.3327 | 76% |

Observed available-space gain: 39,092,379,648 bytes (36.4076 GiB). Allocated bytes use regular-file `st_blocks*512` per path; df can differ because of directory blocks, metadata, concurrent writes and retained records. Initial and final after snapshots are preserved.

Exact executed phase commands follow. Cwd was `/data/ycfeng`; child phase stdout/stderr and exit codes are retained in `execution_commands.json` and `logs/`. The child commands shown after `run_cleanup.py` were launched by that driver, not a second deletion pass.

```bash
timeout 600 python3 -u /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/inventory.py
timeout 600 python3 -u /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/context.py
timeout 600 python3 -u /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/duplicates.py
timeout 180 python3 -u /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/select_entries.py
PYTHONDONTWRITEBYTECODE=1 timeout 180 python3 -u /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/protect_inputs.py setup
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/source_pins.py source_pins_before.json
PYTHONDONTWRITEBYTECODE=1 timeout 1800 python3 -u /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/run_cleanup.py
PYTHONDONTWRITEBYTECODE=1 timeout 300 python3 -u /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/prepare_deletion.py
PYTHONDONTWRITEBYTECODE=1 timeout 600 python3 -u /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/protect_inputs.py protected_hash_before.json
PYTHONDONTWRITEBYTECODE=1 timeout 300 python3 -u /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/delete_entries.py
PYTHONDONTWRITEBYTECODE=1 timeout 600 python3 -u /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/protect_inputs.py protected_hash_after.json
PYTHONDONTWRITEBYTECODE=1 timeout 300 python3 -u /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/verify_cleanup.py
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/source_pins.py source_pins_after.json
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/finalize_manifest.py
PYTHONDONTWRITEBYTECODE=1 timeout 300 python3 -u /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/verify_cleanup.py
df -B1 /data/ycfeng
PYTHONDONTWRITEBYTECODE=1 python3 /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/cleanup_20261107/write_report.py
```

Preflight corrections were completed before removal: the selector initially treated SQLite's serialized `"[]"` errors field as truthy and selected nothing; decoding it corrected the selection. Protection setup initially classified a synthetic pytest `download/` as native data and rejected the overlap; the resolver was corrected. Report generation later encountered a quoting SyntaxError, which was fixed; final syntax and generation passed. No failed rm was retried. Verification commands executed and reported PARTIAL retained-state acceptance rather than a successful overall acceptance.

## Remaining work and decisions for Claude

Strict protected-state acceptance remains incomplete because these changes were observed:

- `h200_mainline/build_package.py`: 56,371 → 57,042 bytes; SHA256 `ef8d750444eeda63006d27d6056b236112b1c93c9fba7c429a2242dba16214a0` → `980d8ab3042b49361f642cd0f156472f512cd344a0c073b67253e5d17e87e212`.
- Main worktree regular-file count: 39,591 → 39,672.
- `h200_mainline/p2_64_formal_retry06/` regular-file count: 65 → 96.
- Protected `/data/ycfeng/tmp/ds4_runs/p64-formal6-submit-1107j/execution.json` inode: 1078146678 → 1078146685; its `exit` inode: 1078146682 → 1078146678, on device 64784.

These paths are neither deletion entries nor executor write targets. Concurrent lead/mainline activity is an inference supported by the exact operation receipts and observed generation06 growth. Restoring them would overwrite concurrent work and exceed the assigned write scope. Claude should review the builder change and generation06 activity, then decide how to accept the cleanup against those external changes or establish a new protected baseline.

There are 89 retained candidate/disposition entries with evidence in `deletion_manifest.json.uncertain_retained`. They include protected/recent or unproven-owned Pi labels, Git-bearing pytest trees, large residual native/accepted diagnostic trees, and these three permission-denied roots:

- `/data/ycfeng/tmp/glm53_dsv4/native_capture_scope_root_cctbt1_m`
- `/data/ycfeng/tmp/glm53_dsv4/native_capture_scope_root_3ov6bjro`
- `/data/ycfeng/tmp/glm53_dsv4/profile04_terminal_ybbq4f2m`

Those roots were retained without chmod or deletion workarounds. Additional deletion requires proof of invalidity/duplicates or resolution of their read-permission prerequisites. The 25 candidate simulator traces were distinct despite equal byte sizes and were retained.

Unique raw measurements and profiling tables, baseline, formal replay/P2-32 evidence, R5 source, current packages, promotion/workload inputs, environments, credentials and registered worktrees remain retained. Seven planned post-run paths were already absent before cleanup and are recorded in `protection_plan.json.absent_before`.

The process scans cover only this execution namespace. Zero visible references are not a host-wide process guarantee. No cloud job phase, GPU-health or resource-control conclusion is claimed.

## Review handoff

Review the exact final manifest and embedded receipts, retained evidence and 104 fidelity-summary copies, ownership/terminal proof for Pi caches, and before/after hashes/counts. Inspect `verification.json` for the protected-state differences and the original execution records for the deliberately retained cache.

Rerun the read-only `verify_cleanup.py` and `source_pins.py source_pins_after.json`, and inspect current capacity with `df -B1 /data/ycfeng`. Later verification can observe further live/mainline changes. Do not rerun `delete_entries.py` or `run_cleanup.py`; the deletion driver intentionally refuses an existing receipt. Final acceptance remains with Claude.
