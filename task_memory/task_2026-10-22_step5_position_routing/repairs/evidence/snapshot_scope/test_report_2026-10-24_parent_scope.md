# Parent verification of the snapshot scope audit

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Verify the independent114-path snapshot audit against primary Git/files, then archive the original records unchanged. |

## Result

PASS_SNAPSHOT_BASELINE_ONLY. Owner d267db2d is archived as ae55a5f5; all seven original record blobs match. Parent verifies six artifact hashes plus the inventory hash,112 manifest bindings,105 superproject path numstats, eight native base/derived file hashes and eight fixed image-evidence hashes. Every current primary source/test/config blob equals the audited snapshot. The114 rows give CORE4173/4404, PRICING3940/3956, native1350/1582, source9463/9942 and global9477/10000;113 paths changed and no violation was found.

`parent_receipt.json` carries the exact source/root commits and counts. The independent report owns original ledger/15-overlay replay and in-memory native patch application; parent verified its bindings and current counts rather than rerunning that entire audit. The key implementation is excluded, so its two-path delta and final union still require verification.

## Commands and environment

The stdin-based parent checker ran under:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- /data/ycfeng/envs/frontier-py310/bin/python -B -
```

It loaded the owner inventory, receipt and blob manifest by their recorded absolute paths, checked each SHA256/size, and used these exact command forms with values from those records:

```sh
git diff --name-only e200bb780d3b20ec6f5ff0a067ed8b9ddd505662 HEAD -- . ':!task_memory/**'
git rev-parse HEAD:<manifest-path>
git cat-file blob <manifest-blob-oid>
git --git-dir=/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06/frontier/cc_backend/backends/collective-sim/.git rev-parse 3dac53c544e546574cb6e8078cd14053d0916113:<submodule-relative-path>
git --git-dir=/data/ycfeng/Frontier/.worktrees/verify-step5-d29-integrated-06/frontier/cc_backend/backends/collective-sim/.git cat-file blob <manifest-blob-oid>
git diff --numstat --no-renames 1aee0af11cd58691e64603b99eca0ac459a5b350 HEAD -- <105 kind=superproject paths from snapshot_baseline_receipt.json>
```

The first diff is empty; all bindings and per-path additions/deletions match. The105-path list is supplied explicitly as subprocess argv, not a filesystem wildcard. Stdlib hashlib reads only those bound files and the named native/image evidence; no profiling CSV or trace scan. Python3.10.6, exit0, cgroup scope `run-rca153e6bc2a44f8280685378bd5d151b.scope`. No product tests, network, GPU or package rebuild.

Archive commands then verified the changed-file set, cherry-picked d267db2d, compared all seven Git blob IDs and ran `git diff --check HEAD~1 HEAD`; all pass. New parent records are checked separately before commit. Final key/frozen regression/full-CLI/T2/native gates remain open.
