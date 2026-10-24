# D29 Ledger Candidate Composed Scope and Binding Audit

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 (UTC) | Recompute all 116 allocated paths, native image patch and pinned collective-sim; preserve two audit failures and the final clean PASS. |

The final audit is **PASS_FULL_COMPOSED_SCOPE_BINDINGS**, with zero violations. This is an independent source accounting and binding audit by the resumed primary session, separate from the original writer owner and the carried-usage handoff checker. It does not admit implementation quality, regressions, native runtime, whole-workload RSS or full CLI.

Frozen source is `1f6b431478e2351e3ca6819f707cccbc3e4336e2`; clean records checkout is `97ae8d9b64f4728ba3d3c9af0647832e83ee0335` at `/data/ycfeng/Frontier/.worktrees/d29-ledger-final-1f6b4314`. Cumulative billing base is `1aee0af11cd58691e64603b99eca0ac459a5b350`; accepted predecessor is `b02772872d0c99ceb5ebb31a8e041926f12f7676`. The ordinary local independent submodule clone is clean at `3dac53c544e546574cb6e8078cd14053d0916113`; no binary was built.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Original approved allocation | 107 paths; unchanged approval rows and SHA | Exact match | PASS |
| Overlay reconstruction | 18 ordered overlays, all owner transfers zero-sum | Exact match | PASS |
| Current allocation | 116 paths; CORE51, PRICING46, native19 | Exact match | PASS |
| CORE | At most 4404 gross | 4377 | PASS |
| PRICING | At most 3956 gross | 3955 | PASS |
| native | At most 1582 gross | 1350 | PASS |
| Source/global | At most 9942/10000 gross | 9682/9696 | PASS |
| Four-file step | At most135 gross; each cap36/20/65/14 | 124 gross, net96; per-file33/15/62/14 | PASS |
| Cumulative growth | Measured separately from step/net | 116 | PASS |
| Current source/native/evidence bindings | All current working bytes and pinned objects agree | 126 bindings | PASS |
| Prior accepted usage | Recompute all 114 prior allocated rows | 114 agree | PASS |
| Scope change | Exactly four approved source/test files; donors unchanged | Exact match | PASS |
| Native source | Apply four patches in memory without fuzz; verify eight actual base/derived files and installed inventory | Exact match | PASS |
| Source state | Frozen checkout/submodule clean; indices unchanged during final audit | Exact match | PASS |

Commands were run from the frozen worktree. The successful invocation was:

```bash
task_audit_root=/data/ycfeng/tmp/d29_ledger_composed_20261024
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONDONTWRITEBYTECODE=1 TMPDIR="$task_audit_root" XDG_CACHE_HOME="$task_audit_root/cache" MPLCONFIGDIR="$task_audit_root/mpl" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B "$task_audit_root/audit03.py"
```

Python is `/data/ycfeng/envs/frontier-py310/bin/python`, version3.10.6/GCC11.3.0, uid10250. The configured 30GiB systemd scope starts, with an outer60s timeout and single math threads. Effective cgroup peak is not measured. The audit receipt records only this metadata process's own `ru_maxrss`; it is not simulator or worker RSS evidence. Final exit0; body duration2.746721s. Exact UTC timestamps, all Git argv/result hashes, per-path counts, allocation transitions and source bindings are in `evidence/composed_scope_2026-10-24/receipt.json`.

The first attempt stops on four externally modified handoff documents in the original candidate worktree. The modifications are late copy/init failure recovery; no source changes. Those edits are preserved. An independent clean frozen checkout is created at the same records commit before rerunning. The managed worktree tool also created an unused checkout outside the project's requested `.worktrees` root; it is retained unchanged, and all accepted audit/test work uses the project-local frozen path above.

The second attempt finishes the accounting assertions but its command receipt contains a `Path` value and JSON serialization raises TypeError. The final script converts command arguments to strings at command construction. It does not change any source, count, condition or admission predicate. All three scripts, raw logs and exits are retained; only the third attempt has a PASS receipt.

Published image and native installed-source identity are checked against retained manifests and actual exported source files. No registry, image rebuild, platform, GPU, cloud, pricing or required-shape operation occurs. Historical step receipts outside the new four-file repair retain their original commit boundaries. Current source usage is recomputed in full rather than inherited from those receipts.

Candidate remains UNADMITTED pending independent Standards/Spec, clean frozen strict77/full-unit/supplements and accepted-production integration. Accepted production remains source9566/global9580.
