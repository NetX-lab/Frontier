# T2 coverage after the shared-input snapshot

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Locate actual T2 source consumption and verify its 26 supplemental bindings against the snapshot. |

## Result

**PASS_STATIC_COVERAGE_NOT_REBOUND.** No new bound files or helper changes are required by the snapshot. Controlled T2 imports from the entire pinned repository mounted read-only, with `PYTHONPATH=$REPO`; its 26 file hashes supplement that revision and do not define the import closure.

The explorer (`ses_e9593d566ffewode6CXodumj9I`) reports 43 Frontier modules in the static closure, unchanged from `79f1a3fd`. Neither `collective_profile.py` nor the prediction-manager/backend paths are reached. Dynamic imports identified by that inspection are Torch and collective-sim schema/scenario modules. `FRONTIER_STEP5_PROFILE_MANIFEST` is unset, so its conditional profiling observer is disabled. This is static evidence, not execution in the native image.

Primary independently checked all 26 disk SHA256 values and the corresponding Git blob IDs at `79f1a3fd35d42b751cd1520c764ddac717b923d9` and `1a365ebc8f27090e48f9223de80807fc68fe13b0`: **26/26 match**. The parent also read the actual mount/PYTHONPATH recipe. It did not rerun the separate 63-checksum package or reproduce the 43-module closure analysis.

## Decisive paths

Paths below are relative to `task_memory/task_2026-10-22_step5_position_routing/`.

- `resumed_parent/t2_e41_binding_receipt.json:8–35`: exact 26-path hash map.
- `resumed_performance/t2_binding_receipt.json`: latest static receipt for `79f1a3fd` and intended integrated-06 target.
- `../task_2026-10-22_step5_d29_native_balanced/t2_preparation/execution_proposal.md:144–155,175–183,232`: whole repository mount, explicit PYTHONPATH and existing entry script.
- The proposal currently present in primary still names `1be91bad`/integrated-04. The later static receipt is not proof that newer packet records have been staged there.
- `t2_snapshot_coverage.json`: current parent result and evidence limits.

## Parent verification

From the primary worktree, Python 3.10.6, `timeout 60s`, no product imports:

```python
bindings = json.loads(Path('task_memory/task_2026-10-22_step5_position_routing/resumed_parent/t2_e41_binding_receipt.json').read_text())['full_repository_files']
assert len(bindings) == 26
for path, expected in bindings.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    assert subprocess.check_output(['git', 'rev-parse', '79f1a3fd35d42b751cd1520c764ddac717b923d9:' + path]) == subprocess.check_output(['git', 'rev-parse', '1a365ebc8f27090e48f9223de80807fc68fe13b0:' + path])
```

Exit 0. No fixture CSV, model weights or trace scan; no package edit, source rebind, seal, network or platform operation. Final key source, CPU workflow, record staging, final 26-file check, seal and native/platform qualification remain pending.
