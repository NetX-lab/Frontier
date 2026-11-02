# Primary scope/lifetime CPU-source verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded independent50-case candidate/merged replays and raw binding closure. |

Commands, Python3.12.13, environment, junit and actual controller are fully bound in
root_cpu_replay_owner_worktree.json and root_cpu_replay_merged_worktree.json.
Driver: `/data/ycfeng/envs/asyncopd-py312/bin/python -B
/data/ycfeng/tmp/glm53_dsv4/scope_lifetime_root_20261024/verify.py <worktree>`;
worktrees are `/data/ycfeng/Frontier/.worktrees/v4-scope-lifetime-probe`
and `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.
Actual suites: tests/unit/test_cupti_scope_lifetime.py and test_cupti_graph_resource_payload.py,
with pytest -q -p no:cacheprovider inside timeout60s/systemdMemoryMax30GiB.
Expected50/0failures/errors/skips; actual50/0/0/0 on e081d5e8 and mergedd3f9bbcd: PASS.
Each exports25fixtureJSON; actual memory.limit32212254720bytes, CUDAuninitialized;
selected source/native/runtime/45sealed04 inputs unchanged: PASS.
Root additionally rehashes26retained bindings across22closed raw artifacts: PASS.
First pre-entry gate used historical48-source receipt and stopped on the two intentional
source/test version mismatches, before pytest; authoritative tuple_repair_receipt resolves it.
Source252/test179 within265/185 and both full13-file staticreviews closed: PASS.
No actual image/SDK/TLS/defaultFalse lifetime/GPU/nativeintegration/ownership/cost qualification.
