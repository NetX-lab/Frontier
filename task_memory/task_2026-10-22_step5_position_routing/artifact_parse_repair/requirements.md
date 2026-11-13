# D29 collective artifact parsing requirements

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Record delegated exact-text experiment,conditional implementation and unchanged acceptance boundaries. |

TaskID D29-COLLECTIVE-ARTIFACT-PARSE. Original PRICING owner;isolated
`/data/ycfeng/Frontier/.worktrees/step5-d29-artifact-parse`,branch
`feat/step5-d29-artifact-parse`,immutable base
`24f970cbd6cae059467f68e0e1cf771b07aeecfe`.

[Original Request] Main07 offline/online timed out2400s on reviewed e41abc07.
Read current requirements,plan,input_validation_and_artifact_parsing.md,
collective_artifact_parse_scope_amendment.json and collective_json_scope report.
CORE group-resolution105-gross repair runs independently;do not edit integrated05
or existing work. First modify only a new scratch copy of retained loop.py for an
exact-text single-variable control;retain original script and fixed group reuse.
No global json patch. Require equal complete costs/proofs/plan keys,removed large
decodes,material measured benefit and actual timing/residency before implementation.
Stat-keyed scout timings are not evidence for this selected exact-text design.

[Selected Design] Read current artifact text for every plan call. Keep only one
backend-local latest-text/parsed-result pair;reuse parsed result only when actual
text equals retained text;replace after successful parse. No hash/stat/inode/global
cache,flag,cost-cache repurposing,native cache-key change,shared API or CORE edit.
Preserve live TP/DP,CPU/native qualification,count/context/buffer/composition
checks and current read/parse errors. Existing consumers read the artifact dict
without mutation. Same-instance rewrites remain visible,even same-size/restored
mtime. Stop/report if control does not show material benefit or reveals a new
semantic tradeoff;do not silently select another design.

[Allocation] Only frontier/cc_backend/backends/collective_sim_cc_backend.py and
tests/unit/test_collective_sim_cc_backend.py plus this artifact_parse_repair record
directory. New source<=15,test<=35,total<=50 gross;retained caps93/233. PRICING3956
unchanged,current3879,77 remaining;effective111 paths/global10000 unchanged.
The root's50-line zero-sum overlay is collective_artifact_parse_scope_amendment.json;
old CPU-plan120 and other historical step caps are not new authority. No other
code path,peer import/edit,GPU/platform/full CLI or delegation. Scratch under
/data/ycfeng/tmp;diagnostic commands<=60s/30GiB. Submodule3dac53c is root-owned,
unchanged;root owns final review,freeze,full run and qualification.

[Acceptance] Source RED/GREEN must exercise real constructor/current-repo CPU
writer and actual predict_collective_plan;cover reuse,current read on hits,
same-size changed text with original mtime,invalid current JSON and read errors,
existing same-instance count rewrite. All45 old backend tests and relevant trace
consumers PASS;retain all negative tests and original numerical criteria. Commit
verified source and records separately;deliver exact patch,counters,full payload
hashes,residency,test commands,JUnit and exact retained/step budgets.
