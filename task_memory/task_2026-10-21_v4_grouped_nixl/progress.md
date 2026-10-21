# Progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Created task and isolated branch/worktree from 6d88700. |

- completed: read current parent requirements, plan, ticket07 and V4 bounded unit
  brief. Created v4-grouped-nixl worktree at exact 6d88700.
- completed: pinned native retention/connector body controls; native_rules.json
  records three boundary prompts and exact source hashes/line ranges.
- completed: grouped source snapshots, remote slot reservation through the existing
  allocator, structural payload selection, completion preflight and release.
- completed: final supported/unsupported CPU checks and record consolidation.
- in-progress: implementation commit.
- pending: one clean strict matrix and final receipt.

Observed: source currently restricts NIXL to ungrouped DSA/block64. Grouped
capacity already exposes get_group_size_bytes and allocator block IDs, so no
capacity schema extension has been demonstrated necessary.

## Native-rule decision and implementation

The native scheduler prunes before `_connector_finished` exchange. The full
6144 prefill initially owns [24,96,96,1536,768] pages; after the hook's processed
frontier pruning it owns [24,2,2,2,16]. Connector exchange includes null slots
([24,3,3,3,17]), while D filters nulls and the worker takes the source suffix
matching D: [24,2,2,2,16]. No Mamba rule applies to compressor state.

Existing `get_group_size_bytes` supplies component-page bytes. Existing grouped
allocation needed the native zero-local-work/external-computed-token input:
pretending to schedule q1 at prompt-1 would reserve the wrong window boundary
for prompt191. The new keyword flows through the same allocation plan and pool,
with no second allocation owner. Remote reservation retains only actual pages;
the explicit scheduler frontier is prompt-1 for the first D replay.

Verification iterations (raw logs `/data/ycfeng/tmp/v4_grouped_nixl_focused_0N.log`):
- 01: 69 existing tests PASS; 23 new cases failed due to missing DP peer context
  in the test fixture and the 6145+1024 max-length boundary. The component test now
  runs the actual one-owner scheduling iteration, not an unprovided DP busy loop;
  max length is7169 for the +1 boundary case.
- 02: actual predictor rejected a whole-model bind of mixed compression variants.
  Fixed the grouped branch to validate all layer families before scalar binding.
- 03: fixture lacked KV_CONNECTOR=nixl, so Batch contexts used logical progress.
  Added isolated global context matching the real NIXL consumer.
- 04: one test literal had the wrong byte sum. Inspected each existing capacity
  group:1002240/823680/786240/870912/656640 B; exact sum is39521664 B.
- 05:23 new cases PASS, including three complete1024-output loops.
- 06:363 affected tests PASS in18.00s, including GLM 96-page/569106432B context
  regression, grouped allocator, V4 source/layout/parameter/cache-zero and P2P
  extraction checks. Added bounded non-NIXL rejection/remaining execution checks
  afterwards; final run is pending.
- Native source-body control initially lacked block_hash on its null fixture;
  fixed the fixture field, then all three prompts passed. No native code changed.
- Final affected regression:371 PASS in16.95s, no skips, exit0. Durable JUnit
  cpu_tests.xml includes actual group/byte/output/free-pool receipts for all three
  complete loops. Log: /data/ycfeng/tmp/v4_grouped_nixl_focused_receipt.log.
- Read accepted parent H200 G0-B report/runtime_verification.json and H200
  admission root_acceptance.json. No new native runtime observation is claimed.
- Module-size check: engine1799 lines before the final bounded NIXL-only guard;
  cluster base1969; allocation1063. All touched modules remain below2000.
- Exact staged source/test diff is `source.diff`, generated with `git diff
  --cached --binary --unified=0 -- frontier tests`. Zero context avoids Git's
  whitespace checker treating raw patch context markers as trailing whitespace;
  no production whitespace or check configuration was changed.

Remaining acceptance gaps: real multi-owner compute/DP synchronization, native
wire descriptors/timing/lifetime, graphs, measured execution and formal four
metrics are NOT verified by these component tests. Existing execution guards stay.
