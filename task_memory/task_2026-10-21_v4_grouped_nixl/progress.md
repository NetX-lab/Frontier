# Progress

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Created task and isolated branch/worktree from 6d88700. |
| 2026-10-21 | Closed P2 actual callback RED/GREEN,470 focused PASS and one clean5428bd7 strict77/77 identical run. |

- completed: read current parent requirements, plan, ticket07 and V4 bounded unit
  brief. Created v4-grouped-nixl worktree at exact 6d88700.
- completed: pinned native retention/connector body controls; native_rules.json
  records three boundary prompts and exact source hashes/line ranges.
- completed: grouped source snapshots, remote slot reservation through the existing
  allocator, structural payload selection, completion preflight and release.
- completed: final supported/unsupported CPU checks and record consolidation.
- completed: implementation commit1a65e4985091f5c85c14830f86385800a614e4e6.
- completed: original clean strict77 matrix and final records commit17e3b42.

Initial observation at6d88700: source restricted NIXL to ungrouped DSA/block64. Grouped
capacity already exposes get_group_size_bytes and allocator block IDs, so no
capacity schema extension has been demonstrated necessary.

## Original slice: native-rule decision and implementation

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
- Post-commit clean matrix: one measure_commit invocation at1a65e49, label
  candidate_v4_grouped_nixl_1a65e49, against the required absolute baseline.
  Actual77/77 identical,0 failures/missing evidence/definition/provenance/cache
  filename differences. Checked all77 clean per-case source stamps, retained
  comparator JSON and candidate checkout cleanliness. Raw log:
  /data/ycfeng/tmp/v4_grouped_nixl_strict77.log. No checks relaxed or matrix rerun.
- cpu_receipt.json records the canonical actual39521664B payload,6143..7166
  contexts,7167 final computed frontier,1024 outputs and3999+null free pools.
- Final bounded source read confirmed native async P max_tokens1 avoids extra
  speculative scheduling, drains in-flight6144 before finish and invokes the
  pruning/connector hook; source/test code is unchanged after measured commit.

Remaining acceptance gaps: real multi-owner compute/DP synchronization, native
wire descriptors/timing/lifetime, graphs, measured execution and formal four
metrics are NOT verified by these component tests. Existing execution guards stay.

## P2 real MetricsStore/op tracing repair

- completed: original failure reproduced before the fix through real MetricsStore
  and enabled TraceStore. Canonical6144 test fails at start callback with
  ValueError: Mixed compression variants require an explicit global layer id;
  no transfer-end event is returned. p2_red.xml,1 failed in2.22s;
  /data/ycfeng/tmp/v4_grouped_nixl_p2_red.log.
- Diagnosis predictions, before the fix: (1) grouped metadata incorrectly reaches
  scalar binding, so early dispatch should eliminate this ValueError; (2) the
  builder lacks snapshots, so forward existing transfer_info and actual scheduler
  block size; (3) a later trace/completion error would fail the full output loop.
- completed: dispatch existing grouped payload metadata before scalar binding;
  reuse existing capacity selection and get_grouped_nixl_payload validation.
  MetricsStore only forwards two existing inputs; no new allocation state,
  timing, model binding or execution admission behavior. Groups, component page
  bytes and P/D/selected block snapshots remain separate in the trace.
- completed: same real callback test GREEN on first fix,1 PASS in5.40s;
  p2_green.xml. It receives a transfer-end event, writes/reads actual TraceStore
  JSONL, completes1024 outputs from6143 and returns both pools to3999+null.
- completed:470 focused checks PASS in21.45s, no skips. p2_focused.xml includes
  actual V4 trace/lifecycle properties; existing GLM metadata test now compares
  every prior field to a real MetricsStore callback and completes the transfer.
- completed: native_retention.md now retains12 exact executed native method
  bodies and9 callsite/spec excerpts from pinned git show. Complete-file hashes
  of the12 methods match native_rules.json. No extraction rule or native code
  changed; no dependence on native working HEAD for review.
- Module size: op_trace_utils.py1089 lines; MetricsStore5744 lines. Existing
  docs/general/module-size-exceptions.md already records the metrics boundary
  and future ledger split. The P2 change adds only two argument-forwarding lines
  there; metadata logic remains in op_trace_utils, avoiding unrelated cleanup.
- completed: p2_receipt.json and p2_source.diff retain current RED/GREEN and exact
  repair delta; source.diff is the cumulative source/test delta from6d88700.
- Artifact-only correction: raw RED JUnit includes traceback trailing spaces.
  Replaced those lexical spaces with XML character references; parsed XML is
  byte-identical before/after serialization. Raw log, failure text, production
  source and whitespace check configuration are unchanged.
- completed: focused-tested P2 commit5428bd7. Both exact diff artifacts match
  that commit; source tree identities are recorded in p2_receipt.json.
- completed: the unique new clean strict77 candidate_v4_grouped_nixl_p2_5428bd7
  against the required absolute baseline. Actual77/77 identical; zero failures,
  missing evidence, definition/provenance/cache filename differences. All77
  source stamps are5428bd7/source_dirty=false; the detached checkout remains
  clean. No dirty matrix, relaxed check or repeat was run.
- completed: p2_fidelity_comparison.json preserves the exact comparator result;
  p2_receipt.json records original failure, actual callback lifecycle and final
  committed-source/cache evidence. Code/test diffs exactly match5428bd7; binding,
  timing, profiling and strict fidelity harness files are unchanged. Final
  records-only changes do not alter the measured frontier/tests trees.
- No unfinished P2 acceptance item remains. Native timing/protocol/full-path and
  formal accuracy gaps listed above are unchanged and are not claimed as passed.
