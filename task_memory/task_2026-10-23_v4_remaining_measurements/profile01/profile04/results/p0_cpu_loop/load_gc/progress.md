# Loading-GC comparison progress

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Started the separately bounded source/control preparation. |

- in-progress: <=40-line loading-GC wrapper, no parser/predicate/frozen-source change.
- preserved harness failure: first decode-control argv used a nonexistent trace filename and exited1 before loading/GC-disable; stderr SHA256b8da0cfb0105769858b547cc322faf16a0446c59e0be1569ffd614c2e71395a3. Derive the corrected EP8/rank0 trace path directly from the completed freeze-control receipt; preserve all first files. This consumes no P0pilot invocation.
- pending: exact decode control, then at mostone same-limit P0rank0 pilot and inspection.
- pending: report complete versus incomplete output; no all32/reference/cost/GPU acceptance.

- completed decode control:142567kernel rows/61useful steps, exact payload/index equality to retained freeze run; GC disabled only during loading and re-enabled afterward,0GCrows inside loading.
- completed sole P0pilot: exit124/60.314844seconds, loaded in31.384seconds versus47.129freeze baseline; kernel pass ends52.788seconds, stream pass58.187seconds, sync58.201seconds and512steps/39useful58.218seconds. Last heartbeat is final metrics sorted intervals; no final audit JSON, so P0/full32 acceptance remains INCOMPLETE.
- pending next allocation: isolate final metrics cost from existing source/outputs before any further full-input run; original failures, complete kernel prefix and all raw/source bytes remain retained.
