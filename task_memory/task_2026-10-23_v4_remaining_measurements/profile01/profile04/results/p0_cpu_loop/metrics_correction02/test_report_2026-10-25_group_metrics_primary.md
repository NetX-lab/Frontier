# Primary group-metrics source and complete-control verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 +08:00 | Accepted the frozen CPU correction after independent reviews, one primary unit and two fresh complete exact controls; affected P0 remains pending. |

**ACCEPTED_CPU_GROUP_METRICS_SOURCE_WITH_EXACT_CONTROLS**. The correction replaces per-event mutable membership lists with integer positions in one stable global interval order. Original group count/sum order, duplicate members and the frozen union recurrence remain exact. It targets measured construction/GC cost; full P0 improvement is not yet observed.

Authority and resources: [allocation](primary/allocation.json), [control inputs](primary/control_inputs.json), [execution allocation](primary/control_execution_allocation.json), and [comparison allocation](primary/control_comparison_allocation.json). Source80/220, tests194/200 and owner Markdown100/100 gross lines. Source SHA2564dec265e20426a20e437588a4e44828e257247c82897cc8188621f93369a5251; test SHA256d9747f5b8bc4371a0ed60c309632a367bb7931ef8c6677d370ef88c10f713827.

System /usr/bin/python3, Python3.10.6/GCC11.3.0, -u -B, standard library, no conda. Cwd /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028. Receipts retain exact UTC2026-10-24 timestamps; filenames use the supplied Hong Kong2026-10-25 task date. Both full controls ran concurrently with at most2 CPU ranks, each timeout60s and actual memory.limit_in_bytes32212254720. No limit increase or retry.

## Commands and actual checks

```text
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -u -B /data/ycfeng/tmp/glm53_dsv4/p0_metrics_group_controls_20261025_01/run_primary_unit.py
/usr/bin/python3 -u -B /data/ycfeng/tmp/glm53_dsv4/p0_metrics_group_controls_20261025_01/run_fresh_controls.py
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -u -B /data/ycfeng/tmp/glm53_dsv4/p0_metrics_group_controls_20261025_01/compare_exact_controls.py
```

The driver launches each full control through `sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -u -B tests/analysis/v4_trace_audit_group_metrics.py RELATIVE_TRACE FRESH_PREFIX`. Exact absolute child argv, original trace/profile/inventory hashes and all output hashes are retained in [complete_controls.json](primary/complete_controls.json) and each archived receipt. Both controls are new allocated invocations; no old driver was imported or executed.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Owner unit, separate earlier invocation |9 tests, no failure/error/skip; consistent copied evidence |9 tests/42 rows,0/0/0,0.272824s; used1/1 |PASS retained owner evidence |
| Primary unit |9 tests and42 consistent serialized expected/actual rows |9/42,0/0/0,0.253358s,actual30GiB; used1/1 |PASS |
| Frozen source/AST |Original grouping, attribution, pipeline/main and union recurrence unchanged |Eight exact pins;10 named AST/composition checks; no per-event mutable list literal |PASS |
| decodeDP1/rank0 |Every non-clock JSON field and decompressed index exact;61 useful steps |361 metric groups float-hex exact;142567 rows/10462613 decompressed bytes exact;61 useful |PASS |
| prefillDP1/rank0 |Every non-clock JSON field and decompressed index exact;4 useful steps |240 metric groups float-hex exact;570964 rows/43277745 decompressed bytes exact;4 useful |PASS |
| Execution budget |60s/actual30GiB/max2/no retry |decode6.096341s/prefill29.475656s enclosing,each exit0;actual32212254720 bytes |PASS |
| Unchanged matrix |Explicit binding, no fresh capture or simulator change |77 identical cases; unchanged domains verified against prior fixed baseline |PASS binding only |

Audit-only elapsed times are5.687352s/28.040537s; maxRSS731764/3238656KiB. Artifact comparison took0.726559s; its duration is not an audit or serving measurement. All count/sum/union/hierarchy/leaf/per-stream values are exact. Only elapsed_seconds,maxrss_kib,kernel_sample_index.path and compressed sha256 are excluded from cross-run equality; fresh paths/hashes and numeric clock fields are checked independently.

| Control | Kernel count | Exact sum_ms | Exact union_ms |
| --- | --- | --- | --- |
| decodeDP1/rank0 |142567 |1377.599441999903 |1215.560419921875 |
| prefillDP1/rank0 |570964 |8427.020689000585 |7639.143890869141 |

## Independent review and retained evidence

| Axis | Reviewer | Outcome | Evidence |
| --- | --- | --- | --- |
| Spec |Sartre, independent source/receipt inspection |No findings |[review_spec.md](review_spec.md) |
| Standards |Socrates, independent source/receipt inspection |No findings |[review_standards.md](review_standards.md) |

Each axis reviewed the staged two-file diff at fixed point4ae81052 and inspected primary receipts without running tests or traces. Spec0 findings; Standards0 findings; neither axis has an unresolved issue. Primary inspected actual source, command bindings and archived closed evidence, independently executed its unit/controls and streamed the exact index comparison.

[Source acceptance](primary/source_acceptance.json), [source/AST review](primary/source_review.json), [unit receipt](primary/primary_unit.receipt.json), [full AST](primary/primary_unit.ast.json), [control archive binding](primary/control_archive_binding.json), [matrix binding](primary/matrix_binding.json) and [owner delivery](owner/delivery_manifest.json) retain provenance. Fourteen compact control logs/receipts were copied byte-identically; full JSON/index artifacts remain under the fresh scratch root and are referenced by hash. Initial read-only shell Python/path lookup errors consumed no experiment allowance and are recorded separately; original failed P0/tiny/timing evidence is unchanged.

This accepts the CPU correction and its two complete controls only. Both full-control allowances and owner/primary units are now spent. Fresh affected invocations are0/8, with rank0 required as a complete pilot before the other seven. R5 remains24/32. No native serving ownership, dependency, GPU performance, non-perturbation, cost or reference admission follows. The scope GPU attempt remains spent1/1 and is not repeated.
