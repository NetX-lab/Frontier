# Fresh group-metrics pilot failure

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-25 Hong Kong | Archived the one fresh group pilot and retained full-output failure without starting seven conditional ranks. |

**FAILED_P0_PILOT_NO_FINAL_JSON**. Source8b4d6993 is accepted CPU source; the fresh actual P0 pilot still exceeds60s. R5 remains24/32. Source acceptance is not revoked or upgraded; seven conditional ranks remainNOT_RUN/BLOCKED and the fresh affected allowance is1/8 used.

Environment: /usr/bin/python3 3.10.6/GCC11.3.0,-u -B,no conda. Cwd /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028. UTC receipt2026-10-24; filename uses Hong Kong2026-10-25. Source SHA4dec265e20426a20e437588a4e44828e257247c82897cc8188621f93369a5251; frozen allocation [affected_allocation.json](primary/affected_allocation.json) SHAf5830ece5508b339bfe8de26695c28c9f8fe21d674e66a85683964e9e7b04511.

## Exact commands

```text
/usr/bin/python3 -u -B /data/ycfeng/tmp/glm53_dsv4/p0_metrics_group_affected_20261025_01/run_affected.py pilot
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -u -B /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/analysis/v4_trace_audit_group_metrics.py torch_traces/prefill/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792842834745705872.pt.trace.json.gz /data/ycfeng/tmp/glm53_dsv4/p0_metrics_group_affected_20261025_01/prefill_dp0_rank0
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 -u -B /data/ycfeng/tmp/glm53_dsv4/p0_metrics_group_affected_20261025_01/inspect_failed_pilot.py
```

The first command launches the second once. Input hashing precedes the60s child; output hashes are computed after child exit. The third only reads/copies closed logs and streams the already produced index; it does not load or audit the raw trace. Its recorded allocation permits one artifact inspection and zero raw audits.

| Check | Expected | Actual | Result |
| --- | --- | --- | --- |
| Complete P0 final JSON/index and >=3useful steps | All present, identities/fields valid | Final JSON absent; useful steps UNKNOWN | FAIL |
| Child/driver exit |0/0 after complete validation |124/1;60.324082s enclosing | FAIL |
| Native CPU budget |60s/actual30GiB/max2/no retry |60s/32212254720bytes,chosen concurrency1,one attempt | PASS budget only |
| Original source/input binding | Frozen accepted source and original triplet | Driver rehashed all three inputs; frozen dependencies/inventory unchanged | PASS binding only |
| Closed diagnostic index | Clean EOF,complete kernel-pass row count |1215919rows;decompressed SHA678fb91782cc85a5068cf6bccbd51749562449abdfc90c628c0a084dddf68667 | PASS index only |
| New metrics stage | Recorded completion | Start55.992s;end absent | CENSORED |
| Metrics gen2 pauses | Observe rather than infer |0callbacks after batch start | OBSERVED |
| Remaining seven | Start only after complete pilot |No receipts/launches | BLOCKED |

Observed stages: loading0.493→30.565s(30.072s); event pass done37.203s;correlation done40.177s; kernel pass40.178→55.532s(15.354s);batch start55.992s for1215919events/240groups/5331778memberships. Sort/order/group/union durations are UNKNOWN because only batch boundaries were instrumented. The remaining four-second interval is not a complete measured batch time. Heartbeat's last captured stack is still kernel-side and does not locate the open metrics segment.

[Failure](primary/pilot_failure.json) retains exact commands, seven closed artifact hashes, stage rows, actual cgroup, full diagnostic-index count/hash and no-final-output status. [Inspection allocation](primary/pilot_failure_inspection_allocation.json) and [source](primary/pilot_failure_inspection_source.txt) bind the70-line read-only check; elapsed0.386061s. Full index remains in scratch; small logs/receipt are copied byte-identically into primary/. No earlier unit/control/diagnostic/GPU run was repeated.

This attempt records no metrics gen2 callbacks, while the earlier construction diagnosis recorded two pauses. The difference does not establish sufficient overall speed or non-perturbation. A separate segment-only diagnostic is allocated to distinguish global-order, per-group sort/union and pre-metrics/post-metrics budget hypotheses before another correction. No source/loader/GC/raw/predicate/native/package limit change and no cost/reference/native-ownership admission.
