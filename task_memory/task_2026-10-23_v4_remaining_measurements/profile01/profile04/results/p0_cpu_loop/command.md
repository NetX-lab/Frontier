# W128-P0-AUDIT-CPU-LOOP commands

Scratch `/data/ycfeng/tmp/glm53_dsv4/p0_audit_cpu_20261024`, Python `/usr/bin/python3` 3.10.6, stdlib only. Frozen inputs are hash-asserted inside every run (auditor `7ed0ea45`, reader `fe4c3f82`, `stage_audit.py` `f7f1253c`; full hashes in `receipt.json`). `ROOT` is the retained profile04 run root; trace paths are relative to it. Scratch source 128 new lines (limit 180).

```bash
cd /data/ycfeng/tmp/glm53_dsv4/p0_audit_cpu_20261024
# 1. GREEN control on a completed rank, then output-equivalence check
timeout 120s /usr/bin/env PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 stage_audit.py run \
  torch_traces/decode/dp1/dp1_pp0_tp0_dcp0_ep8_rank0.1792842735368033988.pt.trace.json.gz \
  control_decode_dp1_rank0.json control_decode_dp1.obs.jsonl control_decode_dp1.hb.log   # exit 0, 6.69 s
/usr/bin/python3 stage_audit.py compare control_decode_dp1_rank0.json \
  /data/ycfeng/tmp/glm53_dsv4/profile04_trace_of9lmcmi/decode_dp1_rank0.json
# {"payload_equal": true, "kernel_rows_equal": true, "rows": [142567, 142567], "useful_steps": [61, 61]}
# 2. The one P0 full-input diagnostic, baseline ceiling unchanged (60 s / 30 G)
time sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/env PYTHONDONTWRITEBYTECODE=1 \
  /usr/bin/python3 /data/ycfeng/tmp/glm53_dsv4/p0_audit_cpu_20261024/stage_audit.py run \
  torch_traces/prefill/dp0/dp0_pp0_tp0_dcp0_ep0_rank0.1792842834745705872.pt.trace.json.gz \
  /data/ycfeng/tmp/glm53_dsv4/p0_audit_cpu_20261024/p0_prefill_dp0_rank0.json \
  /data/ycfeng/tmp/glm53_dsv4/p0_audit_cpu_20261024/p0_prefill_dp0_rank0.obs.jsonl \
  /data/ycfeng/tmp/glm53_dsv4/p0_audit_cpu_20261024/p0_prefill_dp0_rank0.hb.log
# EXIT=124, 1:00.46, 54.77 s user + 5.67 s sys, scope run-rd01d862de93a48cb89ccd21d60e12667.scope
# 3. Read-only controls
gzip -l <p0 rank0 trace>.gz                       # 142288791 -> 2221133654 bytes
# gzip-only decode, completed D0 traces: 209168680 B in 0.53 s (392 MB/s); 375706178 B in 0.88 s (425 MB/s)
```

Baseline kept: exit 124 at 60.40 s, 71,638 rows from index 1,386,499. Completed controls read from retained `aggregate.json`/rank outputs without re-parse: decode dp0 12.7 s / 256,303 kernels / 1.23 GB RSS; decode dp1 6.7 s / 142,567 / 0.70 GB; prefill dp1 33.4 s / 570,964 / 3.09 GB. No limit was raised, no prlimit substitute, no second full-input attempt, no GPU retry. Outputs and hashes: `receipt.json`.
