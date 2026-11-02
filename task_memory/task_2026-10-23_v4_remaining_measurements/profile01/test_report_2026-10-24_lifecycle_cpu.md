# Native shutdown lifecycle CPU controls

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded two deterministic controls against the unchanged worker and retained the original command/receipt. |

## Command and environment

```bash
PYTHONPATH=/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028 PYTHONDONTWRITEBYTECODE=1 \
  /data/ycfeng/envs/frontier-py310/bin/python -B \
  /data/ycfeng/tmp/glm53_dsv4/profile03_lifecycle_control_jmg_ngi0/control.py
```

Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Python3.10.6/GCC11.3.0; no conda activation. Recorded execution2026-10-23T18:42:51Z, exit0. `lifecycle_cpu/command.json` contains the exact argv/environment/source revision; `receipt.json` contains raw fake-clock observations and source/script hashes. The temporary script uses the existing `test_native_pd_worker.exercise` seam.

| Control | Expected | Actual | Result |
| --- | --- | --- | --- |
| Marker later than stop budget | stop10/peer60/marker20: headless evidence wait expires at10 | wait10; missing-marker observation; native/workload return0 | PASS reproduction |
| Delayed enumeration | four direct paths visible at0; glob withheld until20 | quorum wait ends20; all sampled direct-path checks true | PASS reproduction |
| Input preservation | worker source unchanged | SHA256 `9ec701ec12f7a5da0f442f994a353f98c5ae1d2880b9feca38106ea536fd5cab` before/after | PASS |

The first case uses the existing naturally completed headless fixture. The second injects a filesystem-listing delay. Neither proves the real shared-filesystem cause, GPU stop ordering or cross-host clock alignment. No production source/configuration or runtime behavior was changed. Proposed source corrections and their required GREEN checks are in [profile03_design.md](profile03_design.md).
