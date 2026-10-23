# Profile02 CPU evidence audit commands

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Added the durable raw-audit archive and root-reproduction receipts; original command evidence remains unchanged. |
| 2026-10-23 | W128-PROFILE02-DATA: recorded local-only audit commands and source/schema boundaries. |

Task: `W128-PROFILE02-DATA`. No network, cloud, Docker, GPU, StepMind, Git write, source edit, admission edit or delegation was performed.

Native-reference, serving-total and operator-cost admissions remain **UNVERIFIED**; this file records CPU audit execution, not primary acceptance.

`data_audit.json` and the32 compressed kernel indices are durably stored under the cloud root in `archive_receipt.json` (`results/` and `audit/` respectively). Local temporary paths below describe the executed commands. Root's additional exact32-trace reproduction is in `root_verification.json`; primary acceptance is in `primary_acceptance.json`.

## Inputs and execution environment

- Repository: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.
- Download: `/data/ycfeng/tmp/glm53_dsv4/profile02_results_readonly_7ba56n6z/artifacts`.
- Download inventory: `/data/ycfeng/tmp/glm53_dsv4/profile02_results_readonly_7ba56n6z/remote_inventory.json`.
- Terminal receipt: `/data/ycfeng/tmp/glm53_dsv4/profile02_terminal_ewudq62n/receipt.json`.
- Frozen package identity: `../profile02_source_binding.json`, runtime `05952396b0b8e5a794c53e10f9a21213beb5193d`, native `2cf0a6915ce544dc493a0990f2ea38d81601128a`.
- Local Python: `/usr/bin/python3`, Python 3.10.6, GCC 11.3.0. Standard library only; no conda environment and no torch/vLLM import.
- Temporary adapter and logs: `/data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/`.

## Exact per-trace commands

`data_audit.json → execution.commands.commands[]` and temporary `commands.json` retain all 32 actual argument arrays, exit statuses, wall times, logs, output paths, source hashes and expected trace hashes. A two-thread CPU runner executes these independently; each process has its own 60-second timeout and 30-GiB systemd scope. It never opens more than two traces concurrently.

```bash
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s \
  env PYTHONDONTWRITEBYTECODE=1 python3 \
  /data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/audit.py \
  <inventory-relative-trace-path> \
  /data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/<role>-dp<lane>-rank<local-rank>.json
```

The adapter imports only `trace_header` and `load_trace` from `tests/comparison/calibration/forward_device_timeline.py`. Its incompatible `Forward` CLI, peer-nearest-timestamp matching and world-directory assumptions are not used. Adapter and reader SHA256 values are retained in the audit. No reusable parser was added to the repository.

## Additional actual CPU checks

All identity, client and snapshot inspections ran as `sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s [env PYTHONDONTWRITEBYTECODE=1] python3 -` with inline standard-library code recorded in the execution transcript. Their complete generated evidence is retained as temporary `identity.json` and `snapshot_census.json`, included in `data_audit.json`.

1. Stream SHA256 over each of 226 downloaded files and compare bytes/hash with `remote_inventory.json`.
2. Rehash all 45 package paths from `profile02_source_binding.json`; compare all 40 run-manifest input entries with their actual local package files.
3. Compare all 87 observed native source hashes with the pinned native checkout; the approved `nvidia/model.py` overlay is compared with the frozen package's `overlay_model.py` rather than the unmodified native file.
4. Read all 96 collector snapshots (`before_profile`, `profile_stopped`, `pre_shutdown`) and independently retain window presence, counts, drops, errors, suppression and original status.
5. Check all 432 client rows, unique IDs, HTTP/status, per-repeat phase counts, output-token count, validation errors and stream completion; retain the four profiling RPC attempts and drain fields.
6. Compare exact per-step TP signatures within each `(role,lane)`. No cross-DP timestamp or ordinal inference is used.

Assembly and independent raw-index verification:

```bash
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s \
  env PYTHONDONTWRITEBYTECODE=1 python3 \
  /data/ycfeng/tmp/glm53_dsv4/W128-PROFILE02-DATA/assemble.py
```

The final inline verifier used the same scoped `python3 -` invocation to read all 32 compressed kernel-index files, compare each SHA256, require unique in-range original-event indices, compare all status histograms with the per-trace partition, and require 7,600,008 retained kernel rows. It passed in 33.370841 seconds; exact stdin is retained in the execution transcript and the complete result in temporary `raw_index_verification.json`.

## Schema handling and preservation

- Identity uses `(role,lane,local rank)`: collector/filename rank is 0–7 in every lane; `distributedInfo.rank` and filename EP rank are 0–15 per role. All values and the node host are retained and cross-checked, not substituted for one another.
- Annotations join by the complete `frontier_native_<category>|<id>|<scope>` name. Collector IDs remain distinct even when category/scope repeat.
- Every launch is associated with all enclosing native ranges on its own host thread. Hierarchy metrics are non-additive memberships. Innermost scopes are used only when unique; graph replay is kept as captured-segment UNKNOWN, never as captured per-operation ownership.
- Kernel `count`, duration `sum_ms` and interval `union_ms` remain separate. Neither cross-stream nor cross-rank sums establish serving latency.
- All original files remain hash-identical. Every kernel has a retained raw-index row referencing its original `traceEvents` entry, including unowned, uncorrelated, ambiguous and graph kernels. No median selection, sample removal, scaling or gap classification occurs.
- Collector event pair checks use preceding object/generation/source records only. CUDA event-to-trace-stream mapping is UNKNOWN when `trace_stream_id` is null; no handle or graph-ID mapping is invented.

## Retained inspection failures

- Unprivileged `systemd-run --user` failed; system scope without sudo required interactive authentication. The approved non-interactive sudo system scope worked. No memory-heavy inspection ran without the required scope/timeout.
- A preliminary inspector assumed `launcher_receipt.json` existed; its absence is retained, not repaired.
- A preliminary inspector assumed every annotation had `launch_arguments`; actual rows have three observed key sets, so argument samples are read only when the field is present.
- A preliminary snapshot inspector assumed `window` existed before profiling; all 32 `before_profile` snapshots omit it. The corrected census records field presence separately and makes no startup-window assumption.
- A preliminary TP inspector assumed synchronized descriptor/mode existed on every step; 109 zero-token lane steps omit both. The corrected comparison preserves absent fields, excludes zero-token steps from useful samples, and proves complete required-field agreement only on useful steps.
- One display-only aggregation attempted to JSON-encode tuple Counter keys; assembly uses explicit string keys. No trace parsing or measurements were rerun for that formatting failure.
- These are CPU inspection failures, not workload failures or evidence modifications. Platform and native-reference qualification remain separate.
