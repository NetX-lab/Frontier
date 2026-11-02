# CPU data-audit commands

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded W128-NATIVE-STREAM-PROBE-DATA final commands and independent rerun. |

Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`.
Host Python: `/usr/bin/python3`, 3.10.6, GCC 11.3.0; `CONDA_PREFIX` unset. Standard library only; no Torch/CUDA imports.
Script: `/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_data_audit_20261024.py`.
Script SHA256: `0d399e59f0f6da42a9e300a2686789c40e97e2b50933b6372d81bd9e60a5809d`.

Final execution, exit0, 162/162 checks PASS:
```bash
python3 -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_data_audit_20261024.py --worktree /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028 --download /data/ycfeng/tmp/glm53_dsv4/native_stream_probe01_results_sxh2vn9h --output /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/stream_probe/results/data_audit.json
```
Fresh-process repeat, exit0, 162/162 checks PASS:
```bash
python3 -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_data_audit_20261024.py --worktree /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028 --download /data/ycfeng/tmp/glm53_dsv4/native_stream_probe01_results_sxh2vn9h --output /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_data_audit_rerun_20261024.json
```
Exact equality/inspection command, exit0, `BYTE_IDENTICAL`:
```bash
python3 -B - <<'PY'
from pathlib import Path
import hashlib,json
p=Path('task_memory/task_2026-10-23_v4_remaining_measurements/profile01/stream_probe/results/data_audit.json')
r=Path('/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_data_audit_rerun_20261024.json')
assert p.read_bytes()==r.read_bytes()
d=json.loads(p.read_text())
print('rerun: BYTE_IDENTICAL')
print('script',d['script'])
print('audit_json',hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_size)
print('trace_metadata_loss',d['loss_and_limits']['cycle2_missing_metadata_keys'])
print('trace_details',json.dumps({c:{k:v for k,v in row.items() if k in ('sha256','decompressed_sha256','decompressed_bytes','event_count','categories')} for c,row in d['traces'].items()}))
print('capture_runtime_args',{label:[e['args'] for e in row['runtime']] for label,row in d['cycles']['1']['capture_ops'].items()})
PY
```
The script also runs read-only `git --no-optional-locks show <source_revision>:<path>` for the four exact `source_lock.json` source entries. No Git writes, source edits, GPU, Docker, platform, network or cloud operations occurred.
Original exploratory errors: broad glob `**/vllm/profiler/wrapper.py` under `/data/ycfeng` timed out after30seconds; exact native path was then read successfully. Initial `results/` read returned `File not found`; only the three allocated result files were created. Final audit runs had no errors. Root's expected CPU `RuntimeError: No CUDA GPUs are available` is retained verbatim in `data_audit.json.original_errors`, not rerun.
