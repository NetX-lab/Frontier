# D21-NATIVE-REPLAY verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded actual-source red/green caller regression and125 CPU checks. |
| 2026-10-21 | Parent executes generation03 and verifies actual runtime FULL rows on both decode lanes; full G3/G4 remain open. |

Environment: local Linux, Python3.12.13 at `/data/ycfeng/envs/asyncopd-py312/bin/python`; existing simulator-export fixture uses `/data/ycfeng/envs/frontier-py310/bin/python` through its documented override. No conda activation, GPU, auth, network, SSH or further delegation. CPU tests are not native replay evidence.

The paragraph above describes the original source/CPU slice. The following parent-only native continuation is additional observed evidence; earlier receipts remain unchanged.

## Parent generation03 native result

- Submitted with `setsid bash /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification03/stage/submit.sh`, output redirected to the enclosing `launcher.log`. Source/config is the sealed6838bc1 generation,32H800/codesign/non-preemptible. Personal source/mount identity is in `parent_identity02.json` beside the stage.
- Terminal command, after sourcing `/data/ycfeng/tmp/step5_native_pd_r1/stepmind_env.sh`: `"$STEPMIND_PYTHON" /data/ycfeng/tmp/step5_native_pd_r1/status.py exp-1021-205523-285491 > /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification03/parent_terminal03.json`. Observed **Succeeded**, all four tasks succeeded, finish2026-10-21T13:09:48Z, creator i-fengyicheng, non-preemptible.
- Parent Python3.10.6 read all24 `stage/evidence/replay/client_requests.jsonl` records:24 unique IDs;16 warmup/8 concurrent;6144 requested/accepted prompt tokens;1024 requested/generated/token-ID deltas;completed streams, length finishes and no validation errors. Both idle boundaries are verified for both roles/lanes; no observation errors. **PASS for exact requests/idle**.
- Parent scans all four complete `stage/evidence/nodes/*/server.log` files. The nearest preceding `Engine NNN` associates each runtime table with its lane; match only five-column numeric rows, excluding capture/config text. **99 runtime rows** retained in `generation03_runtime_rows.csv`, including55 FULL rows: lane0 has30 rows/2224 forwards, lane1 has25 rows/2152 forwards. P lanes each log12 NONE forwards at6128 tokens; D additionally logs20/18 PIECEWISE observations. **PASS for actual FULL replay on both decode lanes**.
- Source evidence: locked `compilation/cuda_graph.py:120–124` logs then resets interval stats, so counts can be summed across tables. Locked `v1/worker/gpu/cudagraph_utils.py:550–563` executes `graph.replay()` for FULL. The logger chain and runtime identity are pinned by the sealed stage. Counts are observed intervals, not a claim that every rank/request/forward is recorded.
- `generation03_receipt.json` binds12 input/source files by SHA-256 and records the observations and limits. **Basic remains PARTIAL; formal-ready NO** pending G3 independent content/lifetime/reference, complete G4 useful/dummy-forward and rank accounting, admitted profiles and three clean128-request repetitions. No repeat basic run is needed merely for the corrected logger.

Recheck the retained inputs and source-line addresses from the repository root:

```bash
python3 - <<'PY'
import csv, hashlib, json
from pathlib import Path
root = Path('task_memory/task_2026-10-21_step5_priority_runs/native_replay')
receipt = json.loads((root / 'generation03_receipt.json').read_text())
assert all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest() == r['sha256'] for r in receipt['inputs'])
with (root / 'generation03_runtime_rows.csv').open() as handle:
    rows = list(csv.DictReader(handle))
for row in rows:
    line = Path(row['source']).read_bytes().decode(errors='replace').split('\n')[int(row['line']) - 1]
    assert '| ' + row['unpadded'] in line and row['mode'] in line
assert len(rows) == 99 and receipt['request_count'] == 24
print('PASS: 12 input hashes and 99 LF-delimited runtime row addresses')
PY
```

Parent executed these checks successfully. Raw LF-delimited line numbers preserve carriage-return progress lines without counting them as new log lines.

Working directory for commands: `/data/ycfeng/Frontier/.worktrees/step5-priority-runs`.

## Red signal and correction

Read-only inline `python3 -B` over P0 `stage/evidence/nodes/*/server.log` asserted at least one `Engine\\s+\\d{3}` row. Actual: four complete files, zero rows, assertion failure/exit1. This retained-run symptom intentionally remains unchanged after the fix: historical logs are immutable.

Deterministic CPU loop, before the caller correction:

```bash
env PYTHONDONTWRITEBYTECODE=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_native_source_contract.py -k canonical_caller -q --tb=short -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d21_native_replay_red02
```

Expected/observed: **2 failed in0.33s**, DP2/DP4 each `Multiple API clients disable native EngineNNN/FULL graph logging`, zero LoggingStatLogger adapters while Prometheus survives. It exercises the real builder/node_plan/server_command, actual locked ServeSubcommand.cmd, and actual locked StatLoggerManager constructor with CPU-only process/logger factories. It checks the suppression condition at the real caller seam, not just existence of source strings. Existing actual-source tests separately exercise MRv2 replay dispatch and graph table emission. First test-fixture attempt failed before that seam with `KeyError: skip_tokenizer_init`; using the existing builder's normalized engine fixed the fixture, not production behavior.

Minimal correction: canonical config freezes `api_server_count=1`; native command emits it only for API heads. Headless nodes omit it because native rejects positive counts with headless. All TP/DP/EP values, request populations, NIXL behavior, original logging level, auxiliary error isolation and metric gates remain unchanged. Frozen runtime bindings now include frontend CLI, API entry and periodic logger caller.

## Green and affected regressions

```bash
env PYTHONDONTWRITEBYTECODE=1 STEP5_NATIVE_SOURCE_ROOT=/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm FRONTIER_SIMULATOR_PYTHON=/data/ycfeng/envs/frontier-py310/bin/python /data/ycfeng/envs/asyncopd-py312/bin/python -m pytest tests/unit/test_step5_native_source_contract.py tests/unit/test_step5_priority_runs.py tests/unit/test_step5_canonical_native.py tests/unit/test_step5_native_pd.py tests/unit/test_native_pd_worker.py -q --tb=short -p no:cacheprovider --basetemp=/data/ycfeng/tmp/d21_native_replay_green02
git diff --check
```

Expected: single frontend with per-engine logger covering everyDP lane, no API flag on headless peers, all prior exact-request/idle/worker/soft-observation checks unchanged. Actual: **125 passed in86.13s — PASS**; diff whitespace checkPASS. No tests skipped. This exceeds50 concrete affected scenarios without touching simulator files.

Preserved environment error: first broad invocation used the same asyncopd interpreter without `FRONTIER_SIMULATOR_PYTHON`; actual17 passed/32 setup errors (`ModuleNotFoundError: plotly`) in3.15s. The unchanged fixture's documented existing interpreter override resolved it; no install or production fix. No GPU replay is claimed from the green CPU result.

## Fresh offline stage

Source `6838bc10135faf5bea1bc48321837caa938f85a1`. Exact build command:

```bash
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B tests/integration/step5_pd/canonical_stage.py build --run-definition tests/integration/step5_pd/canonical_run_h200.json --kind qualification --case h800_32 --model-dir /data/ycfeng/tmp/step5_native_pd_h800_r1w/reference_inputs --image-source-hashes /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/native_pd_20261020/image_source/sha256_in_image.txt --probe-dir /data/ycfeng/tmp/step5_native_pd_h800_r1w/stage02 --runtime-source-root /data/ycfeng/tmp/step5_profiling_20261020/image_src --launcher-scope /data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-15_step5_preview_h800/native_role_tp16/launch_capacity_20261018_03/launcher_scope.py --workspace /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification03 --stage-dir /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification03/stage --output-dir /mnt/codesign-exp/ycfeng/frontier/step5_native_pd/d21_h800_32_qualification03 --job-tag d21h80032qual03 > task_memory/task_2026-10-21_step5_priority_runs/native_replay/fresh_stage_build.json
env PYTHONDONTWRITEBYTECODE=1 python3 -B /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification03/stage/tests/integration/step5_pd/canonical_stage.py verify --stage-dir /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification03/stage > task_memory/task_2026-10-21_step5_priority_runs/native_replay/fresh_stage_verify.json
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/asyncopd-py312/bin/python -B /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification03/stage/tests/integration/step5_pd/submit_native_pd.py --config /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification03/stage/config.json --launcher-scope /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification03/stage/launcher_scope.py --dry-run > task_memory/task_2026-10-21_step5_priority_runs/native_replay/fresh_stage_dry_run.json
bash -n /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification03/stage/submit.sh
```

Expected/actual:37 tool pins and50 native source pins; both API heads have count1, both headless peers omit the count; four8-GPU H800 replicas/codesign/non-preemptible; basic qualification bypasses formal-only accuracy checks while native/runtime/reference stayUNVERIFIED. Build, standalone verify, exact dry-run and shell syntax each exit0 — **PASS**. Config SHA256: `7708e7c55d2361872f5054305cb427d77cd6bd22affa457b99370f14de595579`. No submit.sh execution or launcher/SDK construction occurred.

## Retained business evidence checks

Read-only inline Python3.10.6 checked old64 JSON/JSONL rows, prompt hashes, exact token-ID sums and usage, both later idle populations,64 unique H200 UUIDs,32 loaded EP ranks per role, MRv2 and capacity log lines, NIXL/prompt counter deltas, and lifecycle records. **24 exact requests/all lanes/postwarm+postrun idle PASS**; original summary remainsREFERENCE_UNSTABLE, actual replay/G3/G4UNVERIFIED. Concurrent8:64 reads/8,952,774,656 bytes; P49,024 local, D49,024 external+128 local. Full retained server logs have no EngineNNN/runtime FULL rows; both native heads explicitly suppress the logger.

Current exported bytes match all47 old64 native source pins; original tools27/27 and metadata3/3PASS. All6 existing D21 qualification01/02 tool manifests also match every stored pin (28/36/37 files as recorded in `evidence_receipt.json`), so sealed inputs were not rewritten. Missing startup gauges were not substituted with zeros. Actual shutdown EngineDeadError records after requested stop are retained and scoped in `report.md`; they were not relabelled clean success.

Verification limits: no fresh native run, no content/known-state comparison, no finite-state/useful/dummy-forward census, no clean128-request repetitions, admitted-profile evidence or four-metric comparison. Do not infer replay from source/CPU tables or formal readiness from platform/request success.

Final retained-file integrity recheck (executed,90/90 hashesPASS;8 package checks retained):

```bash
python3 -B - <<'PY'
import json,pathlib,hashlib
p=pathlib.Path('task_memory/task_2026-10-21_step5_priority_runs/native_replay/evidence_receipt.json')
r=json.loads(p.read_text())
assert all(hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()==digest for path,digest in r['sha256'].items())
assert not r['ready_for_formal'] and not r['fresh_stage']['submitted']
print('PASS:',len(r['sha256']),'evidence/source hashes;',len(r['package_pin_checks']),'package pin checks; formal-ready NO; fresh stage not submitted.')
PY
```
