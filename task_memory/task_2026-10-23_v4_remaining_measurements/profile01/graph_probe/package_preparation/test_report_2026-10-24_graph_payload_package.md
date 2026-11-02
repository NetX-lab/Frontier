# Graph payload package CPU verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded immutable package/source checks, real launcher CPU checks and explicitly substituted command/SDK controls. |

Task ID: `W128-GRAPH-PAYLOAD-PACKAGE`.
Workdir: `/data/ycfeng/Frontier/.worktrees/v4-graph-payload-package`.
Existing Python `/data/ycfeng/envs/oneshot-opd-py312/bin/python`,3.12.13; no conda activation/dependency changes. Common environment `PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode`. No Docker/network/SSH/cloud/platform/auth/GPU/actual StepMind operation.

## Exact build and generated entry commands

Let `P=/data/ycfeng/Frontier/.worktrees/v4-graph-payload-package/task_memory/task_2026-10-23_v4_remaining_measurements/profile01/graph_probe`.

```bash
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/oneshot-opd-py312/bin/python \
  "$P/package_preparation/build_package.py"
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/oneshot-opd-py312/bin/python \
  "$P/h800_graph_probe_01/launch.py" --help
env PYTHONDONTWRITEBYTECODE=1 /data/ycfeng/envs/oneshot-opd-py312/bin/python \
  "$P/h800_graph_probe_01/launch.py"
```

The actual dry-run command was passed unchanged to `bash -n -c <command>` for syntax only, not execution. `generated_entry_cpu.json` retains both exact Python argument vectors/stdout/stderr and budgets; its dry-run stdout includes the full actual generated command. Every package Python source passed stdlib `ast.parse` and `compile(text,path,"exec")` without imports/pyc writes; all package JSON parsed.

Raw logs: `/data/ycfeng/tmp/w128_graph_package_{build,help,dry_run,standins}.log`. Exact selected evidence is in `generated_entry_cpu.json`, `standin_cpu_receipt.json` and `package_inventory.json`.

## Scoped expectations and outcomes

| Check | Passing condition | Actual | Verdict |
| --- | --- | --- | --- |
| Fresh builder | Exact source1aa1e4a5; no overwrite; no seal/submission |8 inputs+manifest, PREPARED_NOT_SEALED, no launch receipts | PASS |
| Real generated launcher | --help/dry-run/bash syntax all return0, no SDK/native imports | Return0 for all;4 Python sources compile;5 JSON parse | PASS CPU entry only |
| Diagnostic/output order | Fake native entry sees sibling diagnostics, absent output, exact run/output/image CLI | Both success and failure controls assert these states and local TMPDIR/cache | PASS substituted native only |
| Native return | Exit0/17 preserved |0/17 exactly | PASS |
| Initial tool failure | Fake nvidia-smi7 prevents native entry |7 exactly, no diagnostic/output entry | PASS |
| Runtime integrity | Tampered scratch input rejects before diagnostic/native entry |1/AssertionError, no diagnostic/output | PASS |
| Existing diagnostic directory | No overwrite/retry on an existing sink directory |1/FileExistsError | PASS CPU negative only |
| Host integrity | Tampered scratch input rejects on real copied launcher dry-run |1/Prepared graph package input changed | PASS |
| Existing package | Builder refuses to overwrite current fresh package |1/FileExistsError, all actual package hashes unchanged | PASS CPU negative only |
| SDK wiring | Explicit fake ResourceConfig/spawn capture resource/env/command and receipts |GPU1/replica1/CPU16/mem64/non-preemptible/H800, exact image/mount/codesign/workspace; scratch launch receipts | PASS stand-in only |
| Original preservation | Old10 stream and23 frozen graph files retain initial hashes |33/33 unchanged | PASS |
| Existing matrix | Exact complete77 comparison hash and unchanged simulator-domain diff |77 identical/no mismatches; no new capture | PASS binding only |

There are7 command/integrity/exclusive controls plus one SDK capture control. The first CPU-harness invocation failed outer Python parsing with nested triple-single quotes (line8, SyntaxError, exit1) before any control executed. Distinct quote delimiters fixed only the temporary invocation; no builder/launcher/package/source was altered. `cpu_standin_initial_failure.json` preserves this failure and correction.

## Explicit substitutions and reproducibility

Scratch root: `/data/ycfeng/tmp/w128_graph_package_cpu_ip4s7q7c`. Each control copied the exact generated launcher, rewrote the entire cloud/output/diagnostics/worker tmp/cache/workspace/script binding to its own fresh local roots, and recomputed only that scratch copy's manifest. The real package remained unchanged.

- `python3` in the scratch PATH is a link to the existing3.12.13 interpreter.
- `nvidia-smi` in that PATH is an explicit script that prints `CPU fake nvidia-smi` and returns the requested test status. No real GPU tool was invoked.
- Scratch `probe_diagnostic.py` is a stdlib-only fake native command asserting diagnostic-before-argparse and absent output, creating its own output and returning0/17. It is **not** frozen1aa1e4a5 and establishes no callback/CUDA behavior.
- `steptron.exp.base_exp.ResourceConfig` and `steptron.utils.stepmind.spawn_tasks` are explicit in-memory stand-ins in the --launch entry control. They capture values only, return job `cpu-stand-in`/status `CPU_STAND_IN_COMPLETED`, and never submit, mount, authenticate or execute the captured worker command. The five fake module entries/environment/cwd are restored afterwards.
- Only scratch copies get launcher_started/result receipts. The real unsealed package has none.

The exact executed control stdin, fake source text, commands, stdout/stderr, resource/spawn capture and environment are in `standin_cpu_receipt.json`. To replay those same CPU controls without adding a wrapper or touching the real package:

```bash
env PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('task_memory/task_2026-10-23_v4_remaining_measurements/profile01/graph_probe/package_preparation/standin_cpu_receipt.json')
exec(compile(json.loads(p.read_text())['executed_stdin'], '<w128-package-cpu-controls>', 'exec'))
PY
```

## Budget, source and immutable inventory

`build_package.py`113 lines+`profile_plan.json`20=133/220 maintenance lines. Even additionally counting generated config29 gives162/220. Generated launcher73/100 lines (template65, delta12 additions+4 deletions). Frozen probe669 lines/34078B, SHA256 `a9d6a24e0aff13ef61088d75b5f693b599f42e8c50c2702342fbb84d3754d1a0`; zero source changes. `package_inventory.json` lists every maintenance/generated source's exact bytes/lines/hash. `generated_launcher.diff` retains the full transformation.

Manifest SHA256 `ec816706edeb0930756ad7077684a2872b93c7728ae8851264e3e2592409c146`;8 inputs plus manifest,9 files total,0 tracked package files. Both host and worker verify the input manifest and probe hash; no manifest signature/seal is claimed. Original33 pins and current package hashes were rechecked after the command controls.

## Verification limits and root gates

The existing probe API report retains library path/version/exports/mapped names, **not binary hashes**. No source helper was added. Actual locked-image/library/SDK behavior, current personal auth/quota, local NFS worker mount, fresh cloud destination, independent frozen-source reviews, seal/launch/lifecycle and native strictU1–U4 outcome remain root-owned and unverified here. No actual package --launch, cloud write, GPU operation or retry occurred. Existing source/stage/subscriber/9-node5-kernel4-event semantics and paused GLM scope are unchanged.
