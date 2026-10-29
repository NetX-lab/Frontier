# Native stream probe CPU and execution verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Root accepted targeted stream/restart evidence after a byte-identical complete audit and independent review; graph ownership remains UNKNOWN. |
| 2026-10-24 | Added root final-source CPU/image/SDK checks and verified single-H800 execution, with data acceptance pending. |
| 2026-10-24 | Appended original-byte RED and repaired-source GREEN evidence for both P2 findings; original CPU errors retained. |
| 2026-10-24 | Recorded final source checks, exact CPU commands and unmodified local compatibility failures. |

## Environment and observed result

Working directory: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Python `/data/ycfeng/envs/asyncopd-py312/bin/python`,3.12.13; local Torch2.9.1+cu128/CUDA build12.8, vLLM0.16.0, Transformers4.57.6. `CUDA_VISIBLE_DEVICES=` and `PYTHONDONTWRITEBYTECODE=1` on imports/help. No conda activation, installation, CUDA initialization, Docker, network, platform operation or Git write.

| Check | Expected | Actual | Status |
| --- | --- | --- | --- |
| Source AST and budget | Parse; at most250 gross lines; direct collector calls and no excluded API |249lines; direct `observer.stream/trace_stream` and native wrapper start present; excluded APIs absent | PASS |
| Native config AST witness | Probe constructor keywords are fields of the actual native `ProfilerConfig` | All keywords match the pinned native source | PASS, source-only |
| Minimal CLI | `--help` exits0 with only output-dir plus standard help | Exactly those options, no output directory created | PASS |
| Final fresh imports | Probe/collector/parser/native types/CUDA binding import, without CUDA initialization | All imported; `torch.cuda.is_initialized() == False`; GPU UUID descriptor exists | PASS, local versions only |
| Direct pinned-checkout import in local environment | Actual native sources import if local prerequisites match | Exit1; Transformers4.57.6 refused, native requires v5; missing generated `vllm._version` warning precedes failure | FAIL, local environment limitation |
| Local installed native-config construction | Actual configured keywords supported if native version matches | Exit1; vLLM0.16.0 rejects warmup/wait keyword arguments | FAIL, local version limitation |
| Actual locked-image/native wrapper construction and CUDA cycles | Root must verify pinned2286e853 image/native source and GPU outputs | Not run by source owner | NOT RUN |

Source SHA256 `237a82f1176a6acefff82b0e329ee09679b602383f4045b4c4751f1df85759d7`. Read-only numstat: `249\t0\t/dev/null => tests/comparison/calibration/native_profile_probe.py`. Root-owned collector SHA256 remains `17d6dea11d35a2dc9849a11986fe5df46b45adb84d2d2bdeb958642a851b26ef`.

## Exact final commands

```bash
/data/ycfeng/envs/asyncopd-py312/bin/python -B - <<'PY'
import ast, hashlib, subprocess
from pathlib import Path
probe = Path('tests/comparison/calibration/native_profile_probe.py')
source = probe.read_bytes()
tree = ast.parse(source)
calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
assert len(source.splitlines()) <= 250
assert 'observer.stream' in calls and 'observer.trace_stream' in calls and 'wrapper.start' in calls
assert not any(any(name in call for name in ('get_graph_data', 'raw_cuda_graph', 'cuptiSubscribe', 'cudaGraphGetNodes')) for call in calls)
native = Path('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69')
config_tree = ast.parse((native / 'vllm/config/profiler.py').read_bytes())
config_class = next(node for node in config_tree.body if isinstance(node, ast.ClassDef) and node.name == 'ProfilerConfig')
fields = {node.target.id for node in config_class.body if isinstance(node, ast.AnnAssign)}
config_call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'ProfilerConfig')
assert {keyword.arg for keyword in config_call.keywords} <= fields
print({'ast': 'PASS', 'gross_lines': len(source.splitlines()), 'probe_sha256': hashlib.sha256(source).hexdigest(), 'native_config_keyword_witness': 'PASS', 'collector_sha256': hashlib.sha256(Path('tests/comparison/calibration/native_serve_profile.py').read_bytes()).hexdigest(), 'numstat': subprocess.run(['git', 'diff', '--no-index', '--numstat', '/dev/null', str(probe)], capture_output=True, text=True).stdout.strip()})
PY
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /data/ycfeng/envs/asyncopd-py312/bin/python -B -c "import sys, inspect, importlib.metadata; sys.path.insert(0,'tests/comparison/calibration'); import native_profile_probe, native_serve_profile, forward_device_timeline, torch; from vllm.config import ProfilerConfig; from vllm.profiler.wrapper import TorchProfilerWrapper; from cuda.bindings import runtime; assert not torch.cuda.is_initialized(); print({'fresh_imports':'PASS','cuda_initialized':torch.cuda.is_initialized(),'gpu_uuid_descriptor':hasattr(torch._C._CudaDeviceProperties,'uuid'),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'vllm':importlib.metadata.version('vllm'),'transformers':importlib.metadata.version('transformers'),'wrapper_source':inspect.getfile(TorchProfilerWrapper),'runtime_binding_source':runtime.__file__})"
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /data/ycfeng/envs/asyncopd-py312/bin/python -B tests/comparison/calibration/native_profile_probe.py --help
```

The native-config witness parses actual source only; it does not execute substituted config/profiler classes. No persistent mocked test is added.

## Exact failing feasibility commands and preserved errors

```bash
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69 /data/ycfeng/envs/asyncopd-py312/bin/python -B -c "import sys, inspect, importlib; sys.path.insert(0, 'tests/comparison/calibration'); import native_profile_probe, native_serve_profile, forward_device_timeline, torch; from vllm.config import ProfilerConfig; from vllm.profiler.wrapper import TorchProfilerWrapper; config=ProfilerConfig(profiler='torch',torch_profiler_dir='/data/ycfeng/tmp/native_stream_probe_cpu_not_created',torch_profiler_with_stack=False,torch_profiler_dump_cuda_time_total=False,torch_profiler_use_gzip=True,delay_iterations=0,max_iterations=0,warmup_iterations=0,wait_iterations=0); inspect.signature(TorchProfilerWrapper).bind(config,'native_stream_probe',0,['CPU','CUDA']); print({'import_and_signature':'PASS','wrapper_origin':inspect.getfile(TorchProfilerWrapper),'config_origin':inspect.getfile(ProfilerConfig),'cuda_initialized':torch.cuda.is_initialized(),'gpu_uuid_descriptor':hasattr(torch._C._CudaDeviceProperties,'uuid'),'torch':torch.__version__,'cuda_build':torch.version.cuda})"
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /data/ycfeng/envs/asyncopd-py312/bin/python -B -c "import sys, inspect, importlib, importlib.metadata; sys.path.insert(0, 'tests/comparison/calibration'); import native_profile_probe, native_serve_profile, forward_device_timeline, torch; from vllm.config import ProfilerConfig; from vllm.profiler.wrapper import TorchProfilerWrapper; config=ProfilerConfig(profiler='torch',torch_profiler_dir='/data/ycfeng/tmp/native_stream_probe_cpu_not_created',torch_profiler_with_stack=False,torch_profiler_dump_cuda_time_total=False,torch_profiler_use_gzip=True,delay_iterations=0,max_iterations=0,warmup_iterations=0,wait_iterations=0); inspect.signature(TorchProfilerWrapper).bind(config,'native_stream_probe',0,['CPU','CUDA']); from cuda.bindings import runtime; print({'import_and_signature':'PASS','wrapper_origin':inspect.getfile(TorchProfilerWrapper),'config_origin':inspect.getfile(ProfilerConfig),'cuda_bindings_origin':runtime.__file__,'cuda_initialized':torch.cuda.is_initialized(),'gpu_uuid_descriptor':hasattr(torch._C._CudaDeviceProperties,'uuid'),'torch':torch.__version__,'cuda_build':torch.version.cuda,'vllm':importlib.metadata.version('vllm'),'transformers':importlib.metadata.version('transformers'),'python':sys.version})"
```

First command, exit1:

```text
/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/__init__.py:7: RuntimeWarning: Failed to read commit hash:
No module named 'vllm._version'
  from .version import __version__, __version_tuple__  # isort:skip
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/config/__init__.py", line 23, in <module>
    from vllm.config.model import (
  File "/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/config/model.py", line 31, in <module>
    from vllm.transformers_utils.config import (
  File "/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69/vllm/transformers_utils/config.py", line 56, in <module>
    raise ImportError(
ImportError: Support for Transformers v4 is deprecated and was removed in vLLM v0.24.0. Please upgrade to Transformers v5: pip install --upgrade transformers
```

Second command, exit1:

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/data/ycfeng/envs/asyncopd-py312/lib/python3.12/site-packages/pydantic/_internal/_dataclasses.py", line 121, in __init__
    s.__pydantic_validator__.validate_python(ArgsKwargs(args, kwargs), self_instance=s)
pydantic_core._pydantic_core.ValidationError: 2 validation errors for ProfilerConfig
warmup_iterations
  Unexpected keyword argument [type=unexpected_keyword_argument, input_value=0, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/unexpected_keyword_argument
wait_iterations
  Unexpected keyword argument [type=unexpected_keyword_argument, input_value=0, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/unexpected_keyword_argument
```

No directory was created by config validation or help. These two failures describe the local environment, not a repaired constructor. No compatibility fallback or dependency change is authorized or attempted. Initial environment discovery also found bare `python` unavailable (exit127, `zsh:1: command not found: python`); substantive checks use the existing project interpreter above.

## Runtime acceptance limits and root integration

Root packages `native_profile_probe.py`, unchanged `native_serve_profile.py`, unchanged `native_full_structure.py`, and unchanged `forward_device_timeline.py` together with actual pinned native source/runtime. Leave `FRONTIER_NATIVE_SERVING_EVIDENCE_DIR` and `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` absent to avoid unrelated automatic collector/routing installation. Supply the existing `FRONTIER_NATIVE_PROFILE_LIMITS` from launch config. Use the already approved driver compat configuration and Torch-preloaded CUPTI library; no new path flag is needed.

An eventual `--output-dir` must be fresh and root-approved worker-local or `/mnt/codesign-exp/ycfeng/...`. Expected outputs are original `native_traces/*.pt.trace.json.gz` (one per cycle), `probe_report.json`, `native_profiler.log`, and `proc_maps.txt`. Source/library paths and SHA256s, Torch/native/runtime versions, GPU properties/UUID, all stream snapshots, complete target annotations/runtime/kernel fields and separated priming rows are retained. The external root launch receipt must bind the actual image digest and platform/GPU identity to these artifacts; this CPU delivery supplies only the expected image from requirements, not an observed image.

For eager PASS, all four targets in both cycles need unique CPU annotations, nonempty exact-correlation kernels on the queried CUPTI stream; all eight snapshot stages need both distinct streams with cached/direct CUPTI agreement. The script records unchanged wrapper/underlying-profiler object IDs around both cycles. A native start/stop failure or incomplete trace is retained and does not prove mapping.

Graph PASS requires directly attributed positive exact `(graph id, graph node id)` pairs under each captured CPU op, unique owners, and exact complete coverage of both replay traces. Missing IDs, source/exec namespace differences, capture/replay/restart failures or incomplete correspondence yield UNKNOWN, independently of eager acceptance. No name/order matching, numeric conversion, topology inspection, callbacks or tools-ID APIs are used. Exit0 means only the script's eager predicate passed; it never establishes graph mapping or serving costs. No GPU performance metrics exist in this delivery.

## Reviewed P2 repair — final source and exact fault controls

The preceding commands/results describe the original249-line delivery and remain historical evidence. Root subsequently accepted the collector at `0d745706`/records `6ef05dd8` and passed actual original-source locked-image checks. This follow-up does not rewrite or rerun those checks or the original CPU failures.

Final source `tests/comparison/calibration/native_profile_probe.py`:262gross lines under the root275-line repair allocation, SHA256 `2830d15c79ccf8f3a6280bc388c93ac423ecacefe9d73e053412a405ae4fd7c3`. Repair diff21 additions/8 deletions, retained as `source_repair.diff` in this directory. Only the two reviewed verdict defects were changed; actual stream/native profiler/capture mechanisms, CLI and source/library identity retention are unchanged.

Temporary CPU evidence root: `/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_review_20261024/`.

- `original_native_profile_probe.py`: exclusive fresh byte snapshot, SHA256 `237a82f1176a6acefff82b0e329ee09679b602383f4045b4c4751f1df85759d7`.
- `temporary_cpu_fault_controls.py`: SHA256 `45dd28b31f951e86163e2312fe4564b78e0e662b5e3112756953dfe5cd089cbc`; identical in RED and GREEN.
- `red_original/receipt.json`: source-bound raw cases, errors and inputs; exit1,2PASS/6FAIL.
- `green_repaired/receipt.json`: source-bound raw cases, errors and inputs; exit0,8PASS/0FAIL.
- Both directories retain every generated `*.synthetic.pt.trace.json.gz` input. These inputs are synthetic CPU fixtures, not native CUDA traces.

The controls import actual `joined` and `graph_correspondence` from the specified source bytes. They compile unchanged extracted `run.analyze`, `run.attempt` and the final eager-verdict statements from the same AST, then execute those bodies using the actual existing `forward_device_timeline.load_trace` on synthetic gzip inputs. No Torch/GPU function is called or replaced. A plain namespace supplies only observer-error state; snapshot IDs7/17 and graph/node IDs19/21/22 are explicitly invented fixture fields and are not CUPTI or graph equivalence evidence.

| Control | Expected | Original RED | Repaired GREEN |
| --- | --- | --- | --- |
| Valid exact graph positive | PASS | PASS | PASS |
| Capture did not return successfully | UNKNOWN | PASS, defect reproduced | UNKNOWN |
| Cycle1 replay synchronization did not return | UNKNOWN | PASS, defect reproduced | UNKNOWN |
| Cycle2 replay synchronization did not return | UNKNOWN | PASS, defect reproduced | UNKNOWN |
| Valid eager positive, both cycles/four rows each | PASS | PASS | PASS |
| Valid eager, malformed cycle1 replay annotation | eager PASS | eager FAIL, defect reproduced | eager PASS; graph UNKNOWN; exact parser error retained |
| Valid eager, malformed cycle2 replay annotation | eager PASS | eager FAIL, defect reproduced | eager PASS; graph UNKNOWN; exact parser error retained |
| Valid eager, malformed optional priming annotation | eager PASS | eager FAIL, same optional-analysis seam | eager PASS; independent valid synthetic graph PASS; priming error retained |

For malformed graph negatives, the runtime launch still exists but its containing CPU annotation lacks `dur`. Original `joined` raises `KeyError: 'dur'` after all four eager rows already pass. The repaired code stores the eager verdict first and retains the error in `analyze_graph_cycle1/2`. For malformed priming, it retains the same parser error in `analyze_priming_cycle1`; the separate graph analysis still executes. GREEN receipts were additionally inspected to assert every eager row and both cycle statuses stay PASS, every schema error remains present, and malformed graph correspondence is UNKNOWN.

### Exact snapshot and RED/GREEN commands

```bash
/data/ycfeng/envs/asyncopd-py312/bin/python -B - <<'PY'
import hashlib
from pathlib import Path
source = Path('tests/comparison/calibration/native_profile_probe.py').read_bytes()
expected = '237a82f1176a6acefff82b0e329ee09679b602383f4045b4c4751f1df85759d7'
assert hashlib.sha256(source).hexdigest() == expected
snapshot = Path('/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_review_20261024/original_native_profile_probe.py')
with snapshot.open('xb') as retained:
    retained.write(source)
print({'snapshot': str(snapshot), 'sha256': expected, 'gross_lines': len(source.splitlines())})
PY
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /data/ycfeng/envs/asyncopd-py312/bin/python -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_review_20261024/temporary_cpu_fault_controls.py --source /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_review_20261024/original_native_profile_probe.py --phase RED --output /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_review_20261024/red_original
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /data/ycfeng/envs/asyncopd-py312/bin/python -B /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_review_20261024/temporary_cpu_fault_controls.py --source /data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/tests/comparison/calibration/native_profile_probe.py --phase GREEN --output /data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_review_20261024/green_repaired
```

RED ran before any production-source edit. The same8 controls ran GREEN without changing control code, fixtures or expected verdicts. Output directories are fresh/exclusive; retained evidence must not be overwritten on a future rerun.

### Exact final AST/budget/help commands

```bash
/data/ycfeng/envs/asyncopd-py312/bin/python -B - <<'PY'
import ast, hashlib, difflib
from pathlib import Path
original = Path('/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_review_20261024/original_native_profile_probe.py')
source = Path('tests/comparison/calibration/native_profile_probe.py')
blob = source.read_bytes()
ast.parse(blob)
assert len(blob.splitlines()) <= 275
print({'ast': 'PASS', 'gross_lines': len(blob.splitlines()), 'source_sha256': hashlib.sha256(blob).hexdigest(), 'original_sha256': hashlib.sha256(original.read_bytes()).hexdigest()})
print(''.join(difflib.unified_diff(original.read_text().splitlines(True), blob.decode().splitlines(True), fromfile=str(original), tofile=str(source))))
PY
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= /data/ycfeng/envs/asyncopd-py312/bin/python -B tests/comparison/calibration/native_profile_probe.py --help
```

Actual AST PASS/262lines and help exit0. Python remains the existing3.12.13 project interpreter; no dependency/environment edit or GPU initialization. No Git command/write, Docker/network/platform operation or delegation was used in the follow-up.

### Existing root image evidence and remaining binding

Read-only existing receipt `/data/ycfeng/tmp/glm53_dsv4/native_stream_probe_image_oj706s8g/receipt.json` has SHA256 `08e8b3aa8c522c162fcb40884d43a48b7aeb1cd76e116bcd0891ca08fe90d2cc`, status `PASS_LOCKED_IMAGE_CPU_ENTRY`, bound to the original SHA256 `237a82f1176a6acefff82b0e329ee09679b602383f4045b4c4751f1df85759d7`.

It records actual native constructor PASS, help PASS and full original CLI reaching expected no-GPU failure/exit2. Runtime Python3.12.3, Torch2.13.0+cu130, vLLM0.28.0; native config SHA256 `78af917ec2ea2ba3bacda15f58fda9782b8ff82c0b810fd5c9563f4203aa5ba8`, wrapper SHA256 `eb0db0fd1c0a95c320fbaf9bdd66b6222a2d26ba9c1ac97efbfe1b196b3fbf15`; already mapped CUPTI SHA256 `e2f9ed861fe27c492b8bb52b5e3220ef5120f3edcda36312e96b7fd8a186be3e`. That receipt was neither rerun nor rewritten.

Root must re-review and bind final repaired source/package evidence. Actual H800 stream mapping and graph attribution remain unverified. CPU fault controls prove only the repaired completion/verdict behavior, not successful native GPU capture, replay or kernel attribution.

### Exact post-control evidence inspection and final binding checks

```bash
/data/ycfeng/envs/asyncopd-py312/bin/python -B - <<'PY'
import hashlib, json, difflib
from pathlib import Path
base = Path('/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_review_20261024')
red = json.loads((base / 'red_original/receipt.json').read_text())
green = json.loads((base / 'green_repaired/receipt.json').read_text())
assert red['passes'] == 2 and red['failures'] == 6
assert green['passes'] == 8 and green['failures'] == 0
assert red['control_sha256'] == green['control_sha256']
for case in green['cases']:
    if case['name'].startswith('valid_eager_malformed'):
        evidence = case['evidence']
        assert evidence['eager_stream_status'] == 'PASS'
        assert all(cycle['status'] == 'PASS' and all(row['status'] == 'PASS' for row in cycle['eager'].values()) for cycle in evidence['cycles'].values())
        assert any("KeyError: 'dur'" in error['traceback'] for error in evidence['errors'])
        if 'graph' in case['name']:
            assert evidence['graph_correspondence']['status'] == 'UNKNOWN'
        print({'case': case['name'], 'eager': evidence['eager_stream_status'], 'graph': evidence['graph_correspondence']['status'], 'retained_error_stages': [error['stage'] for error in evidence['errors']]})
original = (base / 'original_native_profile_probe.py').read_text().splitlines(True)
source_path = Path('tests/comparison/calibration/native_profile_probe.py')
source = source_path.read_text().splitlines(True)
diff = ''.join(difflib.unified_diff(original, source, fromfile='original_native_profile_probe.py', tofile=str(source_path)))
artifact = Path('task_memory/task_2026-10-23_v4_remaining_measurements/profile01/stream_probe/source_repair.diff')
with artifact.open('x') as retained:
    retained.write(diff)
lines = diff.splitlines()
print({'control_sha256': green['control_sha256'], 'source_sha256': green['source_sha256'], 'gross_lines': len(source), 'diff_added': sum(line.startswith('+') and not line.startswith('+++') for line in lines), 'diff_deleted': sum(line.startswith('-') and not line.startswith('---') for line in lines), 'diff_artifact': str(artifact), 'root_original_image_receipt_sha256': hashlib.sha256(Path('/data/ycfeng/tmp/glm53_dsv4/native_stream_probe_image_oj706s8g/receipt.json').read_bytes()).hexdigest()})
PY
/data/ycfeng/envs/asyncopd-py312/bin/python -B - <<'PY'
import ast, hashlib, json
from pathlib import Path
source = Path('tests/comparison/calibration/native_profile_probe.py').read_bytes()
root = Path('/data/ycfeng/tmp/opencode-v2-tmp/opencode/w128_native_stream_probe_review_20261024')
green = json.loads((root / 'green_repaired/receipt.json').read_text())
assert hashlib.sha256(source).hexdigest() == green['source_sha256']
assert hashlib.sha256((root / 'original_native_profile_probe.py').read_bytes()).hexdigest() == '237a82f1176a6acefff82b0e329ee09679b602383f4045b4c4751f1df85759d7'
assert hashlib.sha256(Path('/data/ycfeng/tmp/glm53_dsv4/native_stream_probe_image_oj706s8g/receipt.json').read_bytes()).hexdigest() == '08e8b3aa8c522c162fcb40884d43a48b7aeb1cd76e116bcd0891ca08fe90d2cc'
ast.parse(source)
print({'source_sha256': green['source_sha256'], 'gross_lines': len(source.splitlines()), 'green_passes': green['passes'], 'original_source_and_root_receipt_unchanged': True})
PY
```

Both commands exited0. The first created the fresh repair diff once and confirmed21 additions/8 deletions; do not rerun its exclusive write against the retained artifact. The final read-only check confirms repaired source exactly matches GREEN, original source remains unchanged and the root original-image receipt is unchanged. All raw exception evidence remains in the temporary receipts.

## Root final-source checks and executed job

Root CPU commands and exact arguments are retained in `root_cpu.json` and `root_verdict_controls.json`; original source-owner results above are historical. CPU environment: Python3.10.6 StepMind SDK, Python3.12.13 fault controls; locked image Python3.12.3/Torch2.13.0+cu130/vLLM0.28.0. Final source262/275lines at5f7d0950; seven input hashes and retained Git-source bytes match.

| Check | Expected | Observed | Result |
| --- | --- | --- | --- |
| Identical root fault controls | Original defects reproduce; repaired eight controls pass | RED2/6, GREEN8/0 | PASS |
| Final locked-image entry/constructor/help | Actual native types; entry stops only at absent GPU | Constructor/help exit0, entry expected exit2; pinned config/wrapper/CUPTI hashes match | PASS, CPU scope |
| SDK/resource construction |1H800,16CPU,mem_gb64,non-preemptible,600s | Exact configured values | PASS |
| Platform/container | Correct identity/image/NFS; success without restart | Succeeded,23s,exit0,restart0 | PASS |
| Worker GPU identity | nvidia-smi and Torch identify the same H800 | UUID c459bcfe-2bea-e452-274c-beb947928449 | PASS after correcting the inspector's prefix-sensitive comparison |
| Sealed inputs/download | Seven inputs unchanged; every output hash matches |7/7 inputs;5 outputs/228175bytes | PASS |
| Raw stream/graph evidence | Independent reconstruction determines qualification | Root162-check rerun byte-identical; independent review No findings | PASS for targeted stream/restart; graph UNKNOWN |

Executed local launcher: `/data/ycfeng/tmp/stepmind-env/bin/python -u -B <package>/launch.py --launch`, with package `h800_stream_probe_01` under this directory, `STEPMIND_BACKEND=rjob`, personal credentials and the runbook proxy environment. `root_submission.json` records exact argv/cwd/PID/resources/log; `h800_stream_probe_01/launcher_result.json` records creation-to-completion. The worker uses the generated launcher's exact command (see `generated_launcher.diff`) and the sealed config. Read-only platform query and SSH-tar inventory/extraction commands are recorded by `platform_terminal.json`, `download.json` and the session transcript. No cloud bytes or permissions were modified during retrieval.

Raw launcher log: `/data/ycfeng/tmp/glm53_dsv4/native_stream_probe01_launch_20261024.log`; redacted worker log and terminal receipt: `/data/ycfeng/tmp/glm53_dsv4/native_stream_probe01_terminal_z176kwmi/`. `root_run_check.json` retains the original `AssertionError` from comparing `GPU-<uuid>` with `<uuid>` and the corrected `uuid.UUID` equality. This was an inspector-format error, not a worker failure. Source/package reviews are static evidence; root executions provide independent CPU/platform checks. Final raw-data commands/results are in `results/{command.md,test_report_2026-10-24_data.md,root_verification.json}`; `results/primary_acceptance.json` accepts only targeted stream identity and observed two-cycle reuse.
