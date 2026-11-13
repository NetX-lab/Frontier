# EP Ledger Actual Caller Follow-up

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-25 | Record the authorized real save/load repair, deterministic controls, resolved checkout failures and UNADMITTED handoff target. |

## Outcome and source

The requested small caller repair and focused verification are complete.
**The committed candidate is UNADMITTED.** Primary/new-session independent clean
composed audit, Standards/Spec review and final frozen regressions still precede
accepted-production integration.

- Task: `D29-EP-LANE-LEDGER-IMPLEMENTATION`.
- Session: `ses_e91675a86ffegKCQdHRPQFOypp`.
- Worktree: `/data/ycfeng/Frontier/.worktrees/step5-d29-ledger-stream`.
- Branch: `fix/d29-ledger-stream`.
- Initial component: `6c2f179c9117256bb7086441b0c11a1459f48674`; initial records: `66e5011619e21a04e7ea6ad1a6f91c6af7e62afc`.
- Authorization: `886945f3b757194b30be4aa0d314a63b772b465e`, imported as `924427c05e9b77fb26952253badafe9d7f38ba50`.
- Corrected source: `1f6b431478e2351e3ca6819f707cccbc3e4336e2`.

Only `tests/performance/step5_actual06_iteration.py` changes in the source
successor: existing persistent IDs carry cumulative count and bind the fresh
writer; file-position calculation permits a restored count before lazy output
exists. No stream, old row list, new API or error-suppressing fallback is added.
Both implementation and authorization documentation intents survive the two
cherry-pick conflicts. Root files were changed only by importing primary's commit.

## Commands and resource envelope

Working directory for all commands is the worktree above. Python is
`/data/ycfeng/envs/frontier-py310/bin/python`,3.10.6/GCC11.3.0. Each execution
uses an outer60s timeout and configured systemd `MemoryMax=30G` (32,212,254,720B).
Every scope starts. Effective cgroup limit and peak are **not measured**; no v2
memory.max assumption or uncapped fallback is used. Single OPENBLAS/OMP/MKL/NUMEXPR
threads, `-B`, TMPDIR/cache/MPL directories below `/data/ycfeng/tmp` are used.

The exact common Python/pytest command form is:

```bash
scratch=/data/ycfeng/tmp/d29_ledger_stream_followup_<suffix>
mkdir -p "$scratch"
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH="$PWD:$PWD/tests/unit" PYTHONDONTWRITEBYTECODE=1 TMPDIR="$scratch" XDG_CACHE_HOME="$scratch/cache" MPLCONFIGDIR="$scratch/mpl" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B <arguments below> > "$scratch/raw.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$scratch/exit.txt"
cat "$scratch/raw.log"
exit "$rc"
```

| suffix | Python arguments | Raw result |
| --- | --- | --- |
| actual_caller | `- <<'PY'` with the exact GREEN input below | exit0; actual save/load count0/3 PASS; control-body0.054s excludes imports |
| checkout_tests | `-m pytest tests/integration/test_step5_measured_main.py -q -p no:cacheprovider -k position_command_prices_exact --basetemp="$scratch/pytest" --junitxml="$scratch/junit.xml"` | exit0;4PASS/68deselected;2.18s |
| ep | `-m pytest tests/unit/test_stage_reporting_contract.py tests/unit/test_ep_wave_trace_context.py tests/unit/test_ep_trace.py tests/unit/test_typed_ep_trace_contract.py -q -p no:cacheprovider --basetemp="$scratch/pytest" --junitxml="$scratch/junit.xml"` | exit0;30PASS;2.39s |
| integration | `-m pytest tests/integration/test_step5_measured_main.py -q -p no:cacheprovider --basetemp="$scratch/pytest" --junitxml="$scratch/junit.xml"` | exit0;66PASS/6SKIP;50.43s |
| red_committed | `- <<'PY'` with the exact RED input below | harness exit0; exact prior committed save_state raises PicklingError as expected |

The ordinary setup commands use the same60s/systemd scope with
`env TMPDIR="$scratch" XDG_CACHE_HOME="$scratch/cache"` followed by:

```bash
# suffix checkout_clone, initially empty inspected destination
git clone --local --no-hardlinks --no-checkout /data/ycfeng/Frontier/.worktrees/step5-preview/frontier/cc_backend/backends/collective-sim frontier/cc_backend/backends/collective-sim
# suffix checkout_pin
git -C frontier/cc_backend/backends/collective-sim checkout --detach 3dac53c544e546574cb6e8078cd14053d0916113
# Read-only identity checks
git -C frontier/cc_backend/backends/collective-sim rev-parse HEAD
git -C frontier/cc_backend/backends/collective-sim status --porcelain
```

Both setup exits are0; HEAD is the pinned commit and status is empty. This is an
independent local checkout, not a fake path/symlink or backend override. No network,
build, backend source, donor or superproject gitlink edit occurs. The four original
constructors run unchanged through the actual resolver and all pass. Retain this
checkout as an active prerequisite for the new session.

## Exact executed GREEN stdin

This fenced verification transcript is not runtime code loaded from task records.
The state shell is synthetic and has no events/predictors/models; MetricsStore,
its writer and save_state/load_state are actual existing implementations. Only
the exact current main count/position/digest statements are executed separately
to avoid a prohibited production pricing/fullCLI wave.

```python
import ast, hashlib, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from tests.performance import step5_actual06_iteration as perf
from tests.unit.test_stage_reporting_contract import _store
from frontier.types import EventType
scratch=Path(os.environ['TMPDIR'])
started=time.perf_counter()
fixtures=Path('task_memory/task_2026-10-22_step5_position_routing/ledger_memory/evidence/implementation_2026-10-25/legacy_successful.jsonl')
rows=[json.loads(line) for line in fixtures.read_text().splitlines()]
main=next(n for n in ast.parse(Path(perf.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
def assignment_index(name):
    return next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name)
i=assignment_index('ledger_before')
j=assignment_index('ledger_digest')
before_code=compile(ast.Module(body=main.body[i:i+3],type_ignores=[]),'<actual-main-before-count-and-position>','exec')
after_code=compile(ast.Module(body=main.body[j:j+2],type_ignores=[]),'<actual-main-incremental-digest>','exec')
def simulator(output):
    metrics,_=_store(output,enable_op_level_tracing=False,store_frontier_stage_batch_ledger=True)
    return SimpleNamespace(_execution_time_prediction_model_manager=SimpleNamespace(_diagnostic_models={}),_predictors={},_global_scheduler=SimpleNamespace(_cluster_schedulers={}),_event_queue=[],_metric_store=metrics,_all_requests=[SimpleNamespace(num_emitted_decode_tokens=1)],_requests_by_session_id={},_time=0.,_event_trace=[dict(event_type=EventType.GLOBAL_BATCH_END.value)],_clusters={},_diagnostic_iteration_boundary=True,_config=SimpleNamespace(metrics_config=metrics._config),_trace_store=None)
results=[]
for count in (0,3):
    case=scratch/f'count_{count}'
    source=simulator(case/'source')
    original=source._metric_store._frontier_ep_wave_lane_ledger_rows
    source_path=Path(original._path)
    for index in range(count): original.append(rows[index%len(rows)])
    source_bytes=source_path.read_bytes() if count else b''
    state_path=case/'state.pkl'
    perf.save_state(source,state_path)
    assert (original._stream is None) if not count else (not original._stream.closed)
    destination=simulator(case/'destination')
    fresh=destination._metric_store._frontier_ep_wave_lane_ledger_rows
    destination_path=Path(fresh._path)
    assert destination_path!=source_path and not destination_path.exists()
    path_open=Path.open
    def no_source_open(path,*args,**kwargs):
        assert path!=source_path, 'load_state reopened old ledger output'
        return path_open(path,*args,**kwargs)
    with patch.object(Path,'open',no_source_open): receipt=perf.load_state(destination,state_path,case/'destination')
    ledger=destination._metric_store._frontier_ep_wave_lane_ledger_rows
    assert ledger is fresh and ledger is not original and ledger._stream is None
    assert len(ledger)==count and Path(ledger._path)==destination_path and not destination_path.exists()
    caller=dict(simulator=destination,Path=Path,hashlib=hashlib)
    exec(before_code,caller)
    assert caller['ledger_before']==count and caller['ledger_position']==0
    exec(after_code,caller)
    assert caller['ledger_digest'].hexdigest()==hashlib.sha256(b'').hexdigest()
    new_rows=[{**rows[k], 'batch_id':batch_id} for k,batch_id in ((1,101),(0,102),(1,103))]
    expected=b''.join((json.dumps(row,sort_keys=True)+'\n').encode() for row in new_rows)
    for row in new_rows: ledger.append(row)
    exec(after_code,caller)
    assert destination_path.read_bytes()==expected and len(ledger)==count+3
    assert caller['ledger_digest'].hexdigest()==hashlib.sha256(expected).hexdigest()
    exec(before_code,caller)
    assert caller['ledger_position']==len(expected) and caller['ledger_before']==count+3
    final_row={**rows[0], 'batch_id':104}
    final_bytes=(json.dumps(final_row,sort_keys=True)+'\n').encode()
    ledger.append(final_row)
    exec(after_code,caller)
    assert caller['ledger_digest'].hexdigest()==hashlib.sha256(final_bytes).hexdigest()
    assert len(ledger)-caller['ledger_before']==1 and len(original)==count
    assert destination_path.read_bytes()==expected+final_bytes
    assert source_path.read_bytes()==source_bytes if count else not source_path.exists()
    destination._metric_store._write_frontier_stage_batch_ledger()
    source._metric_store._write_frontier_stage_batch_ledger()
    assert destination_path.read_bytes()==expected+final_bytes
    assert source_path.read_bytes()==source_bytes if count else not source_path.exists()
    results.append(dict(status='PASS',historical_count=count,restored_count=count,terminal_count=len(ledger),fresh_writer_identity=True,fresh_output_path=str(destination_path),old_output_path=str(source_path),old_output_absent=count==0,old_output_unchanged=True,source_sha256=hashlib.sha256(source_bytes).hexdigest(),new_output_bytes=len(expected+final_bytes),new_output_sha256=hashlib.sha256(expected+final_bytes).hexdigest(),ordered_three_row_digest=hashlib.sha256(expected).hexdigest(),later_single_row_digest=hashlib.sha256(final_bytes).hexdigest(),zero_increment_digest=hashlib.sha256(b'').hexdigest(),state_receipt=receipt))
result=dict(task_id='D29-EP-LANE-LEDGER-IMPLEMENTATION',status='PASS_ACTUAL_SAVE_LOAD_CALLER',source_mode='current actual save_state/load_state; real MetricsStore writer; exact current main digest AST, not a patched proposal',scenarios=results,wall_s=time.perf_counter()-started,python=sys.version,executable=sys.executable,configured_memory_max_bytes=32212254720,effective_cgroup_limit_and_peak='not measured',limitations=['Synthetic reporting state with no event/pricing execution','No fullCLI, native, G3, E2E, runtime or full-workload RSS acceptance'])
(scratch/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
```

## Exact executed RED stdin

The earlier actual defect records remain unchanged. This additional explicit
trace captures the exact committed pre-repair module's save_state with the same
real writer, not a hypothetical altered function. Harness exit0 means the
expected exception was captured; the tested prior save_state **fails**.

```python
import hashlib, json, os, pickle, subprocess, sys, traceback
from pathlib import Path
from types import ModuleType, SimpleNamespace
from tests.unit.test_stage_reporting_contract import _store
scratch=Path(os.environ['TMPDIR'])
revision='6c2f179c9117256bb7086441b0c11a1459f48674'
path='tests/performance/step5_actual06_iteration.py'
source=subprocess.check_output(['git','show',revision+':'+path])
prior=ModuleType('d29_committed_pre_caller_repair')
prior.__file__=str(Path(path).resolve())
exec(compile(source,'<committed-6c2f179c-performance>','exec'),vars(prior))
metrics,_=_store(scratch/'source',enable_op_level_tracing=False,store_frontier_stage_batch_ledger=True)
row=json.loads(Path('task_memory/task_2026-10-22_step5_position_routing/ledger_memory/evidence/implementation_2026-10-25/legacy_successful.jsonl').read_text().splitlines()[0])
ledger=metrics._frontier_ep_wave_lane_ledger_rows
ledger.append(row)
simulator=SimpleNamespace(_execution_time_prediction_model_manager=SimpleNamespace(_diagnostic_models={}),_predictors={},_global_scheduler=SimpleNamespace(_cluster_schedulers={}),_event_queue=[],_metric_store=metrics,_all_requests=[],_requests_by_session_id={},_time=0.,_event_trace=[],_clusters={},_diagnostic_iteration_boundary=False)
try:
    prior.save_state(simulator,scratch/'failed_state.pkl')
except pickle.PicklingError as error:
    failure=traceback.format_exc()
    assert 'Cannot pickle files that are not opened for reading: w' in str(error)
    (scratch/'red_traceback.txt').write_text(failure)
    print(failure)
    result=dict(task_id='D29-EP-LANE-LEDGER-IMPLEMENTATION',status='EXPECTED_RED_CAPTURED',tested_save_state_revision=revision,tested_source_sha256=hashlib.sha256(source).hexdigest(),actual_save_state_outcome='RAISED_PICKLING_ERROR',exception_type=type(error).__name__,exception_message=str(error),real_store=True,nonempty_count=len(ledger),writer_mode=ledger._stream.mode,python=sys.version,executable=sys.executable,configured_memory_max_bytes=32212254720,effective_cgroup_limit_and_peak='not measured',limitations=['Expected-failure capture of exact committed pre-repair function, not candidate acceptance','No pricing, event execution or native/fullCLI run'])
    (scratch/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
else:
    raise AssertionError('Committed pre-repair save_state did not reproduce its real writer defect')
metrics._write_frontier_stage_batch_ledger()
```

## Expected and actual values

| Control | Expectation | Actual |
| --- | --- | --- |
| Prior committed save_state | Real nonempty writer raises PicklingError | PASS expected RED; raw traceback retained |
| Fresh restored writer | New writer identity/path; no old stream/row retention | PASS for historical counts0/3; no old ledger reopen during load |
| Restored count/no file | Counts0/3 before lazy output creation | PASS; before-position0 and empty increment SHA256 |
| New output | Complete sorted values/proofs/keys/order and count | Four new rows7124B in each scenario; terminal counts4/7 |
| Ordered new bytes | Exact SHA256 | `f63c0297e9dffe14d25d929fc3488d6d7c0f9eb6b1ff15393f24d49fe03e34a7` |
| Digest slices | Three new rows then one later row; no old bytes | PASS; full digest values in actual_caller/result.json |
| Original file | Byte-identical after save/load/new writes/close; zero case absent | PASS; nonempty SHA256 `6be83f0e97cc1ae315a3b108ece4b185ac1b06517375a3d3b58cedf3d9102d30` |
| Old checkout failures | Four original tests unchanged, real resolver succeeds |4PASS; previous raw4FAIL preserved |
| Full focused parity | Keep all93 old IDs/statuses and9 newPASS IDs |96PASS/6SKIP across102 IDs; no old ID/status delta; four prior setupFAIL→PASS |

No full-unit/strict77 run is performed. Earlier broad raw FAIL/ERROR/SKIP suites
remain historical evidence and are not relabeled PASS by the focused controls.

## Exact four-path accounting

Step baseline: `0ca540251ae1c2bdbe436a2884d48adf29ed955a`.
Cumulative baseline: `1aee0af11cd58691e64603b99eca0ac459a5b350`.
Commands: `git diff --no-renames --numstat <baseline> <source commit> -- <path>`.

| File | Step +/− | Step gross/cap | Cumulative +/− | Cumulative gross/cap |
| --- | ---: | ---: | ---: | ---: |
| frontier/metrics/metrics_store.py |26/7|33/36|26/7|33/36|
| tests/integration/run_step5_measured_main.py |12/3|15/20|52/4|56/67|
| tests/integration/test_step5_measured_main.py |61/1|62/65|320/0|320/325|
| tests/performance/step5_actual06_iteration.py |11/3|14/14|11/3|14/14|
| Total |110/14|124/135|409/14|423/four-path caps|

Follow-up commit alone is5insertions/1deletion; the overall performance path stays
14gross because the file-position expression was already a new line. Increment
116 projects CORE4377/4404,source9682/9942,global9696/10000. This is committed
**local four-path accounting**, not primary's independent composed audit. Other
three allocated sources, producer and gitlink are unchanged since initial delivery.

## Evidence and handoff limits

Canonical evidence root:
`task_memory/task_2026-10-22_step5_position_routing/ledger_memory/evidence/caller_followup_2026-10-25/`.

- `scope.json`, `complete_scoped_code.diff`, `caller_successor.diff`.
- `focused_status_comparison.json`, `test_ids.json`, `checkout.json`.
- `artifact_manifest.json`:27 byte-verified copied artifacts, including raw commands/results and native state/output files.
- `raw/actual_caller/result.json`: actual GREEN; `caller/count_0/` and `caller/count_3/` contain native state and complete new output; count3 also contains unchanged source output.
- `raw/red_committed/red_traceback.txt`, `failed_state.pkl`, `source_output.jsonl`: actual expected RED.
- `raw/integration/`, `raw/ep/`, `raw/checkout_tests/`: raw exits/logs/JUnit.

Keep the pinned local checkout, retained legacy fixture and cited caller state
inputs. Cleanup removed only four completed byte-identical setup-log/exit scratch
duplicates and their two empty directories. `cleanup.json` records the retained
hashes; no other-session or active input artifacts were removed.

No production pricing wave, required shape, fullCLI, model/CSV/trace census,
full-unit/strict77, GPU/platform/network/SSH/worker operation is performed. YC's
sufficient codesign OSS quota confirmation is recorded, not exercised here.
Remaining: independent composed audit, Standards/Spec, final-source freeze/full
regressions and accepted-production integration. FullCLI/native/G3/E2E/runtime,
whole-workload RSS/storage qualification and other root workflow items remain
pending for the new session. This owner does not write the root continuation plan.
