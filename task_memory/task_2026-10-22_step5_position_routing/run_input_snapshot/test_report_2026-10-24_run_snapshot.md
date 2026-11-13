# D29 run-fixed collective input verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Close component verification with real-wave parity, parent hash checks, independent reviews and source commit db581ff6. |
| 2026-10-24 | Record primary RED/GREEN, independent trainer checks and configured-path lifetime checks; full wave and independent review remain pending. |

## Scope and source

Requirements: `../requirements.md` lines261-301; interface and acceptance: `design.md`; allocation: `../repairs/run_input_snapshot_scope_amendment.json`.

Worktree: `/data/ycfeng/Frontier/.worktrees/step5-d29-input-snapshot`. Source parent `ecccf83fa6f2c24a14fa5397ef567fbb32b06bef` is the delivered byte-repair cherry-pick. The reviewed and verified snapshot is committed as `db581ff6ad43e1ce57361ed00fc4be0256e85a45`. The component receipt records exact source hashes and step/retained counts; donor source and the native numerical cache key are unchanged.

Environment: `/data/ycfeng/envs/frontier-py310/bin/python`, Python3.10.6. Commands run from the candidate worktree, use a30G systemd scope, and keep evidence under `/data/ycfeng/tmp`. Except the initial RED, Python runs as `brainpp`; bytecode writes are disabled throughout.

## Commands

Behavioral backend RED, before snapshot source changes:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G env PYTHONPATH=$PWD:$PWD/tests/unit PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_collective_sim_cc_backend.py::test_collective_artifact_reuses_only_identical_current_bytes -q --basetemp=/data/ycfeng/tmp/d29_snapshot_red01 --junitxml=/data/ycfeng/tmp/d29_snapshot_red01.xml > /data/ycfeng/tmp/d29_snapshot_red01.log 2>&1
```

Behavioral constructor RED after fixing the test's architecture configuration:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH=$PWD:$PWD/tests/unit PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_moe_routing_runtime.py::test_collective_input_binds_both_role_constructors_and_refreshes_next_run -q -o cache_dir=/data/ycfeng/tmp/d29_snapshot_pytest_cache --basetemp=/data/ycfeng/tmp/d29_snapshot_red03 --junitxml=/data/ycfeng/tmp/d29_snapshot_red03.xml > /data/ycfeng/tmp/d29_snapshot_red03.log 2>&1
```

Affected GREEN selection:

```sh
timeout 240s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH=$PWD:$PWD/tests/unit PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_collective_sim_cc_backend.py tests/unit/test_moe_routing_runtime.py tests/unit/test_prediction_measurement_bindings.py tests/unit/test_step5_measured_ffn.py -q -o cache_dir=/data/ycfeng/tmp/d29_snapshot_pytest_cache --basetemp=/data/ycfeng/tmp/d29_snapshot_green01 --junitxml=/data/ycfeng/tmp/d29_snapshot_green01.xml > /data/ycfeng/tmp/d29_snapshot_green01.log 2>&1
```

Separate both-trainer/preflight and independent actual-fit/cache selection:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env -u FRONTIER_TEST_CPU_WRITER_FILE -u FRONTIER_TEST_CPU_WRITER_SHA256 PYTHONPATH=$PWD:$PWD/tests/unit PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp FRONTIER_TMP_ROOT=/data/ycfeng/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/ycfeng/envs/frontier-py310/bin/python -B -m pytest tests/unit/test_moe_grouped_gemm_padded_features.py::test_selected_sources_reach_real_preflights_loader_and_both_trainers_before_old_helpers tests/unit/test_moe_grouped_gemm_padded_features.py::test_independent_selected_actual_fit_cache_binds_native_context -q -o cache_dir=/data/ycfeng/tmp/d29_snapshot_pytest_cache --basetemp=/data/ycfeng/tmp/d29_snapshot_trainers01 --junitxml=/data/ycfeng/tmp/d29_snapshot_trainers01.xml > /data/ycfeng/tmp/d29_snapshot_trainers01.log 2>&1
```

Configured-path probe:

```sh
timeout 60s sudo -n systemd-run --scope -p MemoryMax=30G runuser -u brainpp -- env PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp /data/ycfeng/envs/frontier-py310/bin/python -B /data/ycfeng/tmp/d29_snapshot_input_lifetime_probe.py --output /data/ycfeng/tmp/d29_snapshot_input_lifetime01 > /data/ycfeng/tmp/d29_snapshot_input_lifetime01.log 2>&1
```

## Observed results

| Check | Expectation | Observation | Verdict |
| --- | --- | --- | --- |
| Backend behavioral RED | Existing backend must retain original cost after file replacement | Old source returned0.04 instead of0.03; one assertion failed | Expected RED |
| Constructor behavioral RED | P/D must retain the version established before the between-role rewrite | Old source returned[0.04,0.04], expected[0.03,0.03] | Expected RED |
| Four affected files | All constructor/lifetime/admission/point/replacement checks pass |82 PASS;0 FAIL/ERROR/SKIP;154.23s in log | PASS |
| Both-trainer selection | Existing shared and independent trainer checks remain valid |2 PASS;0 FAIL/ERROR/SKIP;7.788s JUnit time; kept separate from82 | PASS |
| Configured paths | Normalized aliases share one entry; distinct device/directory inputs remain separate; symlink retarget is observed only by a new owner | Five named checks PASS | PASS |
| Whitespace | No diff whitespace errors | `git diff --check` exit0 | PASS |
| Full real wave |208 complete events/costs/proofs/order/keys and exact JSONL/ledger; one input read and zero warm reads | Complete equality; binding1/cold0/warm0 reads | PASS |
| Independent review | Standards and Spec have no actionable findings | No findings on both axes; `review.md` | PASS |
| Parent wave receipt verification | Retained outputs and reviewed source match recorded hashes |28 manifest,41 source,6 script,6 raw artifact hashes; complete JSONL/ledger equality | PASS |

The P/D test executes actual manager and MoE predictor constructors. It uses an empty manager configuration to avoid training and stubs unrelated parent training/routing allocation; it is constructor-wiring evidence, not a full Simulator run. The trained CORE query test writes invalid input between query and backend pricing and verifies retained plans still price consistently; new lifetimes reject the invalid input. Native importer tests use synthetic CPU-side schema evidence and do not qualify native measurements.

Initial missing/permission/UTF8/JSON errors are exercised before successful publication, including repeated failure and recovery on the same backend. The existing per-point invalid plan/count/context/buffer tests remain. Private estimator, row-tuple and measurement-identity replacement checks remain outside the new file-lifetime rule.

## Limits and operational evidence

The full-wave commands and complete observations are in `wave/test_report_2026-10-24_run_snapshot_wave.md` and its evidence file. The parent hash check iterated that manifest, both source-hash maps and script map with SHA256; it independently compared complete baseline/candidate JSONL and ledger files. All assertions passed with Python3.10.6, from the candidate worktree, within60s. Warm medians0.340867→0.162715s are executor measurements; binding0.089499s and cold1.080278s are separate. Numeric lookups288, fresh proofs144 and collective prices64 remain per wave. Constructor checks and cached-manager wave reconstruction are complementary component evidence, not a full Simulator run.

Exact parent receipt check, tool timeout60000ms, candidate cwd:

```sh
/data/ycfeng/envs/frontier-py310/bin/python -B - <<'PY'
import hashlib,json
from pathlib import Path
root=Path.cwd();wave=root/'task_memory/task_2026-10-22_step5_position_routing/run_input_snapshot/wave'
e=json.loads((wave/'evidence/evidence.json').read_text());m=json.loads((wave/'evidence/manifest.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
for p,d in m['sha256'].items():assert sha(wave/p)==d,p
n=0
for arm in e['source_freeze'].values():
 for p,d in arm['hashes'].items():assert sha(Path(arm['root'])/p)==d,p;n+=1
for p,d in e['scripts'].items():assert sha(p)==d,p
for arm in e['artifacts'].values():
 for key in ('wave_result','whole_jsonl','ledger'):assert sha(arm[key])==arm[key+'_sha256'],arm[key]
for key in ('whole_jsonl','ledger'):
 assert Path(e['artifacts']['baseline'][key]).read_bytes()==Path(e['artifacts']['candidate'][key]).read_bytes(),key
assert e['verdict']=='PASS' and e['command_exits']==[0,0,0,0]
print(json.dumps({'manifest':len(m['sha256']),'source_hashes':n,'scripts':len(e['scripts']),'whole_jsonl_and_ledger_equal':True,'candidate_report_keys':list(e['reports']['candidate'])}))
PY
```

`evidence/parent_component_receipt.json` records212/340 gross (73/120 source,139/220 tests), net+86, and every path within its step/retained cap. Reconciliation against the delivered-byte planning receipt gives CORE4173/4404, PRICING3940/3956, native1350/1582,9463 source and9477 global. This is not a fresh114-path final union audit. The supplied review diff is `evidence/bytes_and_snapshot.diff`, based on2cfbb344 and including the uncommitted overlay plus the new module.

`red02.log` records an incorrect initial test setup (`SYS_ARCH` was not `pd-disaggregation`), corrected before the constructor RED. It also records pytest-cache permission warnings caused by root-owned entries created in `red01`. Later commands use the explicit scratch cache. No cleanup or permission changes were made.

This report does not establish full6144/1024 CLI completion, the2400s limit, strict77/full-unit union acceptance, T2 readiness, native qualification or E2E accuracy. The separate private-encoder and config-scan candidates remain unallocated; no numerical/whole-proof cache was added.
