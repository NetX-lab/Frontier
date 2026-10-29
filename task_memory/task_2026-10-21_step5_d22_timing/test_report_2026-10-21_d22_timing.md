# D22 independent timing reference: CPU and staging report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record real-seam RED->GREEN,873 affected passes and verified fresh full-probe stage. |

## Result and limits

Source/CPU correction PASS; native timing remains UNVERIFIED. The original D18 H200 probe and H800 Smoke03 remain FAIL. No old receipt was recalculated. No GPU, auth, network, cloud, SSH or Docker operation occurred. No nested delegation or compute-branch merge occurred.

`collect_replay_timings` now executes one candidate window and a distinct reference window for each sample index. Every window allocates and primes its own outer event pair before restore, with no internal event record during clean replays. The two cohorts are interleaved in candidate/reference order, not shared or derived from each other. Their replay counts match. If either window is shorter than the declared minimum, both are discarded and their common count increases from the shorter observed duration. There is no mean/CV/agreement-dependent retry. Both retained cohorts have the requested sample count.

The independent reference median is computed from normalized reference windows. Candidate CV and mean-versus-reference remain <=5%; reference CV is not a new gate. Boundary single-replay samples/statistics are exported separately as `instrumentation_diagnostics_only`, never as window samples or agreement references. Scoped operator samples and eager GPU-active samples keep their original protocols. No linear source file or simulator behavior changed.

## Implementation evidence

- `frontier/profiling/step5/timing.py:65`: existing collection/measurement seam owns independent clean cohorts and minimum-window treatment. `collect_scoped_timings` is unchanged.
- `frontier/profiling/step5/rows.py:28`: optional independently measured layer reference; per-iteration statistics remain real per-invocation operator/GPU-active samples. Default behavior is unchanged.
- `frontier/profiling/step5/vllm_wrapper.py:460`: actual reference is passed into quality calculation; reference durations/counts/normalized observations and boundary diagnostics are exported separately.
- `frontier/profiling/step5/main.py:104`: each native cell is measured once even on quality failure; original initial-statistics columns are retained with `stability_attempts=1`.
- Existing strict `tests/integration/step5_profiling/judge_attention_smoke.py` recomputes the unchanged 5% checks, verifies reference normalization/count/minimum and diagnostic labels, and does not gate on boundary diagnostic disagreement. Old frozen judges remain untouched.
- Existing common stage resource description and launcher accept explicit preemptibility; existing defaults are unchanged. The new D22 stage selects preemptible H200 to avoid using any of D21's <=32 personal non-preemptible H200 allowance. Parent still owns fresh quota and priority checks.

## Environment and exact checks

CWD: `/data/ycfeng/Frontier/.worktrees/step5-profiling`.
Python: `/data/ycfeng/envs/oneshot-opd-py312/bin/python`, **3.12.13**, Clang22.1.3. No conda environment creation or dependency install. Existing dependencies: `/data/ycfeng/tmp/step5_profiling_20261020/extra_site`.

Real-seam RED command (same command run twice with output suffix01/02):

```bash
env PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu TMPDIR=/data/ycfeng/tmp \
  PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-profiling:/data/ycfeng/tmp/step5_profiling_20261020/extra_site \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python -B -m pytest \
  tests/unit/test_step5_attention_timing.py::test_independent_outer_windows_ignore_internal_boundary_event_overhead \
  -q -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d22_timing_red02.xml
```

Expected RED: missing independently executed cohort. Actual02: only five windows, not the seven required minimum, FAIL. Actual01 first exposed absent reference fields. Retained evidence: `evidence/red01.xml`, `evidence/red02.xml`.

Injected replay work **34.7us** and event-record cost **3.33us** reproduce boundary singles **38.03us**. GREEN retains the measured extra event cost: independent outer windows include one outer end-event cost divided by replay count, with no subtraction. Candidate/reference agreement passes while boundary-single versus candidate disagreement stays >5%. The exact call trace proves distinct event pairs, preparation outside each window and separate native invocations. Additional tests require short-reference count increases for both cohorts and reject invalid duration in either cohort.

Focused command:

```bash
env PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu TMPDIR=/data/ycfeng/tmp \
  PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-profiling:/data/ycfeng/tmp/step5_profiling_20261020/extra_site \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python -B -m pytest \
  tests/unit/test_step5_attention_timing.py tests/unit/test_step5_attention_profiling.py \
  tests/unit/test_step5_attention_judgment.py tests/unit/test_step5_fixed_probe.py \
  -q -p no:cacheprovider --junitxml=/data/ycfeng/tmp/d22_focused01.xml
```

Actual: **253 PASS in19.82s** before the final eleven explicit reference-duration/minimum tests. Counts overlap with affected regression; do not add them.

Final affected command (the full exact child Python command is also retained in `evidence/baseline_comparison02.json`):

```bash
env PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu TMPDIR=/data/ycfeng/tmp \
  WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 /data/ycfeng/envs/oneshot-opd-py312/bin/python - <<'PY'
import json, os, pathlib, subprocess, sys
root = pathlib.Path('/data/ycfeng/tmp/d22_timing_20261021')
files = json.loads(pathlib.Path('/data/ycfeng/tmp/d18_profile_20261021/profile_commands.json').read_text())['files'] + ['test_step5_fixed_probe', 'test_step5_boundary_probe']
args = [*(f'tests/unit/{name}.py' for name in files), '-q', '-p', 'no:cacheprovider', f'--junitxml={root}/affected02.xml']
code = "import sys,types,pathlib; p=types.ModuleType('tests'); p.__path__=[str(pathlib.Path.cwd()/'tests')]; sys.modules['tests']=p; import pytest; raise SystemExit(pytest.main(" + repr(args) + "))"
with (root / 'affected02.log').open('x') as stream:
    run = subprocess.run([sys.executable, '-B', '-c', code], env={**os.environ, 'PYTHONPATH': f'{pathlib.Path.cwd()}:/data/ycfeng/tmp/step5_profiling_20261020/extra_site'}, stdout=stream, stderr=subprocess.STDOUT)
raise SystemExit(run.returncode)
PY
```

Use a fresh suffix when rerunning exclusive-create logs. Actual final: **873 PASS**,14 pre-existing Torch deprecation warnings, **51.20s**. JUnit ID/verdict comparison against `/data/ycfeng/tmp/d18_profile_20261021/boundary_affected01.xml`: **851 baseline IDs preserved/PASS,22 added,0 missing,0 changed verdicts**. Prior affected checkpoint862 PASS overlaps. `git diff --check` PASS.

CPU tests include actual launcher execution with inert module doubles and fake credential strings; no StepMind client/auth environment is loaded or contacted.

## Sealed stage

Source commit: `74df0100ca0f39d6ff134d999909e863c3db6962`. Stage: `/data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01`. Manifest SHA256: `a1dc28ceca71235b000e7f7448464a5ff76d27e76d2d51ee63fe0811908b5312`.

Exact stage command:

```bash
env PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu TMPDIR=/data/ycfeng/tmp \
  PYTHONPATH=/data/ycfeng/Frontier/.worktrees/step5-profiling:/data/ycfeng/tmp/step5_profiling_20261020/extra_site \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python -B -m frontier.profiling.step5.job \
  --stage /data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01 \
  --run-id fixed_h200_d22_probe_20261021_01 \
  --dependency-site /data/ycfeng/tmp/d18_profile_20261021/h200_small_probe_stage02/offline_site \
  --fixed-workload-probe --preemptible
```

Exact frozen preflight command:

```bash
env PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu TMPDIR=/data/ycfeng/tmp \
  PYTHONPATH=/data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01:/data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01/offline_site \
  /data/ycfeng/envs/oneshot-opd-py312/bin/python -B \
  /data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01/tests/integration/step5_profiling/verify_fixed_probe.py \
  --stage /data/ycfeng/tmp/d22_timing_20261021/h200_full_probe_stage01 \
  --output /data/ycfeng/tmp/d22_timing_20261021/package_preflight01.json
```

Actual PASS:3407 payload hashes, exact inventory, workload checksums,18 event+4 GPU-active attention rows,16 total linear rows, both actual linear CLI dry runs, shell syntax and complete native probe policy. All525 tracked staged payloads match committed source; critical hashes are retained in `evidence/source_stage_audit01.json`. Native attention arguments, policy, full linear plan/workload and worker command exactly match original D18 after replacing run ID only. All13 staged linear protocol files are byte-identical; scoped timing function AST is unchanged.

Original `/data/ycfeng/tmp/d18_profile_20261021/h200_small_probe_stage02`: all3407 payload hashes and exact inventory PASS, manifest remains `42545748def4f40947037a61ca68541d4f3736cd5e8faaa60a8ec3698be91077`. Old receipt verdicts were not recomputed. New stage is preemptible; original stage remains non-preemptible historical evidence. Preflight/source audit JSONs are retained locally under this task's `evidence/`.

Source/docs whitespace check PASS excluding original raw RED XML. A first staged-wide `git diff --check` found four whitespace-only lines embedded in the original pytest failure text; raw XML bytes are intentionally preserved rather than rewritten. No product whitespace error exists.

No submission occurred. [handoff.md](handoff.md) records exact parent-only launch/judge commands, quotas, budget and limits. There is no independent review claim; this subtask forbids nested delegation. Native D22 acceptance remains pending the parent's fresh unminimized run.
