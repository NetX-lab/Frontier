# D22 independent timing reference: CPU and staging report

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Record real-seam RED->GREEN, affected regression and pending sealed-stage verification. |

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

Pending source commit and fresh-stage preflight. No submission is performed by this task. The parent-only handoff will record the immutable stage path, manifest hash, all payload/source identity checks, full18+4/native-policy dry run, both original linear dry runs, exact launch/judge commands and one-H200 runtime budget. Original stage02 will be rehashed without reading cloud or modifying any receipt.
