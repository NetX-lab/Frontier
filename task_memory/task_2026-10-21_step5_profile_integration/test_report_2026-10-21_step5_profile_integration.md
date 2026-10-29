# Step profiling integration verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Archive affected tests, complete fidelity and corrected comparison invocation. |

## Scope and result

Source f0ab2eb merges b0018c4 and5fa1926,37 files. Independent source review has zero actionable Standards/Spec findings. **Affected tests787 PASS; legacy fidelity77/77 identical; cache filenames identical.** Native timing, full canonical coverage, typed admission and the existing CPU performance budget remain separate and unverified.

## Affected tests

Environment: Linux, Python3.12.13 `/data/ycfeng/envs/oneshot-opd-py312/bin/python`, Torch2.13.0+cu130, no conda/dependency changes. Workdir `/data/ycfeng/Frontier/.worktrees/step5-profile-integration`.

```bash
env PYTHONDONTWRITEBYTECODE=1 VLLM_TARGET_DEVICE=cpu TMPDIR=/data/ycfeng/tmp WANDB_DISABLED=true VIDUR_DISABLE_WANDB=1 PYTHONPATH="$PWD:/data/ycfeng/tmp/step5_profiling_20261020/extra_site" /data/ycfeng/envs/oneshot-opd-py312/bin/python -B - <<'PY'
import json,pathlib,sys,types,pytest
receipt=pathlib.Path('/data/ycfeng/Frontier/.worktrees/step5-profiling/task_memory/task_2026-10-15_step5_preview_h800/d18_profile_20261021/evidence/profile_commands.json')
files=json.loads(receipt.read_text())['files']+['test_step5_fixed_probe']
package=types.ModuleType('tests');package.__path__=[str(pathlib.Path.cwd()/'tests')];sys.modules['tests']=package
raise SystemExit(pytest.main([*(f'tests/unit/{name}.py' for name in files),'-q','-p','no:cacheprovider','--basetemp=/data/ycfeng/tmp/d18_profile_integration_tests01']))
PY
```

Expected: all affected cases pass. Actual:787 passed,14 existing Torch deprecation warnings,54.87s. Log `/data/ycfeng/tmp/d18_profile_integration_tests01.log`. The namespace selection prevents an unrelated installed `tests` package from shadowing the repository tests.

## Fidelity

Python3.10.6 `/data/ycfeng/envs/frontier-py310/bin/python`. Baseline is the completed clean b1f032d run at `/data/ycfeng/tmp/step5_comm_fidelity_20261021/candidate`; b0018c4 changes only documentation, verified by an empty `git diff b1f032d b0018c4 -- frontier data tests`. No baseline artifacts or manifests modified.

```bash
PYTHONPATH=$PWD TMPDIR=/data/ycfeng/tmp/opencode-v2-tmp/opencode /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py run --repo-root /data/ycfeng/Frontier/.worktrees/step5-profile-integration --label candidate --output-root /data/ycfeng/tmp/step5_profile_fidelity_20261021 --python-bin /data/ycfeng/envs/frontier-py310/bin/python --jobs 4 --clean-cache --continue-on-failure
PYTHONPATH=$PWD /data/ycfeng/envs/frontier-py310/bin/python -B tests/e2e/refactor_fidelity/run_matrix.py compare --output-root /data/ycfeng/tmp/step5_profile_fidelity_compare02 --baseline-label /data/ycfeng/tmp/step5_comm_fidelity_20261021/candidate --candidate-label /data/ycfeng/tmp/step5_profile_fidelity_20261021/candidate
```

Both full populations contain77 cases, clean caches and identical case definitions. Final comparison exit0:77 identical, zero mismatches/failures/missing evidence, zero differing cache filenames. Report `/data/ycfeng/tmp/step5_profile_fidelity_compare02/comparison.json`.

Initial invocation through a relocated baseline symlink returned77 mismatches. Every difference was exactly `config.json:metrics_config/output_dir`: the unchanged comparator's substitutions used the symlink label rather than the original absolute output root. The original failed report remains `/data/ycfeng/tmp/step5_profile_fidelity_20261021/comparison.json`. Correcting the invocation to actual absolute label paths resolved all77 differences without changing source, artifacts, tolerances or comparison rules. No simulation rerun was necessary.
