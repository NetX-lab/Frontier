"""Run one CSA FULL cause analysis from the committed source and write its receipt."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028')
HERE = Path(__file__).resolve().parent
TASK = HERE.parent
PROBE = TASK / 'a6_repeatability_20261101'
RAW = Path('/data/ycfeng/tmp/glm53_dsv4/a6_repeatability_20261101_download/A6_REPEATABILITY/bridge/FULL_AND_PIECEWISE/raw')
PYTHON = '/data/ycfeng/envs/frontier-py310/bin/python'
ENVIRONMENT = {'PYTHONPATH': f'{REPO}:{REPO}/tests/analysis', 'PYTHONDONTWRITEBYTECODE': '1',
               'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'TMPDIR': '/data/ycfeng/tmp'}
ANALYSES = {
    'stream_overlap': ('tests/analysis/v4_attention_stream_overlap.py', 600, [
        '--package-plan', str(PROBE / 'package_plan.json'), '--case-mapping', str(PROBE / 'case_mapping.json'),
        '--raw', str(RAW / 'dsv4_raw_e0.jsonl'), str(RAW / 'dsv4_raw_e1.jsonl'),
        '--output', str(HERE / 'stream_overlap.json')]),
    'phase_heldout': ('tests/analysis/v4_attention_phase_heldout.py', 7200, [
        '--result', str(TASK / 'a6_codex_20261101/result.json'),
        '--scratch-dir', '/data/ycfeng/tmp/glm53_dsv4/a6_phase_heldout_scratch',
        '--output', str(HERE / 'phase_heldout.json')]),
}


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis', choices=sorted(ANALYSES))
    name = parser.parse_args().analysis
    script, seconds, arguments = ANALYSES[name]
    receipt_path, log_path = HERE / f'{name}_command.json', HERE / f'{name}.log'
    for path in (receipt_path, log_path):
        if path.exists():
            raise FileExistsError(path)
    dirty = [line for line in git('status', '--porcelain', '--', 'frontier', 'tests/analysis').splitlines() if line]
    command = ['env', *(f'{k}={v}' for k, v in ENVIRONMENT.items()), 'timeout', f'{seconds}s', PYTHON, script, *arguments]
    receipt = {'command': command, 'cwd': str(REPO), 'python': sys.version.split()[0],
               'source_commit': git('rev-parse', 'HEAD'), 'source_dirty_paths': dirty,
               'script_sha256': hashlib.sha256((REPO / script).read_bytes()).hexdigest(),
               'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    started = time.monotonic()
    with log_path.open('x') as log:
        exit_code = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                   env={**os.environ, **ENVIRONMENT}).returncode
    output = Path(arguments[arguments.index('--output') + 1])
    receipt.update(exit_code=exit_code, elapsed_seconds=time.monotonic() - started,
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   output=str(output), output_sha256=hashlib.sha256(output.read_bytes()).hexdigest()
                   if output.exists() else None)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('exit_code', 'elapsed_seconds', 'source_commit', 'source_dirty_paths')}))
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
