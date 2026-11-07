"""Poll one exact RJob and its local launcher until the launcher exits; read-only.

Usage (through monitor_job.sh, which sets the StepMind environment):
    monitor_job.sh <job name> <launcher pid> <launcher wrapper> <output dir> <interval s> <max hours>

Each poll runs read_job.py once and appends one summary line to monitor.jsonl; the full read is kept
as latest.json and also saved whenever the phase or replica phases change. The launcher is observed
through /proc only (no signal). After the launcher exits, one final read is saved as final.json.
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

JOB, PID, WRAPPER, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
INTERVAL_S, MAX_HOURS = float(sys.argv[5]), float(sys.argv[6])
READER = Path(__file__).resolve().parent / 'read_job.py'
STEPMIND_PYTHON = '/data/ycfeng/tmp/stepmind-env/bin/python'
READ_TIMEOUT_S = 60


def launcher_alive():
    try:
        argv = Path(f'/proc/{PID}/cmdline').read_bytes().decode(errors='replace').split('\0')
    except OSError:
        return False
    return WRAPPER in argv or any(arg.endswith('/launch.py') for arg in argv)


def read(path):
    try:
        done = subprocess.run([STEPMIND_PYTHON, str(READER), JOB, str(path)],
                              capture_output=True, text=True, timeout=READ_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None, f'timeout after {READ_TIMEOUT_S}s'
    if done.returncode != 0:
        return None, (done.stdout + done.stderr)[-2000:]
    return json.loads(path.read_text()), None


def summary(identity, error, alive):
    job = (identity or {}).get('job') or {}
    rows = ((identity or {}).get('replicas') or {}).get('rows') or []
    return {'observed_utc': datetime.now(timezone.utc).isoformat(), 'launcher_alive': alive,
            'phase': job.get('phase'), 'startTime': job.get('startTime'), 'finishTime': job.get('finishTime'),
            'queue_status': job.get('queue_status'), 'stop_annotations': job.get('stop_annotations'),
            'replica_phases': sorted(r.get('phase') or 'None' for r in rows),
            'read_error': error or (identity or {}).get('errors') or None,
            'replica_errors': ((identity or {}).get('replicas') or {}).get('errors') or None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + MAX_HOURS * 3600
    previous = None
    poll = 0
    while True:
        poll += 1
        alive = launcher_alive()
        identity, error = read(OUT / 'latest.json')
        line = summary(identity, error, alive)
        with open(OUT / 'monitor.jsonl', 'a') as f:
            f.write(json.dumps(line, sort_keys=True) + '\n')
        state = (line['phase'], tuple(line['replica_phases']))
        if identity and state != previous:
            (OUT / f'change_{poll:04d}.json').write_text(json.dumps(identity, indent=2, sort_keys=True) + '\n')
            previous = state
        if not alive:
            time.sleep(10)
            final, error = read(OUT / 'final.json')
            with open(OUT / 'monitor.jsonl', 'a') as f:
                f.write(json.dumps({**summary(final, error, launcher_alive()), 'final': True}, sort_keys=True) + '\n')
            return 0
        if time.time() >= deadline:
            with open(OUT / 'monitor.jsonl', 'a') as f:
                f.write(json.dumps({'observed_utc': datetime.now(timezone.utc).isoformat(),
                                    'monitor_deadline_reached': True}) + '\n')
            return 4
        time.sleep(INTERVAL_S)


if __name__ == '__main__':
    sys.exit(main())
