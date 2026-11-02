#!/usr/bin/env python3
"""Observe the newly failed prefill control's native fault once, without retry."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, payload):
    with Path(path).open('x') as stream:
        json.dump(payload, stream, indent=2)
        stream.write('\n')


def main():
    allocation = Path(sys.argv[1])
    plan = json.loads(allocation.read_text())
    prefix = Path(plan['prefix'])
    assert not prefix.parent.exists(), 'Native observation directory already exists'
    assert {key: os.environ.get(key) for key in plan['selected_environment']} == plan['selected_environment']
    for pin in plan['pins']:
        assert sha(pin['path']) == pin['sha256'], pin['path']
    prefix.parent.mkdir()
    receipt = dict(status='STARTING', allocation_sha256=sha(allocation), command=plan['command'],
                   started_utc=datetime.now(timezone.utc).isoformat(), used=1, ceiling=1,
                   python=sys.version, environment=plan['selected_environment'], automatic_retry=False)
    write(prefix.with_suffix('.started.json'), receipt)
    started = time.monotonic()
    with prefix.with_suffix('.stdout').open('x') as output, prefix.with_suffix('.stderr').open('x') as errors:
        result = subprocess.run(plan['command'], stdout=output, stderr=errors, cwd=plan['workspace'])
    receipt.update(returncode=result.returncode, enclosing_seconds=time.monotonic() - started,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    native = [line.split('V4_PREFILL_FAULT_STATUS ', 1)[1]
              for line in prefix.with_suffix('.stdout').read_text().splitlines()
              if line.startswith('V4_PREFILL_FAULT_STATUS ')]
    assert len(native) <= 1, 'More than one native status marker'
    receipt['native_status'] = json.loads(native[0]) if native else None
    cgroup = prefix.with_suffix('.cgroup.json')
    receipt['child_cgroup'] = json.loads(cgroup.read_text()) if cgroup.exists() else None
    if receipt['child_cgroup']:
        assert receipt['child_cgroup']['limits']['memory.limit_in_bytes'] == '32212254720'
    receipt['frozen_pins_unchanged'] = {pin['path']: sha(pin['path']) == pin['sha256'] for pin in plan['pins']}
    receipt['status'] = ('CENSORED' if result.returncode == 124 else
                         'NATIVE_STOP' if receipt['native_status'] and receipt['native_status']['observed_stop_signal'] else
                         'INFERIOR_EXIT' if receipt['native_status'] and receipt['native_status']['inferior_exited'] else
                         'FAILED_UNOBSERVED')
    receipt['closed_outputs'] = [dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
                                 for path in sorted(prefix.parent.glob(prefix.name + '.*'))]
    receipt['limits'] = ['Native detection site only; corruption writer UNKNOWN',
                         'Does not replace failed control or imply allocator cure/R5/cost/reference acceptance',
                         'Compressed index is hash-only; no EOF/row acceptance by this controller']
    write(prefix.with_suffix('.receipt.json'), receipt)
    print(json.dumps(receipt, indent=2), flush=True)
    return 0 if result.returncode == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
