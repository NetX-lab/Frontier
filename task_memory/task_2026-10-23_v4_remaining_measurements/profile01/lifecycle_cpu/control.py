"""Reproduce retained shutdown evidence gaps through the existing CPU seam."""

import hashlib
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

from tests.unit.test_native_pd_worker import exercise, native_pd_worker


ROOT = Path(__file__).parent
SOURCE = Path(native_pd_worker.__file__)
SOURCE_SHA = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
REAL_WRITE = Path.write_text
REAL_GLOB = Path.glob


class Worker:
    def __getattr__(self, name):
        return getattr(native_pd_worker, name)

    def run(self, config):
        config['peer_wait_seconds'] = 60
        assert config['stop_timeout_s'] == 10
        return native_pd_worker.run(config)


clock = [0]
published = []


def delayed_marker(path, text, *args, **kwargs):
    if path.name == 'native_stopped.json' and path.parent.name == 'host0':
        clock[0] += 1
        if clock[0] < 20:
            return len(text)
        published.append(clock[0])
    return REAL_WRITE(path, text, *args, **kwargs)


with patch.object(native_pd_worker.time, 'monotonic', side_effect=lambda: clock[0]), \
        patch.object(Path, 'write_text', delayed_marker):
    delayed = exercise(Worker(), ROOT, 'canonical_clean_natural_headless_exit')
assert delayed['worker_returncode'] == delayed['native_server_returncode'] == 0
assert not delayed['shutdown_order']['head_stopped_before_headless_stop']
assert delayed['observation_errors'] == [
    'Head shutdown marker missing; native stop order unverified']
assert clock[0] == 10 and published == []

clock[0] = 0
enumeration = []


def stale_listing(path, pattern):
    if pattern == 'nodes/*/drained.json':
        markers = [path / 'nodes' / f'host{i}' / 'drained.json' for i in range(4)]
        enumeration.append(dict(at=clock[0], direct_paths_exist=all(p.is_file() for p in markers)))
        if clock[0] < 20:
            return iter(())
    return REAL_GLOB(path, pattern)


def advance(seconds):
    clock[0] += seconds


with patch.object(native_pd_worker.time, 'monotonic', side_effect=lambda: clock[0]), \
        patch.object(native_pd_worker.time, 'sleep', side_effect=advance), \
        patch.object(Path, 'glob', stale_listing):
    listing = exercise(Worker(), ROOT, 'canonical_clean')
assert listing['worker_returncode'] == listing['native_server_returncode'] == 0
assert not listing['observation_errors']
assert clock[0] == 20 and enumeration[0] == dict(at=0, direct_paths_exist=True)
assert all(row['direct_paths_exist'] for row in enumeration)
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
result = dict(status='PASS_REPRODUCED_CPU_CONTROLS', python=sys.version,
    platform=platform.platform(), worker_source=str(SOURCE), worker_sha256=SOURCE_SHA,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    marker_delay=dict(stop_timeout_s=10, peer_wait_seconds=60, marker_available_at=20,
        wait_returned_at=10, result=delayed),
    enumeration_delay=dict(visibility_at=20, head_wait_returned_at=clock[0],
        samples=enumeration, result=listing),
    limits=['CPU process/filesystem/time stand-ins, not GPU or cloud measurements.',
        'First control uses the existing naturally completed headless fixture.',
        'Second control demonstrates a possible enumeration delay; it does not identify the real cloud-delay cause.',
        'No source or configuration file was changed.'])
(ROOT / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'receipt': str(ROOT / 'receipt.json')}))
