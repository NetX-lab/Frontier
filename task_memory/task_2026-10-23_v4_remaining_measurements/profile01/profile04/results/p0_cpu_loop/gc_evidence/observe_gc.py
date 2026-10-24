"""Observe generation-two collections without changing the frozen trace audit."""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

SOURCE = Path('/data/ycfeng/tmp/glm53_dsv4/p0_audit_cpu_20261024/stage_audit.py')
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == 'f7f1253ce9ebae5dcb3b60e2aeb97d7f17f58f34a7d68af56611918f809b9fe4'
spec = importlib.util.spec_from_file_location('p0_stage_audit', SOURCE)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def run(relative, output):
    prefix = Path(output)
    descriptor = os.open(str(prefix) + '.gc.jsonl', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    cgroup = Path('/proc/self/cgroup').read_text()
    memory_path = next(line.split(':', 2)[2] for line in cgroup.splitlines() if 'memory' in line.split(':', 2)[1].split(','))
    mount = next(line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines() if ' - cgroup ' in line and 'memory' in line.split(' - ', 1)[1].split()[2].split(','))
    group = Path(mount[4]) / Path(memory_path).relative_to(mount[3])
    limits = {name: (group / name).read_text().strip() for name in ['memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.failcnt']}
    assert limits['memory.limit_in_bytes'] == '32212254720', limits
    prefix.with_suffix('.cgroup.json').write_text(json.dumps(dict(cgroup=cgroup, memory_mount_root=mount[3], group=str(group), limits=limits), indent=2) + '\n')

    def collection(phase, info):
        if info['generation'] == 2:
            row = dict(t=time.monotonic() - stage.T0, phase=phase, generation=2, collected=info.get('collected'), uncollectable=info.get('uncollectable'))
            os.write(descriptor, (json.dumps(row) + '\n').encode())

    gc.callbacks.append(collection)
    try:
        stage.run(relative, prefix.with_suffix('.json'), prefix.with_suffix('.obs.jsonl'), prefix.with_suffix('.hb.log'))
    finally:
        gc.callbacks.remove(collection)
        os.close(descriptor)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
