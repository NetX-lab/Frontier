import gc, json, os, signal, sys
from pathlib import Path
ALLOCATOR_KEYS = ('PYTHONMALLOC', 'PYTHONMALLOCSTATS', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'GLIBC_TUNABLES', 'MALLOC_CHECK_', 'MALLOC_PERTURB_', 'MALLOC_ARENA_MAX')
def metadata():
    cgroup = Path('/proc/self/cgroup').read_text()
    memory_path = next(line.split(':', 2)[2] for line in cgroup.splitlines() if 'memory' in line.split(':', 2)[1].split(','))
    mount = next(line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines() if ' - cgroup ' in line and 'memory' in line.split(' - ', 1)[1].split()[2].split(','))
    group = Path(mount[4]) / Path(memory_path).relative_to(mount[3])
    return dict(pid=os.getpid(), cgroup=cgroup, memory_mount_root=mount[3], group=str(group),
                memory_limit_bytes=int((group / 'memory.limit_in_bytes').read_text()),
                python_executable=sys.executable, python_version=sys.version,
                allocator={name: os.environ.get(name) for name in ALLOCATOR_KEYS},
                gc_enabled=gc.isenabled(), gc_thresholds=list(gc.get_threshold()))

print('V4_SYNTHETIC_META ' + json.dumps(metadata(), sort_keys=True), flush=True)
sys.exit(0)
