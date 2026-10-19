"""Apply the approved overlay before importing the installed native runtime."""
import hashlib
import json
import os
import sys
from importlib.machinery import PathFinder
from pathlib import Path

PACKAGE = Path('/usr/local/lib/python3.12/dist-packages/vllm')
FILES = ('v1/frontier_trace.py', 'v1/worker/gpu_worker.py',
    'v1/worker/gpu_model_runner.py', 'v1/engine/core.py', 'v1/core/sched/output.py',
    'compilation/cuda_graph.py')
V4_MODEL = 'models/deepseek_v4/nvidia/model.py'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def verify_hash(path, expected):
    require(path.is_absolute() and sha256(path) == expected, f'Hash mismatch: {path}')

def bootstrap(config_path, *, package=PACKAGE, launch=os.execve):
    require(not any(n.split('.')[0] in ('torch', 'vllm') for n in sys.modules),
            'Torch or vLLM already loaded')
    spec = PathFinder.find_spec('vllm', sys.path)
    require(spec is not None and spec.origin == str(package / '__init__.py')
            and package.resolve() == package, 'Installed vLLM origin mismatch')
    config = json.loads(config_path.read_text())
    paths = [config_path] + [Path(config[key]) for key in
                            ('script', 'overlay_manifest', 'observer_manifest')]
    require(all(path.is_absolute() for path in paths), 'Input paths must be absolute')
    overlay = json.loads(paths[2].read_text())
    observer = json.loads(paths[3].read_text())
    staged = Path(overlay['staged_dir'])
    require(staged.is_absolute() and staged.is_dir()
            and all(path.parent == staged for path in paths), 'Staged paths mismatch')
    files = FILES + ((V4_MODEL,) if V4_MODEL in overlay['files'] else ())
    require(overlay['installed_package'] == str(package)
            and set(overlay['files']) == set(files),
            'Six approved files required; only the pinned V4 model may be added')
    if V4_MODEL in overlay['files']:
        require(config['expected_source_hashes']['vllm/' + V4_MODEL]
                == overlay['files'][V4_MODEL]['sha256'], 'V4 diagnostic source mismatch')
    fields = ('run_id', 'source_sha', 'instrumentation_sha', 'config_id',
              'replica_id', 'role', 'expected_ranks', 'output_dir')
    require(set(observer) == set(fields), 'Observer identity fields mismatch')
    require(all(type(observer[k]) is str and observer[k]
                for k in fields if k != 'expected_ranks'), 'Nonempty strings required')
    require(all(len(observer[k]) == 40 and set(observer[k]) <= set('0123456789abcdef')
                for k in ('source_sha', 'instrumentation_sha')), 'Malformed SHA')
    ranks = observer['expected_ranks']
    require(type(ranks) is list and ranks and len(set(ranks)) == len(ranks)
            and all(type(r) is int and r >= 0 for r in ranks), 'Malformed ranks')
    output = Path(observer['output_dir'])
    require(output.is_absolute()
            and output.resolve().is_relative_to('/mnt/codesign-exp/ycfeng'),
            'Observer output outside personal cloud root')
    observer.update(mode='light', capacity_bytes=4194304,
                    startup_memory={'records': 12, 'bytes': 65536})
    require(config['script'] in config['tool_hashes'], 'Diagnostic hash missing')
    for name, expected in config['tool_hashes'].items():
        verify_hash(Path(name), expected)
    writes = []
    for name in files:
        row, target = overlay['files'][name], package / name
        source = Path(row['source'])
        verify_hash(source, row['sha256'])
        require(target.parent.resolve().is_relative_to(package)
                and not target.is_symlink(), f'Target escapes package: {name}')
        require((not target.exists() and row['base_sha256'] is None)
                if name == FILES[0] else sha256(target) == row['base_sha256'],
                f'Installed base mismatch: {name}')
        writable = target if target.exists() else target.parent
        require(writable.stat().st_mode & 0o222 and os.access(writable, os.W_OK),
                f'Unwritable target: {name}')
        writes.append((target, source.read_bytes(), row['sha256']))
    binaries = overlay['binary_sha256']
    required = {str(package / name) for name in
                ('_C_stable_libtorch.abi3.so', '_moe_C_stable_libtorch.abi3.so')}
    require(required <= binaries.keys()
            and {str(p) for p in package.rglob('*.so')} <= binaries.keys(),
            'Installed binary manifest incomplete')
    for name, expected in binaries.items():
        verify_hash(Path(name), expected)
    for target, content, expected in writes:
        target.write_bytes(content)
        verify_hash(target, expected)
    for name, expected in binaries.items():
        verify_hash(Path(name), expected)
    os.chdir(staged)
    env = dict(os.environ, PYTHONPATH=str(staged), PYTHONDONTWRITEBYTECODE='1',
               VLLM_FRONTIER_OBSERVER=json.dumps(observer))
    launch(sys.executable,
           [sys.executable, '-B', config['script'], '--config', str(config_path)], env)

if __name__ == '__main__':
    require(sys.flags.isolated and len(sys.argv) == 3 and sys.argv[1] == '--config',
            'Bootstrap requires python -I and --config ABSOLUTE_PATH')
    bootstrap(Path(sys.argv[2]))
