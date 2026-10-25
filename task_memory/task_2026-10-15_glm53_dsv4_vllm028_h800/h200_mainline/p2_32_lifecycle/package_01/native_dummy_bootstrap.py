"""Deploy only the previously accepted C26 model leaf before native imports."""
import hashlib
import json
import os
import sys
from importlib.machinery import PathFinder
from pathlib import Path

PACKAGE = Path('/usr/local/lib/python3.12/dist-packages/vllm')
V4_MODEL = 'models/deepseek_v4/nvidia/model.py'
FILES = (V4_MODEL,)

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
    paths = [config_path] + [Path(config[key]) for key in ('script', 'overlay_manifest')]
    require(all(path.is_absolute() for path in paths), 'Input paths must be absolute')
    overlay = json.loads(paths[2].read_text())
    staged = Path(overlay['staged_dir'])
    require(staged.is_absolute() and staged.is_dir()
            and all(path.parent == staged for path in paths), 'Staged paths mismatch')
    require(overlay['installed_package'] == str(package)
            and set(overlay['files']) == set(FILES), 'Only the accepted V4 model leaf is allowed')
    require(config['expected_source_hashes']['vllm/' + V4_MODEL]
            == overlay['files'][V4_MODEL]['sha256'], 'V4 diagnostic source mismatch')
    require(config['script'] in config['tool_hashes'], 'Diagnostic hash missing')
    for name, expected in config['tool_hashes'].items():
        verify_hash(Path(name), expected)
    writes = []
    for name in FILES:
        row, target = overlay['files'][name], package / name
        source = Path(row['source'])
        verify_hash(source, row['sha256'])
        require(target.parent.resolve().is_relative_to(package)
                and not target.is_symlink(), f'Target escapes package: {name}')
        require(sha256(target) == row['base_sha256'], f'Installed base mismatch: {name}')
        require(target.stat().st_mode & 0o222 and os.access(target, os.W_OK),
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
    env = dict(os.environ, PYTHONPATH=str(staged), PYTHONDONTWRITEBYTECODE='1')
    launch(sys.executable,
           [sys.executable, '-B', config['script'], '--config', str(config_path)], env)

if __name__ == '__main__':
    require(sys.flags.isolated and len(sys.argv) == 3 and sys.argv[1] == '--config',
            'Bootstrap requires python -I and --config ABSOLUTE_PATH')
    bootstrap(Path(sys.argv[2]))
