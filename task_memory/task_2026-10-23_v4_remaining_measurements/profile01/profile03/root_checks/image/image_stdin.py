"""Verify the accepted bootstrap in the locked CPU-only container."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
stage = Path(sys.argv[1])
config = json.loads((stage / 'config.json').read_text())
overlay = json.loads((stage / 'overlay.json').read_text())
spec = importlib.util.spec_from_file_location('prefill_bootstrap', stage / 'native_dummy_bootstrap.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)

def inspect_launch(executable, argv, env):
    assert executable == sys.executable
    assert argv == [sys.executable, '-B', config['script'], '--config', str(stage / 'config.json')]
    assert env['PYTHONPATH'] == bootstrap.os.pathsep.join((str(stage), str(stage / 'tests/comparison/dp_placement_pp'))) and Path.cwd() == stage
    assert not any((name.split('.')[0] in ('torch', 'vllm') for name in sys.modules))
    for (name, digest) in config['expected_source_hashes'].items():
        assert bootstrap.sha256(bootstrap.PACKAGE.parent / name) == digest, name
    assert len(config['expected_source_hashes']) == 87
    assert len(config['tool_hashes']) == 23
    assert len(overlay['binary_sha256']) == 20
    print(json.dumps({'status': 'PASS', 'python': sys.version.split()[0], 'source_pins': 87, 'tool_pins': 23, 'binary_pins': 20, 'torch_vllm_imported': False, 'cuda_initialized': False, 'launch_executable': executable, 'stage': str(stage)}))
'Additional callback for the existing locked-image source/binary verifier.'
import subprocess
image_launch_check = inspect_launch

def inspect_launch(executable, argv, env):
    child = "\nimport hashlib, json, sys\nfrom pathlib import Path\npackage = Path(sys.argv[1]).resolve()\nassert not any(name in sys.modules for name in ('native_pd_worker', 'pd_replay', 'vllm_replay'))\nsys.path.insert(0, str(package))\nimport native_pd_worker\nimport vllm_replay\nassert Path(native_pd_worker.__file__).resolve() == package / 'native_pd_worker.py'\nassert Path(vllm_replay.__file__).resolve() == package / 'tests/comparison/dp_placement_pp/vllm_replay.py'\nprint(json.dumps(dict(status='PASS', kind='bootstrap_fresh_child_entry',\n    worker_sha256=hashlib.sha256(Path(native_pd_worker.__file__).read_bytes()).hexdigest(),\n    replay_sha256=hashlib.sha256(Path(vllm_replay.__file__).read_bytes()).hexdigest(),\n    torch_imported='torch' in sys.modules, vllm_imported='vllm' in sys.modules,\n    preloaded_dependencies=False, worker_main_executed=False, platform_operations=0)))\n"
    result = subprocess.run([executable, '-B', '-c', child, str(stage)], env=env, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError('Fresh bootstrap child failed: ' + result.stderr)
    receipt = json.loads(result.stdout)
    assert receipt['status'] == 'PASS'
    assert not receipt['torch_imported'] and (not receipt['vllm_imported'])
    print(json.dumps(receipt), flush=True)
    image_launch_check(executable, argv, env)
bootstrap.bootstrap(stage / 'config.json', launch=inspect_launch)
