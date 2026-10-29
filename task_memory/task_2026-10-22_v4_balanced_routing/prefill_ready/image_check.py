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
    assert env['PYTHONPATH'] == str(stage) and Path.cwd() == stage
    assert not any((name.split('.')[0] in ('torch', 'vllm') for name in sys.modules))
    for (name, digest) in config['expected_source_hashes'].items():
        assert bootstrap.sha256(bootstrap.PACKAGE.parent / name) == digest, name
    assert len(config['expected_source_hashes']) == 60
    assert len(config['tool_hashes']) == 519
    assert len(overlay['binary_sha256']) == 20
    print(json.dumps({'status': 'PASS', 'python': sys.version.split()[0], 'source_pins': 60, 'tool_pins': 519, 'binary_pins': 20, 'torch_vllm_imported': False, 'cuda_initialized': False, 'launch_executable': executable, 'stage': str(stage)}))
bootstrap.bootstrap(stage / 'config.json', launch=inspect_launch)
