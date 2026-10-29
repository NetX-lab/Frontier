"""Regression checks for non-blocking RDMA command timeout observations."""
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
from unittest.mock import patch

PROBE = Path(__file__).with_name('inspect_rdma.py')
spec = importlib.util.spec_from_file_location('inspect_rdma', PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_command_timeout_is_an_observation():
    error = subprocess.TimeoutExpired(['ibv_devinfo'], 30, output=b'partial output', stderr=b'partial error')
    with patch.object(probe.shutil, 'which', return_value='/usr/bin/ibv_devinfo'), \
            patch.object(probe.subprocess, 'run', side_effect=error):
        record = probe.command(['ibv_devinfo'])
    assert record['timeout_seconds'] == 30
    assert record['stdout'] == 'partial output'
    assert record['stderr'] == 'partial error'


def test_main_writes_json_after_command_timeout(tmp_path):
    output = tmp_path / 'rdma.json'
    with patch.object(sys, 'argv', [str(PROBE), '--output', str(output)]), \
            patch('subprocess.run', side_effect=subprocess.TimeoutExpired(['probe'], 30)), \
            patch('ctypes.util.find_library', return_value=None):
        runpy.run_path(str(PROBE), run_name='__main__')
    record = json.loads(output.read_text())
    assert any(row.get('timeout_seconds') == 30 for row in record['commands'])
    assert 'hostname' in record
