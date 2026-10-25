#!/usr/bin/python3
"""Run two synthetic cases once after primary source and command binding."""
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

CONTROL = Path(__file__).resolve()
OWNER = CONTROL.parent
ROOT = CONTROL.parents[8]
GDB_SOURCE = ROOT / 'tests/analysis/v4_trace_audit_native_stack.gdb'
INPUTS = OWNER.parent / 'primary/control_inputs.json'
MEMORY_BYTES = 32212254720
ALLOCATOR_KEYS = ('PYTHONMALLOC', 'PYTHONMALLOCSTATS', 'LD_PRELOAD', 'LD_LIBRARY_PATH',
                  'GLIBC_TUNABLES', 'MALLOC_CHECK_', 'MALLOC_PERTURB_', 'MALLOC_ARENA_MAX')
CAPTURE_COMMANDS = ['thread apply all bt 24', 'info sharedlibrary', 'info proc mappings']


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


def write_json(name, value):
    with (OWNER / name).open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def marker(stdout, stderr, prefix):
    rows = [line[len(prefix):] for line in (stdout + stderr).splitlines()
            if line.startswith(prefix)]
    assert len(rows) == 1, (prefix, len(rows))
    return json.loads(rows[0])


def run_case(name, body, parent, pins):
    source = ('import gc, json, os, signal, sys\nfrom pathlib import Path\n'
              + 'ALLOCATOR_KEYS = ' + repr(ALLOCATOR_KEYS) + '\n' + inspect.getsource(metadata)
              + "\nprint('V4_SYNTHETIC_META ' + json.dumps(metadata(), sort_keys=True), flush=True)\n"
              + body + '\n')
    command = ['/usr/bin/gdb', '--nx', '--nh', '--batch',
               '-iex', 'set debuginfod enabled off', '-iex', 'set auto-load off',
               '-x', str(GDB_SOURCE), '--args', '/usr/bin/python3', '-u', '-B', '-c', source]
    write_json(name + '.command.json', dict(argv=command, source_pins=pins,
               inferior_source_sha256=hashlib.sha256(source.encode()).hexdigest()))
    stdout_path, stderr_path = OWNER / (name + '.stdout'), OWNER / (name + '.stderr')
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    stdout, stderr = stdout_path.read_text(), stderr_path.read_text()
    result = dict(gdb_returncode=completed.returncode, verdict='FAIL')
    try:
        native = marker(stdout, stderr, 'V4_NATIVE_STATUS ')
        inferior = marker(stdout, stderr, 'V4_SYNTHETIC_META ')
        result.update(native_status=native, inferior_metadata=inferior)
        for key in ('cgroup', 'memory_mount_root', 'group', 'memory_limit_bytes',
                    'python_executable', 'python_version', 'allocator', 'gc_enabled', 'gc_thresholds'):
            assert inferior[key] == parent[key], (key, inferior[key], parent[key])
        assert inferior['memory_limit_bytes'] == MEMORY_BYTES, inferior
        assert inferior['pid'] > 0 and native['observed_pid'] == inferior['pid'], native
        assert native['run_error'] is None and native['capture_errors'] == {}, native
        if name == 'exit0':
            assert completed.returncode == 0, completed.returncode
            assert native['inferior_exited'] and native['inferior_exit_code'] == 0, native
            assert native['observed_stop_signal'] is None and native['stop_kind'] is None, native
            assert native['inferior_state'] == 'exited' and native['batch_quit_action'] == 'none', native
            assert native['capture_commands'] == [], native
        else:
            assert completed.returncode == 1, completed.returncode
            assert not native['inferior_exited'] and native['inferior_exit_code'] is None, native
            assert native['observed_stop_signal'] == 'SIGABRT' and native['stop_kind'] == 'SignalEvent', native
            assert native['inferior_state'] == 'stopped', native
            assert native['batch_quit_action'] == 'kill_stopped_inferior', native
            assert native['capture_commands'] == CAPTURE_COMMANDS, native
    except (AssertionError, KeyError, ValueError) as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
    else:
        result['verdict'] = 'PASS'
    write_json(name + '.result.json', result)
    return result


def main():
    raw_inputs = INPUTS.read_bytes()
    inputs = json.loads(raw_inputs)
    pins = {row['path']: row['sha256'] for row in inputs['source_pins']}
    assert len(inputs['source_pins']) == len(pins) == 2, 'Expected two distinct source pins'
    assert set(pins) == {str(CONTROL), str(GDB_SOURCE)}, pins
    for path, digest in pins.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    command = ['/usr/bin/python3', '-u', '-B', str(CONTROL)]
    assert inputs['command'] == command and sys.orig_argv == command, sys.orig_argv
    assert sys.version_info[:3] == (3, 10, 6), sys.version
    write_json('control_command.json', dict(command=command, source_pins=pins,
               inputs_path=str(INPUTS), inputs_sha256=hashlib.sha256(raw_inputs).hexdigest()))
    parent = metadata()
    assert parent['memory_limit_bytes'] == MEMORY_BYTES, parent
    cases = [run_case('exit0', 'sys.exit(0)', parent, pins),
             run_case('sigabrt', 'os.kill(os.getpid(), signal.SIGABRT)', parent, pins)]
    result = dict(verdict='PASS' if all(case['verdict'] == 'PASS' for case in cases) else 'FAIL',
                  controller_metadata=parent, cases=cases)
    write_json('control_result.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
