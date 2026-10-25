# The caller supplies --nx --nh --batch and disables debuginfod before file load.
set pagination off
set confirm off
set debuginfod enabled off
set auto-load off
set disable-randomization off
set startup-with-shell off
set print frame-arguments none
set print elements 16
set print raw-values on
set may-call-functions off
handle SIGABRT stop print nopass
handle SIGSEGV stop print nopass
python
import gdb
import json
from pathlib import Path

status = {
    'inferior_exited': False,
    'inferior_exit_code': None,
    'observed_stop_signal': None,
    'observed_pid': None,
    'stop_kind': None,
    'run_error': None,
    'capture_commands': [],
    'capture_errors': {},
}


def record_pid(event):
    status['observed_pid'] = event.inferior_thread.ptid[0]


def record_exit(event):
    status['inferior_exited'] = True
    if hasattr(event, 'exit_code'):
        status['inferior_exit_code'] = event.exit_code


def record_stop(event):
    status['observed_pid'] = gdb.selected_inferior().pid
    status['stop_kind'] = type(event).__name__
    if isinstance(event, gdb.SignalEvent):
        status['observed_stop_signal'] = event.stop_signal


gdb.events.new_thread.connect(record_pid)
gdb.events.exited.connect(record_exit)
gdb.events.stop.connect(record_stop)
try:
    gdb.execute('run')
except gdb.error as error:
    status['run_error'] = str(error)

if not status['inferior_exited'] and status['stop_kind'] is not None:
    for command in ('x/i $pc', 'info symbol $pc', 'thread apply all bt 24', 'info sharedlibrary', 'info proc mappings'):
        status['capture_commands'].append(command)
        try:
            gdb.execute(command)
        except gdb.error as error:
            status['capture_errors'][command] = str(error)

if not status['inferior_exited'] and status['observed_stop_signal'] is not None:
    proc = Path('/proc') / str(status['observed_pid'])
    status['linux_limits'] = (proc / 'limits').read_text()
    status['linux_cgroup'] = (proc / 'cgroup').read_text()

status['inferior_state'] = ('exited' if status['inferior_exited'] else
                            'stopped' if status['stop_kind'] is not None else 'unobserved')
status['batch_quit_action'] = ('none' if status['inferior_exited'] else
                              'kill_stopped_inferior' if status['stop_kind'] is not None
                              else 'kill_if_alive')
success = (status['inferior_exited'] and status['inferior_exit_code'] == 0
           and status['observed_stop_signal'] is None and status['run_error'] is None)
gdb.write('V4_PREFILL_FAULT_STATUS ' + json.dumps(status, sort_keys=True) + '\n')
gdb.execute('quit %d' % (0 if success else 1))
end
