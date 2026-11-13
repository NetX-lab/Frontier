"""CPU-only checks for the future two-node capacity lifecycle, never a GPU run."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import os
PATH = Path(os.environ.get('STEP5_TP16_TEST_SOURCE_ROOT', ROOT))/'role_capacity_supervision.py'
spec = importlib.util.spec_from_file_location('capacity_supervision', PATH)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


@pytest.fixture
def root(tmp_path):
    return tmp_path


@pytest.fixture(autouse=True)
def fake_supervisor_identity(root, monkeypatch):
    original = module.node_identity
    def identify(env, boot_path):
        node = original(env, boot_path)
        return dict(node, pid=1000+node['node_rank'])
    monkeypatch.setattr(module, 'node_identity', identify)


def environment(rank):
    return dict(NODE_RANK=str(rank), NODE_COUNT='2', PROC_PER_NODE='8', MASTER_ADDR='10.1.2.3')


def identity(root, rank):
    boot = root/f'boot_{rank}'
    if not boot.exists(): boot.write_text(f'boot-{rank}')
    node = module.node_identity(environment(rank), boot_path=boot)
    return dict(node, pid=1000+rank)


@pytest.mark.parametrize('key,value', [('MASTER_ADDR', '127.0.0.1'), ('MASTER_ADDR', '0.0.0.0'),
    ('MASTER_ADDR', '::1'), ('MASTER_ADDR', '::'), ('MASTER_ADDR', '::ffff:127.0.0.1'),
    ('MASTER_PORT', '0'), ('MASTER_PORT', '65536'),
    ('MASTER_PORT', 'bad'), ('MASTER_PORT', '-1'), ('NODE_COUNT', '0'), ('NODE_COUNT', '1'), ('NODE_COUNT', '3'),
    ('PROC_PER_NODE', '0'), ('PROC_PER_NODE', '4'), ('PROC_PER_NODE', '16'), ('NODE_RANK', '-1'), ('NODE_RANK', '2'),
    ('MASTER_ADDR', '127.3.2.1'), ('MASTER_ADDR', '::ffff:0.0.0.0')])
def test_invalid_endpoint_or_allocation_rejected(root, key, value):
    boot = root/'boot'; boot.write_text('boot')
    env = environment(0); env[key] = value
    with pytest.raises(ValueError): module.node_identity(env, boot_path=boot)


def test_default_port_and_exclusive_markers(root, monkeypatch):
    node = identity(root, 0)
    assert node['endpoint'] == dict(address='10.1.2.3', port=5678, port_source='platform_default')
    row = module.publish_marker(root, 'g1', node, 'prepared', 'PASS', 'ready', clock=lambda: 2)
    assert row['time_ns'] == 2000000000 and row['boot_id'] == 'boot-0'
    env = dict(environment(0), MASTER_PORT='6000')
    assert module.node_identity(env, root/'boot_0')['endpoint']['port_source'] == 'environment'
    original = (root/'node_0_prepared.json').read_bytes()
    with pytest.raises(FileExistsError):
        module.publish_marker(root, 'g1', node, 'prepared', 'FAIL', 'duplicate')
    assert (root/'node_0_prepared.json').read_bytes() == original
    def full_disk(*args): raise OSError('disk full')
    monkeypatch.setattr(module.os, 'fsync', full_disk)
    with pytest.raises(OSError, match='disk full'):
        module.publish_marker(root, 'g1', node, 'completion', 'FAIL', 'write failed')
    assert not (root/'node_0_completion.json').exists()
    assert not list(root.glob('.*.json'))
    with pytest.raises(ValueError):
        module.publish_marker(root, 'g1', identity(root, 1), 'workload_terminal', 'PASS', 'wrong rank')

class Clock:
    def __init__(self): self.now, self.events = 0, []
    def __call__(self): return self.now
    def sleep(self, seconds):
        self.now += seconds
        for at, call in list(self.events):
            if self.now >= at:
                self.events.remove((at, call)); call()

class Socket:
    def __init__(self, calls): self.calls = calls
    def bind(self, endpoint): self.calls.append(('bind', endpoint))
    def listen(self, backlog): self.calls.append('listen')
    def settimeout(self, seconds): assert seconds > 0
    def accept(self): self.calls.append('accept'); return Socket(self.calls), ('peer', 1234)
    def recv(self, count): return b''
    def connect(self, endpoint): self.calls.append(('connect', endpoint))
    def close(self): self.calls.append('connection_close')

def prepared(root, rank, generation='g1', **changes):
    node = identity(root, rank)
    fields = dict(endpoint=node['endpoint'], engine_settings={'tp': 16}); fields.update(changes)
    return module.publish_marker(root, generation, node, 'prepared', 'PASS', 'ready', **fields)


@pytest.mark.parametrize('rank', [0, 1])
def test_preparation_handshake_precedes_native_start(root, rank):
    clock, calls = Clock(), []
    clock.events.append((1, lambda: prepared(root, 1-rank)))
    exchange = module.preparation_exchange(root, 'g1', identity(root, rank), {'tp': 16}, 5,
        clock=clock, sleep=clock.sleep, socket_factory=lambda *args: Socket(calls))
    assert exchange['result'] == 'PASS' and exchange['peer']['node_rank'] == 1-rank and clock() == 1
    assert ('accept' in calls) if rank == 0 else calls.count(('connect', ('10.1.2.3', 5678))) == 1
    assert 'TCP' in module.HANDSHAKE_SCOPE and 'not NCCL' in module.HANDSHAKE_SCOPE


@pytest.mark.parametrize('change', ['boot', 'rank', 'endpoint', 'settings', 'generation'])
def test_peer_preparation_must_match(root, change):
    prepared(root, 1)
    path = root/'node_1_prepared.json'; row = json.loads(path.read_text())
    if change == 'boot': row['boot_id'] = 'boot-0'
    elif change == 'rank': row['node_rank'] = 0
    elif change == 'endpoint': row['endpoint']['port'] = 6000
    elif change == 'settings': row['engine_settings']['tp'] = 8
    else: row['generation'] = 'old'
    path.write_text(json.dumps(row))
    clock = Clock()
    row = module.preparation_exchange(root, 'g1', identity(root, 0), {'tp': 16}, 5,
        clock=clock, sleep=clock.sleep, socket_factory=lambda *args: Socket([]))
    assert row['result'] == 'FAIL'


def test_prepared_deadline_is_finite(root):
    clock = Clock()
    row = module.preparation_exchange(root, 'g1', identity(root, 1), {'tp': 16}, 3,
        clock=clock, sleep=clock.sleep, socket_factory=lambda *args: Socket([]))
    assert row['result'] == 'FAIL' and row['timeout'] and clock() == 3


def worker_stat(root, start):
    path = root/'proc/99/stat'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('99 (native worker) S '+'0 '*18+str(start))

@pytest.mark.parametrize('case,verdict', [('gone', 'PASS'), ('live', 'FAIL'), ('reuse', 'PASS'),
    ('query_exit', 'UNVERIFIED'), ('query_exception', 'UNVERIFIED'), ('gpu', 'FAIL'),
    ('stat_error', 'UNVERIFIED'), ('malformed_query', 'UNVERIFIED'), ('expired', 'UNVERIFIED'),
    ('query_timeout', 'UNVERIFIED')])
def test_residual_worker_identity_and_compute_query(root, case, verdict):
    from subprocess import TimeoutExpired
    from types import SimpleNamespace
    clock = Clock(); uuids = [f'GPU-{index}' for index in range(8)]
    if case in ('live', 'reuse', 'stat_error'):
        worker_stat(root, 10 if case != 'reuse' else 20)
        if case == 'stat_error': (root/'proc/99/stat').write_text('unreadable stat')
    def run(argv, **kwargs):
        assert argv == module.COMPUTE_QUERY and 0 < kwargs['timeout'] <= 3
        if case == 'query_exception': raise OSError('query unavailable')
        if case == 'query_timeout': raise TimeoutExpired(argv, kwargs['timeout'], output='partial', stderr='late')
        return SimpleNamespace(returncode=1 if case == 'query_exit' else 0, stderr='query stderr',
            stdout='GPU-0, 88, native, 0\n' if case == 'gpu' else 'broken' if case == 'malformed_query'
                   else 'GPU-unallocated, 88, other, 100\n')
    row = module.verify_residual([(99, 10)], uuids, 0 if case == 'expired' else 3, proc_root=root/'proc',
                                 clock=clock, sleep=clock.sleep, run=run)
    assert row['verdict'] == verdict and row['query']['command'] == module.COMPUTE_QUERY
    assert row['workers'][0]['exit_code'] == 'not_waitable_by_supervisor'
    if case == 'query_timeout': assert row['query']['stdout'] == 'partial' and row['query']['stderr'] == 'late'
    if case in ('live', 'gpu'): assert clock() == 3
    if case == 'stat_error': assert row['workers'][0]['gone'] is None

class Child:
    def __init__(self, clock, exit_at=3, code=0):
        self.clock, self.exit_at, self.code, self.pid, self.signals = clock, exit_at, code, 42, []
    def poll(self): return self.code if self.clock() >= self.exit_at else None
    def send_signal(self, number): self.signals.append((self.clock(), number)); self.exit_at = self.clock()+1

def finish(root, rank, clock, result='PASS'):
    module.publish_marker(root, 'g1', identity(root, rank), 'completion', result, 'finished', clock=clock,
                          final_status=dict(result=result))


def supervise(root, rank, clock, child):
    from types import SimpleNamespace
    prepared(root, 1-rank)
    argv = ['native', 'LLM'] if rank == 0 else ['vllm', 'serve', '--headless', '--node-rank', '1']
    def popen(arguments):
        assert arguments == argv and (root/f'node_{rank}_prepared.json').exists()
        return child
    if rank == 1:
        native_started(root, clock)
    return module.supervise(root, 'g1', environment(rank), {'tp': 16}, argv, [(99, 10)],
        [f'GPU-{i}' for i in range(8)], boot_path=root/f'boot_{rank}', proc_root=root/'proc',
        clock=clock, sleep=clock.sleep, popen=popen,
        run=lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''),
        socket_factory=lambda *args: Socket([]))


@pytest.mark.parametrize('native_code', [0, 7])
def test_node0_missing_terminal_is_failed_and_peer_still_waited(root, native_code):
    clock = Clock(); identity(root, 0)
    clock.events.append((5, lambda: finish(root, 1, clock)))
    row = supervise(root, 0, clock, Child(clock, code=native_code))
    terminal = module.read_marker(root, 'g1', 0, 'workload_terminal')
    assert terminal['result'] == 'FAIL' and row['raw_exit_code'] == native_code
    assert row['return_code'] == 1 and clock() == 5
    own = module.read_marker(root, 'g1', 0, 'completion')
    assert own['time_ns'] < row['peer_completion']['time_ns']


def test_node1_exit_zero_before_terminal_is_premature_failure(root):
    clock = Clock(); identity(root, 1); child = Child(clock, exit_at=2)
    clock.events.append((4, lambda: finish(root, 0, clock)))
    row = supervise(root, 1, clock, child)
    assert row['return_code'] == 1 and row['raw_exit_code'] == 0 and child.signals == []
    assert module.read_marker(root, 'g1', 1, 'premature_exit')['time_ns'] == 2000000000


@pytest.mark.parametrize('missing_peer', [False, True])
def test_node1_repeated_terminal_sends_one_sigterm_and_completion_precedes_wait(root, missing_peer):
    import signal
    clock = Clock(); identity(root, 1); child = Child(clock, exit_at=9999, code=-signal.SIGTERM)
    def shutdown(number): child.signals.append((clock(), number)); child.exit_at = clock()+3
    child.send_signal = shutdown
    def terminal():
        workload_terminal(root, clock)
    clock.events.append((2, terminal))
    if not missing_peer: clock.events.append((7, lambda: finish(root, 0, clock)))
    row = supervise(root, 1, clock, child)
    assert child.signals == [(2, signal.SIGTERM)] and row['raw_signal'] == signal.SIGTERM
    assert row['return_code'] == int(missing_peer) and row['timeouts']['peer_completion'] == missing_peer
    assert module.read_marker(root, 'g1', 1, 'completion')['time_ns'] == 5000000000
    assert clock() == (2400 if missing_peer else 7)


@pytest.mark.parametrize('mode', ['normal', 'native_failure', 'peer_premature'])
def test_two_node_sequence(root, mode):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    import signal
    clocks = [Clock(), Clock()]; connected = Event(); children = [Child(clocks[0]), Child(clocks[1], 9999)]
    for rank in (0, 1): identity(root, rank)
    children[0].code = 9 if mode == 'native_failure' else 0
    if mode == 'peer_premature': children[1].exit_at = 1
    def terminal():
        workload_terminal(root, clocks[0])
    if mode == 'normal': clocks[0].events.append((2, terminal))
    class Exchange(Socket):
        def accept(self): assert connected.wait(2); return super().accept()
        def connect(self, endpoint): super().connect(endpoint); connected.set()
    def node(rank):
        from types import SimpleNamespace
        clock = clocks[rank]
        def sleep(seconds): clock.sleep(seconds); Event().wait(0.001)
        return module.supervise(root, 'g1', environment(rank), {'tp': 16}, ['native', str(rank)], [(99, 10)],
            [f'GPU-{i}' for i in range(8)], boot_path=root/f'boot_{rank}', proc_root=root/'proc',
            clock=clock, sleep=sleep, popen=lambda argv: children[rank],
            run=lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''),
            socket_factory=lambda *args: Exchange([]))
    with ThreadPoolExecutor(max_workers=2) as pool: rows = list(pool.map(node, (0, 1)))
    assert [row['return_code'] for row in rows] == ([0, 0] if mode == 'normal' else [1, 1])
    assert len(children[1].signals) == (0 if mode == 'peer_premature' else 1)
    if mode == 'peer_premature': assert rows[0]['peer_premature_exit']['result'] == 'FAIL'
    else: assert children[1].signals[0][1] == signal.SIGTERM
    for rank in (0, 1):
        saved = json.loads((root/f'node_{rank}'/'status.json').read_text())
        assert saved['scope'] == 'capacity_only_not_G2' and saved['peer_final_status'] != 'missing'
        saved_marker = module.read_marker(root, 'g1', rank, 'completion')
        assert saved_marker['final_status']['peer_final_status'] == 'missing'


@pytest.mark.parametrize('rank', [0, 1])
def test_shutdown_budget_has_no_extra_kill(root, rank):
    clock = Clock(); identity(root, rank); child = Child(clock, exit_at=9999)
    child.send_signal = lambda number: child.signals.append((clock(), number))
    if rank == 1:
        clock.events.append((2, lambda: workload_terminal(root, clock)))
    row = supervise(root, rank, clock, child)
    assert row['return_code'] == 1 and row['timeouts']['shutdown'] and len(child.signals) == 1
    assert row['cleanup_start_time_ns'] == (2370000000000 if rank == 0 else 62000000000)
    assert clock() == 2400

def test_handshake_failure_skips_native_and_waits_after_completion(root):
    clock = Clock(); identity(root, 1); prepared(root, 0)
    class Closed(Socket):
        def connect(self, endpoint): raise OSError('connection refused')
    row = module.supervise(root, 'g1', environment(1), {'tp': 16}, ['native'], [],
        [f'GPU-{i}' for i in range(8)], boot_path=root/'boot_1', clock=clock, sleep=clock.sleep,
        socket_factory=lambda *a: Closed([]), popen=lambda *a: pytest.fail('native started'),
        run=lambda *a, **k: type('Query', (), dict(returncode=0, stdout='', stderr=''))())
    assert row['return_code'] == 1 and clock() == 2400
    assert module.read_marker(root, 'g1', 1, 'completion')['time_ns'] == 0

def test_import_does_not_load_native_packages_or_start_children(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Import attempted a child, socket or native query')
    monkeypatch.setattr(module.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(module.subprocess, 'run', forbidden)
    monkeypatch.setattr(module.socket, 'socket', forbidden)
    names = set(sys.modules)
    spec = importlib.util.spec_from_file_location('capacity_supervision_import', PATH)
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)
    assert not any(name.split('.')[0] in ('torch', 'vllm', 'steptron') for name in set(sys.modules)-names)



def workload_terminal(root, clock, pid=42, **fields):
    node = dict(identity(root, 0), pid=pid)
    return module.publish_marker(root, 'g1', node, 'workload_terminal', 'PASS',
                                 'no further workload', clock=clock, **fields)


def native_started(root, clock, child_pid=42):
    return module.publish_marker(root, 'g1', identity(root, 0), 'native_started', 'PASS',
                                 'native child spawned', clock=clock, child_pid=child_pid)


def fake_attempt(root, rank, clock, child, *, workers=((99, 10),), run=None):
    from types import SimpleNamespace
    for local_rank in (0, 1): identity(root, local_rank)
    prepared(root, 1-rank)
    if rank == 1: native_started(root, clock)
    return module.supervise(root, 'g1', environment(rank), {'tp': 16}, ['FAKE', str(rank)],
        list(workers), [f'GPU-{i}' for i in range(8)], clock=clock, sleep=clock.sleep,
        boot_path=root/f'boot_{rank}', proc_root=root/'proc', popen=lambda argv: child,
        socket_factory=lambda *args: Socket([]),
        run=run or (lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr='')))


@pytest.mark.parametrize('fault', ['generation', 'malformed', 'read_io', 'write_io'])
def test_marker_fault_after_start_still_shuts_down_and_finishes(root, monkeypatch, fault):
    import signal
    clock = Clock(); child = Child(clock, exit_at=9999)
    clock.events.append((5, lambda: finish(root, 1, clock)))
    if fault in ('generation', 'malformed'):
        def bad_marker():
            path = root/'node_0_workload_terminal.json'
            path.write_text('{' if fault == 'malformed' else json.dumps(dict(
                generation='old', node_rank=0, kind='workload_terminal', boot_id='boot-0',
                pid=child.pid, result='PASS', time_ns=2000000000, reason='stale')))
        clock.events.append((2, bad_marker))
    elif fault == 'read_io':
        original = module.read_marker
        def read(*args, **kwargs):
            if args[3] == 'workload_terminal' and clock() >= 2:
                raise OSError('FAKE marker read unavailable')
            return original(*args, **kwargs)
        monkeypatch.setattr(module, 'read_marker', read)
    else:
        original = module.publish_marker
        ordinary_sleep = clock.sleep
        def first_late_sleep(seconds):
            ordinary_sleep(seconds+2310)
        def publish(*args, **kwargs):
            if args[3] == 'workload_terminal':
                clock.sleep = ordinary_sleep
                raise OSError('FAKE marker write unavailable')
            if args[3] == 'native_started': clock.sleep = first_late_sleep
            return original(*args, **kwargs)
        monkeypatch.setattr(module, 'publish_marker', publish)
    row = fake_attempt(root, 0, clock, child)
    assert row['return_code'] == 1 and row['evidence_verdict'] == 'UNVERIFIED'
    assert row['evidence_errors'] and len(child.signals) == 1
    assert child.signals[0][1] == signal.SIGTERM and row['raw_exit_code'] == 0
    assert row['cleanup_start_time_ns'] is not None and row['residual']['verdict'] == 'PASS'
    assert (root/'node_0_completion.json').exists() and (root/'node_0/status.json').exists()
    assert clock() <= module.SECONDS


def test_terminal_publication_fault_preserves_native_exit9(root, monkeypatch):
    clock = Clock(); child = Child(clock, code=9)
    clock.events.append((5, lambda: finish(root, 1, clock)))
    original = module.publish_marker
    def publish(*args, **kwargs):
        if args[3] == 'workload_terminal': raise OSError('FAKE terminal storage unavailable')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'publish_marker', publish)
    row = fake_attempt(root, 0, clock, child)
    assert row['raw_exit_code'] == 9 and row['raw_signal'] is None
    assert 'native exit 9' in row['reasons'] and row['return_code'] == 1
    assert row['evidence_verdict'] == 'UNVERIFIED' and row['residual']['verdict'] == 'PASS'
    saved = json.loads((root/'node_0/status.json').read_text())
    assert saved['raw_exit_code'] == 9 and (root/'node_0_completion.json').exists()


@pytest.mark.parametrize('fault', ['completion_write', 'completion_read', 'premature_read', 'status_write'])
def test_final_evidence_fault_is_nonblocking_and_preserves_native_failure(root, monkeypatch, capsys, fault):
    clock = Clock(); clock.events.append((5, lambda: finish(root, 1, clock)))
    original_read, original_publish = module.read_marker, module.publish_marker
    def read(*args, **kwargs):
        if args[3] == ('premature_exit' if fault == 'premature_read' else 'completion') and \
                fault in ('premature_read', 'completion_read'):
            raise OSError('FAKE final marker read unavailable')
        return original_read(*args, **kwargs)
    def publish(*args, **kwargs):
        if args[3] == 'completion' and args[2]['node_rank'] == 0 and fault == 'completion_write':
            raise OSError('FAKE completion storage unavailable')
        return original_publish(*args, **kwargs)
    monkeypatch.setattr(module, 'read_marker', read)
    monkeypatch.setattr(module, 'publish_marker', publish)
    if fault == 'status_write':
        def save(*args): raise OSError('FAKE status storage unavailable')
        monkeypatch.setattr(module, 'save', save)
    row = fake_attempt(root, 0, clock, Child(clock, code=9))
    assert row['raw_exit_code'] == 9 and 'native exit 9' in row['reasons']
    assert row['return_code'] == 1 and row['evidence_verdict'] == 'UNVERIFIED'
    assert row['evidence_errors'] and row['cleanup_end_time_ns'] is not None
    assert 'CAPACITY_EVIDENCE_UNVERIFIED' in capsys.readouterr().out
    if fault != 'status_write': assert (root/'node_0/status.json').exists()


def test_query_timeout_bytes_remains_serializable_and_retains_native_failure(root):
    from subprocess import TimeoutExpired
    clock = Clock(); clock.events.append((2, lambda: workload_terminal(root, clock)))
    clock.events.append((4, lambda: finish(root, 1, clock)))
    def run(argv, **kwargs):
        assert kwargs['text'] is True
        raise TimeoutExpired(argv, kwargs['timeout'], output=b'partial\xff', stderr=b'late')
    row = fake_attempt(root, 0, clock, Child(clock, code=9), run=run)
    query = row['residual']['query']
    assert isinstance(query['stdout'], str) and query['stderr'] == 'late'
    assert query['stdout'].startswith('partial') and row['residual']['verdict'] == 'UNVERIFIED'
    assert row['raw_exit_code'] == 9 and 'native exit 9' in row['reasons']
    assert json.loads((root/'node_0/status.json').read_text())['raw_exit_code'] == 9
    json.dumps(row)


@pytest.mark.parametrize('phase,observed', [('workload', 2311), ('shutdown', 61)])
def test_late_observation_cannot_pass_expired_deadline(root, phase, observed):
    clock = Clock(); child = Child(clock, exit_at=observed)
    if phase == 'shutdown': workload_terminal(root, clock)
    def late_sleep(seconds):
        clock.now = observed
        if phase == 'workload': workload_terminal(root, clock)
        finish(root, 1, clock)
    clock.sleep = late_sleep
    row = fake_attempt(root, 0, clock, child)
    assert row['return_code'] == 1 and row['timeouts'][phase]
    assert row['raw_exit_code'] == 0 and row['exit_time_ns'] == observed*10**9


@pytest.mark.parametrize('kind', ['workload_terminal', 'completion', 'premature_exit'])
def test_wrong_publisher_pid_is_rejected_after_start(root, kind):
    clock = Clock(); child = Child(clock)
    def wrong_marker():
        rank = 0 if kind == 'workload_terminal' else 1
        node = dict(identity(root, rank), pid=987654)
        module.publish_marker(root, 'g1', node, kind, 'PASS', 'foreign publisher', clock=clock,
                              final_status={'result': 'PASS'})
    clock.events.append((2, wrong_marker))
    if kind != 'workload_terminal': clock.events.append((1, lambda: workload_terminal(root, clock)))
    if kind != 'completion': clock.events.append((5, lambda: finish(root, 1, clock)))
    row = fake_attempt(root, 0, clock, child)
    assert row['return_code'] == 1 and row['evidence_verdict'] == 'UNVERIFIED'
    assert any('pid' in error['error'].lower() for error in row['evidence_errors'])


def test_started_native_without_worker_identities_has_unverified_residual(root):
    clock = Clock(); clock.events.append((2, lambda: workload_terminal(root, clock)))
    clock.events.append((4, lambda: finish(root, 1, clock)))
    row = fake_attempt(root, 0, clock, Child(clock), workers=())
    assert row['residual']['verdict'] == 'UNVERIFIED' and row['residual']['workers'] == []
    assert 'worker identities' in '; '.join(row['residual']['errors']).lower()
    assert row['raw_exit_code'] == 0 and row['return_code'] == 1


@pytest.mark.parametrize('change', ['pid', 'child_pid', 'missing', 'delayed'])
def test_node1_binds_terminal_to_started_child(root, change):
    import signal
    clock = Clock(); child = Child(clock, exit_at=9999)
    for rank in (0, 1): identity(root, rank)
    prepared(root, 0)
    if change not in ('missing', 'delayed'):
        native_started(root, clock, child_pid=123 if change == 'child_pid' else 42)
        if change == 'pid':
            path = root/'node_0_native_started.json'
            row = json.loads(path.read_text()); row['pid'] = 987654
            path.write_text(json.dumps(row))
    if change == 'delayed': clock.events.append((3, lambda: native_started(root, clock)))
    clock.events.append((2, lambda: workload_terminal(root, clock)))
    clock.events.append((5, lambda: finish(root, 0, clock)))
    row = module.supervise(root, 'g1', environment(1), {'tp': 16}, ['FAKE'], [(99, 10)],
        [f'GPU-{i}' for i in range(8)], boot_path=root/'boot_1', proc_root=root/'proc',
        clock=clock, sleep=clock.sleep, popen=lambda argv: child,
        run=lambda *a, **k: type('Query', (), dict(returncode=0, stdout='', stderr=''))(),
        socket_factory=lambda *a: Socket([]))
    assert len(child.signals) == 1 and child.signals[0][1] == signal.SIGTERM
    assert row['raw_exit_code'] == 0 and row['return_code'] == int(change != 'delayed')
    if change != 'delayed': assert row['evidence_verdict'] == 'UNVERIFIED'
    assert clock() <= module.SECONDS


@pytest.mark.parametrize('result,explicit', [('FAIL', True), ('PASS', True), ('FAIL', False)])
def test_supervisor_terminal_fallback_is_explicit_fail_only(root, result, explicit):
    clock = Clock(); child = Child(clock, exit_at=9999)
    def fallback():
        module.publish_marker(root, 'g1', identity(root, 0), 'workload_terminal', result,
                              'supervisor fallback', clock=clock, supervisor_fallback=explicit)
    clock.events.append((2, fallback)); clock.events.append((5, lambda: finish(root, 0, clock)))
    row = fake_attempt(root, 1, clock, child)
    assert row['return_code'] == 1 and len(child.signals) == 1
    assert row['evidence_verdict'] == ('PASS' if result == 'FAIL' and explicit else 'UNVERIFIED')


@pytest.mark.parametrize('start', [None, '10', -1])
def test_missing_worker_start_time_is_unverified(root, start):
    from types import SimpleNamespace
    clock = Clock()
    row = module.verify_residual([(99, start)], [f'GPU-{i}' for i in range(8)], 30,
        clock=clock, sleep=clock.sleep, proc_root=root/'proc',
        run=lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''))
    assert row['verdict'] == 'UNVERIFIED' and row['workers'][0]['gone'] is None
    assert row['errors'] and row['workers'][0]['start_time'] == start


def test_skipped_native_has_no_expected_worker_inventory(root):
    from types import SimpleNamespace
    clock = Clock()
    row = module.verify_residual([], [f'GPU-{i}' for i in range(8)], 30, native_started=False,
        clock=clock, sleep=clock.sleep, proc_root=root/'proc',
        run=lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''))
    assert row['verdict'] == 'PASS' and row['native_started'] is False and row['workers'] == []


def test_remote_monotonic_value_is_not_a_local_deadline(root):
    clock = Clock(); child = Child(clock, exit_at=9999)
    clock.events.append((2, lambda: workload_terminal(root, lambda: 10**12)))
    clock.events.append((5, lambda: finish(root, 0, lambda: 10**12)))
    row = fake_attempt(root, 1, clock, child)
    assert row['return_code'] == 0 and child.signals[0][0] == 2
    assert row['terminal_observed_time_ns'] == 2*10**9 and row['terminal_time_ns'] == 10**21


def test_native_started_write_fault_still_observes_native_exit(root, monkeypatch):
    clock = Clock(); child = Child(clock, exit_at=9999, code=9)
    clock.events.append((5, lambda: finish(root, 1, clock)))
    original = module.publish_marker
    def publish(*args, **kwargs):
        if args[3] == 'native_started': raise OSError('FAKE start identity storage unavailable')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'publish_marker', publish)
    row = fake_attempt(root, 0, clock, child)
    assert row['raw_exit_code'] == 9 and 'native exit 9' in row['reasons']
    assert row['evidence_verdict'] == 'UNVERIFIED' and len(child.signals) == 1
    assert (root/'node_0/status.json').exists() and (root/'node_0_completion.json').exists()


def test_marker_fault_with_live_child_uses_one_bounded_shutdown(root, monkeypatch):
    clock = Clock(); child = Child(clock, exit_at=9999)
    child.send_signal = lambda number: child.signals.append((clock(), number))
    original = module.read_marker
    def read(*args, **kwargs):
        if args[3] == 'workload_terminal': raise OSError('FAKE unavailable marker')
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'read_marker', read)
    row = fake_attempt(root, 0, clock, child)
    assert row['timeouts']['shutdown'] and row['raw_exit_code'] is None
    assert row['cleanup_start_time_ns'] == module.SHUTDOWN_SECONDS*10**9
    assert len(child.signals) == 1 and clock() == module.SECONDS
    assert row['return_code'] == 1 and row['evidence_verdict'] == 'UNVERIFIED'


def test_native_poll_fault_is_not_suppressed_as_evidence(root):
    clock = Clock(); child = Child(clock)
    def poll(): raise RuntimeError('FAKE native poll defect')
    child.poll = poll
    with pytest.raises(RuntimeError, match='native poll defect'):
        fake_attempt(root, 0, clock, child)


def test_marker_wait_rejects_late_visible_completion(root):
    clock = Clock()
    def late(seconds):
        clock.now = 6; finish(root, 1, clock)
    with pytest.raises(TimeoutError):
        module.wait_marker(root, 'g1', 1, 'completion', 5, clock=clock, sleep=late, expected_pid=1001)


@pytest.mark.parametrize('case', ['complete', 'partial', 'fault', 'malformed', 'duplicate', 'native_failure'])
def test_supported_inventory_callback_precedes_residual_and_completion(root, monkeypatch, case):
    from types import SimpleNamespace
    clock = Clock(); child = Child(clock, code=9 if case == 'native_failure' else 0)
    for rank in (0, 1): identity(root, rank)
    prepared(root, 1)
    clock.events.append((2, lambda: workload_terminal(root, clock)))
    clock.events.append((4, lambda: finish(root, 1, clock)))
    known = [(99, 10)]; observed = []
    def inventory():
        observed.append('inventory')
        assert not (root/'node_0_completion.json').exists()
        if case == 'fault': raise OSError('FAKE inventory I/O fault')
        pairs = [(100+i, 10+i) for i in range(7)]
        if case == 'partial': pairs = pairs[:2]
        if case == 'malformed': pairs = [(100, None)]
        if case == 'duplicate': pairs = [(100, 10), (100, 11)]
        return dict(workers=pairs, complete=case != 'partial', rows=[], errors=[])
    original = module.verify_residual
    def residual(workers, *args, **kwargs):
        observed.append(('residual', list(workers)))
        assert known[0] == (99, 10) and not (root/'node_0_completion.json').exists()
        return original(workers, *args, **kwargs)
    monkeypatch.setattr(module, 'verify_residual', residual)
    row = module.supervise(root, 'g1', environment(0), {'tp': 16}, ['FAKE'], known,
        [f'GPU-{i}' for i in range(8)], clock=clock, sleep=clock.sleep,
        boot_path=root/'boot_0', proc_root=root/'proc', popen=lambda argv: child,
        run=lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''),
        socket_factory=lambda *a: Socket([]), worker_inventory=inventory)
    assert observed[0] == 'inventory' and observed[1][0] == 'residual'
    assert (99, 10) in observed[1][1] and row['residual']['verdict'] == 'PASS'
    marker = module.read_marker(root, 'g1', 0, 'completion')
    bad = case not in ('complete', 'native_failure')
    assert row['return_code'] == int(case != 'complete')
    assert marker['result'] == ('PASS' if case == 'complete' else 'FAIL')
    assert marker['final_status']['evidence_verdict'] == ('UNVERIFIED' if bad else 'PASS')
    assert row['raw_exit_code'] == (9 if case == 'native_failure' else 0)
    if case == 'partial': assert len(observed[1][1]) == 3
    if case == 'native_failure': assert 'native exit 9' in row['reasons']
    assert child.signals == [] and clock() == 4


def test_inventory_callback_unserializable_fault_still_retains_raw_failure_and_completion(root):
    from types import SimpleNamespace
    clock = Clock(); child = Child(clock, code=9)
    for rank in (0, 1): identity(root, rank)
    prepared(root, 1)
    clock.events.append((2, lambda: workload_terminal(root, clock)))
    clock.events.append((4, lambda: finish(root, 1, clock)))
    def inventory():
        return dict(workers=object(), complete=False, rows=[], errors=['fake fault'])
    row = module.supervise(root, 'g1', environment(0), {'tp': 16}, ['FAKE'], [(99, 10)],
        [f'GPU-{i}' for i in range(8)], clock=clock, sleep=clock.sleep,
        boot_path=root/'boot_0', proc_root=root/'proc', popen=lambda argv: child,
        run=lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''),
        socket_factory=lambda *a: Socket([]), worker_inventory=inventory)
    marker = module.read_marker(root, 'g1', 0, 'completion')
    assert row['raw_exit_code'] == 9 and row['evidence_verdict'] == 'UNVERIFIED'
    assert row['residual']['workers'][0]['pid'] == 99 and marker['result'] == 'FAIL'
    assert clock() == 4


def test_malformed_inventory_preserves_every_known_valid_worker_pair(root, monkeypatch):
    from types import SimpleNamespace
    clock = Clock(); child = Child(clock)
    for rank in (0, 1): identity(root, rank)
    prepared(root, 1)
    clock.events.append((2, lambda: workload_terminal(root, clock)))
    clock.events.append((4, lambda: finish(root, 1, clock)))
    row = module.supervise(root, 'g1', environment(0), {'tp': 16}, ['FAKE'], [(99, 10)],
        [f'GPU-{i}' for i in range(8)], clock=clock, sleep=clock.sleep,
        boot_path=root/'boot_0', proc_root=root/'proc', popen=lambda argv: child,
        run=lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''),
        socket_factory=lambda *a: Socket([]), worker_inventory=lambda: dict(
            workers=[(100, 11), (101, None)], complete=False, rows=[], errors=[]))
    assert {item['pid'] for item in row['residual']['workers']} == {99, 100}
    assert row['return_code'] == 1 and row['evidence_verdict'] == 'UNVERIFIED'
    assert module.read_marker(root, 'g1', 0, 'completion')['result'] == 'FAIL'


@pytest.mark.parametrize('verdict', ['PASS', 'UNVERIFIED', 'exception'])
@pytest.mark.parametrize('native_code', [0, 9])
def test_review_finalization_runs_after_residual_before_any_durable_pass(root, monkeypatch, verdict, native_code):
    from types import SimpleNamespace
    clock = Clock(); child = Child(clock, code=native_code)
    for rank in (0, 1): identity(root, rank)
    prepared(root, 1)
    clock.events.append((2, lambda: workload_terminal(root, clock)))
    clock.events.append((4, lambda: finish(root, 1, clock)))
    observed = []
    def finalize(status):
        observed.append('finalize')
        assert status['cleanup_end_time_ns'] is not None and status['residual']['verdict'] == 'PASS'
        assert not (root/'node_0_completion.json').exists() and not (root/'node_0/status.json').exists()
        assert status['raw_exit_code'] == native_code
        if verdict == 'exception': raise OSError('FAKE finalization I/O failure')
        return dict(verdict=verdict, retained_files=['fake.raw'], retained_bytes=10)
    row = module.supervise(root, 'g1', environment(0), {'tp': 16}, ['FAKE'], [(99, 10)],
        [f'GPU-{i}' for i in range(8)], clock=clock, sleep=clock.sleep,
        boot_path=root/'boot_0', proc_root=root/'proc', popen=lambda argv: child,
        run=lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''),
        socket_factory=lambda *a: Socket([]), finalize_evidence=finalize)
    marker = module.read_marker(root, 'g1', 0, 'completion')
    durable = json.loads((root/'node_0/status.json').read_text())
    assert observed == ['finalize'] and row['raw_exit_code'] == native_code
    expected = 'PASS' if verdict == 'PASS' and native_code == 0 else 'FAIL'
    assert marker['result'] == durable['result'] == row['result'] == expected
    assert row['evidence_verdict'] == ('PASS' if verdict == 'PASS' else 'UNVERIFIED')
    if verdict != 'exception': assert marker['final_status']['evidence_finalization']['verdict'] == verdict


@pytest.fixture(autouse=True)
def forbid_real_supervision_operations(monkeypatch):
    import builtins
    def forbidden(*args, **kwargs):
        pytest.fail('Fake supervisor test attempted a real child, command, socket, signal or native import')
    monkeypatch.setattr(module.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(module.subprocess, 'run', forbidden)
    monkeypatch.setattr(module.socket, 'socket', forbidden)
    monkeypatch.setattr(module.os, 'kill', forbidden)
    original_import = builtins.__import__
    def fake_only_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'vllm', 'steptron', 'stepmind'):
            forbidden()
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', fake_only_import)


@pytest.mark.parametrize('receipt', [None, {}, dict(verdict='maybe')])
def test_review_finalization_requires_explicit_receipt_not_success_fallback(root, receipt):
    from types import SimpleNamespace
    clock = Clock()
    for rank in (0, 1): identity(root, rank)
    prepared(root, 1, engine_settings={}); workload_terminal(root, clock)
    clock.events.append((4, lambda: finish(root, 1, clock)))
    row = module.supervise(root, 'g1', environment(0), {}, ['FAKE'], [(99, 10)],
        [f'GPU-{i}' for i in range(8)], clock=clock, sleep=clock.sleep,
        boot_path=root/'boot_0', proc_root=root/'proc', popen=lambda *a: Child(clock),
        run=lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''),
        socket_factory=lambda *a: Socket([]), finalize_evidence=lambda status: receipt)
    assert row['raw_exit_code'] == 0 and row['result'] == 'FAIL' and row['evidence_verdict'] == 'UNVERIFIED'
    assert module.read_marker(root, 'g1', 0, 'completion')['result'] == 'FAIL'


@pytest.mark.parametrize('completion_deadline', [True, float('nan'), float('inf'), 2099, 2401])
def test_review_completion_deadline_never_restarts_the_approved_budget(root, completion_deadline):
    clock = Clock()
    with pytest.raises(ValueError, match='Completion deadline'):
        module.supervise(root, 'g1', environment(0), {}, [], [], [f'GPU-{i}' for i in range(8)],
                         clock=clock, deadline=2100, completion_deadline=completion_deadline)
    assert list(root.iterdir()) == []


@pytest.mark.parametrize('deadline', [True, float('nan'), float('inf'), 90, 2401])
def test_review_supervisor_deadline_cannot_expand_or_remove_cleanup_reserve(root, deadline):
    clock = Clock()
    with pytest.raises(ValueError, match='deadline'):
        module.supervise(root, 'g1', environment(0), {}, [], [], [f'GPU-{i}' for i in range(8)],
                         clock=clock, deadline=deadline)
