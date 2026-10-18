"""Supervise a future two-node native capacity attempt; no launch is authorized."""
import csv
import ipaddress
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time

from check_role_capacity import diagnostic, save

SECONDS, SHUTDOWN_SECONDS, RESIDUAL_SECONDS = 2400, 60, 30
HANDSHAKE_SCOPE = 'TCP master endpoint only; not NCCL or native message queues'
KINDS = ('prepared', 'native_started', 'workload_terminal', 'premature_exit', 'completion')
COMPUTE_QUERY = ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                 '--format=csv,noheader,nounits']


def node_identity(env, boot_path=Path('/proc/sys/kernel/random/boot_id'), resolve=socket.getaddrinfo):
    rank, count, local = (int(env[name]) for name in ('NODE_RANK', 'NODE_COUNT', 'PROC_PER_NODE'))
    if rank not in (0, 1) or count != 2 or local != 8:
        raise ValueError('NODE_RANK must be 0 or 1, NODE_COUNT=2 and PROC_PER_NODE=8')
    port = int(env.get('MASTER_PORT', '5678'))
    if not 1 <= port <= 65535: raise ValueError('Invalid MASTER_PORT')
    address = env['MASTER_ADDR']
    try:
        addresses = [ipaddress.ip_address(address)]
    except ValueError:
        addresses = [ipaddress.ip_address(row[4][0]) for row in resolve(address, port, type=socket.SOCK_STREAM)]
    if not addresses or any(ip.is_loopback or ip.is_unspecified or
            (ip.version == 6 and ip.ipv4_mapped and
             (ip.ipv4_mapped.is_loopback or ip.ipv4_mapped.is_unspecified)) for ip in addresses):
        raise ValueError('MASTER_ADDR must not be loopback or unspecified')
    boot_id = Path(boot_path).read_text().strip()
    if not boot_id: raise ValueError('Empty boot id')
    return dict(node_rank=rank, hostname=socket.gethostname(), boot_id=boot_id, pid=os.getpid(),
        endpoint=dict(address=address, port=port,
                      port_source='environment' if 'MASTER_PORT' in env else 'platform_default'))


def publish_marker(root, generation, node, kind, result, reason, *, clock=time.monotonic, **fields):
    if kind not in KINDS or (kind in ('native_started', 'workload_terminal') and node['node_rank'] != 0):
        raise ValueError('Invalid marker kind or publisher rank')
    row = dict(fields, generation=generation, node_rank=node['node_rank'], boot_id=node['boot_id'],
               pid=node['pid'], time_ns=int(clock()*1e9), kind=kind, result=result, reason=reason)
    root = Path(root)
    fd, temporary = tempfile.mkstemp(dir=root, prefix='.', suffix='.json')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(row, stream); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, root/f"node_{node['node_rank']}_{kind}.json")
    finally:
        os.unlink(temporary)
    return row


def read_marker(root, generation, rank, kind, boot_id=None, expected_pid=None):
    try:
        row = json.loads((Path(root)/f'node_{rank}_{kind}.json').read_text())
    except FileNotFoundError:
        return None
    if (row['generation'], row['node_rank'], row['kind']) != (generation, rank, kind):
        raise ValueError('Marker generation, rank or kind mismatch')
    if boot_id is not None and row['boot_id'] != boot_id: raise ValueError('Marker boot id mismatch')
    if type(row['pid']) is not int or row['pid'] <= 0:
        raise ValueError('Marker pid must be a positive integer')
    if row['result'] not in ('PASS', 'FAIL') or not isinstance(row['reason'], str) or \
            type(row['time_ns']) is not int or row['time_ns'] < 0:
        raise ValueError('Invalid marker result, reason or time_ns')
    if kind == 'native_started' and (row['result'] != 'PASS' or
            type(row['child_pid']) is not int or row['child_pid'] <= 0):
        raise ValueError('native_started requires PASS and positive child pid')
    if expected_pid is not None and row['pid'] != expected_pid:
        raise ValueError('Marker publisher pid mismatch')
    return row


def wait_marker(root, generation, rank, kind, deadline, *, clock=time.monotonic, sleep=time.sleep,
                boot_id=None, expected_pid=None):
    while True:
        if clock() >= deadline: raise TimeoutError('Deadline waiting for '+kind)
        row = read_marker(root, generation, rank, kind, boot_id, expected_pid)
        if clock() >= deadline: raise TimeoutError('Deadline waiting for '+kind)
        if row is not None: return row
        sleep(min(1, deadline-clock()))


def preparation_exchange(root, generation, node, engine_settings, deadline, *,
                         clock=time.monotonic, sleep=time.sleep, socket_factory=socket.socket):
    endpoint = node['endpoint']; address = (endpoint['address'], endpoint['port'])
    result = dict(result='PASS', peer=None, reason='', timeout=False)
    connection = None
    try:
        family = socket.AF_INET6 if ':' in address[0] else socket.AF_INET
        connection = socket_factory(family, socket.SOCK_STREAM)
        # Publish only after listen: the peer makes exactly one connection attempt.
        if node['node_rank'] == 0: connection.bind(address); connection.listen(1)
    except OSError as error:
        result.update(result='FAIL', reason=repr(error))
    try:
        publish_marker(root, generation, node, 'prepared', result['result'], result['reason'], clock=clock,
                       endpoint=endpoint, engine_settings=engine_settings)
        if result['result'] == 'FAIL': return result
        try:
            peer = wait_marker(root, generation, 1-node['node_rank'], 'prepared', deadline, clock=clock, sleep=sleep)
            if peer['boot_id'] == node['boot_id'] or peer['result'] != 'PASS':
                raise ValueError('Distinct boot ids and successful preparation required')
            if any(peer['endpoint'][key] != endpoint[key] for key in ('address', 'port')) or \
                    peer['engine_settings'] != engine_settings:
                raise ValueError('Peer endpoint or engine settings mismatch')
            result['peer'] = peer
            remaining = deadline-clock()
            if remaining <= 0: raise TimeoutError('TCP handshake deadline')
            connection.settimeout(remaining)
            if node['node_rank'] == 0:
                accepted, _ = connection.accept()
                try:
                    remaining = deadline-clock()
                    if remaining <= 0: raise TimeoutError('TCP close deadline')
                    accepted.settimeout(remaining)
                    # Let the client close first, avoiding server TIME_WAIT on the native master port.
                    accepted.recv(1)
                finally:
                    accepted.close()
            else:
                connection.connect(address)
        except (OSError, ValueError) as error:
            result.update(result='FAIL', reason=repr(error), timeout=isinstance(error, TimeoutError))
    finally:
        if connection is not None: connection.close()
    return result


def evidence(status, operation, call):
    """Keep evidence faults visible without replacing native lifecycle observations."""
    value = None
    def attempt():
        nonlocal value
        try:
            value = call()
        except Exception as error:
            status['evidence_verdict'] = 'UNVERIFIED'
            status['evidence_errors'].append(dict(operation=operation, error=repr(error)))
            status['result'] = 'FAIL'; status['return_code'] = 1
            status['reasons'].append('evidence unavailable: '+operation)
            raise
    diagnostic(attempt)
    return value


def verify_residual(workers, gpu_uuids, deadline, *, proc_root=Path('/proc'),
                    clock=time.monotonic, sleep=time.sleep, run=subprocess.run, native_started=True):
    if len(gpu_uuids) != 8 or len(set(gpu_uuids)) != 8:
        raise ValueError('Exactly eight allocated GPU UUIDs required')
    query = dict(command=list(COMPUTE_QUERY), exit_code=None, stdout='', stderr='')
    missing = ['Native started without known worker identities (pid and start_time)'] if native_started and not workers else []
    row = dict(verdict='UNVERIFIED', workers=[dict(pid=pid, start_time=start, gone=None,
        exit_code='not_waitable_by_supervisor') for pid, start in workers], compute_apps=[], query=query,
        timeout=False, native_started=native_started, errors=missing)
    while clock() < deadline:
        identities, errors = [], list(missing)
        for pid, start in workers:
            current, error_text = None, None
            if type(pid) is not int or pid <= 0 or type(start) is not int or start < 0:
                error_text = 'Missing worker identities: integer pid and start_time required'
                errors.append(error_text)
            try:
                if error_text is None:
                    # comm may contain spaces or ')'; starttime follows the closing ')'.
                    current = int((Path(proc_root)/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()[19])
            except FileNotFoundError:
                pass
            except (OSError, ValueError, IndexError) as error:
                error_text = repr(error); errors.append(error_text)
            identities.append(dict(pid=pid, start_time=start, observed_start_time=current,
                gone=None if error_text else current != start, error=error_text, exit_code='not_waitable_by_supervisor'))
        row.update(workers=identities, errors=errors)
        remaining = deadline-clock()
        if remaining <= 0: break
        query = dict(command=list(COMPUTE_QUERY), exit_code=None, stdout='', stderr='')
        apps = []
        try:
            response = run(list(COMPUTE_QUERY), capture_output=True, text=True, timeout=min(10, remaining))
            query.update(exit_code=response.returncode, stdout=response.stdout, stderr=response.stderr)
            if response.returncode == 0:
                for fields in csv.reader(response.stdout.splitlines()):
                    if len(fields) != 4: raise ValueError('Malformed compute-app query row')
                    uuid, pid, name, memory = [value.strip() for value in fields]
                    if uuid in gpu_uuids: apps.append(dict(gpu_uuid=uuid, pid=int(pid), name=name, used_gpu_memory=memory))
            else:
                errors.append('Compute-app query failed')
        except Exception as error:
            query['error'] = repr(error); errors.append(repr(error))
            if isinstance(error, subprocess.TimeoutExpired):
                query.update(stdout=error.stdout.decode('utf-8', errors='backslashreplace')
                             if isinstance(error.stdout, bytes) else error.stdout or '',
                             stderr=error.stderr.decode('utf-8', errors='backslashreplace')
                             if isinstance(error.stderr, bytes) else error.stderr or '')
        verdict = 'UNVERIFIED' if errors else 'FAIL' if apps or any(not r['gone'] for r in identities) else 'PASS'
        row.update(verdict=verdict, workers=identities, compute_apps=apps, query=query, errors=errors)
        if verdict in ('PASS', 'UNVERIFIED'):
            row['timeout'] = clock() >= deadline
            return row
        remaining = deadline-clock()
        if remaining > 0: sleep(min(1, remaining))
    row['timeout'] = True
    return row


def completion_exchange(root, generation, node, status, deadline, *, clock=time.monotonic, sleep=time.sleep):
    status['completion_time_ns'] = int(clock()*1e9)
    status['return_code'] = int(status['result'] != 'PASS')
    evidence(status, 'publish completion', lambda: publish_marker(root, generation, node,
        'completion', status['result'], '; '.join(status['reasons']), clock=clock, final_status=status))
    peer_prepared = status['preparation']['peer']
    peer_boot = peer_prepared['boot_id'] if peer_prepared else None
    peer_pid = peer_prepared['pid'] if peer_prepared else None
    def receive_completion():
        try:
            peer = wait_marker(root, generation, 1-node['node_rank'], 'completion', deadline,
                               clock=clock, sleep=sleep, boot_id=peer_boot, expected_pid=peer_pid)
        except TimeoutError:
            status['timeouts']['peer_completion'] = True
            status['result'] = 'FAIL'; status['reasons'].append('peer completion deadline')
            return
        status['peer_completion'] = peer
        status['peer_final_status'] = peer['final_status']
        if peer['result'] != 'PASS': status['result'] = 'FAIL'; status['reasons'].append('peer failure')
    evidence(status, 'read peer completion', receive_completion)
    if node['node_rank'] == 0:
        premature = evidence(status, 'read peer premature_exit', lambda: read_marker(
            root, generation, 1, 'premature_exit', peer_boot, expected_pid=peer_pid))
        if premature is not None:
            status['peer_premature_exit'] = premature; status['premature_time_ns'] = premature['time_ns']
            status['result'] = 'FAIL'; status['reasons'].append('peer premature exit')
    status['return_code'] = int(status['result'] != 'PASS')
    evidence(status, 'save final status', lambda: save(Path(root)/f"node_{node['node_rank']}"/'status.json', status))
    return status


def supervise(root, generation, env, engine_settings, argv, workers, gpu_uuids, *,
              clock=time.monotonic, sleep=time.sleep, popen=subprocess.Popen, run=subprocess.run,
              boot_path=Path('/proc/sys/kernel/random/boot_id'), proc_root=Path('/proc'), socket_factory=socket.socket):
    started = clock(); end = started+SECONDS
    workload_end, shutdown_end = end-SHUTDOWN_SECONDS-RESIDUAL_SECONDS, end-RESIDUAL_SECONDS
    if len(gpu_uuids) != 8 or len(set(gpu_uuids)) != 8:
        raise ValueError('Exactly eight allocated GPU UUIDs required')
    node = node_identity(env, boot_path); rank = node['node_rank']; root = Path(root)
    root.mkdir(parents=True, exist_ok=True); (root/f'node_{rank}').mkdir()
    status = dict(node, scope='capacity_only_not_G2', generation=generation, handshake_scope=HANDSHAKE_SCOPE,
        time_scope='node_monotonic_clock_not_cross_node_order',
        engine_settings=engine_settings, native_argv=list(argv), child_pid=None, raw_exit_code=None, raw_signal=None,
        started_time_ns=int(started*1e9), premature_time_ns=None, terminal_time_ns=None, signal_time_ns=None,
        terminal_observed_time_ns=None, exit_time_ns=None, cleanup_start_time_ns=None, cleanup_end_time_ns=None,
        signal_error=None, peer_final_status='missing', peer_completion='missing',
        evidence_verdict='PASS', evidence_errors=[], result='PASS', reasons=[],
        timeouts=dict(prepared=False, workload=False, shutdown=False, residual=False, peer_completion=False))
    def fail(reason):
        status['result'] = 'FAIL'; status['reasons'].append(reason)
    exchange = evidence(status, 'preparation exchange', lambda: preparation_exchange(
        root, generation, node, engine_settings, workload_end,
        clock=clock, sleep=sleep, socket_factory=socket_factory))
    if exchange is None:
        exchange = dict(result='FAIL', peer=None, reason='preparation evidence unavailable', timeout=False)
    status['preparation'] = exchange
    peer = exchange['peer']
    peer_boot, peer_pid = (peer['boot_id'], peer['pid']) if peer else (None, None)
    child, shutdown_deadline, native_start = None, None, None
    if exchange['result'] != 'PASS':
        fail(exchange['reason']); status['timeouts']['prepared'] = exchange['timeout']
    else:
        try:
            child = popen(list(argv)); status['child_pid'] = child.pid
        except OSError as error:
            fail('native start: '+repr(error))
    # Private publishers: preparation/completion/premature_exit belong to supervisors.
    # The node0 supervisor binds native_started to the exact spawned workload child;
    # workload_terminal belongs to that child, except an explicit supervisor FAIL.
    if child is not None and rank == 0:
        native_start = evidence(status, 'publish native_started', lambda: publish_marker(
            root, generation, node, 'native_started', 'PASS', 'native child spawned',
            clock=clock, child_pid=child.pid))
    def failed_terminal(reason):
        row = evidence(status, 'publish workload_terminal fallback', lambda: publish_marker(
            root, generation, node, 'workload_terminal', 'FAIL', reason,
            clock=clock, supervisor_fallback=True))
        if row is not None:
            status['terminal_time_ns'] = row['time_ns']
            status['terminal_observed_time_ns'] = int(clock()*1e9)
        return row
    if child is None and rank == 0:
        failed_terminal('native not started')
    def terminal():
        nonlocal native_start
        boot, supervisor_pid = (node['boot_id'], node['pid']) if rank == 0 else (peer_boot, peer_pid)
        row = read_marker(root, generation, 0, 'workload_terminal', boot)
        if row is None: return None
        if rank == 1 and native_start is None:
            native_start = read_marker(root, generation, 0, 'native_started', boot, expected_pid=supervisor_pid)
        fallback = row.get('supervisor_fallback', False)
        if fallback:
            if fallback is not True or row['result'] != 'FAIL' or row['pid'] != supervisor_pid:
                raise ValueError('Supervisor fallback requires FAIL and prepared supervisor pid')
        else:
            expected_pid = child.pid if rank == 0 else native_start['child_pid'] if native_start else None
            if expected_pid is None:
                if clock() >= workload_end:
                    raise ValueError('Missing native_started child pid at workload deadline')
                return None
            if row['pid'] != expected_pid:
                raise ValueError('Workload terminal publisher pid does not match native child pid')
        return row
    while child is not None:
        code = child.poll()
        observed = clock()
        # Poll must retain the native result even when the following marker read fails.
        if code is not None:
            status.update(raw_exit_code=code, raw_signal=-code if code < 0 else None,
                          exit_time_ns=int(observed*1e9))
            if code != 0 and not (rank == 1 and code == -signal.SIGTERM and status['signal_time_ns'] is not None):
                fail('native exit '+str(code))
        if rank == 0 and 'peer_premature_exit' not in status and not status['evidence_errors']:
            premature = evidence(status, 'read peer premature_exit', lambda: read_marker(
                root, generation, 1, 'premature_exit', peer_boot, expected_pid=peer_pid))
            if premature is not None:
                status['peer_premature_exit'] = premature; status['premature_time_ns'] = premature['time_ns']
                fail('peer premature exit')
        row = None
        if status['terminal_time_ns'] is None and not status['evidence_errors']:
            row = evidence(status, 'read workload_terminal', terminal)
        observed = clock()
        if shutdown_deadline is None and observed >= workload_end:
            status['timeouts']['workload'] = True; fail('workload deadline')
            shutdown_deadline = min(observed+SHUTDOWN_SECONDS, shutdown_end)
            if rank == 0 and row is None and not status['evidence_errors']:
                failed_terminal('workload deadline')
        elif shutdown_deadline is not None and observed >= shutdown_deadline:
            status['timeouts']['shutdown'] = True; fail('native shutdown deadline')
        if row is not None:
            status['terminal_time_ns'] = row['time_ns']
            status['terminal_observed_time_ns'] = int(observed*1e9)
            if row['result'] != 'PASS': fail(row['reason'])
            if shutdown_deadline is None:
                shutdown_deadline = min(observed+SHUTDOWN_SECONDS, shutdown_end)
        if code is not None:
            if rank == 1 and status['signal_time_ns'] is None:
                fail('headless child exited before shutdown request')
                premature = evidence(status, 'publish premature_exit', lambda: publish_marker(
                    root, generation, node, 'premature_exit', 'FAIL',
                    'headless child exited before shutdown request', clock=clock))
                if premature is not None: status['premature_time_ns'] = premature['time_ns']
            if rank == 0 and status['terminal_time_ns'] is None and not status['evidence_errors']:
                fail('native child exited without workload_terminal')
                failed_terminal('native child exited without workload_terminal')
            break
        if status['evidence_errors'] and shutdown_deadline is None:
            shutdown_deadline = min(observed+SHUTDOWN_SECONDS, shutdown_end)
        terminate = status['timeouts']['workload'] or bool(status['evidence_errors']) or \
                    (rank == 1 and status['terminal_time_ns'] is not None)
        if status['signal_time_ns'] is None and terminate:
            status['signal_time_ns'] = int(clock()*1e9)
            try:
                child.send_signal(signal.SIGTERM)
            except OSError as error:
                status['signal_error'] = repr(error); fail('SIGTERM send failed')
        deadline = workload_end if shutdown_deadline is None else shutdown_deadline
        if clock() >= deadline:
            if not status['timeouts']['shutdown']:
                status['timeouts']['shutdown'] = True; fail('native shutdown deadline')
            break
        sleep(min(1, deadline-clock()))
    status['cleanup_start_time_ns'] = int(clock()*1e9)
    status['residual'] = verify_residual(workers, gpu_uuids, min(clock()+RESIDUAL_SECONDS, end),
        proc_root=proc_root, clock=clock, sleep=sleep, run=run, native_started=child is not None)
    status['cleanup_end_time_ns'] = int(clock()*1e9); status['timeouts']['residual'] = status['residual']['timeout']
    if status['residual']['verdict'] == 'UNVERIFIED': status['evidence_verdict'] = 'UNVERIFIED'
    if status['residual']['verdict'] != 'PASS': fail('residual '+status['residual']['verdict'])
    return completion_exchange(root, generation, node, status, end, clock=clock, sleep=sleep)
