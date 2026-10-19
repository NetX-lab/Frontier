#!/usr/bin/env python3
"""Private TP16 capacity caller; preflight is inert and capacity is not G2."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

import role_capacity_observation as observation
import role_capacity_supervision as supervision
from check_role_capacity import SOURCE_MANIFESTS, TASK, diagnostic, save

GEN4_MANIFEST_SHA = 'f12f6630c92450f5b1b3bb8207503546d438df88720f17810aa166f899ea01d7'
VERSION = '0.1.dev22912+g8224bb82f'
GPU_GROUPS = {'H800': 'codesign', 'H200': 'step_main'}
REMAINING_G2 = ('prefill', 'subsequent_chunks', 'decode', 'changed_input_all_rank_graph_replay',
                'finite_content', 'precision_and_exclusion_binding', 'full_physical_scope_census')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(config, *, new_generation=True):
    """Validate explicit approved inputs without starting native, SDK, or platform code."""
    if Path(config['task_root']).resolve() != TASK:
        raise ValueError('Exact trusted task root required')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', config['generation']):
        raise ValueError('Generation must be a nonempty path component')
    resources = config['resources']
    gpu_type = resources.get('gpu_type')
    if gpu_type not in GPU_GROUPS or resources != dict(
            nodes=2, gpu_per_node=8, cpu_per_node=32, memory_mi_per_node=128000,
            worker_seconds=2400, gpu_type=gpu_type, charged_group=GPU_GROUPS[gpu_type]):
        raise ValueError('Only the approved two-node GPU and quota-group pairs are supported')
    expected_image = ('a.z01.normandyfunc.com/docker-public/vllm/vllm-openai'
        '@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20')
    if config['image'] != expected_image or config['vllm_version'] != VERSION:
        raise ValueError('Native software and image must remain locked')
    if config['workload'] != dict(stage='capacity_initialization'):
        raise ValueError('This caller observes capacity/census only; faithful G2 replay is not implemented')
    engine = config['engine']
    if engine != dict(max_model_len=8192, max_num_batched_tokens=512, max_num_seqs=1):
        raise ValueError('Preserve the reviewed native capacity workload limits')
    for name in ('rpc_seconds', 'archive_reserve_seconds', 'record_bytes', 'archive_bytes'):
        if name not in config:
            raise ValueError(name+' must be explicitly configured')
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(name+' must be finite and positive')
    for name in ('record_bytes', 'archive_bytes'):
        if type(config[name]) is not int:
            raise ValueError(name+' must be an integer byte limit')
    workload_seconds = supervision.SECONDS-config['archive_reserve_seconds']-supervision.SHUTDOWN_SECONDS-supervision.RESIDUAL_SECONDS
    if workload_seconds <= 0 or config['rpc_seconds'] >= workload_seconds:
        raise ValueError('Archive reserve and RPC deadline must fit the approved workload ceiling')
    roots = {name: Path(config[name]) for name in ('marker_root', 'worker_local_root', 'archive_root')}
    if any(not path.is_absolute() or path != path.resolve() for path in roots.values()):
        raise ValueError('Generation roots must be absolute canonical paths')
    cloud = Path('/mnt/codesign-exp/ycfeng')
    if not roots['archive_root'].is_relative_to(cloud) or roots['archive_root'] == cloud:
        raise ValueError('Archive must be in the personal cloud subtree')
    if any(roots[name].is_relative_to(Path('/mnt/codesign-exp')) for name in ('marker_root', 'worker_local_root')):
        raise ValueError('Markers and high-frequency records must not use cloud storage')
    if not roots['marker_root'].is_relative_to(TASK) or roots['marker_root'] == TASK:
        raise ValueError('Markers must use a new task-local shared NFS generation')
    if any(roots[a].is_relative_to(roots[b]) or roots[b].is_relative_to(roots[a])
           for a, b in (('marker_root', 'worker_local_root'), ('marker_root', 'archive_root'),
                        ('worker_local_root', 'archive_root'))):
        raise ValueError('Generation roots must be disjoint')
    if new_generation and any(roots[name].exists() for name in ('marker_root', 'worker_local_root')):
        raise ValueError('Preflight requires exclusive new local generation roots')
    if not Path(config['native_cli']).is_absolute():
        raise ValueError('Explicit installed native CLI executable path required')
    for filename, key in (('check_role_tp16.py', 'caller_sha256'),
                          ('role_capacity_observation.py', 'observer_sha256'),
                          ('role_capacity_supervision.py', 'supervisor_sha256')):
        if sha256(Path(__file__).with_name(filename)) != config[key]:
            raise ValueError('Caller/observer/supervisor source identity mismatch: '+filename)
    manifests = dict(SOURCE_MANIFESTS)
    manifests['native_role_sources/observed_generation4'] = GEN4_MANIFEST_SHA
    sources = {}
    for relative, expected in manifests.items():
        path = TASK/relative/'source_manifest.json'
        if sha256(path) != expected:
            raise ValueError('Frozen native source manifest mismatch: '+relative)
        for row in json.loads(path.read_text())['files']:
            name, digest = row['relative_path'], row.get('sha256', row.get('source_sha256'))
            if name in sources and sources[name] != digest:
                raise ValueError('Conflicting native source identities: '+name)
            sources[name] = digest
    lock = json.loads((TASK/'reference_lock.json').read_text())
    for row in lock['files']:
        path = TASK/'reference_inputs'/row['name']
        if path.stat().st_size != row['size_bytes'] or sha256(path) != row['sha256']:
            raise ValueError('Frozen model/tokenizer identity mismatch: '+row['name'])
    return dict(scope='inert_preflight_not_native_qualification', sources=sources,
                resources=resources, workload=config['workload'], remaining_g2=list(REMAINING_G2))


def installed_sources(config, sources, distribution=importlib.metadata.distribution):
    package = distribution('vllm')
    if package.version != config['vllm_version']:
        raise ValueError('Installed vLLM version mismatch')
    for relative, expected in sources.items():
        if sha256(package.locate_file(relative)) != expected:
            raise ValueError('Installed native source mismatch: '+relative)


def node_record(config, env, *, proc_root=Path('/proc'), identify=supervision.node_identity):
    if 'MASTER_PORT' not in env:
        raise ValueError('MASTER_PORT is required; no guessed rendezvous default')
    node = identify(env, proc_root/'sys/kernel/random/boot_id')
    pid_namespace = os.readlink(proc_root/'self/ns/pid')
    return dict(node, generation=config['generation'], pid_namespace=pid_namespace,
                observer_sha256=config['observer_sha256'], resources=dict(config['resources']))


def engine_kwargs(config, node):
    return dict(model=str(TASK/'reference_inputs'), load_format='dummy', skip_tokenizer_init=False,
        tensor_parallel_size=16, data_parallel_size=1, pipeline_parallel_size=1,
        nnodes=2, node_rank=node['node_rank'], master_addr=node['endpoint']['address'],
        master_port=node['endpoint']['port'], enable_chunked_prefill=True,
        enable_prefix_caching=False, speculative_config=None,
        worker_cls='role_capacity_observation.CapacityWorker', **config['engine'])


def headless_argv(config, node):
    if node['node_rank'] != 1:
        raise ValueError('Only node1 runs native headless')
    kwargs = engine_kwargs(config, node)
    flags = ('tensor_parallel_size', 'data_parallel_size', 'pipeline_parallel_size', 'nnodes',
             'node_rank', 'master_addr', 'master_port', 'max_model_len', 'max_num_batched_tokens',
             'max_num_seqs', 'load_format', 'worker_cls')
    return [config['native_cli'], 'serve', kwargs['model'], '--headless',
            '--enable-chunked-prefill', '--no-enable-prefix-caching'] + [part
        for key in flags for part in ('--'+key.replace('_', '-'), str(kwargs[key]))]


def gpu_inventory(resources, run=subprocess.run):
    argv = ['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader,nounits']
    result = run(argv, capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise ValueError('Allocated GPU inventory query failed: '+result.stderr)
    rows = []
    expected_name = 'NVIDIA ' + resources['gpu_type']
    for line in result.stdout.splitlines():
        index, uuid, name = [value.strip() for value in line.split(',')]
        if name != expected_name or not uuid.startswith('GPU-'):
            raise ValueError('Allocated GPU name or UUID differs from the configured hardware')
        rows.append((int(index), uuid))
    rows.sort()
    count = resources['gpu_per_node']
    if [row[0] for row in rows] != list(range(count)) or len({row[1] for row in rows}) != count:
        raise ValueError('Allocated GPU count, indices or unique UUIDs differ from the configured resources')
    return [row[1] for row in rows]


def worker_record(path, node):
    """Parse observed worker identity separately from parent admission."""
    row = json.loads(path.read_text())
    for key in ('generation', 'node_rank', 'boot_id', 'pid_namespace', 'observer_sha256'):
        if row[key] != node[key]:
            raise ValueError('Worker identity mismatch: '+key)
    rank, local, pid, start = (row[key] for key in ('rank', 'local_rank', 'pid', 'start_time'))
    if any(type(value) is not int for value in (rank, local, pid, start)) or pid <= 0 or start < 0:
        raise ValueError('Real worker PID/start_time and integer ranks required')
    if rank != node['node_rank']*8+local or not 0 <= local < 8:
        raise ValueError('Worker rank outside allocated node')
    if path.name != f'worker_{rank}_{pid}.json':
        raise ValueError('Worker identity filename mismatch')
    if pid == node['pid']:
        raise ValueError('Supervisor used as worker')
    return row


def worker_parent(row):
    parent, parent_start = row['parent_pid'], row['parent_start_time']
    if type(parent) is not int or parent <= 0 or type(parent_start) is not int or parent_start < 0 or parent == row['pid']:
        raise ValueError('Distinct live native worker parent identity required')


class WorkerInventory:
    """Update the existing supervisor's mutable PID list from supported live records."""
    def __init__(self, output, node):
        self.output, self.node, self.workers, self.rows, self.errors = output, node, [], {}, []
        self.child_pid = None
        self.known_pairs = set()

    def refresh(self):
        try:
            paths = sorted(Path(self.output).glob('worker_*.json'))
        except Exception:
            self.errors.append(traceback.format_exc())
            return
        for path in paths:
            try:
                row = worker_record(path, self.node)
                if row['pid'] == self.child_pid:
                    raise ValueError('Direct native child is not a GPU worker')
                self.known_pairs.add((row['pid'], row['start_time']))
                worker_parent(row)
                if self.child_pid is not None and self.node['node_rank'] == 1 and row['parent_pid'] != self.child_pid:
                    raise ValueError('Headless workers must belong to the actual native direct child')
                if self.child_pid is not None and self.node['node_rank'] == 0 and row['parent_pid'] in (self.child_pid, self.node['pid']):
                    raise ValueError('Node0 workers require their separate local EngineCore parent')
                key = row['rank']
                if key in self.rows and row != self.rows[key]:
                    raise ValueError('Worker identity changed in one generation')
                if any(other['pid'] == row['pid'] and rank != key for rank, other in self.rows.items()):
                    raise ValueError('Duplicate worker PID across ranks')
                self.rows[key] = row
            except Exception:
                message = traceback.format_exc()
                if message not in self.errors:
                    self.errors.append(message)
                    diagnostic(lambda: print('TP16_WORKER_INVENTORY_UNVERIFIED '+message, flush=True))
        self.workers[:] = sorted(self.known_pairs)

    def snapshot(self):
        self.refresh()
        parents = {(row['parent_pid'], row['parent_start_time']) for row in self.rows.values()}
        return dict(workers=list(self.workers), complete=len(self.rows) == 8 and len(self.workers) == 8 and len(parents) == 1 and not self.errors,
                    rows=[row for _, row in sorted(self.rows.items())], errors=list(self.errors))

    def spawn(self, argv, popen=subprocess.Popen):
        child = popen(argv)
        self.child_pid = child.pid
        inventory = self
        class NativeChild:
            pid = child.pid
            def poll(self):
                result = child.poll()
                inventory.refresh()
                return result
            def send_signal(self, number):
                return child.send_signal(number)
        return NativeChild()


def all_rank_census(rows, nodes, expected):
    if len(rows) != 16 or {row.get('rank') for row in rows} != set(range(16)):
        raise ValueError('All sixteen native worker census records are required')
    if nodes[0]['boot_id'] == nodes[1]['boot_id']:
        raise ValueError('Two different physical node boots are required')
    allocated = [{uuid.removeprefix('GPU-') for uuid in node['gpu_uuids']} for node in nodes]
    if any(len(uuids) != 8 for uuids in allocated) or len(allocated[0] | allocated[1]) != 16:
        raise ValueError('Eight per node and sixteen total unique allocated GPU UUIDs required')
    seen = set()
    for row in rows:
        node = nodes[row['rank']//8]
        for key in ('generation', 'node_rank', 'boot_id', 'pid_namespace', 'observer_sha256'):
            if row[key] != node[key]:
                raise ValueError('All-rank worker binding mismatch: '+key)
        identity = (row['boot_id'], row['pid_namespace'], row['pid'], row['start_time'])
        if identity in seen or type(row['pid']) is not int or row['pid'] <= 0 or \
                type(row['start_time']) is not int or row['start_time'] < 0 or row['pid'] == node['pid']:
            raise ValueError('Distinct real worker PID/start_time identities required')
        seen.add(identity)
        parent, parent_start = row['parent_pid'], row['parent_start_time']
        if type(parent) is not int or parent <= 0 or type(parent_start) is not int or parent_start < 0 or \
                parent in (row['pid'], node['pid']):
            raise ValueError('Distinct native local parent PID/start_time required')
        if len({(item['parent_pid'], item['parent_start_time']) for item in rows
                if item['node_rank'] == node['node_rank']}) != 1:
            raise ValueError('Each node requires one native local worker parent')
        local = row['rank'] % 8
        uuids = [item['gpu_uuid'].removeprefix('GPU-') for item in rows
                 if item['node_rank'] == node['node_rank']]
        if set(uuids) != allocated[row['rank']//8] or len(uuids) != 8:
            raise ValueError('Actual worker GPU UUIDs must cover the allocated eight devices')
        if row['local_rank'] != local or row['gpu_uuid'].removeprefix('GPU-') not in allocated[row['rank']//8]:
            raise ValueError('All-rank actual GPU UUID mapping mismatch')
        if row['status'] != 'metadata_only_not_G2' or set(row['completed_stages']) != set(observation.STAGES) or \
                row['resource_evidence_errors']:
            raise ValueError('All native initialization phases and resource records must be observed on every rank')
        parallel = row['parallel_config']
        if any(parallel[key] != value for key, value in dict(tensor_parallel_size=16, data_parallel_size=1,
                pipeline_parallel_size=1, enable_expert_parallel=False, nnodes=2, node_rank=node['node_rank']).items()):
            raise ValueError('Actual native parallel topology mismatch')
        if row['configuration'] != dict(model_dtype='torch.bfloat16', gpu_memory_utilization=0.92,
                                       speculative=False, prefix_caching=False):
            raise ValueError('Actual native model/cache configuration mismatch')
        classes = row['module_classes']
        for name, count in expected['module_counts'].items():
            if sum(value for cls, value in classes.items() if cls.rsplit('.', 1)[-1] == name) != count:
                raise ValueError('Full wrapper/backbone census mismatch: '+name)
        if len(row['routed_experts']) != expected['routed_layers'] or any(
                item['count'] != expected['experts_per_layer'] for item in row['routed_experts']):
            raise ValueError('Reduced native routed expert census')
    return dict(status='capacity_initialization_and_metadata_census_observed_not_G2', ranks=16,
                remaining_g2={name: 'UNMEASURED' for name in REMAINING_G2})


def expected_census():
    config = json.loads((TASK/'reference_inputs/config.json').read_text())
    text = config['text_config']
    # The pinned inventory supplies construction expectations, not invented runtime facts.
    return dict(module_counts={'Step5DecoderLayer': text['num_hidden_layers'],
        'Step5Attention': text['num_hidden_layers'], 'Step5ForCausalLM': 1,
        'Step5ForConditionalGeneration': 1, 'PerceptionEncoder': 1,
        'PerceptionEncoderVisionBlock': config['vision_config']['layers']},
        routed_layers=len(text['moe_layers_enum'].split(',')), experts_per_layer=text['moe_num_experts'])


def publish_terminal(config, node, result, reason, publish=supervision.publish_marker):
    actual = dict(node, pid=os.getpid())
    return publish(Path(config['marker_root']), config['generation'], actual,
                   'workload_terminal', result, reason)


def execute_leader(config, node, sources, *, llm_factory=None, verify=installed_sources,
                   clock=time.monotonic, publish=publish_terminal):
    """Native failures re-raise; missing diagnostic evidence never fabricates G2 PASS."""
    if node['node_rank'] != 0:
        raise ValueError('Only node0 owns an LLM')
    if type(node['workload_deadline']) not in (int, float) or not math.isfinite(node['workload_deadline']):
        raise ValueError('Finite local monotonic workload deadline required')
    end = node['workload_deadline']
    try:
        verify(config, sources)
        if llm_factory is None:
            from vllm import LLM
            llm_factory = LLM
        llm = llm_factory(**engine_kwargs(config, node))
        result = 'FAIL'; reason = 'Capacity returned; census evidence unverified; not G2'
        remaining = end-clock()
        if remaining <= 0:
            raise TimeoutError('Capacity/census observation workload deadline')
        snapshots = llm.collective_rpc('capacity_snapshot', timeout=min(remaining, config['rpc_seconds']))
        try:
            save(Path(node['local_output'])/'all_rank_snapshots.json', dict(workers=snapshots))
            if clock() >= end:
                raise TimeoutError('Late capacity/census observation')
            nodes = [json.loads((Path(config['marker_root'])/f'node_{rank}_inventory.json').read_text())
                     for rank in (0, 1)]
            census = all_rank_census(snapshots, nodes, expected_census())
            save(Path(node['local_output'])/'all_rank_census.json', dict(census=census, workers=snapshots))
            result = 'PASS'; reason = 'Native capacity and all-rank metadata census returned; remaining G2 unmeasured'
        except Exception:
            diagnostic(lambda: save(Path(node['local_output'])/'census_unverified.json',
                                    dict(status='UNVERIFIED', traceback=traceback.format_exc())))
        diagnostic(lambda: publish(config, node, result, reason))
        return llm
    except BaseException:
        failure = traceback.format_exc()
        diagnostic(lambda: save(Path(node['local_output'])/'native_failure.json',
                                dict(status='BLOCKED_ON_RUNTIME', traceback=failure)))
        diagnostic(lambda: publish(config, node, 'FAIL', 'BLOCKED_ON_RUNTIME: '+failure))
        raise


def archive_records(local, destination, byte_limit, *, deadline, clock=time.monotonic):
    """Retain captured byte ranges, not future status, markers, or native log tails."""
    def check_deadline():
        if clock() >= deadline:
            raise TimeoutError('Archive copy exceeded the absolute worker deadline')
    check_deadline()
    local, destination = Path(local), Path(destination)
    source_root = local.resolve()
    if destination != destination.resolve():
        raise ValueError('Archive destination must remain a canonical path')
    files = []
    for path in sorted(local.rglob('*')):
        check_deadline()
        relative = path.relative_to(local)
        if 'cache' in relative.parts:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(source_root):
            raise ValueError('Archive evidence contains a symlink or escaped source')
        if path.is_file():
            files.append(dict(path=str(relative), bytes=path.stat().st_size))
    receipt = dict(verdict='PASS', scope='captured_bytes_only_not_final_status_or_G2',
        retained_files=files, retained_bytes=sum(row['bytes'] for row in files),
        byte_count_scope='source files listed above; archive_manifest.json is additional metadata',
        omitted=['completion markers published after this copy',
                 'peer completion exchange and durable final status after this copy',
                 'native stdout/stderr tails beyond the listed captured byte counts'])
    manifest = (json.dumps({key: value for key, value in receipt.items() if key != 'verdict'}, indent=2)+'\n').encode()
    if receipt['retained_bytes']+len(manifest) > byte_limit:
        raise ValueError('Archive evidence and manifest exceed approved byte limit')
    check_deadline()
    destination.mkdir(parents=True, exist_ok=False)
    for row in files:
        check_deadline()
        source_path, target = local/row['path'], destination/row['path']
        if source_path.is_symlink() or not source_path.resolve().is_relative_to(source_root):
            raise ValueError('Archive source escaped before copy')
        target.parent.mkdir(parents=True, exist_ok=True)
        with source_path.open('rb') as source, target.open('xb') as sink:
            remaining = row['bytes']
            while remaining:
                check_deadline()
                block = source.read(min(1024*1024, remaining))
                if not block:
                    raise ValueError('Archive source changed before its captured bytes were retained')
                sink.write(block); remaining -= len(block)
                check_deadline()
    check_deadline()
    with (destination/'archive_manifest.json').open('xb') as stream:
        stream.write(manifest)
    check_deadline()
    return dict(receipt, manifest_bytes=len(manifest))


def run_node(config, sources, config_path, *, env=None, run=subprocess.run, popen=subprocess.Popen,
             supervise=supervision.supervise, proc_root=Path('/proc'), deadline=None):
    entry = time.monotonic()
    archive_deadline = entry+supervision.SECONDS if deadline is None else deadline
    if isinstance(archive_deadline, bool) or not isinstance(archive_deadline, (int, float)) or \
            not math.isfinite(archive_deadline) or archive_deadline > entry+supervision.SECONDS:
        raise ValueError('Absolute worker deadline must not expand the approved ceiling')
    supervisor_deadline = archive_deadline-config['archive_reserve_seconds']
    env = dict(os.environ if env is None else env)
    if not os.path.ismount('/mnt/codesign-exp'):
        raise ValueError('Required actual cloud archive mount missing before native start')
    node = node_record(config, env, proc_root=proc_root)
    node['workload_deadline'] = supervisor_deadline-supervision.SHUTDOWN_SECONDS-supervision.RESIDUAL_SECONDS
    if node['workload_deadline']-time.monotonic() <= config['rpc_seconds']:
        raise ValueError('Node setup consumed the remaining workload/RPC budget')
    rank = node['node_rank']
    if (Path(config['archive_root'])/f'node_{rank}').exists():
        raise ValueError('Native archive node generation already exists')
    local = Path(config['worker_local_root'])/f'node_{rank}'
    local.mkdir(parents=True, exist_ok=False)
    records = local/'workers'; records.mkdir()
    node.update(gpu_uuids=gpu_inventory(config['resources'], run), local_output=str(local), observer_output=str(records),
                record_bytes=config['record_bytes'])
    node['cache_root'] = str(local/'cache')
    Path(node['cache_root']).mkdir()
    root = Path(config['marker_root']); root.mkdir(parents=True, exist_ok=True)
    save(root/f'node_{rank}_inventory.json', node)
    node_path = local/'node.json'; save(node_path, node)
    env.update(STEP5_CAPACITY_TASK=str(TASK), STEP5_CAPACITY_NODE_FILE=str(node_path),
               STEP5_CAPACITY_OBSERVER_SHA=config['observer_sha256'])
    env['PYTHONPATH'] = str(Path(__file__).parent)+os.pathsep+env.get('PYTHONPATH', '')
    env.update(XDG_CACHE_HOME=node['cache_root'], TMPDIR=str(local))
    inventory = WorkerInventory(records, node)
    argv = [sys.executable, '-B', str(Path(__file__).resolve()), 'leader' if rank == 0 else 'headless',
            '--config', str(config_path), '--node-file', str(node_path)]
    with (local/'native.stdout').open('xb') as out, (local/'native.stderr').open('xb') as err:
        if node['workload_deadline']-time.monotonic() <= config['rpc_seconds']:
            raise ValueError('Node setup consumed the remaining workload/RPC budget')
        def spawn(arguments):
            return inventory.spawn(arguments, lambda command: popen(command, env=env, stdout=out, stderr=err))
        def finalize_evidence(status):
            try:
                out.flush(); err.flush()
                supervision.evidence(status, 'save pre-completion status', lambda: save(
                    local/'pre_completion_status.json', dict(status, result='UNVERIFIED', return_code=None,
                    evidence_verdict='UNVERIFIED', phase='before_archive_completion_and_peer_exchange',
                    final_result_retained=False)))
                if not os.path.ismount('/mnt/codesign-exp'):
                    raise ValueError('Required cloud archive mount missing during finalization')
                return archive_records(local, Path(config['archive_root'])/f'node_{rank}', config['archive_bytes'],
                                       deadline=archive_deadline, clock=time.monotonic)
            except Exception:
                failure = traceback.format_exc()
                diagnostic(lambda: save(local/'archive_unverified.json', dict(status='UNVERIFIED', traceback=failure)))
                diagnostic(lambda: print('TP16_ARCHIVE_UNVERIFIED '+failure, flush=True))
                raise
        status = supervise(root, config['generation'], env, dict(engine=config['engine'], tp=16, dp=1, pp=1,
            ep=False, nnodes=2, workload=config['workload']), argv, inventory.workers, node['gpu_uuids'],
            popen=spawn, run=run, proc_root=proc_root, boot_path=proc_root/'sys/kernel/random/boot_id',
            worker_inventory=inventory.snapshot, deadline=supervisor_deadline,
            completion_deadline=archive_deadline, finalize_evidence=finalize_evidence)
    return status


def main():
    deadline = time.monotonic()+supervision.SECONDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'node', 'leader', 'headless'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--node-file', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.mode == 'preflight':
        print(json.dumps(preflight(config), indent=2)); return 0
    # Runtime validation must precede a native import or exec and accepts only the frozen receipt.
    receipt = preflight(config, new_generation=False)
    sources = receipt['sources']
    if sources != config['native_sources']:
        raise ValueError('Frozen preflight native source ledger mismatch')
    if args.mode == 'node':
        return run_node(config, sources, args.config, deadline=deadline)['return_code']
    node = json.loads(args.node_file.read_text())
    if args.mode == 'headless':
        installed_sources(config, sources)
        argv = headless_argv(config, node)
        os.execv(argv[0], argv)
    else:
        execute_leader(config, node, sources)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
