"""CPU composition checks for the private native request caller; no GPU qualification."""
import builtins
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import check_role_execution as caller


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'vllm', 'stepmind', 'steptron') and name not in sys.modules:
            raise AssertionError('CPU caller checks cannot import native or platform packages')
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError('CPU caller checks cannot start a child, query GPUs, or send signals')

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    monkeypatch.setattr(subprocess, 'Popen', forbidden)
    monkeypatch.setattr(subprocess, 'run', forbidden)
    monkeypatch.setattr(os, 'execv', forbidden)
    monkeypatch.setattr(os, 'kill', forbidden)


@pytest.fixture
def config(tmp_path):
    capacity = caller.capacity
    return dict(task_root=str(capacity.TASK), generation='cpu-request-caller',
        resources=dict(nodes=2, gpu_per_node=8, cpu_per_node=32, memory_mi_per_node=128000,
                       worker_seconds=2400, gpu_type='H800', charged_group='codesign'),
        image='a.z01.normandyfunc.com/docker-public/vllm/vllm-openai'
              '@sha256:69308454cff61240995e0a17b7baf62b84d1e5ac49afd224e718715adb6a7f20',
        vllm_version=capacity.VERSION,
        workload=dict(stage='native_requests', requests=2, prompt_tokens=4105, output_tokens=9),
        engine=dict(max_model_len=8192, max_num_batched_tokens=512, max_num_seqs=1),
        rpc_seconds=60, record_bytes=512*1024**2, archive_bytes=4*1024**3,
        archive_reserve_seconds=300,
        marker_root=str(capacity.TASK/'cpu_request_caller_unused_markers'),
        worker_local_root=str(tmp_path/'worker'),
        archive_root='/mnt/codesign-exp/ycfeng/step5-preview/cpu_request_caller_unused',
        native_cli='/cpu-fixture/bin/vllm',
        caller_sha256=capacity.sha256(HERE/'check_role_tp16.py'),
        observer_sha256=capacity.sha256(HERE/'role_capacity_observation.py'),
        supervisor_sha256=capacity.sha256(HERE/'role_capacity_supervision.py'),
        request_caller_sha256=capacity.sha256(HERE/'check_role_execution.py'),
        request_observer_sha256=capacity.sha256(HERE/'role_execution_observation.py'),
        model_input_observer_sha256=capacity.sha256(HERE/'role_model_input_observation.py'),
        request_manifest=str(capacity.TASK/'request_input_candidate_20261018_01.json'),
        request_manifest_sha256=capacity.sha256(capacity.TASK/'request_input_candidate_20261018_01.json'))


def test_preflight_preserves_capacity_contract_and_binds_request_sources(config):
    original = copy.deepcopy(config)
    receipt = caller.preflight(config)
    assert config == original
    assert receipt['scope'] == 'inert_request_preparation_not_native_qualification'
    assert receipt['capacity_preflight']['workload'] == {'stage': 'capacity_initialization'}
    assert receipt['request_inputs']['prompt_tokens'] == 4105
    assert receipt['request_inputs']['output_tokens'] == 9
    assert receipt['request_inputs']['changed_prompt_positions'] == [4104]
    assert receipt['request_inputs']['native_input_acceptance'] == 'UNMEASURED'
    assert receipt['sources']['vllm/compilation/cuda_graph.py'] == (
        '6e34824d2e62cd3bbd2eaee615ebfae3db542ce2d6be9b14d1b012398087fe2e')
    assert not Path(config['marker_root']).exists()
    with pytest.raises(ValueError, match='capacity/census only'):
        caller.capacity.preflight(config)


def retained_methods(relative, owner, names, namespace):
    """Execute only selected retained CPU method bodies, with external boundaries supplied."""
    import ast
    path = caller.capacity.TASK/relative
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                             *methods], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def original_cpu_method(relative, owner, name, namespace, defaults=()):
    """Load the pinned original code object with injected CPU globals, without importing its module."""
    from types import FunctionType
    path = caller.capacity.TASK/relative
    compiled = compile(path.read_bytes(), str(path), 'exec', dont_inherit=True,
                       optimize=sys.flags.optimize)
    code = caller.model_observation.nested_code(compiled, owner, name)
    return FunctionType(code, dict(__name__=__name__, **namespace), name, defaults)


class CpuSampling:
    """Native argument validation body with no native package or sampling kernels."""
    def __init__(self, **kwargs):
        self.__dict__.update(dict(n=1, presence_penalty=0.0, frequency_penalty=0.0,
            repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0,
            seed=None, max_tokens=16, min_tokens=0, ignore_eos=False, stop=[], stop_token_ids=[],
            extra_args=None, stream_interval=None, logprobs=None, prompt_logprobs=None,
            bad_words=[], detokenize=True))
        self.__dict__.update(copy.deepcopy(kwargs))
        self._verify_args()

    @classmethod
    def from_optional(cls, **kwargs):
        return cls(**kwargs)


import math
CpuSampling._verify_args = retained_methods(
    'native_role_sources/observed/vllm/sampling_params.py', 'SamplingParams', ('_verify_args',),
    dict(math=math, VLLMValidationError=ValueError,
         _verify_num_sequences=lambda value, name: None))['_verify_args']


class CpuRequestOutput:
    def __init__(self, request_id, tokens, output_tokens):
        from types import SimpleNamespace
        self.request_id, self.prompt_token_ids, self.finished = request_id, list(tokens), True
        self.encoder_prompt_token_ids = None
        self.outputs = [SimpleNamespace(index=0, token_ids=list(output_tokens), text='CPU fixture',
                                        finish_reason='length', stop_reason=None)]


def startup_fixture(config, tmp_path):
    fixture_path = HERE.parents[1]/'fixtures/step5_environment/tp16_capacity03_uuid_census.json'
    fixture = json.loads(fixture_path.read_text())
    nodes = copy.deepcopy(fixture['nodes'])
    for node in nodes:
        node.update(generation=config['generation'], observer_sha256=config['observer_sha256'],
                    workload_deadline=2010, local_output=str(tmp_path/'native_output'),
                    endpoint=dict(address='10.4.5.6', port=19540))
    nodes[0].update(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    pid_namespace=os.readlink('/proc/self/ns/pid'))
    rows = [copy.deepcopy(dict(fixture['common_worker_fields'],
        parallel_config=fixture['nodes'][row['node_rank']]['worker_parallel_config'], **row))
        for row in fixture['workers']]
    for row in rows:
        node = nodes[row['node_rank']]
        row.update(generation=config['generation'], observer_sha256=config['observer_sha256'],
                   boot_id=node['boot_id'], pid_namespace=node['pid_namespace'])
    root = tmp_path/'markers'; root.mkdir()
    config['marker_root'] = str(root)
    local = Path(nodes[0]['local_output']); local.mkdir()
    (local/'workers').mkdir()
    for node in nodes:
        (root/f"node_{node['node_rank']}_inventory.json").write_text(json.dumps(node))
    return nodes, rows


class CpuCoreRequest:
    def __init__(self, request_id, tokens, params):
        self.request_id, self.external_req_id = request_id, None
        self.prompt_token_ids = list(tokens)
        self.sampling_params, self.pooling_params = params, None
        self.prompt_embeds, self.resumable = None, False

    @property
    def params(self):
        return self.sampling_params


class CpuRequestState:
    @staticmethod
    def from_new_request(*, request, parent_req, request_index, **kwargs):
        from types import SimpleNamespace
        params = request.sampling_params
        return SimpleNamespace(request_id=request.request_id, external_req_id=request.external_req_id,
            prompt_token_ids=request.prompt_token_ids, prompt_embeds=request.prompt_embeds,
            parent_req=parent_req, request_index=request_index, max_tokens_param=params.max_tokens,
            n=params.n, top_p=params.top_p, temperature=params.temperature)


class CpuOutputProcessor:
    def __init__(self):
        from collections import defaultdict
        self.request_states, self.external_req_ids, self.parent_requests = {}, defaultdict(list), {}
        self.tokenizer, self.log_stats, self.stream_interval, self.admission_stats = None, False, 1, None

    def _update_streaming_request_state(self, state, request, prompt):
        self.updated_state = state


for name, defaults in [('add_request', (None, 0, None)), ('_update_admission_stats', ()),
                       ('_finish_request', ())]:
    setattr(CpuOutputProcessor, name, original_cpu_method(
        'native_role_sources/observed_generation5/vllm/v1/engine/output_processor.py',
        'OutputProcessor', name, dict(RequestState=CpuRequestState), defaults))


class CpuEngine:
    def __init__(self, llm):
        from types import SimpleNamespace
        self.llm, self.active, self.steps, self.output = llm, False, [], None
        self.output_processor = CpuOutputProcessor()
        self.input_processor = SimpleNamespace(assign_request_id=original_cpu_method(
            'native_role_sources/observed_generation5/vllm/v1/engine/input_processor.py',
            'InputProcessor', 'assign_request_id', dict(envs=SimpleNamespace(
                VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=False), random_uuid=lambda: 'cpu-'+str(len(llm.generated)))))
        self.input_processor.resolve_watermarking = lambda params: None
        self.input_processor.apply_watermarking = lambda params, resolved: None
        self.engine_core = SimpleNamespace(add_request=self.dispatch)

    def add(self, prompts, params):
        assert not self.active, 'B cannot overlap A'
        assert len(prompts) == 1 and set(prompts[0]) == {'prompt_token_ids'}
        tokens = prompts[0]['prompt_token_ids']
        self.llm.events.append(('generate', list(tokens), copy.deepcopy(vars(params))))
        request = CpuCoreRequest(str(len(self.llm.generated)), tokens, copy.deepcopy(params))
        self.add_request(request.request_id, request, params)

    def dispatch(self, request):
        # This replaces only the actual EngineCore/device boundary in the CPU fixture.
        state = self.output_processor.request_states[request.request_id]
        assert state.external_req_id == request.external_req_id
        self.current_request = request
        self.active = True
        self.steps = [512]*8+[9]+[1]*8
        self.computed = 0
        self.output = CpuRequestOutput(request.external_req_id, request.prompt_token_ids, [67]*9)
        self.llm.generated.append(self.output)

    def has_unfinished_requests(self):
        return self.active

    def step(self):
        from types import SimpleNamespace
        import numpy as np
        count = self.steps.pop(0)
        request_id = self.current_request.request_id
        work = SimpleNamespace(total_num_scheduled_tokens=count, num_scheduled_tokens={request_id: count})
        prefilling = self.computed < 4105
        for worker in self.llm.workers:
            worker.model_runner.execute_model_state = SimpleNamespace(input_batch=SimpleNamespace(
                req_ids=[request_id], num_reqs=1, num_reqs_after_padding=1,
                num_tokens=count, num_tokens_after_padding=count,
                idx_mapping_np=np.array([0]), num_scheduled_tokens=np.array([count]),
                query_start_loc_np=np.array([0, count]), num_computed_tokens_np=np.array([self.computed]),
                prefill_len_np=np.array([4105]),
                num_computed_prefill_tokens_np=np.array([min(self.computed, 4105)]),
                is_prefilling_np=np.array([prefilling]),
                prefill_runs_as_decode_np=np.array([prefilling and count == 1])))
            assert worker.execute_model(work) is None
            worker.sample_tokens(None)
        self.computed += count
        self.active = bool(self.steps)
        if not self.active:
            self.output_processor._finish_request(self.output_processor.request_states[request_id])
        return [] if self.active else [self.output]


from types import SimpleNamespace
CpuEngine.add_request = original_cpu_method(
    'native_role_sources/observed/vllm/v1/engine/llm_engine.py', 'LLMEngine', 'add_request',
    dict(EngineCoreRequest=CpuCoreRequest, SamplingParams=CpuSampling,
         logger=SimpleNamespace(warning_once=lambda message: None)),
    (None, None, None, None, 0, None, None, None))


class CpuLLM:
    def __init__(self, rows, events, **kwargs):
        from types import SimpleNamespace
        self.kwargs, self.rows, self.events, self.generated = kwargs, rows, events, []
        self.default_sampling_params = None
        self.model_config = SimpleNamespace(runner_type='generate', get_diff_sampling_param=lambda: dict(
            repetition_penalty=1.125, presence_penalty=0.25, frequency_penalty=-0.1,
            temperature=0.75, top_k=17, top_p=0.92, min_p=0.03))
        self.llm_engine = CpuEngine(self)
        self.workers = []
        self.records = []

        def native_execute(worker, scheduler_output):
            return None

        def native_sample(worker, grammar_output):
            worker.model_runner.execute_model_state = None
            return None

        base = type('CpuNativeWorker', (), dict(execute_model=native_execute, sample_tokens=native_sample,
                    **{name: lambda self: None for name in caller.observation.capacity.STAGES}))
        observed = caller.observation.observed_worker(base, record=lambda worker, payload: self.records.append(payload))
        for row in rows:
            worker = observed()
            worker.rank, worker.local_rank = row['rank'], row['local_rank']
            worker._capacity_identity = {key: row[key] for key in ('rank', 'local_rank', 'pid', 'start_time',
                'generation', 'node_rank', 'boot_id', 'pid_namespace', 'observer_sha256',
                'parent_pid', 'parent_start_time')}
            worker.model_runner = SimpleNamespace(execute_model_state=None)
            self.workers.append(worker)

    def _add_completion_requests(self, prompts, params, **kwargs):
        self.llm_engine.add(prompts, params)

    def collective_rpc(self, method, timeout):
        self.events.append(('rpc', method, timeout))
        if method == 'capacity_snapshot':
            return copy.deepcopy(self.rows)
        assert method == 'request_snapshot'
        # PID is an external per-process boundary; each worker still uses the real snapshot API.
        result = []
        original = os.getpid
        try:
            for worker in self.workers:
                os.getpid = lambda: worker._capacity_identity['pid']
                result.append(worker.request_snapshot())
        finally:
            os.getpid = original
        return result


for name, method in retained_methods('native_role_sources/observed/vllm/entrypoints/llm.py', 'LLM',
        ('generate', 'get_default_sampling_params'), dict(SamplingParams=CpuSampling,
                                                        RequestOutput=CpuRequestOutput)).items():
    setattr(CpuLLM, name, method)
for name, method in retained_methods('native_role_sources/observed_generation5/vllm/entrypoints/common/offline.py',
        'OfflineInferenceMixin', ('_run_completion', '_run_engine'), dict(RequestOutput=CpuRequestOutput)).items():
    setattr(CpuLLM, name, method)


def run_leader(config, tmp_path, *, configure=lambda llm: None, clock=lambda: 10):
    nodes, rows = startup_fixture(config, tmp_path)
    events = []
    llm = CpuLLM(rows, events)
    configure(llm)
    returned = caller.execute_leader(config, nodes[0], {}, llm_factory=lambda **kwargs: llm,
        sampling_factory=CpuSampling, verify=lambda *args: None, clock=clock)
    assert returned is llm
    return llm, nodes, events


def test_real_generate_body_composes_current_worker_and_separate_capacity_receipts(config, tmp_path):
    llm, nodes, events = run_leader(config, tmp_path)
    generation_events = [row for row in events if row[0] == 'generate']
    assert len(generation_events) == 2 and len(generation_events[0][1]) == 4105
    assert generation_events[0][1][-1] == 67 and generation_events[1][1][-1] == 68
    assert generation_events[0][2]['temperature'] == 0.75
    assert generation_events[0][2]['top_k'] == 17 and generation_events[0][2]['top_p'] == 0.92
    assert generation_events[0][2]['presence_penalty'] == 0.25
    assert generation_events[0][2]['seed'] is None
    assert generation_events[0][2]['min_tokens'] == generation_events[0][2]['max_tokens'] == 9
    assert llm.default_sampling_params['temperature'] == 0.75
    assert llm.default_sampling_params.get('min_tokens') is None
    local = Path(nodes[0]['local_output'])
    raw = json.loads((local/'all_rank_snapshots.json').read_text())
    census = json.loads((local/'all_rank_census.json').read_text())
    assert raw['workers'] == llm.rows
    assert {row['status'] for row in raw['workers']} == {'metadata_only_not_G2'}
    assert census['census']['status'] == 'capacity_initialization_and_metadata_census_observed_not_G2'
    receipt = json.loads((local/'request_evidence.json').read_text())
    assert receipt['scope'] == 'native_request_results_and_cpu_metadata_only_not_G2'
    assert receipt['status'] == 'COMPLETED_PUBLIC_OUTPUTS_ONLY_NOT_G2'
    assert receipt['completed_requests'] == ['A', 'B']
    assert receipt['terminal_publication'] == 'UNMEASURED'
    assert receipt['terminal_marker'] == str(Path(config['marker_root'])/'node_0_workload_terminal.json')
    assert set(receipt['evidence_status'].values()) == {'UNMEASURED'}
    assert receipt['sampling_observation_scope'] == 'original_frontend_resolved_fields_device_consumption_unmeasured'
    assert len(receipt['frontend_associations']) == 2
    for request, joined in zip(llm.generated, receipt['frontend_associations']):
        assert joined['public_request_id'] == request.request_id
        assert joined['internal_request_id'] != request.request_id
        assert joined['worker_request_join'] == joined['device_consumption'] == 'UNMEASURED'
    assert not llm.llm_engine.output_processor.request_states
    assert not llm.llm_engine.output_processor.external_req_ids
    for name in ('A', 'B'):
        admitted = json.loads((local/f'request_{name}_admission.json').read_text())
        assert admitted['status'] == 'OBSERVED_FRONTEND_ADMISSION_ONLY_NOT_G2'
        assert admitted['source_status'] == 'LOADED_CODE_MATCHES_PINNED_SOURCE'
        row = admitted['records'][0]
        assert row['native_registration_outcome'] == 'returned'
        assert row['engine_core_submission'] == 'ORIGINAL_ENGINE_RETURNED_ID'
        assert row['resolved_matches_submitted'] and row['resolved_is_submitted'] is False
        assert row['resolved_sampling_fields'] == receipt['submitted_sampling_fields']
        assert row['context']['name'] == name
        assert row['context']['manifest_sha256'] == config['request_manifest_sha256']
        assert row['frontend_identity']['pid'] == os.getpid()
        assert row['registration_line'] == 642 and row['engine_registration_line'] == 297
    after_a = json.loads((local/'request_A_snapshots.json').read_text())
    assert len(after_a['workers']) == 16
    assert after_a['workers'][0]['request_metadata']['invocations'] == 17
    assert after_a['workers'][0]['evidence_status']['content'] == 'UNMEASURED'
    assert json.loads(llm.records[0])['prepared_batch']['num_tokens'] == 512
    terminal = json.loads((Path(config['marker_root'])/'node_0_workload_terminal.json').read_text())
    assert terminal['pid'] == os.getpid() and terminal['result'] == 'PASS'
    assert 'request_evidence.json retained sha256='+caller.capacity.sha256(local/'request_evidence.json') in terminal['reason']


class CpuClock:
    def __init__(self, now=100):
        self.now = now

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def native_node_attempt(config, tmp_path, monkeypatch, *, rank=0, archive_bytes=None,
                        setup_seconds=0, raw_code=0, prepare_output=lambda node: None):
    """Use real node setup, PID inventory, supervisor and archive with external CPU boundaries."""
    import signal
    from types import SimpleNamespace
    clock = CpuClock()
    original_config = copy.deepcopy(config)
    config['marker_root'] = str(tmp_path/'shared_markers')
    config['archive_root'] = str(tmp_path/'archive')
    if archive_bytes is not None:
        config['archive_bytes'] = archive_bytes
    node = dict(generation=config['generation'], node_rank=rank, pid=300+rank,
        boot_id='cpu-boot-'+str(rank), pid_namespace='cpu-pid-namespace',
        observer_sha256=config['observer_sha256'], endpoint=dict(address='10.4.5.6', port=19540))
    peer = dict(node_rank=1-rank, pid=301-rank, boot_id='cpu-boot-'+str(1-rank), result='PASS')
    events = []
    env = dict(NODE_RANK=str(rank), NODE_COUNT='2', PROC_PER_NODE='8',
        MASTER_ADDR=node['endpoint']['address'], MASTER_PORT=str(node['endpoint']['port']),
        NCCL_IB_DISABLE='0', NCCL_SOCKET_IFNAME='native-platform-value',
        PYTHONPATH='cpu-original-path')
    monkeypatch.setattr(caller.capacity.time, 'monotonic', clock)
    monkeypatch.setattr(caller.capacity.os.path, 'ismount', lambda path: path == '/mnt/codesign-exp')

    def identify(passed_env, *args, **kwargs):
        assert passed_env['MASTER_PORT'] == env['MASTER_PORT']
        clock.now += setup_seconds
        return node

    monkeypatch.setattr(caller.capacity.supervision, 'node_identity', identify)
    original_node_record = caller.capacity.node_record
    monkeypatch.setattr(caller.capacity, 'node_record',
        lambda *args, **kwargs: original_node_record(*args, **kwargs, identify=identify))
    namespace = tmp_path/'cpu_proc/self/ns/pid'
    namespace.parent.mkdir(parents=True)
    namespace.symlink_to(node['pid_namespace'])

    class Connection:
        def bind(self, address):
            events.append(('bind', address))
        def listen(self, count):
            return None
        def connect(self, address):
            events.append(('connect', address))
        def settimeout(self, seconds):
            assert 0 < seconds <= 2010
        def accept(self):
            return Connection(), ('10.4.5.7', 1)
        def recv(self, count):
            return b''
        def close(self):
            return None

    def socket_factory(*args):
        if not (Path(config['marker_root'])/f'node_{1-rank}_prepared.json').exists():
            caller.capacity.supervision.publish_marker(Path(config['marker_root']), config['generation'],
                peer, 'prepared', 'PASS', 'CPU peer', clock=clock, endpoint=node['endpoint'],
                engine_settings=dict(engine=config['engine'], tp=16, dp=1, pp=1, ep=False,
                    nnodes=2, workload=config['workload'], request_sources=dict(
                        caller_sha256=config['request_caller_sha256'],
                        observer_sha256=config['request_observer_sha256'],
                        model_input_observer_sha256=config['model_input_observer_sha256']),
                    request_manifest_sha256=config['request_manifest_sha256']))
        return Connection()

    def query(argv, **kwargs):
        if '--query-gpu=index,uuid,name' in argv:
            return SimpleNamespace(returncode=0, stdout=''.join(
                f'{i}, GPU-cpu-node-{rank}-device-{i}, NVIDIA H800\n' for i in range(8)), stderr='')
        assert argv == caller.capacity.supervision.COMPUTE_QUERY
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    child = SimpleNamespace(pid=444, signals=[], exited=rank == 0)
    child.poll = lambda: raw_code if child.exited and rank == 0 else -signal.SIGTERM if child.exited else None
    def send_signal(number):
        assert number == signal.SIGTERM
        child.signals.append(number)
        child.exited = True
    child.send_signal = send_signal

    def popen(argv, **kwargs):
        events.append(('popen', list(argv), kwargs['env']))
        assert argv[2] == str(HERE/'check_role_execution.py')
        passed_env = kwargs['env']
        assert passed_env['NCCL_SOCKET_IFNAME'] == 'native-platform-value'
        assert passed_env['NCCL_IB_DISABLE'] == '0'
        assert passed_env['STEP5_REQUEST_CALLER_SHA'] == config['request_caller_sha256']
        assert passed_env['STEP5_REQUEST_OBSERVER_SHA'] == config['request_observer_sha256']
        assert passed_env['STEP5_MODEL_INPUT_OBSERVER_SHA'] == config['model_input_observer_sha256']
        actual_node = json.loads(Path(passed_env['STEP5_CAPACITY_NODE_FILE']).read_text())
        node.update(actual_node)
        assert actual_node['observer_sha256'] == original_config['observer_sha256']
        records = Path(actual_node['observer_output'])
        for local_rank in range(8):
            row = dict(node, rank=rank*8+local_rank, local_rank=local_rank,
                pid=1000+rank*8+local_rank, start_time=77+local_rank,
                parent_pid=child.pid if rank else 555, parent_start_time=88)
            (records/f"worker_{row['rank']}_{row['pid']}.json").write_text(json.dumps(row))
        kwargs['stdout'].write(b'CPU native child stdout\xff\n'); kwargs['stdout'].flush()
        kwargs['stderr'].write(b'CPU native child stderr\xfe\n'); kwargs['stderr'].flush()
        prepare_output(node)
        leader = dict(peer if rank else node, node_rank=0, pid=42 if rank else child.pid)
        if rank:
            caller.capacity.supervision.publish_marker(Path(config['marker_root']), config['generation'],
                dict(peer, node_rank=0), 'native_started', 'PASS', 'CPU peer child', clock=clock, child_pid=42)
        caller.capacity.supervision.publish_marker(Path(config['marker_root']), config['generation'],
            leader, 'workload_terminal', 'PASS', 'CPU generation returned', clock=clock)
        caller.capacity.supervision.publish_marker(Path(config['marker_root']), config['generation'],
            peer, 'completion', 'PASS', 'CPU peer archive done', clock=clock,
            final_status=dict(result='PASS'))
        return child

    def supervise(*args, **kwargs):
        return caller.capacity.supervision.supervise(*args, **kwargs,
            clock=clock, sleep=clock.sleep, socket_factory=socket_factory)

    status = caller.run_node(config, {}, tmp_path/'request_config.json', env=env,
        run=query, popen=popen, supervise=supervise, proc_root=tmp_path/'cpu_proc', deadline=2500)
    return status, node, child, events, clock


@pytest.mark.parametrize('rank', [0, 1])
def test_real_node_lifecycle_retains_request_argv_pins_and_capacity_labels(config, tmp_path, monkeypatch, rank):
    status, node, child, events, clock = native_node_attempt(config, tmp_path, monkeypatch, rank=rank)
    assert status['return_code'] == 0 and status['scope'] == 'capacity_only_not_G2'
    assert status['native_argv'][2] == str(HERE/'check_role_execution.py')
    assert status['engine_settings']['workload']['stage'] == 'native_requests'
    assert status['worker_inventory']['complete']
    assert node['workload_deadline'] == 2110 and clock() <= 2500
    archive = Path(config['archive_root'])/f'node_{rank}'
    assert (archive/'native.stdout').read_bytes() == b'CPU native child stdout\xff\n'
    assert (archive/'native.stderr').read_bytes() == b'CPU native child stderr\xfe\n'
    raw = json.loads((archive/'pre_completion_status.json').read_text())
    assert raw['phase'] == 'before_archive_completion_and_peer_exchange'
    assert raw['final_result_retained'] is False and raw['result'] == 'UNVERIFIED'
    assert len(list((archive/'workers').glob('worker_*.json'))) == 8
    assert not (archive/'cache').exists()
    assert status['raw_exit_code'] == (0 if rank == 0 else -15)
    assert child.signals == ([] if rank == 0 else [15])


@pytest.mark.parametrize('mode', ['preflight', 'node', 'leader', 'headless'])
def test_cli_uses_frozen_source_ledger_and_original_deadline(config, tmp_path, monkeypatch, capsys, mode):
    sources = caller.preflight(config)['sources']
    config['native_sources'] = sources
    config_path = tmp_path/'config.json'
    config_path.write_text(json.dumps(config))
    node_path = tmp_path/'node.json'
    node = dict(node_rank=1 if mode == 'headless' else 0,
                endpoint=dict(address='10.4.5.6', port=19540))
    node_path.write_text(json.dumps(node))
    monkeypatch.setattr(sys, 'argv', ['check_role_execution.py', mode, '--config', str(config_path),
                                     '--node-file', str(node_path)])
    clock = CpuClock()
    monkeypatch.setattr(caller.time, 'monotonic', clock)
    events = []
    original_preflight = caller.preflight

    def preflight(*args, **kwargs):
        events.append(('preflight', kwargs.get('new_generation', True)))
        clock.now += 80
        return original_preflight(*args, **kwargs)

    monkeypatch.setattr(caller, 'preflight', preflight)
    monkeypatch.setattr(caller.capacity, 'installed_sources',
                        lambda *args: events.append(('installed', args)))
    monkeypatch.setattr(caller, 'execute_leader', lambda *args: events.append(('leader', args)))

    def node_entry(*args, **kwargs):
        events.append(('node', args, kwargs))
        assert kwargs['deadline'] == 2500
        return dict(return_code=7)

    monkeypatch.setattr(caller, 'run_node', node_entry)

    class ExecBoundary(Exception):
        pass

    def exec_native(executable, argv):
        events.append(('exec', executable, argv))
        raise ExecBoundary('CPU native exec boundary')

    monkeypatch.setattr(caller.os, 'execv', exec_native)
    if mode == 'headless':
        with pytest.raises(ExecBoundary):
            caller.main()
        assert events[-1][1] == config['native_cli']
        argv = events[-1][2]
        assert argv[argv.index('--worker-cls')+1] == 'check_role_execution.RequestWorker'
        assert events[-2][0] == 'installed'
    else:
        assert caller.main() == (7 if mode == 'node' else 0)
        if mode == 'preflight':
            result = json.loads(capsys.readouterr().out)
            assert result['scope'] == 'inert_request_preparation_not_native_qualification'
            assert events == [('preflight', True)]
        else:
            assert events[-1][0] == mode
            assert events[-1][1][1] == (sources if mode == 'node' else node)
    assert events[0] == ('preflight', mode == 'preflight')


def test_cli_rejects_unfrozen_native_sources_before_native_dispatch(config, tmp_path, monkeypatch):
    config['native_sources'] = {}
    path = tmp_path/'config.json'; path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, 'argv', ['check_role_execution.py', 'node', '--config', str(path)])
    monkeypatch.setattr(caller, 'run_node', lambda *args, **kwargs: pytest.fail('Native dispatch before freeze'))
    with pytest.raises(ValueError, match='native source ledger'):
        caller.main()


@pytest.mark.parametrize('fault', ['source', 'constructor', 'generate_A', 'generate_B',
                                  'initial_rpc', 'request_rpc', 'final_rpc'])
def test_native_and_admission_failure_identity_survives_receipt_and_terminal_storage_faults(config, tmp_path, monkeypatch, fault):
    nodes, rows = startup_fixture(config, tmp_path)
    events = []
    llm = CpuLLM(rows, events)
    native_error = RuntimeError('exact native '+fault)
    original_generate, original_rpc = llm.generate, llm.collective_rpc
    capacity_calls = [0]

    def generate(*args, **kwargs):
        name = 'A' if not llm.generated else 'B'
        if fault == 'generate_'+name:
            raise native_error
        return original_generate(*args, **kwargs)

    def rpc(method, timeout):
        if method == 'capacity_snapshot':
            capacity_calls[0] += 1
        if (fault == 'initial_rpc' and capacity_calls[0] == 1 or
                fault == 'final_rpc' and method == 'capacity_snapshot' and capacity_calls[0] == 2 or
                fault == 'request_rpc' and method == 'request_snapshot'):
            raise native_error
        return original_rpc(method, timeout)

    llm.generate, llm.collective_rpc = generate, rpc
    def factory(**kwargs):
        if fault == 'constructor':
            raise native_error
        return llm
    def verify(*args):
        if fault == 'source':
            raise native_error
    saves = []
    original_save = caller.capacity.save
    def save(path, value):
        saves.append((path.name, copy.deepcopy(value)))
        if path.name in ('native_failure.json', 'request_evidence.json'):
            raise OSError('faulted diagnostic storage')
        return original_save(path, value)
    monkeypatch.setattr(caller.capacity, 'save', save)
    terminals = []
    def publish(*args):
        terminals.append(args)
        raise OSError('faulted terminal storage')
    with pytest.raises(RuntimeError) as caught:
        caller.execute_leader(config, nodes[0], {}, llm_factory=factory,
            sampling_factory=CpuSampling, verify=verify, clock=lambda: 10, publish=publish)
    assert caught.value is native_error
    if fault == 'source':
        admission = next(value for name, value in saves if name == 'request_evidence.json')
        assert admission['status'] == 'UNVERIFIED' and admission['evidence_verdict'] == 'UNVERIFIED'
        failure = admission['failure']
        assert not any(name == 'native_failure.json' for name, _ in saves)
        assert not events and not llm.generated
    else:
        failure = next(value for name, value in saves if name == 'native_failure.json')
        assert failure['status'] == 'BLOCKED_ON_RUNTIME'
    assert failure['engine_kwargs']['worker_cls'] == 'check_role_execution.RequestWorker'
    assert failure['config']['request_manifest_sha256'] == config['request_manifest_sha256']
    expected_phase = dict(source='installed_source_verification', constructor='native_initialization',
        generate_A='native_generate_A', generate_B='native_generate_B',
        initial_rpc='native_initial_capacity_rpc', request_rpc='native_request_A_rpc',
        final_rpc='native_final_capacity_rpc')
    assert failure['phase'] == expected_phase[fault]
    assert str(native_error) in failure['traceback']
    assert terminals[-1][2] == 'FAIL'
    if fault == 'generate_B':
        assert failure['request']['name'] == 'B'
        assert failure['submitted_sampling_fields']['temperature'] == 0.75
        assert failure['submitted_sampling_fields']['seed'] is None


@pytest.mark.parametrize('fault', ['length', 'finish_reason', 'stop_reason', 'prompt', 'completion_count',
                                  'no_output', 'negative_token'])
def test_invalid_public_completion_is_retained_without_fabricating_B_or_native_failure(config, tmp_path, fault):
    def configure(llm):
        original = llm.generate
        def generate(*args, **kwargs):
            outputs = original(*args, **kwargs)
            output = outputs[0]
            if fault == 'length':
                output.outputs[0].token_ids.pop()
            elif fault == 'finish_reason':
                output.outputs[0].finish_reason = 'stop'
            elif fault == 'stop_reason':
                output.outputs[0].stop_reason = 0
            elif fault == 'prompt':
                output.prompt_token_ids[-1] = 68
            elif fault == 'completion_count':
                output.outputs.append(copy.deepcopy(output.outputs[0]))
            elif fault == 'negative_token':
                output.outputs[0].token_ids[0] = -1
            else:
                outputs = []
            return outputs
        llm.generate = generate
    llm, nodes, events = run_leader(config, tmp_path, configure=configure)
    local = Path(nodes[0]['local_output'])
    raw = json.loads((local/'request_A_outputs.json').read_text())
    receipt = json.loads((local/'request_evidence.json').read_text())
    assert raw['outputs'] == caller.public_output([] if fault == 'no_output' else llm.generated)
    assert receipt['status'] == 'UNVERIFIED' and receipt['result'] == 'FAIL'
    assert receipt['native_generation_returns'] == ['A'] and receipt['completed_requests'] == []
    assert len([row for row in events if row[0] == 'generate']) == 1
    assert not (local/'native_failure.json').exists()
    assert json.loads((Path(config['marker_root'])/'node_0_workload_terminal.json').read_text())['result'] == 'FAIL'


@pytest.mark.parametrize('fault', ['missing_rank', 'wrong_pid', 'source', 'missing_metadata',
                                  'collection_stopped', 'no_B_progress', 'runner_observer'])
def test_request_metadata_faults_keep_capacity_originals_and_healthy_native_generation(config, tmp_path, fault):
    def configure(llm):
        original = llm.collective_rpc
        first = [None]
        def rpc(method, timeout):
            rows = original(method, timeout)
            if method == 'request_snapshot':
                if first[0] is None:
                    first[0] = copy.deepcopy(rows)
                elif fault == 'no_B_progress':
                    return copy.deepcopy(first[0])
                if fault == 'missing_rank':
                    rows.pop()
                elif fault == 'wrong_pid':
                    rows[0]['worker_identity']['pid'] += 1
                elif fault == 'source':
                    rows[0]['source']['observer_sha256'] = 'unfrozen'
                elif fault == 'missing_metadata':
                    rows[0].pop('request_metadata')
                elif fault == 'collection_stopped':
                    rows[0]['request_metadata']['collection_stopped'] = True
                elif fault == 'runner_observer':
                    rows[0]['runner_observer_error'] = dict(status='UNVERIFIED')
            return rows
        llm.collective_rpc = rpc
    llm, nodes, events = run_leader(config, tmp_path, configure=configure)
    local = Path(nodes[0]['local_output'])
    receipt = json.loads((local/'request_evidence.json').read_text())
    assert receipt['status'] == 'UNVERIFIED' and receipt['result'] == 'FAIL'
    assert receipt['completed_requests'] == ['A', 'B']
    assert len([row for row in events if row[0] == 'generate']) == 2
    assert {row['status'] for row in json.loads((local/'all_rank_snapshots.json').read_text())['workers']} == {'metadata_only_not_G2'}
    assert json.loads((local/'final_all_rank_census.json').read_text())['census']['ranks'] == 16
    assert not (local/'native_failure.json').exists()
    raw = json.loads((local/'request_A_snapshots.json').read_text())['workers']
    assert len(raw) == (15 if fault == 'missing_rank' else 16)


@pytest.mark.parametrize('fault', ['byte_limit', 'deadline', 'existing_file'])
def test_request_receipt_honors_original_record_budget_and_exclusive_storage(config, tmp_path, fault):
    local = tmp_path/'receipt_storage'
    local.mkdir()
    node = dict(local_output=str(local), workload_deadline=100)
    value = dict(text='漢')
    target = local/'request.json'
    if fault == 'byte_limit':
        # The inherited writer escapes non-ASCII characters; UTF-8 text alone undercounts it.
        config['record_bytes'] = len((json.dumps(value, indent=2, ensure_ascii=False)+'\n').encode('utf-8'))
        with pytest.raises(ValueError, match='record budget'):
            caller.retain(config, node, target.name, value, clock=lambda: 10)
        assert not target.exists()
    elif fault == 'deadline':
        with pytest.raises(TimeoutError, match='original workload deadline'):
            caller.retain(config, node, target.name, value, clock=lambda: 100)
        assert not target.exists()
    else:
        target.write_bytes(b'original capacity bytes')
        with pytest.raises(FileExistsError):
            caller.retain(config, node, target.name, value, clock=lambda: 10)
        assert target.read_bytes() == b'original capacity bytes'


def test_terminal_storage_failure_is_separate_from_successful_native_generation(config, tmp_path, capsys):
    nodes, rows = startup_fixture(config, tmp_path)
    events = []
    llm = CpuLLM(rows, events)
    terminals = []
    def publish(*args):
        terminals.append(args)
        raise OSError('native child terminal storage unavailable')
    assert caller.execute_leader(config, nodes[0], {}, llm_factory=lambda **kwargs: llm,
        sampling_factory=CpuSampling, verify=lambda *args: None, clock=lambda: 10,
        publish=publish) is llm
    local = Path(nodes[0]['local_output'])
    receipt = json.loads((local/'request_evidence.json').read_text())
    assert receipt['completed_requests'] == ['A', 'B']
    assert receipt['terminal_publication'] == 'UNMEASURED'
    assert not Path(receipt['terminal_marker']).exists()
    assert 'native child terminal storage unavailable' in capsys.readouterr().out
    assert not (local/'native_failure.json').exists()
    assert len(terminals) == 1


@pytest.mark.parametrize('filename', ['request_A_outputs.json', 'request_B_snapshots.json'])
def test_request_storage_faults_keep_healthy_generation_and_mark_separate_evidence(config, tmp_path, monkeypatch,
                                                                                filename):
    original_save = caller.capacity.save
    def save(path, value):
        if path.name == filename:
            raise OSError('request receipt storage unavailable')
        return original_save(path, value)
    monkeypatch.setattr(caller.capacity, 'save', save)
    llm, nodes, events = run_leader(config, tmp_path)
    receipt = json.loads((Path(nodes[0]['local_output'])/'request_evidence.json').read_text())
    assert receipt['completed_requests'] == ['A', 'B']
    assert receipt['status'] == 'UNVERIFIED' and receipt['result'] == 'FAIL'
    assert receipt['evidence_verdict'] == 'UNVERIFIED'
    assert len([event for event in events if event[0] == 'generate']) == 2
    assert not (Path(nodes[0]['local_output'])/'native_failure.json').exists()
    terminal = json.loads((Path(config['marker_root'])/'node_0_workload_terminal.json').read_text())
    assert terminal['result'] == 'FAIL'


@pytest.mark.parametrize('fault', ['io', 'deadline', 'bytes'])
def test_final_request_receipt_failure_prevents_unique_terminal_PASS_without_native_failure(
        config, tmp_path, monkeypatch, fault):
    nodes, rows = startup_fixture(config, tmp_path)
    local = Path(nodes[0]['local_output'])
    clock = CpuClock(now=10)
    events = []
    llm = CpuLLM(rows, events)
    final_save_error = OSError('final request receipt I/O unavailable')
    original_save = caller.capacity.save
    attempts = []
    def save(path, value):
        attempts.append(path.name)
        if fault == 'io' and path.name == 'request_evidence.json':
            raise final_save_error
        result = original_save(path, value)
        if path.name == 'final_all_rank_census.json':
            if fault == 'deadline':
                clock.now = nodes[0]['workload_deadline']
            elif fault == 'bytes':
                used = sum(item.stat().st_size for item in local.rglob('*') if item.is_file())
                with (local/'native_shutdown.log').open('xb') as stream:
                    stream.truncate(config['record_bytes']-used)
        return result
    monkeypatch.setattr(caller.capacity, 'save', save)
    publications = []
    def publish(*args):
        assert not (local/'request_evidence.json').exists()
        publications.append(args)
        return caller.capacity.publish_terminal(*args)
    assert caller.execute_leader(config, nodes[0], {}, llm_factory=lambda **kwargs: llm,
        sampling_factory=CpuSampling, verify=lambda *args: None, clock=clock, publish=publish) is llm
    assert len(llm.generated) == 2
    assert [event[1] for event in events if event[0] == 'rpc'] == [
        'capacity_snapshot', 'request_snapshot', 'request_snapshot', 'capacity_snapshot']
    assert (local/'final_all_rank_census.json').exists()
    assert not (local/'native_failure.json').exists()
    assert len(publications) == 1 and publications[0][2] == 'FAIL'
    terminal = json.loads((Path(config['marker_root'])/'node_0_workload_terminal.json').read_text())
    assert terminal['pid'] == os.getpid() and terminal['result'] == 'FAIL'
    assert 'UNVERIFIED' in terminal['reason'] and 'phase=request_evidence_retention' in terminal['reason']
    expected = dict(io=str(final_save_error), deadline='original workload deadline',
                    bytes='original record budget')
    assert expected[fault] in terminal['reason']
    assert attempts.count('request_evidence.json') == (1 if fault == 'io' else 0)


def test_mandatory_request_receipt_is_saved_before_unique_terminal(config, tmp_path, monkeypatch):
    nodes, rows = startup_fixture(config, tmp_path)
    local = Path(nodes[0]['local_output'])
    llm = CpuLLM(rows, [])
    original_save = caller.capacity.save
    published = []
    def save(path, value):
        assert not published, 'No mandatory diagnostic write may follow the unique terminal'
        return original_save(path, value)
    monkeypatch.setattr(caller.capacity, 'save', save)
    def publish(*args):
        receipt = json.loads((local/'request_evidence.json').read_text())
        assert receipt['completed_requests'] == ['A', 'B'] and receipt['result'] == 'PASS'
        assert receipt['terminal_publication'] == 'UNMEASURED'
        assert not Path(receipt['terminal_marker']).exists()
        row = caller.capacity.publish_terminal(*args)
        published.append(row)
        return row
    assert caller.execute_leader(config, nodes[0], {}, llm_factory=lambda **kwargs: llm,
        sampling_factory=CpuSampling, verify=lambda *args: None, clock=lambda: 10, publish=publish) is llm
    assert len(published) == 1 and published[0]['result'] == 'PASS'
    assert caller.capacity.sha256(local/'request_evidence.json') in published[0]['reason']


def test_original_deadline_exhausted_after_A_prevents_B_and_does_not_claim_a_native_failure(config, tmp_path):
    nodes, rows = startup_fixture(config, tmp_path)
    clock = CpuClock(now=10)
    events = []
    llm = CpuLLM(rows, events)
    original_generate = llm.generate
    def generate(*args, **kwargs):
        output = original_generate(*args, **kwargs)
        clock.now = nodes[0]['workload_deadline']
        return output
    llm.generate = generate
    with pytest.raises(TimeoutError, match='original absolute deadline'):
        caller.execute_leader(config, nodes[0], {}, llm_factory=lambda **kwargs: llm,
            sampling_factory=CpuSampling, verify=lambda *args: None, clock=clock)
    assert len([event for event in events if event[0] == 'generate']) == 1
    assert not (Path(nodes[0]['local_output'])/'native_failure.json').exists()
    terminal = json.loads((Path(config['marker_root'])/'node_0_workload_terminal.json').read_text())
    assert terminal['result'] == 'FAIL'
    assert 'UNVERIFIED_WORKLOAD_DEADLINE' in terminal['reason']


def test_each_native_rpc_uses_remaining_original_deadline(config, tmp_path):
    nodes, rows = startup_fixture(config, tmp_path)
    clock = CpuClock(now=1960)
    events = []
    llm = CpuLLM(rows, events)
    original_rpc = llm.collective_rpc
    def rpc(*args, **kwargs):
        output = original_rpc(*args, **kwargs)
        clock.now += 3
        return output
    llm.collective_rpc = rpc
    caller.execute_leader(config, nodes[0], {}, llm_factory=lambda **kwargs: llm,
        sampling_factory=CpuSampling, verify=lambda *args: None, clock=clock)
    assert [event[2] for event in events if event[0] == 'rpc'] == [50, 47, 44, 41]
    assert clock() < nodes[0]['workload_deadline']


def test_request_node_preserves_original_archive_failure_verdict(config, tmp_path, monkeypatch):
    status, node, child, events, clock = native_node_attempt(config, tmp_path, monkeypatch, archive_bytes=1)
    assert status['scope'] == 'capacity_only_not_G2'
    assert status['result'] == 'FAIL' and status['return_code'] == 1
    assert status['raw_exit_code'] == 0
    assert status['evidence_verdict'] == 'UNVERIFIED'
    assert not (Path(config['archive_root'])/'node_0/archive_manifest.json').exists()
    assert (Path(node['local_output'])/'archive_unverified.json').exists()
    completion = json.loads((Path(config['marker_root'])/'node_0_completion.json').read_text())
    assert completion['result'] == 'FAIL'


@pytest.mark.parametrize('key', ['request_manifest', 'request_manifest_sha256', 'request_caller_sha256',
                                 'request_observer_sha256', 'model_input_observer_sha256', 'rpc_seconds'])
def test_preflight_reports_missing_required_inputs_without_native_dispatch(config, key):
    config.pop(key)
    with pytest.raises(ValueError, match=key+' must be explicitly configured'):
        caller.preflight(config)


@pytest.mark.parametrize('key', ['request_caller_sha256', 'request_observer_sha256',
                                 'model_input_observer_sha256'])
def test_request_source_mismatch_is_rejected_by_real_preflight(config, key):
    config[key] = 'unreviewed source bytes'
    with pytest.raises(ValueError, match='Request source identity mismatch'):
        caller.preflight(config)


def test_actual_cli_requires_explicit_frozen_native_sources(config, tmp_path, monkeypatch):
    path = tmp_path/'unfrozen_config.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(sys, 'argv', ['check_role_execution.py', 'node', '--config', str(path)])
    monkeypatch.setattr(caller, 'run_node', lambda *args, **kwargs: pytest.fail('Unfrozen native dispatch'))
    with pytest.raises(ValueError, match='native_sources must be explicitly configured'):
        caller.main()


@pytest.mark.parametrize('fault', ['missing_hash', 'wrong_hash'])
def test_lazy_request_worker_rejects_unbound_environment_before_native_import(config, tmp_path, monkeypatch, fault):
    local = tmp_path/'lazy_worker'
    local.mkdir()
    node = local/'node.json'
    records = local/'workers'
    records.mkdir()
    node.write_text(json.dumps(dict(observer_output=str(records), record_bytes=config['record_bytes'])))
    for name, value in dict(STEP5_CAPACITY_TASK=config['task_root'],
        STEP5_CAPACITY_NODE_FILE=str(node), STEP5_CAPACITY_OBSERVER_SHA=config['observer_sha256'],
        STEP5_REQUEST_CALLER_SHA=config['request_caller_sha256'],
        STEP5_REQUEST_OBSERVER_SHA=config['request_observer_sha256'],
        STEP5_MODEL_INPUT_OBSERVER_SHA=config['model_input_observer_sha256']).items():
        monkeypatch.setenv(name, value)
    if fault == 'missing_hash':
        monkeypatch.delenv('STEP5_REQUEST_OBSERVER_SHA')
        message = 'STEP5_REQUEST_OBSERVER_SHA must be explicitly configured'
    else:
        monkeypatch.setenv('STEP5_REQUEST_OBSERVER_SHA', 'unfrozen observer')
        message = 'Request worker source identity mismatch'
    with pytest.raises(ValueError, match=message):
        getattr(caller, 'RequestWorker')
    assert not list(records.iterdir())


@pytest.mark.parametrize('fault', ['version', 'source'])
def test_installed_source_admission_failure_is_unverified_before_native_construction(
        config, tmp_path, fault):
    from types import SimpleNamespace
    nodes, _ = startup_fixture(config, tmp_path)
    source = tmp_path/'installed_worker.py'
    source.write_text('different installed source body')
    sources = {'vllm/v1/worker/gpu_worker.py': '0'*64}
    package = SimpleNamespace(version='mismatched' if fault == 'version' else config['vllm_version'],
                              locate_file=lambda relative: source)
    original_errors = []
    def verify(actual_config, actual_sources):
        try:
            caller.capacity.installed_sources(actual_config, actual_sources, distribution=lambda name: package)
        except ValueError as error:
            original_errors.append(error)
            raise
    def construct(**kwargs):
        pytest.fail('Source admission must finish before native construction')
    with pytest.raises(ValueError, match='Installed') as caught:
        caller.execute_leader(config, nodes[0], sources, llm_factory=construct,
            sampling_factory=CpuSampling, verify=verify, clock=lambda: 10)
    assert caught.value is original_errors[0]
    local = Path(nodes[0]['local_output'])
    receipt = json.loads((local/'request_evidence.json').read_text())
    assert receipt['status'] == 'UNVERIFIED' and receipt['result'] == 'FAIL'
    assert receipt['evidence_verdict'] == 'UNVERIFIED'
    assert receipt['failure']['phase'] == 'installed_source_verification'
    assert receipt['failure']['error_type'] == 'builtins.ValueError'
    assert receipt['failure']['sources'] == sources
    assert not receipt['native_generation_returns'] and not (local/'native_failure.json').exists()
    terminal = json.loads((Path(config['marker_root'])/'node_0_workload_terminal.json').read_text())
    assert terminal['result'] == 'FAIL' and 'UNVERIFIED' in terminal['reason']
    assert 'installed_source_verification' in terminal['reason']


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('records', ['rank_limits', 'pre_completion', 'within_limit_with_cache'])
def test_both_node_record_budgets_include_logs_and_finalization_metadata_before_archive(
        config, tmp_path, monkeypatch, rank, records):
    config['record_bytes'] = 64*1024
    def prepare_output(node):
        local = Path(node['local_output'])
        output = Path(node['observer_output'])
        if records in ('rank_limits', 'within_limit_with_cache'):
            for identity in output.glob('worker_*.json'):
                worker = json.loads(identity.read_text())
                allowance = config['record_bytes']//8
                size = allowance-identity.stat().st_size if records == 'rank_limits' else allowance//4
                assert size > 0
                path = output/f"rank_{worker['rank']}_pid_{worker['pid']}.jsonl"
                path.write_bytes(b' '*(size-1)+b'\n')
                assert path.stat().st_size+identity.stat().st_size <= allowance
        else:
            used = sum(path.stat().st_size for path in local.rglob('*') if path.is_file())
            with (local/'native.stdout').open('ab') as stream:
                stream.write(b'x'*(config['record_bytes']-used-1))
            assert sum(path.stat().st_size for path in local.rglob('*') if path.is_file()) == config['record_bytes']-1
        cache = Path(node['cache_root'])/'native_cache.bin'
        with cache.open('xb') as stream:
            stream.truncate(config['record_bytes']*2)
    status, node, child, events, clock = native_node_attempt(config, tmp_path, monkeypatch,
        rank=rank, prepare_output=prepare_output)
    assert status['scope'] == 'capacity_only_not_G2'
    assert status['raw_exit_code'] == (0 if rank == 0 else -15)
    assert child.signals == ([] if rank == 0 else [15])
    completion = json.loads((Path(config['marker_root'])/f'node_{rank}_completion.json').read_text())
    archive = Path(config['archive_root'])/f'node_{rank}'
    if records == 'within_limit_with_cache':
        assert status['result'] == completion['result'] == 'PASS'
        assert status['evidence_finalization']['record_budget']['verdict'] == 'PASS'
        assert status['evidence_finalization']['retained_bytes'] <= config['record_bytes']
        assert (archive/'archive_manifest.json').exists() and not (archive/'cache').exists()
    else:
        assert status['result'] == completion['result'] == 'FAIL'
        assert status['return_code'] == 1 and status['evidence_verdict'] == 'UNVERIFIED'
        finalization = status['evidence_finalization']
        assert finalization['verdict'] == 'UNVERIFIED'
        assert finalization['record_budget']['verdict'] == 'UNVERIFIED'
        assert finalization['record_budget']['node_rank'] == rank
        assert finalization['record_budget']['required_bytes'] > config['record_bytes']
        assert not archive.exists()
        assert not (Path(node['local_output'])/'pre_completion_status.json').exists()
    assert clock() <= 2500


@pytest.fixture
def admission_case():
    """Use original pinned CPU method bodies and explicit request context."""
    from types import SimpleNamespace
    from uuid import uuid4
    parent = SimpleNamespace(generated=[])
    engine = CpuEngine(parent)
    params = CpuSampling(**dict(caller.LENGTH_FIELDS))
    tokens = [0, 67, 67, 68]
    request = CpuCoreRequest('public-original', tokens, copy.deepcopy(params))
    fields = {name: copy.deepcopy(getattr(params, name))
              for name in (*caller.POLICY_FIELDS, *caller.LENGTH_FIELDS)}
    identity = dict(pid=os.getpid(), generation='cpu-admission', boot_id='fixture',
                    start_time=1, pid_namespace='fixture', node_rank=0)
    context = dict(name='A', manifest='/cpu-manifest/'+str(uuid4()), manifest_sha256='explicit-fixture',
                   prompt_sha256=caller.token_hash(tokens), prompt_token_ids=tokens)
    sites = caller.model_observation.AdmissionSites.bind(engine)
    scope = caller.model_observation.AdmissionScope(engine, sites, context, params, fields, identity)
    return SimpleNamespace(engine=engine, request=request, params=params, fields=fields,
                           context=context, identity=identity, sites=sites, scope=scope)


def test_original_registration_is_observed_before_enginecore_dispatch_and_retained_after_removal(admission_case):
    case = admission_case
    original = case.engine.engine_core.add_request
    dispatches = []
    def dispatch(request):
        assert case.scope.records[0]['internal_request_id'] == request.request_id
        assert case.scope.records[0]['native_registration_outcome'] == 'returned'
        dispatches.append(request)
        original(request)
    case.engine.engine_core.add_request = dispatch
    with case.scope:
        internal = case.engine.add_request(case.request.request_id, case.request, case.params)
        case.engine.output_processor._finish_request(case.engine.output_processor.request_states[internal])
    assert dispatches == [case.request]
    assert not case.engine.output_processor.external_req_ids
    assert case.scope.record['status'] == 'OBSERVED_FRONTEND_ADMISSION_ONLY_NOT_G2'
    row = case.scope.record['records'][0]
    assert row['public_request_id'] == 'public-original' and row['internal_request_id'] == internal
    assert row['resolved_sampling_id'] == id(case.request.sampling_params)
    case.request.prompt_token_ids[-1] = 1
    case.request.sampling_params.temperature = 0.125
    assert row['prompt_token_ids'][-1] == 68 and row['resolved_sampling_fields']['temperature'] == 1.0


@pytest.mark.parametrize('fault', ['no_engine_call', 'duplicate', 'stream_update', 'mapping',
                                  'changed_prompt', 'changed_sampling', 'wrong_parent', 'budget', 'removed_hook'])
def test_original_admission_faults_preserve_native_return_and_remain_unverified(admission_case, fault):
    case = admission_case
    if fault == 'stream_update':
        request = case.request
        request.request_id = 'public-original-cpu-0'
        case.engine.input_processor.assign_request_id = lambda request: setattr(request, 'external_req_id', 'public-original')
        state = CpuRequestState.from_new_request(request=request, parent_req=None, request_index=0)
        state.external_req_id = 'public-original'
        case.engine.output_processor.request_states[request.request_id] = state
    elif fault == 'mapping':
        case.engine.output_processor.external_req_ids['public-original'] = ['other-internal']
    elif fault == 'changed_prompt':
        case.request.prompt_token_ids[-1] = 67
    elif fault == 'changed_sampling':
        case.request.sampling_params.temperature = 0.125
    elif fault == 'budget':
        case.scope.limits = caller.model_observation.ObservationLimits(scope_bytes=128)
    before = sys.getprofile()
    with case.scope:
        if fault == 'wrong_parent':
            case.request.external_req_id = 'public-original'
            returned = case.engine.output_processor.add_request(case.request, None)
            assert returned is None
        elif fault != 'no_engine_call':
            returned = case.engine.add_request(case.request.request_id, case.request, case.params)
            assert returned == case.request.request_id
            if fault == 'duplicate':
                other = CpuCoreRequest('public-second', case.context['prompt_token_ids'], copy.deepcopy(case.params))
                assert case.engine.add_request(other.request_id, other, case.params) == other.request_id
        if fault == 'removed_hook':
            sys.setprofile(None)
    assert sys.getprofile() is before
    assert case.scope.record['status'] == 'UNVERIFIED'
    assert case.scope.record['observer_errors']
    if fault == 'budget':
        encoded = (json.dumps(case.scope.record, indent=2, allow_nan=False)+'\n').encode('utf-8')
        assert len(encoded) <= case.scope.limits.scope_bytes
    else:
        assert case.scope.record['device_operations_added'] is False
    if fault == 'stream_update':
        assert case.engine.output_processor.updated_state is not None
    if fault == 'changed_sampling':
        row = case.scope.record['records'][0]
        assert row['resolved_sampling_fields']['temperature'] == 0.125
        assert row['submitted_sampling_fields']['temperature'] == 1.0


def test_native_admission_exception_identity_and_original_partial_association_are_preserved(admission_case):
    case = admission_case
    failure = RuntimeError('exact native dispatch failure')
    def fail(request):
        raise failure
    case.engine.engine_core.add_request = fail
    before = sys.getprofile()
    with pytest.raises(RuntimeError) as caught:
        with case.scope:
            case.engine.add_request(case.request.request_id, case.request, case.params)
    assert caught.value is failure and sys.getprofile() is before
    row = case.scope.record['records'][0]
    assert row['native_registration_outcome'] == 'returned'
    assert row['engine_core_submission'] == 'UNVERIFIED'
    assert case.scope.record['native_outcome'] == 'raised'
    assert case.scope.record['status'] == 'UNVERIFIED'


def test_existing_frontend_profile_and_trace_hooks_are_composed_and_restored(admission_case):
    case = admission_case
    before_profile, before_trace = sys.getprofile(), sys.gettrace()
    calls, traced = [], []
    def profile(frame, event, arg):
        if frame.f_code is case.sites.codes['registration']:
            calls.append(event)
    def trace(frame, event, arg):
        if frame.f_code is case.sites.codes['registration']:
            traced.append(event)
        return trace
    try:
        sys.setprofile(profile); sys.settrace(trace)
        scope = caller.model_observation.AdmissionScope(case.engine, case.sites, case.context,
            case.params, case.fields, case.identity)
        with scope:
            case.engine.add_request(case.request.request_id, case.request, case.params)
        assert sys.getprofile() is profile and sys.gettrace() is trace
        assert scope.record['status'] == 'OBSERVED_FRONTEND_ADMISSION_ONLY_NOT_G2'
        assert 'call' in calls and 'return' in calls and 'line' in traced
    finally:
        sys.setprofile(before_profile); sys.settrace(before_trace)


@pytest.mark.parametrize('fault', ['binding', 'identity', 'scope_setup', 'storage', 'missing_record',
                                  'finalization', 'tiny_budget'])
def test_frontend_observer_failures_do_not_prevent_native_generation_or_claim_G2(config, tmp_path, monkeypatch, fault):
    nodes, rows = startup_fixture(config, tmp_path)
    llm = CpuLLM(rows, [])
    kwargs = {}
    def failure(*args, **kwargs):
        raise OSError('faulted frontend '+fault)
    if fault == 'binding':
        kwargs['bind_admission'] = failure
    elif fault == 'identity':
        kwargs['identify_frontend'] = failure
    elif fault == 'scope_setup':
        monkeypatch.setattr(caller.model_observation, 'AdmissionScope', failure)
    elif fault in ('missing_record', 'finalization'):
        original_exit = caller.model_observation.AdmissionScope.__exit__
        def exit(scope, *args):
            returned = original_exit(scope, *args)
            if fault == 'finalization':
                failure()
            scope.record = None
            return returned
        monkeypatch.setattr(caller.model_observation.AdmissionScope, '__exit__', exit)
    elif fault == 'tiny_budget':
        original_scope = caller.model_observation.AdmissionScope
        def scope(*args, **kwargs):
            return original_scope(*args, limits=caller.model_observation.ObservationLimits(scope_bytes=1), **kwargs)
        monkeypatch.setattr(caller.model_observation, 'AdmissionScope', scope)
    else:
        original = caller.capacity.save
        def save(path, value):
            if path.name.endswith('_admission.json'):
                failure()
            return original(path, value)
        monkeypatch.setattr(caller.capacity, 'save', save)
    assert caller.execute_leader(config, nodes[0], {}, llm_factory=lambda **kwargs: llm,
        sampling_factory=CpuSampling, verify=lambda *args: None, clock=lambda: 10, **kwargs) is llm
    receipt = json.loads((Path(nodes[0]['local_output'])/'request_evidence.json').read_text())
    assert receipt['completed_requests'] == ['A', 'B']
    assert receipt['status'] == 'UNVERIFIED' and receipt['evidence_verdict'] == 'UNVERIFIED'
    assert not (Path(nodes[0]['local_output'])/'native_failure.json').exists()
    assert set(receipt['evidence_status'].values()) == {'UNMEASURED'}
    assert json.loads((Path(config['marker_root'])/'node_0_workload_terminal.json').read_text())['result'] == 'FAIL'
    if fault in ('missing_record', 'finalization', 'tiny_budget'):
        assert receipt['frontend_associations'] == []
        assert not (Path(nodes[0]['local_output'])/'request_A_admission.json').exists()
        assert not (Path(nodes[0]['local_output'])/'request_B_admission.json').exists()


def test_completed_public_id_must_match_the_original_retained_admission(config, tmp_path):
    def configure(llm):
        original = llm.generate
        def generate(*args, **kwargs):
            outputs = original(*args, **kwargs)
            outputs[0].request_id = 'public-id-from-some-other-request'
            return outputs
        llm.generate = generate
    llm, nodes, _ = run_leader(config, tmp_path, configure=configure)
    receipt = json.loads((Path(nodes[0]['local_output'])/'request_evidence.json').read_text())
    assert receipt['completed_requests'] == ['A', 'B'] and receipt['frontend_associations'] == []
    assert receipt['status'] == 'UNVERIFIED'
    original = json.loads((Path(nodes[0]['local_output'])/'request_A_admission.json').read_text())
    assert original['records'][0]['public_request_id'] == '0'
    assert not (Path(nodes[0]['local_output'])/'native_failure.json').exists()
