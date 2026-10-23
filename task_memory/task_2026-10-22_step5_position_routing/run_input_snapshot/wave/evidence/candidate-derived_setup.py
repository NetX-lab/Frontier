"""Actual main08 decode EP wave/pricing and emission, frozen source, diagnosis only."""
import argparse
import cProfile
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import pstats
import resource
import subprocess
import sys
import time
resource.setrlimit(resource.RLIMIT_AS, (30 * 1024 ** 3, 30 * 1024 ** 3))
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--profile', action='store_true')
parser.add_argument('--iterations', type=int, default=2)
args = parser.parse_args()
args.output.mkdir(exist_ok=False)
source = Path('/data/ycfeng/Frontier/.worktrees/step5-d29-input-snapshot')
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() == 'ecccf83fa6f2c24a14fa5397ef567fbb32b06bef'
sys.path.insert(0, str(source))
from frontier.config import SimulationConfig
from frontier.types import ClusterType, MeasurementType
from frontier.execution_time_predictor.shared_prediction_model_manager import ExecutionTimePredictionModelManager
from frontier.moe_routing_policy import SourceLaneRouting, TokenInterval, WaveRoutingSource
from frontier.moe_ep_workload import build_contiguous_expert_ownership, materialize_layer_ep_workload
from frontier.entities import Cluster, Batch, Request, EPBatchGroup
from frontier.scheduler.utils.batch_builders import build_ep_lane_batch
from frontier.scheduler.utils.expert_parallel import predict_ep_wave_phase_times, get_ep_phase_times_ms, calculate_ep_wave_timing
from frontier.scheduler.utils.ep_wave import EPWavePlan
from frontier.metrics.metrics_store import MetricsStore
from frontier.metrics.ep_wave_metrics import record_ep_wave
from frontier.metrics.trace_store import TraceStore
import frontier.metrics.trace_store as bound_trace_source
assert Path(bound_trace_source.__file__).resolve() == Path('/data/ycfeng/Frontier/.worktrees/step5-d29-input-snapshot') / 'frontier/metrics/trace_store.py'
assert hashlib.sha256(Path(bound_trace_source.__file__).read_bytes()).hexdigest() == '07d7cbbb081829f558b6a4c101201baf112e5387e2045ee244436853fb463a0d'
from frontier.execution_time_predictor.kernel_gap import read_kernel_gap_ms
from frontier.execution_time_predictor.measurement_input_paths import uses_two_stream_eager_pricing
from tests.unit.test_step5_measured_ffn_replay import _predictor
run = Path('/data/ycfeng/tmp/d29-parent-position-main08/offline')
command = json.loads((run / 'command.json').read_text())
for (i, token) in enumerate(command):
    if token.startswith('--') and (token.endswith('_output_dir') or token.endswith('_cache_dir') or token.endswith('_runner_out_dir') or token.endswith('_log_dir')):
        command[i + 1] = str(args.output / token[2:])
        Path(command[i + 1]).mkdir(parents=True)
sys.argv = command[3:]
config = SimulationConfig.create_from_cli_args()
clusters = config.get_clusters()
role = ClusterType.DECODE
replica = clusters[role].replica_config
cache = Path('/data/ycfeng/tmp/d29-parent-position-fixture03/cache')
bundle_path = cache / 'bundle_594b60e71174496473a041cee333d8f20b5907ddc83b1d3805c07f62f0c7ac26.json'
bundle = json.loads(bundle_path.read_text())
entries = bundle['models']
manager = ExecutionTimePredictionModelManager.__new__(ExecutionTimePredictionModelManager)
manager._cache_dir = str(args.output / 'model_cache')
manager._all_dummy_mode = False
manager._cluster_configs = clusters
for family in ('eager', 'device_event', 'kernel_only'):
    for prefix in ('_trained_models_', '_models_by_precision_', '_model_profiling_precision_'):
        setattr(manager, prefix + family, {})
    for prefix in ('_trained_models_', '_models_by_precision_'):
        setattr(manager, prefix + family + '_by_contract', {})
loaded = []
combined = [(bundle, label, entry) for (label, entry) in entries.items()]
kernel_bundle = json.loads((cache / 'bundle_0d1f560050cac3803a37e544fa99e5ec8fbf062bbec57745c259327f6464dd50.json').read_text())
combined.extend(((kernel_bundle, label, entry) for (label, entry) in kernel_bundle['models'].items()))
for (table, label, entry) in combined:
    entries = table['models']
    name = entry.get('cache_model_name', label)
    data = (cache / f"{name}_{entry['model_hash']}.pkl").read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry['sha256']
    model = pickle.loads(data)
    model._frontier_cache_model_name = name
    if 'kernel_count_model' in entry:
        pair = entries[entry['kernel_count_model']]
        countname = pair.get('cache_model_name', entry['kernel_count_model'])
        data = (cache / f"{countname}_{pair['model_hash']}.pkl").read_bytes()
        assert hashlib.sha256(data).hexdigest() == pair['sha256']
        count = pickle.loads(data)
        count._frontier_operator_name = entry['target_column'].split('.')[1]
        model._frontier_kernel_count_model = count
    manager._active_measurement_type = MeasurementType(entry['measurement_identity']['measurement_type'])
    manager._store_model_precision(name, table['identity']['registrations'][label]['precision'], model)
    loaded.append({'name': name, 'model_hash': entry['model_hash'], 'sha256': entry['sha256']})
predictor = _predictor(replica.model_config, manager)
predictor._replica_config = replica
predictor._cluster_type = role
predictor._config = clusters[role].execution_time_predictor_config
predictor.requires_forward_start_agreement = 'step5_attention' in {s.family_id for s in replica.model_config.get_layer_attention_specs()} and (not predictor._config.enable_dummy_mode)
predictor._two_stream_eager_pricing = uses_two_stream_eager_pricing(predictor._config, '', sys_arch=config.sys_arch, replica_config=replica)
assert not predictor._two_stream_eager_pricing
predictor._models_eager = dict(manager._models_view_for_family('eager', role))
predictor._predictions_eager = {n: {'_on_demand_prediction': True, '_model': m, '_feature_names': tuple(m._frontier_feature_names)} for (n, m) in predictor._models_eager.items()}
del predictor._get_operator_calibration_scale
del predictor._get_optional_operator_phase_calibration_scale
predictor._active_measurement_type = MeasurementType.CUDA_EVENT
predictor._step_measurement_type = MeasurementType.CUDA_EVENT
predictor._cc_backend = Cluster(clusters[role], config.metrics_config, config.request_generator_config).cc_backend
predictor._moe_routing_distribution_type = 'position_cyclic'
predictor._moe_tp_size = replica.moe_tensor_parallel_size
predictor._moe_ep_size = replica.moe_expert_parallel_size
predictor._kernel_gap_ms = read_kernel_gap_ms('/data/ycfeng/tmp/d29-parent-position-fixture03/fixture_kernel_gap.csv', 8)
captured = json.loads(Path('/data/ycfeng/tmp/d29_main08_residual_repro/parent-layer60-source.json').read_text())
actual = captured['meta']['routing_source']
layer_id = captured['meta']['global_layer_ids'][0]
lanes = tuple((SourceLaneRouting(s['lane_id'], s['source_batch_id'], s['source_global_id'], s['forward_group_id'], s['graph_mode'], s['gating_runtime_context'], s['physical_rows'], s['live_rows'], tuple((TokenInterval(**i) for i in s['intervals']))) for s in actual['source_lanes']))
routing = WaveRoutingSource(actual['policy_id'], lanes, actual['gathered_physical_rows'])
workload = materialize_layer_ep_workload(policy_id='position_cyclic', routing_source=routing, target_replica_id=1, global_layer_id=layer_id, routing_token_count=routing.gathered_physical_rows, router_topk=8, total_expert_num=352, moe_expert_parallel_size=16, expert_to_ep=build_contiguous_expert_ownership(352, 16))
request = Request(0.0, 6144, 1024, num_processed_tokens=6147)
batch = Batch(1, [request], [2], is_moe=True, num_context_tokens=[6147], request_is_decoding=[True])
batch._id = 7052
batch.set_global_id(8)

def create_group(requests, tokens, replica_id, ep_id, time, ids, lane):
    return EPBatchGroup(requests, tokens, replica_id, ep_id, time, ids, lane, role, is_moe=True)

def lane_builder(**kw):
    return build_ep_lane_batch(**kw, create_batch_group=create_group, cluster_type=role)

def noop(**kw):
    pass
store = MetricsStore.__new__(MetricsStore)
store._config = config.metrics_config
store._cluster_configs = clusters
store._trace_store = TraceStore(str(args.output), enabled=True)
store._trace_store.initialize({'evidence': 'main08_decode_wave_diagnosis'})
store._frontier_ep_wave_lane_ledger_rows = []
assert not config.metrics_config.store_operation_metrics and (not config.metrics_config.enable_per_layer_expansion)
prof = cProfile.Profile()
timings = []
hashes = []
trace_hashes = []
phases = predict_ep_wave_phase_times(layer_workload=workload, source_batch=batch, stage_id=0, layer_id=layer_id, cluster_type=role, predictor=predictor, lane_builder=lane_builder, phase_getter=get_ep_phase_times_ms, workload_logger=noop, trace_identity=None, batch_id=7052, capture_lane_timings=True)
timing = calculate_ep_wave_timing(start_time_s=captured['meta']['wave_start_time_s'], phases=phases, barrier_logger=noop, wave_logger=noop, cluster_type=role, batch_id=7052, layer_id=layer_id, participant_ep_ids=workload.participant_ep_ids, trace_identity=None)
plan = EPWavePlan(None, workload, phases, timing, None)
