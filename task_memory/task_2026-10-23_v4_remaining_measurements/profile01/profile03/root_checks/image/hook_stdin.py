"""Run the existing CPU native-wrapper probe for the staged profile extension."""
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

package = Path(sys.argv[1])
position = int(sys.argv[2])
config = json.loads((package / 'config.json').read_text())
harness = Path(config['harness_dir'])
sys.path[:0] = [str(package), str(harness)]
from pd_replay import instance_env
from native_pd_worker import node_plan

peers = [dict(host=f'cpu-v4-{i}', ip=f'192.0.2.{i + 1}') for i in range(4)]
plan = node_plan(config, peers, peers[position]['host'], [str(i) for i in range(8)])
directory = Path('/data/ycfeng/tmp/profile03_cpu_startup')
env, changed = instance_env('clean', plan['engine'], directory, dict(os.environ),
    '0,1,2,3,4,5,6,7', observer=None, role=plan['role'])
os.environ.update(env)
os.environ.update(FRONTIER_NATIVE_SERVING_EVIDENCE_DIR=str(directory),
    FRONTIER_NATIVE_RUN_GENERATION='CPU_STARTUP_ONLY_NOT_A_RUN',
    FRONTIER_NATIVE_ROLE=plan['role'],
    FRONTIER_NATIVE_PROFILE_LIMITS=json.dumps(config['serving_profile_limits']))
extension = 'native_serve_profile.NativeServeProfile'
assert plan['engine']['worker_extension_cls'] == extension

probe_path = package / 'tests/integration/native_extension_startup_probe.py'
tree = ast.parse(probe_path.read_bytes())
changes = {'extension': 0, 'import': 0, 'class': 0, 'inventory': 0}
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and node.value == 'native_full_structure.FullModelInventory':
        node.value = extension
        changes['extension'] += 1
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name == 'native_full_structure' and alias.asname == 'module':
                alias.name = 'native_serve_profile'
                changes['import'] += 1
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'module':
        if node.attr == 'FullModelInventory':
            node.attr = 'NativeServeProfile'
            changes['class'] += 1
        elif node.attr == '_NATIVE_SERVING_INVENTORY':
            node.value = ast.Attribute(value=ast.Name(id='module', ctx=ast.Load()),
                                       attr='inventory', ctx=ast.Load())
            changes['inventory'] += 1
assert changes == {'extension': 1, 'import': 2, 'class': 2, 'inventory': 1}, changes
ast.fix_missing_locations(tree)
namespace = {'__name__': 'profile_startup_probe', '__file__': str(probe_path)}
exec(compile(tree, str(probe_path), 'exec'), namespace)

class Receipt:
    def put(self, value):
        self.value = value

receipt = Receipt()
namespace['worker_probe'](receipt, harness / 'native_serve_profile.py', True)
assert receipt.value['ok'], receipt.value
import torch
import native_serve_profile
from vllm.config.profiler import ProfilerConfig
from vllm.model_executor.layers.fused_moe.router.routing_simulator_router import RoutingSimulator

profile = native_serve_profile._PROFILE
assert profile is not None and profile.errors == []
assert profile.limits == config['serving_profile_limits']
assert 'Worker.profile' in profile.hooks
native_config = ProfilerConfig(**plan['engine']['profiler_config'])
assert native_config.max_iterations == config['role_profiler_max_iterations'][plan['role']] and native_config.ignore_frontend
assert native_config.profiler == 'torch'
assert native_config.delay_iterations == native_config.warmup_iterations == native_config.wait_iterations == 0
strategy = RoutingSimulator._routing_strategies['balanced_topk']
origin = Path(inspect.getsourcefile(type(strategy))).resolve()
assert origin == harness / 'balanced_routing.py'
assert hashlib.sha256(origin.read_bytes()).hexdigest() == config['tool_hashes'][str(origin)]
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS', position=position, role=plan['role'],
    native_startup=receipt.value, profile_hooks=profile.hooks, profile_errors=profile.errors,
    profiler_config=plan['engine']['profiler_config'], limits=profile.limits,
    strategy_origin=str(origin), cuda_initialized=False,
    model_hooks_runtime_verified=False, original_probe_sha256=hashlib.sha256(probe_path.read_bytes()).hexdigest(),
    in_memory_probe_changes=changes)))
