"""Observe all four call-local controls in one loaded process with GC enabled."""
import ast
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
RAW = Path(sys.argv[1]).resolve()
ROOT = RAW / 'paired'
ROOT.mkdir(exist_ok=False)
for name in ('freeze.json', 'fixture_wiring.py'):
    (ROOT / name).write_bytes((RAW / 'registry' / name).read_bytes())
frozen = json.loads((ROOT / 'freeze.json').read_text())
target = ROOT / 'baseline/collective-fixture/h800/collective_local.json'
target.parent.mkdir(parents=True)
raw = Path(frozen['collective_original']).read_bytes()
assert hashlib.sha256(raw).hexdigest() == frozen['collective_sha256']
target.write_bytes(raw)
original = HERE.parent / 'runtime_model_view/repair_evidence/run.py'
tree = ast.parse(original.read_bytes())
start = next(node.lineno for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == 'samples' for t in node.targets))
prefix = ast.Module(body=[node for node in tree.body if node.lineno < start], type_ignores=[])
sys.path.insert(0, str(ROOT))
sys.argv = [str(ROOT / 'run.py'), 'baseline']
ns = {'__name__': '__main__', '__file__': str(ROOT / 'run.py')}
exec(compile(prefix, str(original), 'exec'), ns)
from frontier.execution_time_predictor.prediction_model_registry import PredictionModelRegistry
from frontier.execution_time_predictor.moe_routing_workload import MoeRoutingWorkload
slots = [(PredictionModelRegistry, '_get_family_model'),
         (MoeRoutingWorkload, '_step5_plan_candidates'),
         (MoeRoutingWorkload, '_step5_position_queries')]
methods = {'baseline': [getattr(cls, name) for cls, name in slots]}


def select(label):
    for (cls, name), function in zip(slots, methods[label]):
        setattr(cls, name, function)


spec = importlib.util.spec_from_file_location('json_control_install', HERE / 'install_control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)
installations = {}
for label in ('registry', 'point', 'combined'):
    select('baseline')
    installations[label] = control.install(Path(frozen['sources']['baseline']['root']), RAW / label, 'candidate')
    methods[label] = [getattr(cls, name) for cls, name in slots]
select('baseline')
assert gc.isenabled()
exec(ns['wave_code'], ns['setup_namespace'])
expected = ns['payload'](ns['setup_namespace']['phases'])
assert hashlib.sha256(expected).hexdigest() == '3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83'
gc_events = []
current = None
starts = {}


def observe_gc(phase, info):
    generation = info['generation']
    if phase == 'start':
        starts[generation] = (time.perf_counter(), current)
    else:
        began, sample = starts.pop(generation)
        gc_events.append({'sample': sample, 'generation': generation,
            'seconds': time.perf_counter() - began, 'collected': info['collected'],
            'uncollectable': info['uncollectable']})


orders = [
    ['baseline', 'registry', 'point', 'combined'],
    ['combined', 'point', 'registry', 'baseline'],
    ['registry', 'combined', 'baseline', 'point'],
    ['point', 'baseline', 'combined', 'registry'],
    ['baseline', 'point', 'registry', 'combined'],
    ['combined', 'registry', 'point', 'baseline'],
]
samples = []
gc.callbacks.append(observe_gc)
try:
    for order in orders:
        for label in order:
            select(label)
            current = len(samples)
            wall, cpu = time.perf_counter(), time.process_time()
            exec(ns['wave_code'], ns['setup_namespace'])
            cpu_s, wall_s = time.process_time() - cpu, time.perf_counter() - wall
            current = None
            assert ns['payload'](ns['setup_namespace']['phases']) == expected
            samples.append({'variant': label, 'wall_s': wall_s, 'cpu_s': cpu_s})
finally:
    current = None
    gc.callbacks.remove(observe_gc)
    select('baseline')
    json.dumps = ns['real_dumps']
    Path.read_bytes, Path.read_text = ns['real_bytes'], ns['real_text']
for index, sample in enumerate(samples):
    sample['gc_events'] = [event for event in gc_events if event['sample'] == index]
    sample['gc_s'] = sum(event['seconds'] for event in sample['gc_events'])
summary = {}
for label in methods:
    group = [s for s in samples if s['variant'] == label]
    summary[label] = {clock: {'median': statistics.median(s[clock] for s in group),
        'mean': statistics.mean(s[clock] for s in group)} for clock in ('wall_s', 'cpu_s', 'gc_s')}
source = Path(frozen['sources']['baseline']['root'])
for relative, expected_hash in frozen['sources']['baseline']['hashes'].items():
    assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == expected_hash
receipt = {'task_id': 'D29-RUNTIME-JSON-REUSE', 'source': str(source), 'source_revision': frozen['sources']['baseline']['head'],
           'original_driver_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
           'source_bindings': 23, 'pricing_payload_sha256': hashlib.sha256(expected).hexdigest(),
           'gc_enabled_before_and_after': gc.isenabled(), 'gc_threshold': gc.get_threshold(),
           'installations': installations, 'samples': samples, 'gc_events': gc_events, 'summary': summary,
           'limits': ['Timing only; protected output suite not rerun.', 'GC remains enabled; no samples excluded or GC time subtracted.',
                      'Pricing and GC placement do not establish whole-wave/CLI speedup.']}
(ROOT / 'timing.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'samples': len(samples), 'summary': summary, 'gc_events': len(gc_events)}, indent=2))
