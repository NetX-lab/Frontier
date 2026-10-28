"""Compare each engine's realized kv_cache_groups with the 11-layer prediction."""
import json
import sys
from pathlib import Path

from dsv4_graph_bridge_check import graph_mode_flags
from frontier.profiling.attention import dsv4_attention_main
from frontier.profiling.attention.dsv4_attention_plan import require_realized_kv_cache_groups

config = json.loads(Path(sys.argv[1]).read_text())
plan = dsv4_attention_main.build_plan(dsv4_attention_main.build_parser().parse_args(
    ['plan', *config['plan_args'], *graph_mode_flags('FULL_AND_PIECEWISE', tuple(config['cudagraph_capture_sizes']))]))
print('planned layers', [(layer.global_layer_id, layer.attention_variant, layer.plans_tile_schedule) for layer in plan.shape.layers])
failures = 0
for path in sys.argv[2:]:
    engine = next(json.loads(line) for line in open(path) if json.loads(line)['kind'] == 'engine')
    groups = engine.get('kv_cache_groups')
    print(path, 'groups', None if groups is None else [len(group) for group in groups])
    try:
        require_realized_kv_cache_groups(plan, engine)
        print('  equal to the prediction')
    except ValueError as error:
        failures += 1
        print('  MISMATCH', str(error)[:2000])
    print('  layers', engine['layers'])
print('realized_groups', 'FAIL' if failures else 'PASS')
sys.exit(int(failures > 0))
