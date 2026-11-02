import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from frontier.execution_time_predictor.dsv4_composition import compose_attention_phase
from frontier.profiling.kernel_gap.schema import DSV4_EXECUTION_MODES

scratch = Path(__file__).parent
path, = (scratch / 'profiling').rglob('attention_graph_replay_kernel_only.csv')
with path.open() as handle:
    rows = list(csv.DictReader(handle))
results = []
for i, row in enumerate(rows):
    scopes = json.loads(row['operator_scopes'])
    costs = {s: float(row[f'time_stats.{s}.median']) + float(row[f'gap_stats.{s}.median']) for s in scopes}
    counts = {s: int(row[f'time_stats.{s}.kernel_count']) for s in scopes}
    composed = compose_attention_phase(costs, counts, dict.fromkeys(DSV4_EXECUTION_MODES, 0.0),
        variant=row['attention_variant'], graph_runtime_mode=row['graph_runtime_mode'],
        rows=int(row['padded_rows']), multi_stream_gemm_token_threshold=int(row['multi_stream_gemm_token_threshold']),
        indexer_short_context=row['indexer_short_context'] == 'True')
    target = float(row['layer_window_stats.median'])
    error = abs(composed.elapsed_ms - target) / target
    key_names = ['case_id', 'phase', 'attention_variant', 'plans_tile_schedule', 'graph_runtime_mode', 'padded_rows',
                 'dispatch_num_requests', 'prefill_seq_lens', 'prefill_kv_cache_sizes', 'decode_kv_cache_sizes',
                 'global_layer_ids', 'index_scores_executed', 'indexer_short_context']
    results.append(dict(row_index=i, key={k: row.get(k) for k in key_names},
        metadata={k:v for k,v in row.items() if '_metadata.' in k and v != ''},
        operator_scopes=list(scopes), count=int(row['layer_window_stats.count']),
        composed_ms=composed.elapsed_ms, target_ms=target, absolute_relative_error=error,
        window_spread={k:float(row[f'layer_window_stats.{k}']) for k in ['min','max','std']},
        scope_spreads={s: {prefix:{k:float(row[f'{prefix}.{s}.{k}']) for k in ['min','max','std']}
                          for prefix in ['time_stats','gap_stats']} for s in scopes}))
errors = [r['absolute_relative_error'] for r in results]
single = [r['absolute_relative_error'] for r in results if r['count'] == 1]
above = [r for r in results if r['absolute_relative_error'] > 0.05]
report = dict(csv=str(path), row_count=len(rows),
    by_phase=dict(Counter(row['phase'] for row in rows)),
    by_mode=dict(Counter(row['graph_runtime_mode'] for row in rows)),
    layer_call_count=sum(r['count'] for r in results),
    calls_per_row=dict(Counter(r['count'] for r in results)),
    identities={key: sorted({row[key] for row in rows}) for key in ['schema_version','graph_mode','source_revision','model_config_revision']},
    composition=dict(median_absolute_relative_error=float(np.median(errors)),
        p90_absolute_relative_error=float(np.percentile(errors,90)), max_absolute_relative_error=max(errors),
        single_call_row_count=len(single), single_call_max_absolute_relative_error=max(single),
        rows_above_5_percent=len(above), limits_pass=bool(np.median(errors) <= 0.05 and np.percentile(errors,90) <= 0.10)),
    above_5_percent=above)
(scratch/'row_composition.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(scratch/'row_composition_all.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'above_5_percent'},indent=2))
for row in above:
    print('ABOVE_5_PERCENT', json.dumps({k:row[k] for k in ['row_index','key','count','absolute_relative_error','window_spread']},sort_keys=True))
assert len(rows) == 505
assert report['by_phase'] == {'decode':400,'dummy':50,'first_tail':50,'prefill':5}
assert report['by_mode'] == {'FULL':345,'PIECEWISE':150,'NONE':10}
assert report['layer_call_count'] == 1925
assert report['identities']['schema_version'] == ['dsv4_attention_replay.v1']
assert report['identities']['graph_mode'] == ['FULL_AND_PIECEWISE']
assert report['composition']['limits_pass']
