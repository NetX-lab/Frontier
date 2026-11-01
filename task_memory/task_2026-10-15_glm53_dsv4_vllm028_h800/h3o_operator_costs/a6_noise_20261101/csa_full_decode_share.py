"""Share of the measured attention-layer kernel time that the two noisy CSA FULL decode scopes hold.

For every published CSA FULL decode row, divide each scope's kernel-time median by the
sum of all kernel-time medians in that row. The attention layer is only part of a decode
step, so this share is an upper bound on the scope's share of the step time.
"""
import csv
import json
from pathlib import Path
import statistics

TABLE = Path('data/profiling/compute/h200/deepseek-ai/DeepSeek-V4-Flash/attention_graph_replay_kernel_only.csv')
SCOPES = ('dsv4_q_projection', 'dsv4_index_compressor_fused_compress_store')
shares = {(scope, first): [] for scope in SCOPES for first in ('True', 'False')}
with TABLE.open() as handle:
    for row in csv.DictReader(handle):
        if (row['attention_variant'], row['graph_runtime_mode'], row['phase']) != ('csa', 'FULL', 'decode'):
            continue
        medians = {c[len('time_stats.'):-len('.median')]: float(v) for c, v in row.items()
                   if c.startswith('time_stats.') and c.endswith('.median') and v}
        total = sum(medians.values())
        for scope in SCOPES:
            shares[scope, row['plans_tile_schedule']].append(medians[scope] / total)
result = {}
for (scope, first), values in shares.items():
    cuts = statistics.quantiles(values, n=10, method='inclusive')
    result[f"{scope}/{'first' if first == 'True' else 'reuse'}"] = {
        'rows': len(values), 'p10': cuts[0], 'median': statistics.median(values), 'p90': cuts[8]}
    print(f"{scope:45s} {'first' if first == 'True' else 'reuse':6s} rows {len(values)} share p10/p50/p90 "
          f"{100 * cuts[0]:.2f}/{100 * statistics.median(values):.2f}/{100 * cuts[8]:.2f}%")
output = Path(__file__).resolve().parent / 'csa_full_decode_share.json'
assert not output.exists()
output.write_text(json.dumps({'classification': 'DIAGNOSTIC_ATTENTION_LAYER_SHARE', 'table': str(TABLE),
                              'shares': result}, indent=2) + '\n')
