"""Break the four failing A6 attention scopes down by model key for the formal run and three diagnostic reruns."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
runs = {'formal': root.parent / 'a6_codex_20261101/result.json',
        **{name: root / f'a6_{name}/result.json' for name in ('original_shared', 'median3_shared', 'median3_all')}}
scopes = ('dsv4_q_projection', 'dsv4_index_fused_query_prepare',
          'dsv4_compressor_fused_compress_store', 'dsv4_index_compressor_fused_compress_store')
results = {name: json.loads(path.read_text())['attention']['per_key'] for name, path in runs.items()}
breakdown = {}
for scope in scopes:
    keys = sorted(k for k in results['formal'] if json.loads(k)[0] == scope)
    breakdown[scope] = {}
    for key in keys:
        breakdown[scope][key] = {}
        cells = []
        for name, per_key in results.items():
            stats = per_key.get(key, {}).get('joint', {}).get('kernel_time_relative_error') or {}
            median, p90 = stats.get('median_absolute'), stats.get('p90_absolute')
            breakdown[scope][key][name] = {'count': stats.get('count'), 'median': median, 'p90': p90}
            cells.append('-' if p90 is None else f"{100 * median:5.2f}/{100 * p90:6.2f}% n={stats['count']}")
        print(f"{key:80s} " + ' | '.join(cells))
output = root / 'per_key_breakdown.json'
assert not output.exists()
output.write_text(json.dumps({'classification': 'DIAGNOSTIC_MULTIPASS_A6_PER_KEY',
                              'inputs': {name: str(path) for name, path in runs.items()},
                              'columns': list(runs), 'scopes': breakdown}, indent=2) + '\n')
