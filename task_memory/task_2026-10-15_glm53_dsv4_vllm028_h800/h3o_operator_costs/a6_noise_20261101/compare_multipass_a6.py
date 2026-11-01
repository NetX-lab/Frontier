"""Compare the formal A6 attention result with the three diagnostic A6 reruns."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
runs = {'formal': root.parent / 'a6_codex_20261101/result.json',
        **{name: root / f'a6_{name}/result.json' for name in ('original_shared', 'median3_shared', 'median3_all')}}
results = {name: json.loads(path.read_text()) for name, path in runs.items()}
for name, result in results.items():
    attention = result['attention']
    print(f"{name}: table {result['input_table']['sha256'][:12]} units {attention['unit_count']} rows {attention['row_count']} "
          f"legally-zero {attention['legally_zero']['verdict']} commit {result['source_commit'][:8]} dirty {len(result['source_dirty_paths'])}")
comparison = {}
for scope in results['formal']['attention']['per_scope']:
    comparison[scope] = {}
    cells = []
    for name, result in results.items():
        joint = result['attention']['per_scope'][scope]['joint']
        stats = joint['kernel_time_relative_error']
        comparison[scope][name] = {'verdict': joint['verdict'], 'evaluable': joint['evaluable_count'],
                                   'exact_lookup': joint['exact_lookup_count'],
                                   'median': stats['median_absolute'], 'p90': stats['p90_absolute']}
        cells.append('-' if stats['p90_absolute'] is None else
                     f"{joint['verdict'][:4]} {100 * stats['median_absolute']:5.2f}/{100 * stats['p90_absolute']:6.2f}% n={joint['evaluable_count']}")
    print(f"{scope:44s} " + ' | '.join(cells))
output = root / 'multipass_a6_comparison.json'
assert not output.exists()
output.write_text(json.dumps({'classification': 'DIAGNOSTIC_MULTIPASS_A6_COMPARISON',
                              'inputs': {name: str(path) for name, path in runs.items()},
                              'per_scope': comparison}, indent=2) + '\n')
