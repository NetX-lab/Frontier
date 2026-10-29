"""Archive source-bound output comparisons and all paired timing samples."""
import difflib
import hashlib
import json
from pathlib import Path
import pstats
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RAW = Path(sys.argv[1]).resolve()
EVIDENCE = HERE / 'evidence'
EVIDENCE.mkdir(exist_ok=True)
assert not any(EVIDENCE.iterdir())
cases = {'baseline': RAW / 'registry/baseline',
         **{name: RAW / name / 'candidate' for name in ('registry', 'point', 'combined')}}
expected_codec = {'baseline': (1136, 704), 'registry': (976, 704),
                  'point': (1136, 544), 'combined': (976, 544)}
receipts = {name: json.loads((path / 'wave_result.json').read_bytes()) for name, path in cases.items()}
freeze = json.loads((RAW / 'registry/freeze.json').read_bytes())
source = Path(freeze['sources']['baseline']['root'])
head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
assert head == 'dbe19f531ff68d9edafc480418e8927c40eb1c60'
for relative, expected in freeze['sources']['baseline']['hashes'].items():
    assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == expected, relative
assert len(freeze['sources']['baseline']['hashes']) == 23
assert hashlib.sha256(Path(freeze['protected_body']).read_bytes()).hexdigest() == freeze['protected_body_sha256']
output_roots = {name: path / 'accepted-output' / ('baseline' if name == 'baseline' else 'candidate')
                for name, path in cases.items()}
paths = sorted(path.relative_to(output_roots['baseline'])
               for path in output_roots['baseline'].rglob('op_traces.jsonl'))
assert len(paths) == 19
auxiliary_paths = {Path('late-json-failure/op_traces.jsonl'), Path('profile/op_traces.jsonl')}
paths = [path for path in paths if path not in auxiliary_paths]
assert len(paths) == 17
paths.append(Path('ledger.json'))
comparisons = {}
for relative in paths:
    raw = (output_roots['baseline'] / relative).read_bytes()
    for name in ('registry', 'point', 'combined'):
        assert (output_roots[name] / relative).read_bytes() == raw, (name, str(relative))
    comparisons[str(relative)] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
auxiliary = {}
first_row = (output_roots['baseline'] / 'iteration-0/op_traces.jsonl').read_bytes().split(b'\n', 1)[1].splitlines(keepends=True)[0]
assert len(first_row) == 3902
for relative in sorted(auxiliary_paths):
    entries = {}
    baseline_header = json.loads((output_roots['baseline'] / relative).read_bytes().split(b'\n', 1)[0])
    for name, root in output_roots.items():
        raw = (root / relative).read_bytes()
        header_raw, body = raw.split(b'\n', 1)
        header = json.loads(header_raw)
        assert list(header) == list(baseline_header) == ['meta']
        assert list(header['meta']) == list(baseline_header['meta'])
        assert {key: value for key, value in header['meta'].items() if key != 'timestamp'} == {
            key: value for key, value in baseline_header['meta'].items() if key != 'timestamp'}
        assert type(header['meta']['timestamp']) is float
        assert body == (first_row if relative.parts[0] == 'late-json-failure' else b'')
        entries[name] = {'header': header, 'raw_file_sha256': hashlib.sha256(raw).hexdigest(),
                         'body_bytes': len(body), 'body_sha256': hashlib.sha256(body).hexdigest()}
    auxiliary[str(relative)] = entries
payload = (cases['baseline'] / 'pricing_payload.json').read_bytes()
assert hashlib.sha256(payload).hexdigest() == '3d745054625520bf71c4da53c1b5a3de2c72f0b8814e8a6a57acdeb537c5ca83'
counts, profiles = {}, {}
for name, receipt in receipts.items():
    assert (cases[name] / 'pricing_payload.json').read_bytes() == payload
    assert receipt['source_unchanged'] and receipt['fixture_metadata_unchanged']
    assert receipt['protected_body_sha256'] == freeze['protected_body_sha256']
    output = receipt['output']
    assert output['events'] == 208 and output['collective_events_and_cost_objects'] == 64
    assert output['all_event_metadata_proofs_and_order_equal']
    for predicate in ('public_getter_and_pending_snapshot_mutations_isolated',
                      'pending_meta_mutation_does_not_change_source', 'fresh_trace_ledger_mappings_independent'):
        assert output[predicate], (name, predicate)
    assert output['wrong_plan_key_negatives'] == 64
    assert output['late_json_error_restores_buffer_and_prefix_bytes'] == 3902
    counters = receipt['operation_counts']
    assert (counters['loads'], counters['dumps']) == expected_codec[name]
    for operation, expected in {'_get_on_demand_prediction': 288, '_predict_step5_ffn_operator': 144,
                                'predict_collective_plan': 64, '_step5_position_queries': 16,
                                'cache_key': 192, '_resolve_cluster_model_contract': 0}.items():
        assert counters[operation] == expected, (name, operation)
    counts[name] = counters
    stats = pstats.Stats(str(cases[name] / 'pricing.prof'))
    profiles[name] = {'canonical_json_calls': sum(item[1] for key, item in stats.stats.items()
                                                if key[2] == 'canonical_json'),
                      'total_profile_s': stats.total_tt}
    (EVIDENCE / f'{name}_wave.json').write_bytes((cases[name] / 'wave_result.json').read_bytes())
timing = json.loads((RAW / 'paired/timing.json').read_bytes())
assert timing['gc_enabled_before_and_after'] and len(timing['samples']) == 24
assert timing['pricing_payload_sha256'] == hashlib.sha256(payload).hexdigest()
assert all(sum(sample['variant'] == name for sample in timing['samples']) == 6 for name in cases)
(EVIDENCE / 'paired_timing.json').write_bytes((RAW / 'paired/timing.json').read_bytes())
patches = []
for relative in ('frontier/execution_time_predictor/prediction_model_registry.py',
                 'frontier/execution_time_predictor/moe_routing_workload.py'):
    patches.extend(difflib.unified_diff((source / relative).read_text().splitlines(keepends=True),
                   (RAW / 'patches' / relative).read_text().splitlines(keepends=True),
                   fromfile='a/' + relative, tofile='b/' + relative))
(EVIDENCE / 'candidate_source.diff').write_text(''.join(patches))
raw_bindings = {}
for path in [RAW / 'preparation.json', RAW / 'paired/timing.json',
             *[RAW / name for name in ('baseline.log', 'registry.log', 'point.log', 'combined.log', 'paired.log')],
             RAW / 'verify.log',
             *[path / name for path in cases.values() for name in ('wave_result.json', 'pricing.prof')],
             *[RAW / 'patches' / relative for relative in freeze['sources']['baseline']['hashes']
               if (RAW / 'patches' / relative).is_file()]]:
    raw = path.read_bytes()
    raw_bindings[str(path)] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
result = {'task_id': 'D29-RUNTIME-JSON-REUSE', 'status': 'NO_PRODUCTION_ALLOCATION',
          'source_revision': head, 'source_hashes': freeze['sources']['baseline']['hashes'],
          'protected_body_sha256': freeze['protected_body_sha256'], 'output_comparisons': comparisons,
          'auxiliary_trace_checks': auxiliary,
          'pricing_payload_sha256': hashlib.sha256(payload).hexdigest(), 'operation_counts': counts,
          'profile_counts': profiles, 'paired_summary': timing['summary'], 'raw_bindings': raw_bindings,
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'limits': ['Controls are explicit in-memory replacements, not committed production methods.',
                     'Only canonical FP8 DECODE1/layer60/16EP output and protected negatives are verified.',
                     'No extended current-model replacement/error matrix, native or full CLI run.',
                     'Two auxiliary initialization headers have real timestamps; their raw bytes are retained, not claimed equal.',
                     'The3902-byte failure body is exact; profile trace body is empty. Canonical wave files include pinned headers.',
                     'All24 timing samples and enabled GC retained; no GC subtraction or excluded sample.',
                     'Independent-arm large apparent gains are not accepted speedup numbers.',
                     'Small paired medians and GC-sensitive means do not justify allocation or interface expansion.']}
(EVIDENCE / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
bindings = {path.name: {'bytes': path.stat().st_size,
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in EVIDENCE.iterdir() if path.is_file()}
(EVIDENCE / 'artifact_sha256.json').write_text(json.dumps({'algorithm': 'sha256', 'files': bindings}, indent=2) + '\n')
print(json.dumps({'status': result['status'], 'full_file_comparisons_per_candidate': len(comparisons) + 1,
                  'source_bindings': len(result['source_hashes']), 'operation_counts': counts,
                  'paired_summary': timing['summary']}, indent=2))
