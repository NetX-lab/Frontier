"""Verify retained results without rerunning or replacing pricing."""
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
TESTS = ROOT.parent / '01'
frozen = json.loads((ROOT / 'freeze.json').read_text())
results = {arm: json.loads((ROOT / arm / 'wave_result.json').read_text())
           for arm in ('baseline', 'candidate')}
baseline, candidate = results.values()
comparisons = {}
for relative in ('pricing_payload.json', 'accepted-output/{arm}/ledger.json',
                 'accepted-output/{arm}/iteration-0/op_traces.jsonl'):
    paths = [ROOT / arm / relative.format(arm=arm) for arm in results]
    assert all(path.exists() for path in paths), relative
    values = [path.read_bytes() for path in paths]
    assert values[0] == values[1], relative
    comparisons[relative] = {'equal': True, 'bytes': len(values[0]),
        'sha256': hashlib.sha256(values[0]).hexdigest()}
assert baseline['payload_sha256'] == candidate['payload_sha256']
for key in ('sha256', 'whole_jsonl_sha256', 'ledger_sha256', 'events',
            'wrong_plan_key_negatives', 'late_json_error_restores_buffer_and_prefix_bytes'):
    assert baseline['output'][key] == candidate['output'][key], key
for result in results.values():
    assert result['main_run_successful_file_reads'] == 1
    assert all(sample['physical_collective_reads'] == 0 for sample in result['samples'])
    for key in ('public_getter_and_pending_snapshot_mutations_isolated',
                'pending_meta_mutation_does_not_change_source',
                'fresh_trace_ledger_mappings_independent'):
        assert result['output'][key] is True, key
    assert result['source_unchanged'] and result['fixture_metadata_unchanged']
for symbol, count in {'_get_on_demand_prediction': 288, '_predict_step5_ffn_operator': 144,
                      'predict_collective_plan': 64, '_step5_position_queries': 16,
                      'cache_key': 192}.items():
    assert baseline['operation_counts'][symbol] == candidate['operation_counts'][symbol] == count
tests = {}
for name in ('red', 'green', 'affected', 'baseline-tests'):
    tree = ET.parse(TESTS / (name + '.xml'))
    cases = tree.findall('.//testcase')
    counts = {'cases': len(cases), 'failed': 0, 'errors': 0, 'skipped': 0}
    for element, label in (('failure', 'failed'), ('error', 'errors'), ('skipped', 'skipped')):
        counts[label] = sum(case.find(element) is not None for case in cases)
    counts['passed'] = counts['cases'] - counts['failed'] - counts['errors'] - counts['skipped']
    tests[name] = counts
assert tests['red']['failed'] == 1 and tests['red']['errors'] == 0
assert tests['green']['passed'] == 1 and tests['green']['failed'] == 0
assert tests['affected']['passed'] == 124 and tests['affected']['skipped'] == 1
assert tests['affected']['failed'] == tests['affected']['errors'] == 0
assert tests['baseline-tests']['passed'] == 124 and tests['baseline-tests']['skipped'] == 1
for source in frozen['sources'].values():
    root = Path(source['root'])
    for relative, expected in source['hashes'].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative
    pin = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=root / 'frontier/cc_backend/backends/collective-sim', text=True).strip()
    assert pin == source['submodule_pin']
report = {'task_id': 'D29-PLAN-KEY-REUSE', 'status': 'PASS', 'tests': tests,
          'byte_comparisons': comparisons, 'source_hashes_unchanged': True,
          'warm_pricing_medians_s': {arm: value['warm_median_s'] for arm,value in results.items()},
          'emission_medians_s': {arm: value['output']['median_emission_s'] for arm,value in results.items()},
          'actual_key_encodings': {arm: value['actual_key_encodings_by_phase'] for arm,value in results.items()}}
(ROOT / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
