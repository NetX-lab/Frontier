"""Check exact retained P2-64 control reproduction, with a nonblocking receipt."""
import argparse
import csv
import json
from pathlib import Path
import sys

from coverage import save
from compare import TASK, sha

FORMAL = TASK / 'h3o_operator_costs/formal_p2_20261102'


def verify(control):
    from tests.comparison.e2e_gate.gate import read_rows
    result = json.loads((control / 'result.json').read_text())
    if result['status'] != 'TOOLS_FINISHED' or result['r4_status'] != 'PASS':
        raise ValueError('Full control pipeline did not finish with R4 PASS')
    readings = []
    for kind in ('warm', 'empty'):
        source = FORMAL / f'readings/p2_64_{kind}_reading.json'
        path = control / f'{kind}_reading.json'
        old, new = (json.loads(p.read_text()) for p in (source, path))
        # Only source provenance paths/hashes are allowed to differ. Retain the
        # original gate verbatim and every semantic field, including exclusions.
        if {k: v for k, v in old.items() if k != 'sources'} != {k: v for k, v in new.items() if k != 'sources'}:
            raise ValueError(f'{kind} reading changed beyond source provenance')
        readings.append(dict(kind=kind, source=str(source), source_sha256=sha(source),
                             reproduced=str(path), reproduced_sha256=sha(path), status='PASS_EXACT'))
    reading = json.loads((control / 'warm_reading.json').read_text())
    _, sides = read_rows(control / 'gate/rows')
    frontier = next(s for s in sides if s.startswith('frontier/'))
    def window(frame):
        return (float(frame['arrival_s'].min()), float(frame['completion_s'].max()),
                int(frame[['request_num_prefill_tokens', 'request_num_decode_tokens']].to_numpy().sum()))
    with (FORMAL / 'assessment/gate_readings.csv').open() as stream:
        expected = [r for r in csv.DictReader(stream) if r['case'] == 'p2_64']
    if len(expected) != 12:
        raise ValueError('Expected 3 repetitions x 4 P2-64 metrics')
    checked = []
    for row in expected:
        side = next(s for s in reading['warm']['repetitions'] if s.endswith(row['repeat']))
        warm = reading['warm']['repetitions'][side]
        observed = dict(case='p2_64', repeat=row['repeat'], metric=row['metric'],
                        unit='tokens/s' if row['metric'] == 'token_throughput_tps' else 'ms',
                        limit=.1, original_count=128, retained_count=warm['retained_request_count'],
                        excluded_ids=json.dumps(reading['openers'][side]))
        for label, rep in [('warm', warm), ('all', reading['all_request']['repetitions'][side])]:
            metric = next(m for m in rep['metrics'] if m['metric'] == row['metric'])
            observed.update({f'{label}_native': metric['vllm'], f'{label}_frontier': metric['frontier'],
                             f'{label}_relative_error': metric['relative_error'], f'{label}_status': metric['status']})
        for population in ('original', 'retained'):
            for label, source in [('native', side), ('frontier', frontier)]:
                frame = sides[source]
                if population == 'retained':
                    frame = frame.loc[~frame.index.isin(reading['openers'][side])]
                start, end, tokens = window(frame)
                observed.update({f'{population}_{label}_start_s': start, f'{population}_{label}_end_s': end})
                if label == 'native':
                    observed[f'{population}_tokens'] = tokens
        for key, value in observed.items():
            if str(value) != row[key]:
                raise ValueError(f'CSV mismatch {row["repeat"]}/{row["metric"]}/{key}: {value} != {row[key]}')
        if set(row) - set(observed) != {'reading', 'reading_sha256', 'exclusion_evidence'}:
            raise ValueError('Unexpected unchecked CSV fields')
        checked.append(dict(repeat=row['repeat'], metric=row['metric'], fields=len(observed)))
    return dict(status='PASS_EXACT', soft_gate=True, r4='PASS', readings=readings,
                csv_path=str(FORMAL / 'assessment/gate_readings.csv'),
                csv_sha256=sha(FORMAL / 'assessment/gate_readings.csv'), csv_rows=checked,
                all_request_verdict=reading['all_request']['verdict'], warm_verdict=reading['warm']['verdict'])


def main():
    try:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--control', type=Path, required=True)
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args()
        try:
            result = verify(args.control)
        except Exception as error:
            result = dict(status='UNVERIFIED', soft_gate=True, error=f'{type(error).__name__}: {error}')
        save(args.output, result)
        print(result['status'])
    except (Exception, SystemExit) as error:
        print(f'CONTROL_UNVERIFIED {type(error).__name__}: {error}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
