"""Torch-free reducer rejects ambiguous joins and inconsistent per-call work."""

import csv
import json

import pytest

from frontier.execution_time_predictor.dsv4_runner import INPUT_SCOPE, SAMPLING_SCOPE, PREPARATION_COLUMNS
from frontier.profiling.model_runner.dsv4_runner_tables import descriptor_index, reduce_replay_rows, write_table


def fixture(tmp_path, *, bad_count=False, bad_sampling=False, missing=False, ambiguous=False):
    descriptor = dict(zip(PREPARATION_COLUMNS, ('decode', 'FULL', 8, 8, 8, 0, 8, 8)))
    attention = tmp_path / 'attention.csv'
    descriptors = [descriptor | {'case_id': 'other' if missing else 'case'}]
    if ambiguous:
        descriptors.append(descriptor | {'case_id': 'case', 'padded_rows': 16})
    write_table(attention, descriptors)
    rows, records = [], []
    for name, kernels, copies in ((INPUT_SCOPE, 54, 7), (SAMPLING_SCOPE, 7, 2)):
        rows.append(dict(operator=name, row_role='cost', profile_method='record_function',
                         case_id='case', engine_index=0, row_id=name,
                         **{f'kernel_stats.{name}.median': 0.05, f'memcpy_stats.{name}.median': 0.01}))
        for step in range(2):
            for activity, count in (('kernel', kernels - int(bad_count and name == INPUT_SCOPE and step == 1)), ('memcpy', copies)):
                records.extend(dict(kind='interval', scope=name, activity=activity,
                                    case_id='case', step_index=step) for _ in range(count))
    # Validation rows never enter the exact tables.
    rows.append(dict(rows[0], row_role='validation'))
    for step in range(2):
        records.append(dict(kind='step', case_id='case', step_index=step,
                            requests=[{}] * (7 if bad_sampling else 8)))
    write_table(tmp_path / 'dsv4_raw_e0.runner.csv', rows)
    (tmp_path / 'dsv4_raw_e0.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    return attention


def test_replay_cost_rows_join_and_activity_counts(tmp_path):
    attention = fixture(tmp_path)
    rows, inputs, comparisons = reduce_replay_rows(tmp_path, attention, 'fixture')
    assert len(rows) == 2
    assert rows[0]['activity_count'] == 61 and rows[1]['activity_count'] == 9
    assert rows[0]['padded_rows'] == '8' and rows[1]['b'] == 8
    assert all(row['row_role'] == 'cost' for row in rows)
    assert len(comparisons) == 2 and all(row['write_activity'] == 0 for row in comparisons)


@pytest.mark.parametrize('option,match', [('bad_count', 'activity count varies.*case|activity count varies.*preparation'),
                                          ('bad_sampling', 'Sampling b disagrees'),
                                          ('missing', 'no attention descriptor'),
                                          ('ambiguous', 'more than one descriptor')])
def test_replay_rejects_unpriced_or_ambiguous_evidence(tmp_path, option, match):
    attention = fixture(tmp_path, **{option: True})
    with pytest.raises(ValueError, match=match):
        reduce_replay_rows(tmp_path, attention, 'fixture')
