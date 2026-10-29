"""Torch-free reducer rejects ambiguous joins and inconsistent per-call work."""

import json

import pytest

from frontier.execution_time_predictor.dsv4_runner import (
    INPUT_SCOPE, SAMPLING_SCOPE, PREPARATION_COLUMNS, SAMPLING_COLUMNS,
    SAMPLING_BRANCH_KERNEL, LOGITS_PROCESSING_OFF, LOGITS_PROCESSING_ON,
)
from frontier.profiling.model_runner.dsv4_runner_tables import descriptor_index, reduce_replay_rows, write_table


def fixture(tmp_path, *, bad_count=False, bad_sampling=False, missing=False, ambiguous=False,
            branch=LOGITS_PROCESSING_OFF, missing_sampling=False, partial_branch=False):
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
            if missing_sampling and name == SAMPLING_SCOPE and step == 1:
                continue
            for activity, count in (('kernel', kernels - int(bad_count and name == INPUT_SCOPE and step == 1)), ('memcpy', copies)):
                records.extend(dict(kind='interval', scope=name, activity=activity,
                                    case_id='case', step_index=step, name="fixture_kernel") for _ in range(count))
            if name == SAMPLING_SCOPE and branch == LOGITS_PROCESSING_ON and not (partial_branch and step == 1):
                records.append(dict(kind="interval", scope=name, activity="kernel",
                                    case_id="case", step_index=step, name=SAMPLING_BRANCH_KERNEL))
    # Validation rows never enter the exact tables.
    rows.append(dict(rows[0], row_role='validation'))
    for step in range(2):
        records.append(dict(kind='step', phase="decode", case_id='case', step_index=step,
                            requests=[{}] * (7 if bad_sampling else 8)))
    write_table(tmp_path / 'dsv4_raw_e0.runner.csv', rows)
    (tmp_path / 'dsv4_raw_e0.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    return attention


def test_replay_cost_rows_join_and_activity_counts(tmp_path):
    attention = fixture(tmp_path)
    rows, inputs, comparisons = reduce_replay_rows(tmp_path, attention, 'fixture', LOGITS_PROCESSING_OFF)
    assert len(rows) == 2
    assert rows[0]['activity_count'] == 61 and rows[1]['activity_count'] == 9
    assert rows[0]['padded_rows'] == '8' and rows[1]['b'] == 8
    assert rows[1][SAMPLING_COLUMNS[0]] == LOGITS_PROCESSING_OFF
    assert rows[1]['run_tag'] == 'fixture' and rows[1]['case_id'] == 'case'
    assert all(row['row_role'] == 'cost' for row in rows)
    assert len(comparisons) == 2 and all(row['write_activity'] == 0 for row in comparisons)


@pytest.mark.parametrize('option,match', [('bad_count', 'activity count varies.*case|activity count varies.*preparation'),
                                          ('bad_sampling', 'Sampling b disagrees'),
                                          ('missing', 'no attention descriptor'),
                                          ('ambiguous', 'more than one descriptor')])
def test_replay_rejects_unpriced_or_ambiguous_evidence(tmp_path, option, match):
    attention = fixture(tmp_path, **{option: True})
    with pytest.raises(ValueError, match=match):
        reduce_replay_rows(tmp_path, attention, 'fixture', LOGITS_PROCESSING_OFF)


@pytest.mark.parametrize('observed,declared', [
    (LOGITS_PROCESSING_OFF, LOGITS_PROCESSING_ON),
    (LOGITS_PROCESSING_ON, LOGITS_PROCESSING_OFF),
])
def test_declared_branch_mismatch_names_run_case_step(tmp_path, observed, declared):
    attention = fixture(tmp_path, branch=observed)
    with pytest.raises(ValueError, match='Run fixture, case case, step 0: declared needs_logits_processing'):
        reduce_replay_rows(tmp_path, attention, 'fixture', declared)


@pytest.mark.parametrize('option,match', [
    ('partial_branch', 'step 1: declared needs_logits_processing'),
    ('missing_sampling', 'step 1: missing sampling activity'),
])
def test_every_real_step_must_match_declared_branch(tmp_path, option, match):
    attention = fixture(tmp_path, branch=LOGITS_PROCESSING_ON, **{option: True})
    with pytest.raises(ValueError, match=match):
        reduce_replay_rows(tmp_path, attention, 'fixture', LOGITS_PROCESSING_ON)


def test_two_source_promotion_and_manifest(tmp_path):
    import importlib.util
    from pathlib import Path
    from frontier.execution_time_predictor.dsv4_runner import REPLAY_TABLE
    path = Path(__file__).resolve().parents[2] / (
        'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/'
        'h3o_operator_costs/h200_tables/promote_runner_tables.py')
    spec = importlib.util.spec_from_file_location('promote_runner_tables', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    on, off = tmp_path / 'on', tmp_path / 'off'
    on.mkdir()
    off.mkdir()
    attention = fixture(on, branch=LOGITS_PROCESSING_ON)
    fixture(off, branch=LOGITS_PROCESSING_OFF)
    args = dict(output_dir=tmp_path / 'tables', manifest_path=tmp_path / 'manifest.json',
                replay_raw_dir=on, replay_run_tag='on', replay_logits_processing=LOGITS_PROCESSING_ON,
                sampling_raw_dir=off, sampling_run_tag='off', sampling_logits_processing=LOGITS_PROCESSING_OFF,
                attention_table=attention)
    manifest = module.promote_runner_tables(**args)
    rows = module.read_rows(tmp_path / 'tables' / REPLAY_TABLE)
    assert [row['operator'] for row in rows] == [INPUT_SCOPE, SAMPLING_SCOPE, SAMPLING_SCOPE]
    assert [row[SAMPLING_COLUMNS[0]] for row in rows[1:]] == ['1', '0']
    assert [row['run_tag'] for row in rows] == ['on', 'on', 'off']
    assert [int(row['activity_count']) for row in rows] == [61, 10, 9]
    assert manifest['source_run_tags'] == {'block': None, 'replay': 'on', 'sampling': 'off'}
    assert [run['row_counts'] for run in manifest['replay_runs']] == [
        {INPUT_SCOPE: 1, SAMPLING_SCOPE: 1}, {SAMPLING_SCOPE: 1}]
    assert all(item['sha256'] and item['row_count'] for run in manifest['replay_runs'] for item in run['inputs'])
    assert module.promote_runner_tables(**args) == manifest
    args['sampling_logits_processing'] = LOGITS_PROCESSING_ON
    with pytest.raises(ValueError, match='must declare different'):
        module.promote_runner_tables(**args)
