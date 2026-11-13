"""Verify the independent local sampling and admission boundary."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from local_scope_manifest import admit, sample_cases

TASK = Path(__file__).resolve().parents[3] / 'task_memory/task_2026-10-15_step5_preview_h800'


def manifest():
    return json.loads((TASK / 'local_scope_sampling_01.json').read_text())


def test_pinned_boundary_sampling_and_reserved_holdouts():
    m = manifest()
    assert admit(m, TASK) is None
    assert m['geometry'] == {'r': 8, 'B': 16, 'W': 512, 'K': 512}
    lengths = {c['total'] for c in m['cases'] if c['label'].startswith('boundary_')}
    assert lengths == {7, 8, 9, 15, 16, 17, 511, 512, 513, 4095, 4096, 4097}
    assert {c['partition'] for c in m['cases']} == {'sample', 'holdout'}
    assert sample_cases(m['geometry'], m['seed']) == m['cases']


def test_projection_and_branch_ownership_are_exclusive():
    scopes = {s['id']: s for s in manifest()['scopes']}
    assert 'indexer_w' not in scopes['dsa_projection']['includes']
    assert scopes['indexer_w']['includes'] == ['indexer_w']
    assert 'summary_update' in scopes['dsa_decode_prepare']['includes']
    assert 'summary_update' not in scopes['dsa_decode_root']['includes']
    assert 'headwise_gate' in scopes['dsa_decode_root']['includes']
    assert 'headwise_gate' not in scopes['swa_attention']['includes']


@pytest.mark.parametrize('fault', ['identity', 'backend', 'features', 'coverage', 'fusion',
                                   'ownership', 'timing', 'partition', 'target', 'g1',
                                   'backend_contract', 'fusion_contract', 'missing_scope', 'topology'])
def test_admission_rejects_invalid_plans(fault):
    m = deepcopy(manifest())
    if fault == 'identity':
        m['identity']['image'] = 'candidate'
    elif fault == 'backend':
        m['scopes'][0]['backend'] = 'other'
    elif fault == 'features':
        del m['cases'][0]['layout']
    elif fault == 'coverage':
        m['coverage'][0]['status'] = 'complete'
    elif fault == 'fusion':
        m['scopes'][0]['includes'].append('indexer_w')
    elif fault == 'ownership':
        m['scopes'].append(deepcopy(m['scopes'][0]))
    elif fault == 'timing':
        m['measurements']['device_duration_ms'] = 0
    elif fault == 'partition':
        m['cases'][-1]['partition'] = 'sample'
    elif fault == 'target':
        m['exclusive_target']['observed'] = True
    elif fault == 'backend_contract':
        m['scopes'][0]['backend'] = m['scopes'][0]['required_backend'] = 'other'
    elif fault == 'fusion_contract':
        m['scopes'][0]['includes'].append('indexer_w')
        m['scopes'][0]['required_includes'].append('indexer_w')
    elif fault == 'missing_scope':
        m['scopes'].pop()
    elif fault == 'topology':
        m['scopes'][0]['topology']['tp'] = 8
    else:
        m['scopes'][0]['g1'] = 'missing.json'
    with pytest.raises(ValueError):
        admit(m, TASK)


@pytest.mark.parametrize('field,value', [('branches', []), ('owner', 'sparse_indexer_w.forward'),
    ('source', 'native_g1_attention_state/observed_generation2/dsa_long/dsa_long.json'),
    ('g1', 'native_source_export/observed/vllm/models/step5/nvidia/model.py'),
    ('geometry_scope', 'native TP8 rank execution'), ('native_role', 'decode'), ('precision', 'fp8')])
def test_scope_identity_cannot_be_relabeled(field, value):
    m = manifest(); m['scopes'][0][field] = value
    with pytest.raises(ValueError): admit(m, TASK)


@pytest.mark.parametrize('field,value', [('coverage', []), ('measurements', {}), ('construction_sources', {})])
def test_required_plan_sections_cannot_disappear(field, value):
    m = manifest(); m[field] = value
    with pytest.raises(ValueError): admit(m, TASK)
