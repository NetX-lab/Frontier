"""Guard the retained-query census: pairing, every request, every output, dummy purity."""
import gzip
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/input_qualification_20261101/audit_inputs.py'
spec = importlib.util.spec_from_file_location('p2_input_audit_test', AUDIT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def write_forwards(root, decode_count=1024, bad_dummy=False, bad_prefill=False):
    def row(role, ids, step, real=True):
        return dict(role=role, mode='NONE', step_class='real' if real else 'dummy',
                    real=real, request_ids=ids, local_tokens=6144 if role == 'prefill' else len(ids),
                    descriptor=dict(padded_total_tokens=6144 if role == 'prefill' else max(1,len(ids))),
                    step=dict(context_requests=step))
    ids = list(range(128))
    with gzip.open(root / 'forwards.jsonl.gz', 'wt') as stream:
        for i in ids:
            stream.write(json.dumps(row('prefill', [0 if bad_prefill else i], 1)) + '\n')
        for step in range(decode_count):
            stream.write(json.dumps(row('decode', ids, 128 if step == 0 else 0)) + '\n')
        stream.write(json.dumps(row('decode', [0] if bad_dummy else [], 0, False)) + '\n')


def test_coordinate_order_preserves_prefill_context_pairs():
    key = audit.query_key('NONE','prefill',8,[1,2],[9,8],[])
    assert key == audit.query_key('NONE','prefill',8,[2,1],[8,9],[])
    assert key != audit.query_key('NONE','prefill',8,[1,2],[8,9],[])
    with pytest.raises(ValueError, match='pairing'):
        audit.query_key('NONE','prefill',8,[1],[],[])


def test_reconstructs_all_128_requests_and_1024_outputs(tmp_path):
    write_forwards(tmp_path)
    queries, receipt = audit.replay_queries(tmp_path)
    assert receipt['request_count'] == 128 and receipt['decode_tokens_per_request'] == 1024
    assert queries[audit.query_key('NONE','first_tail',128,[1]*128,[6143]*128,[])] == 1
    assert queries[audit.query_key('NONE','decode',128,[],[],[7166]*128)] == 1
    assert sum(queries.values()) == 128 + 1024 + 1


@pytest.mark.parametrize('setting,pattern', [({'decode_count':1023},'Incomplete request'),
    ({'bad_dummy':True},'Dummy forward'), ({'bad_prefill':True},'Prefill request population')])
def test_rejects_missing_tokens_impure_dummy_or_repeated_prefill(tmp_path,setting,pattern):
    write_forwards(tmp_path, **setting)
    with pytest.raises(ValueError,match=pattern):
        audit.replay_queries(tmp_path)


def test_cross_pass_predictor_pools_duplicate_features_and_rejects_interpolation():
    # The proposal must use the production mean reducer, rather than raw row medians.
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import _build_exact_feature_lookup
    lookup = _build_exact_feature_lookup(pd.DataFrame({'width':[1,1,1], 'cost':[1.,2.,6.]}), ['width'],'cost')
    assert lookup[(1.,)] == 3.
    assert (2.,) not in lookup
