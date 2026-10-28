"""CPU admission of exact native-coordinate V4 pilots and frozen legacy plans."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from frontier.profiling.attention import dsv4_attention_main as cli
from frontier.profiling.attention.dsa_plan import DsaMeasurementLimits, build_pilot_cases
from test_dsv4_attention_profiling import _build_plan, hf_config

BRIDGE = Path(__file__).resolve().parents[2] / 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/p1h_bridge'


@pytest.fixture
def payload():
    return json.loads((BRIDGE / 'pilot_cases.json').read_text())


@pytest.fixture
def limits():
    return DsaMeasurementLimits(7168, 267665, 128, (8192,), 256, 2, 8, 1, 3,
                                'graph_replay', 'FULL_AND_PIECEWISE', (1, 2, 4, 8, 16, 24), 200000.0)


def test_package_pilot_exact_engines_phases_allowances_and_windows(payload, limits):
    engines, cases = build_pilot_cases(limits, payload)
    assert [(engine.engine_index, engine.prefill_query_len) for engine in engines] == [(0, 0), (1, 6143)]
    assert all((engine.chunk_budget, engine.max_num_seqs, engine.max_model_len, engine.block_size)
               == (8192, 128, 7168, 256) for engine in engines)
    by_id = {case.case_id: case for case in cases}
    assert len(cases) == 25
    assert {phase: sum(case.phase == phase for case in cases) for phase in ('prefill', 'decode', 'dummy', 'first_tail')} == {
        'prefill': 1, 'decode': 10, 'dummy': 10, 'first_tail': 4}
    for case_id, expected in [('e0-decode-b24-p6144',31), ('e0-decode-b4-p7150',16), ('e0-decode-b16-p6144',25)]:
        assert by_id[case_id].max_tokens == expected
    assert by_id['e0-decode-b4-p7150'].prompt_len + by_id['e0-decode-b4-p7150'].max_tokens == 7166
    for case in cases:
        assert (case.warmup, case.measured) == ((1,3) if case.phase in ('prefill','first_tail') else (2,8))
        if case.phase == 'dummy':
            assert (case.max_tokens, case.estimated_steps) == (0,10)
        elif case.phase == 'decode':
            assert case.estimated_steps == case.max_tokens - 1
    assert by_id['e0-prefill-p6144'].estimated_steps == 4
    assert [by_id[f'e1-first_tail-b{batch}-p6144'].estimated_steps for batch in (0,3,7,10)] == [8,13,16,18]


def test_package_plan_cli_rebuilds_pilot_and_summary(payload):
    spec = json.loads((BRIDGE / 'package_plan.json').read_text())
    from dsv4_graph_bridge_check import graph_mode_flags
    args = cli.build_parser().parse_args(['plan', *spec['plan_args'], '--model_config_path',
        str(BRIDGE / 'package_02/model/config.json'), '--pilot_cases', str(BRIDGE/'pilot_cases.json'),
        *graph_mode_flags('FULL_AND_PIECEWISE', tuple(spec['cudagraph_capture_sizes']))])
    plan = cli.build_plan(args)
    assert plan.to_json()['limits']['step_hold_us'] == 200000
    assert plan.summary()['cases_by_phase'] == {'prefill':1,'decode':10,'first_tail':4,'dummy':10}
    assert [engine['estimated_steps'] for engine in plan.summary()['engines']] == [280,55]
    assert len(spec['cudagraph_capture_sizes']) == 35
    from frontier.profiling.attention.dsv4_attention_rows import build_dsv4_rows
    with pytest.raises(ValueError, match="engine_policy='graph_replay' cannot collect or aggregate"):
        build_dsv4_rows(plan, ())


@pytest.mark.parametrize('target,key,value,message', [
    ('payload','extra',0,'unknown keys'), ('engine','extra',0,'unknown keys'),
    ('case','extra',0,'unknown keys'), ('peer','extra',0,'unknown keys'),
    ('engine','engine_index',1,'contiguous'), ('case','engine_index',2,'missing engine'),
    ('case','case_id','','non-empty'), ('case','phase','other','Unknown pilot phase'),
    ('case','batch_size',True,'positive int'), ('case','prompt_len',0,'positive int'),
    ('case','batch_size',2,'prefill requires batch_size'),
    ('peer','cg_mode','OTHER','Unknown dp_peer'), ('peer','uniform_token_count',None,'FULL dp_peer requires'),
    ('peer','num_tokens',8193,'exceeds chunk budget'), ('peer','num_tokens',0,'positive int'),
    ('peer','uniform_token_count',0,'positive int'), ('peer','max_query_len',False,'positive int'),
])
def test_strict_payload_and_peer_rejections(payload, limits, target, key, value, message):
    row = {'payload':payload, 'engine':payload['engines'][0], 'case':payload['cases'][0],
           'peer':payload['cases'][7]['dp_peer']}[target]
    row[key] = value
    with pytest.raises(ValueError, match=message):
        build_pilot_cases(limits, payload)


@pytest.mark.parametrize('target,key', [('payload','engines'),('engine','prefill_query_len'),('case','dp_peer'),('peer','cg_mode')])
def test_missing_keys_are_rejected(payload, limits, target, key):
    row = {'payload':payload, 'engine':payload['engines'][0], 'case':payload['cases'][0],
           'peer':payload['cases'][7]['dp_peer']}[target]
    row.pop(key)
    with pytest.raises(ValueError, match='missing keys'):
        build_pilot_cases(limits,payload)


def test_duplicate_case_id(payload,limits):
    payload['cases'][1]['case_id'] = payload['cases'][0]['case_id']
    with pytest.raises(ValueError,match='Duplicate'):
        build_pilot_cases(limits,payload)


@pytest.mark.parametrize('mode,peer_mode', [('PIECEWISE','FULL'),('NONE','FULL'),('NONE','PIECEWISE')])
def test_unreachable_peer(payload,limits,mode,peer_mode):
    payload['cases'] = [payload['cases'][7]]
    payload['cases'][0]['dp_peer']['cg_mode'] = peer_mode
    limits = replace(limits,cudagraph_mode=mode,cudagraph_capture_sizes=() if mode=='NONE' else (1,))
    with pytest.raises(ValueError,match='unreachable'):
        build_pilot_cases(limits,payload)


@pytest.mark.parametrize('phase', ['prefill','decode'])
def test_q_engine_refuses_ordinary_cases(payload,limits,phase):
    payload['cases'] = [next(row for row in payload['cases'] if row['phase']==phase)]
    payload['cases'][0]['engine_index'] = 1
    with pytest.raises(ValueError,match='prefill_query_len == 0'):
        build_pilot_cases(limits,payload)


@pytest.mark.parametrize('phase,key,value,message', [
    ('dummy','batch_size',2,'batch_size == 1'), ('dummy','dp_peer',None,'requires prompt_len'),
    ('dummy','prompt_len',1,'requires prompt_len'),
    ('first_tail','prompt_len',6143,'divisible'), ('first_tail','batch_size',1,'estimated_steps'),
    ('first_tail','engine_index',0,'positive engine'),
    ('prefill','prompt_len',7168,'max_context_len'),
    ('decode','prompt_len',7160,'max_context_len'),
])
def test_phase_bounds(payload,limits,phase,key,value,message):
    row = deepcopy(next(row for row in payload['cases'] if row['phase']==phase))
    row[key] = value
    if phase=='first_tail' and message=='estimated_steps':
        limits = replace(limits,max_context_len=6152)
    payload['cases']=[row]
    with pytest.raises(ValueError,match=message):
        build_pilot_cases(limits,payload)


@pytest.mark.parametrize('phase,capacity', [('prefill',6144),('decode',6144),('first_tail',20000)])
def test_kv_bounds(payload,limits,phase,capacity):
    payload['cases'] = [next(row for row in payload['cases'] if row['phase']==phase and (phase!='first_tail' or row['batch_size']>1))]
    with pytest.raises(ValueError,match='max_kv_tokens_per_lane'):
        build_pilot_cases(replace(limits,max_kv_tokens_per_lane=capacity),payload)


def test_first_tail_budget_bound(payload,limits):
    payload['cases']=[payload['cases'][-1]]
    with pytest.raises(ValueError,match='chunk budget'):
        build_pilot_cases(replace(limits,chunk_budgets=(6144,)),payload)


@pytest.mark.parametrize('policy,budgets,message', [('eager_chunked',(8192,),'graph_replay'),('graph_replay',(8192,4096),'exactly one')])
def test_pilot_policy_and_budget_admission(payload,limits,policy,budgets,message):
    limits = replace(limits,engine_policy=policy,chunk_budgets=budgets,step_hold_us=0,
                     cudagraph_mode=None if policy=='eager_chunked' else 'FULL_AND_PIECEWISE',
                     cudagraph_capture_sizes=() if policy=='eager_chunked' else (1,))
    with pytest.raises(ValueError,match=message):
        build_pilot_cases(limits,payload)


@pytest.mark.parametrize('argument,value', [('query_one_context_max',6144),('prefill_prompt_len',6144)])
def test_pilot_mutually_exclusive_serving_options(hf_config,payload,limits,argument,value):
    from frontier.profiling.attention.dsv4_attention_plan import build_dsv4_profile_plan
    with pytest.raises(ValueError,match='mutually exclusive'):
        build_dsv4_profile_plan(model_name='deepseek-ai/DeepSeek-V4-Flash',hf_config=hf_config,
            target_tensor_parallel_size=8,num_layers=6,limits=limits,kv_cache_dtype='fp8',pilot_cases=payload,**{argument:value})


@pytest.mark.parametrize('hold', [-1,float('inf'),float('nan'),True,'1'])
def test_hold_must_be_finite_nonnegative(limits,hold):
    with pytest.raises(ValueError,match='step_hold_us'):
        replace(limits,step_hold_us=hold)


def test_non_graph_hold_rejected_and_legacy_peer_omitted(hf_config,limits):
    with pytest.raises(ValueError,match='Nonzero'):
        replace(limits,engine_policy='eager_chunked',cudagraph_mode=None,cudagraph_capture_sizes=())
    plan = _build_plan(hf_config)
    assert 'step_hold_us' not in plan.to_json()['limits']
    assert all('dp_peer' not in case for case in plan.to_json()['cases'])


def test_pilot_engines_receive_the_v4_startup_query_check(hf_config, payload, limits):
    from frontier.profiling.attention.dsv4_attention_plan import build_dsv4_profile_plan

    payload["cases"] = [next(row for row in payload["cases"] if row["phase"] == "dummy")]
    limited = replace(limits, max_context_len=63)
    with pytest.raises(ValueError, match="V4 exact prefill startup query length 64 exceeds max_model_len=63"):
        build_dsv4_profile_plan(
            model_name="deepseek-ai/DeepSeek-V4-Flash", hf_config=hf_config,
            target_tensor_parallel_size=8, num_layers=6, limits=limited,
            kv_cache_dtype="fp8", pilot_cases=payload,
        )


def test_pilot_peers_serialize_only_when_present(hf_config, payload, limits):
    from frontier.profiling.attention.dsv4_attention_plan import build_dsv4_profile_plan

    plan = build_dsv4_profile_plan(
        model_name="deepseek-ai/DeepSeek-V4-Flash", hf_config=hf_config,
        target_tensor_parallel_size=8, num_layers=6, limits=limits,
        kv_cache_dtype="fp8", pilot_cases=payload,
    )
    cases = {row["case_id"]: row for row in plan.to_json()["cases"]}
    assert "dp_peer" not in cases["e0-decode-b8-p6144"]
    assert cases["e0-decode-b8-p6144-full16"]["dp_peer"] == {
        "num_tokens": 16, "cg_mode": "FULL", "uniform_token_count": 1, "max_query_len": 1,
    }
