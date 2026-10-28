"""CPU checks for bootstrap run invocations, engine layouts and bridge verdicts."""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from frontier.profiling.attention.graph_capture import CaptureRecord, GraphCapture, NodeClone, ScopeEntry
from frontier.profiling.attention.dsv4_graph_replay_runner import CUDAGRAPH_DISPATCH_KEY, LAYER_SCOPE
from frontier.profiling.attention.graph_replay_trace import hold_annotation_name
from test_graph_replay_trace import _activity, _annotation, _launch, _node, _scope, _step

_SCRIPT = Path(__file__).resolve().parents[1] / 'comparison/calibration/dsv4_graph_bridge_check.py'
_SPEC = importlib.util.spec_from_file_location('dsv4_graph_bridge_check', _SCRIPT)
bridge = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = bridge
_SPEC.loader.exec_module(bridge)


@pytest.fixture
def config(tmp_path):
    return bridge.BridgeConfig(
        bootstrap={'script':'accepted/profile_dsv4_prefill.py','source_snapshot':'snapshot',
                   'expected_source_hashes':{'vllm/file.py':'unchanged'},'tool_hashes':{'tool':'unchanged'},
                   'profile_args':['old'],'worker_wheel_site':'wheel','native_package':'native'},
        plan_args=('--model_name','model','--chunk_budgets','8'),
        run_args=('--model_dir','model_dir','--expected_attention_backend','FLASHMLA_SPARSE_DSV4',
                  '--gpu_memory_utilization','0.85','--seed','1234'),
        runs=tuple(bridge.BridgeRun(mode,2,('case',)) for mode in ('FULL_AND_PIECEWISE','PIECEWISE','NONE')),
        cudagraph_capture_sizes=(1,2,4,8),output_root=tmp_path)


def _write_config(config):
    path=config.output_root/'config.json'
    path.write_text(json.dumps(dict(asdict(config),output_root=str(config.output_root))))
    return path


@pytest.mark.parametrize('mode',['FULL_AND_PIECEWISE','PIECEWISE','NONE'])
def test_profile_args_match_mode_and_keep_shared_flags(config,mode):
    run=bridge.BridgeRun(mode,2,('prefill','decode'))
    root=config.output_root/mode
    sizes=[] if mode=='NONE' else ['--cudagraph_capture_sizes','1','2','4','8']
    assert bridge.profile_args(config,run)==['run',*config.plan_args,'--engine_policy','graph_replay',
        '--cudagraph_mode',mode,*sizes,'--engine_index','2',*config.run_args,'--raw_dir',str(root/'raw'),
        '--trace_dir',str(root/'traces'),'--keep_traces','--case_ids','prefill','decode']
    assert bridge.graph_mode_flags(mode,config.cudagraph_capture_sizes)==[
        '--engine_policy','graph_replay','--cudagraph_mode',mode,*sizes]


@pytest.mark.parametrize('failed',[None,'PIECEWISE','OSError'])
def test_run_uses_bootstrap_environment_logs_and_continues_after_failure(config,monkeypatch,failed):
    calls=[]
    monkeypatch.setenv('BRIDGE_PARENT_SENTINEL','retained')
    def invoke(command,**kwargs):
        payload=json.loads(Path(command[-1]).read_text())
        mode=payload['profile_args'][payload['profile_args'].index('--cudagraph_mode')+1]
        calls.append((mode,command,kwargs,payload))
        kwargs['stdout'].write(f'fake {mode}\n')
        if mode=='PIECEWISE' and failed=='OSError':raise OSError('fake unavailable interpreter')
        return SimpleNamespace(returncode=3 if mode==failed else 0)
    monkeypatch.setattr(bridge.subprocess,'run',invoke)
    assert bridge.run_bridge(config)==int(failed is not None)
    results=json.loads((config.output_root/'bridge_run.json').read_text())['runs']
    assert [mode for mode,*_ in calls]==[run.mode for run in config.runs]
    for mode,command,kwargs,payload in calls:
        assert command==[sys.executable,config.bootstrap['script'],'--config',str(config.output_root/mode/'bootstrap_config_e2.json')]
        assert {key:payload[key] for key in config.bootstrap if key!='profile_args'}=={
            key:value for key,value in config.bootstrap.items() if key!='profile_args'}
        assert kwargs['env']['VLLM_USE_BREAKABLE_CUDAGRAPH']=='1'
        assert kwargs['env']['VLLM_USE_V2_MODEL_RUNNER']=='1'
        assert kwargs['env']['BRIDGE_PARENT_SENTINEL']=='retained'
        assert kwargs['stderr'] is bridge.subprocess.STDOUT and kwargs['check'] is False
        assert results[f'{mode}/e2']['duration_seconds']>=0
        assert (config.output_root/mode/'run_e2.log').read_text().startswith(f'fake {mode}')
    if failed=='OSError':
        assert results['PIECEWISE/e2']['exit_code'] is None
        assert 'unavailable interpreter' in results['PIECEWISE/e2']['error']


def _artifacts(config,full_steps=True,attribution_error=False,exit_code=0,hold=False):
    owner=(4,'dsv4_q_projection')
    record=CaptureRecord(captures=(GraphCapture(0,'graph',None,7),),nodes=(_node(10,'kernel',owner=owner),),
        scope_entries=(ScopeEntry(0,(4,LAYER_SCOPE),None),ScopeEntry(0,owner,(4,LAYER_SCOPE))),clones=(NodeClone(1010,10),))
    runs={}
    for run in config.runs:
        root=config.output_root/run.mode
        (root/'raw').mkdir(parents=True,exist_ok=True)
        (root/'traces').mkdir(exist_ok=True)
        saved=CaptureRecord() if run.mode=='NONE' else record
        (root/'raw'/f'dsv4_capture_e{run.engine_index}.json').write_text(json.dumps({'record':saved.to_json(),'summary':{'clone_count':1}}))
        cg_mode=('FULL' if full_steps else 'PIECEWISE') if run.mode=='FULL_AND_PIECEWISE' else run.mode
        (root/'raw'/f'dsv4_raw_e{run.engine_index}.jsonl').write_text(json.dumps({'kind':'step','case_id':run.case_ids[0],
            CUDAGRAPH_DISPATCH_KEY:{'cg_mode':cg_mode,'num_tokens':1,'num_reqs':1}})+'\n')
        if run.mode=='NONE':events=[_step(),_scope(owner,10,50),_launch(1,20,graph=False),_activity(0,'eager',100,20)]
        else:events=[_step(),_launch(1,10),_activity(1010,'mismatch' if attribution_error else 'kernel',100,20)]
        events.extend([_launch(2,200,graph=False),_activity(0,'outside',210,5,correlation=2)])
        if hold:events.extend([_annotation(hold_annotation_name(0),1,5),_launch(3,2,graph=False),_activity(0,'hold',50,20,correlation=3)])
        (root/'traces'/f'{run.case_ids[0]}.json').write_text(json.dumps({'traceEvents':events}))
        runs[run.key]={'exit_code':exit_code}
    (config.output_root/'bridge_run.json').write_text(json.dumps({'runs':runs}))


def test_report_passes_graph_eager_and_hold_attribution(config,capsys):
    _artifacts(config,hold=True)
    assert bridge.report_bridge(config)==0
    report=json.loads((config.output_root/'bridge_report.json').read_text())
    assert report['verdict']=='PASS' and report['reasons']==[]
    assert '3 runs' in capsys.readouterr().out
    for run in config.runs:
        result=report['runs'][run.key]
        trace=result['traces'][0]
        assert trace['error'] is None and trace['outside_kernel_count']==1
        assert trace['hold_interval_count']==1
        assert trace['graph_kernel_interval_count']==(0 if run.mode=='NONE' else 1)
        assert trace['eager_kernel_interval_count']==(2 if run.mode=='NONE' else 1)
        assert trace['unattributed_launch_count']==0


@pytest.mark.parametrize('option',[{'attribution_error':True},{'full_steps':False},{'exit_code':4}])
def test_report_failures_are_explicit(config,option):
    _artifacts(config,**option)
    assert bridge.report_bridge(config)==1
    assert json.loads((config.output_root/'bridge_report.json').read_text())['reasons']


def test_missing_artifacts_produce_fail_report(config):
    assert bridge.report_bridge(config)==1
    report=json.loads((config.output_root/'bridge_report.json').read_text())
    assert all(result['artifact_errors'] for result in report['runs'].values())


@pytest.mark.parametrize('key',['modes','engine_index','case_ids'])
def test_old_top_level_keys_rejected(config,key):
    path=_write_config(config)
    payload=json.loads(path.read_text());payload[key]=[]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError,match='old top-level.*use runs'):bridge.load_config(path)


@pytest.mark.parametrize('runs',[[],[{'mode':'bad','engine_index':0,'case_ids':['case']}],
    [{'mode':'NONE','engine_index':0,'case_ids':['case']}]*2])
def test_config_rejects_invalid_runs(config,runs):
    path=_write_config(config)
    payload=json.loads(path.read_text());payload['runs']=runs
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):bridge.load_config(path)


def test_two_engines_share_mode_without_cross_reading(config):
    config=replace(config,runs=(bridge.BridgeRun('FULL_AND_PIECEWISE',0,('a',)),bridge.BridgeRun('FULL_AND_PIECEWISE',1,('b',))))
    _artifacts(config)
    # The second run contributes PIECEWISE dispatch while sharing the configured mode.
    path=config.output_root/'FULL_AND_PIECEWISE/raw/dsv4_raw_e1.jsonl'
    row=json.loads(path.read_text());row[CUDAGRAPH_DISPATCH_KEY]['cg_mode']='PIECEWISE';path.write_text(json.dumps(row)+'\n')
    assert bridge.main(['--config',str(_write_config(config)),'--report'])==0
    report=json.loads((config.output_root/'bridge_report.json').read_text())
    assert set(report['runs'])=={'FULL_AND_PIECEWISE/e0','FULL_AND_PIECEWISE/e1'}
    assert [Path(report['runs'][run.key]['traces'][0]['path']).name for run in config.runs]==['a.json','b.json']
