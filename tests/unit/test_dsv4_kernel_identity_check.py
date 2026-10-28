"""CPU launch multiset containment, native segmentation and identity CLI gates."""

from collections import Counter
from dataclasses import asdict, replace
import gzip
import json
from pathlib import Path

import pytest

import dsv4_kernel_identity_check as identity
from dsv4_graph_bridge_check import BridgeConfig, BridgeRun, graph_mode_flags
from frontier.profiling.attention import dsv4_attention_main as cli
from frontier.profiling.kernel_gap.dsv4_replay_gaps import ReplayRecords
from test_dsv4_attention_profiling import PLAN_ARGS

K=('attention',(1,1,1),(32,1,1))


def _native_trace(root,role,kind,mode,kernels,repeat=1):
    dispatch={'FULL':'model_dispatch_run_fullgraph','PIECEWISE':'model_dispatch_run_pw_graph','NONE':'model_forward'}[mode]
    events=[]
    for index in range(repeat):
        base=index*1000
        def event(name,cat,ts,dur,args=None,tid=1):
            return dict(ph='X',name=name,cat=cat,ts=base+ts,dur=dur,pid=10,tid=tid,args=args or {})
        events.append(event(f'frontier_native_metadata|{index}|attention_metadata','user_annotation',20,10))
        events.append(event(dispatch,'user_annotation',40,200))
        events.append(event('gloo:all_reduce','user_annotation',5,10,tid=99))
        if kind!='dummy':
            events.append(event(f'frontier_native_ownership|{index}|logits','user_annotation',250,10))
            c,g={'decode':(0,1),'prefill':(1,0),'first_tail_alone':(1,0),'first_tail_with_decodes':(1,3)}[kind]
            events.append(event(f'execute_context_{c}({c})_generation_{g}({g})','user_annotation',0,400))
        for ordinal,((name,grid,block),count) in enumerate(kernels.items()):
            correlation=index*100+ordinal
            events.append(event('cudaLaunchKernel','cuda_runtime',50+ordinal,1,{'correlation':correlation}))
            for _ in range(count):
                events.append(event(name,'kernel',300+ordinal,2,{'correlation':correlation,'stream':7,'grid':list(grid),'block':list(block)},tid=7))
            events.append(event('copy','gpu_memcpy',350,1,{'correlation':correlation,'stream':7},tid=7))
    path=root/role/'dp0'/'dp0_pp0_tp0_synthetic.pt.trace.json.gz'
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt') as handle:json.dump({'traceEvents':events},handle)
    return path


def test_native_dispatch_parsing_deduplicates_forward_counts_and_ignores_copies(tmp_path):
    _native_trace(tmp_path,'decode','decode','FULL',Counter({K:4}),repeat=3)
    result=identity.native_candidates([tmp_path])
    assert len(result)==1
    assert result[0].kernels=={K:4}
    assert (result[0].forward_count,result[0].step_class,result[0].mode,result[0].roles)==(3,'decode','FULL',('decode',))


@pytest.mark.parametrize('phase,batch,kind',[('prefill',1,'prefill'),('decode',8,'decode'),('first_tail',1,'first_tail_alone'),
    ('first_tail',4,'first_tail_with_decodes'),('dummy',1,'dummy')])
def test_native_class_mapping_and_mode_match(phase,batch,kind):
    assert identity.native_step_class(phase,batch)==kind
    candidate=identity.NativeCandidate(kind,'FULL',Counter({K:2}),1,('prefill',) if phase=='dummy' else ('decode',))
    assert identity.match_kernels(Counter({K:1}),kind,'FULL',[candidate])['status']=='PASS'
    assert identity.match_kernels(Counter({K:1}),kind,'PIECEWISE',[candidate])['candidate_count']==0
    assert identity.match_kernels(Counter({K:1}),'other','FULL',[candidate])['candidate_count']==0


def test_missing_and_grid_mismatch_report_closest_candidate():
    other=(K[0],(2,1,1),K[2])
    candidates=[identity.NativeCandidate('decode','FULL',Counter({other:2}),5,('decode',)),
                identity.NativeCandidate('decode','FULL',Counter(),1,('decode',))]
    result=identity.match_kernels(Counter({K:2,other:1}),'decode','FULL',candidates)
    assert result['status']=='FAIL' and result['missing_kernel_count']==2
    assert result['candidate_count']==2 and result['closest_candidate_forward_count']==5
    assert result['grid_mismatches']==[{'name':K[0],'block':list(K[2]),'expected_grid':list(K[1]),'native_grids':[[2,1,1]],'missing_count':2}]


def _role_expansion_inputs():
    from test_dsv4_attention_profiling import _build_plan, CONFIG_PATH
    from frontier.profiling.attention.dsa_plan import load_hf_config
    plan = _build_plan(load_hf_config(CONFIG_PATH))
    variants = {layer.global_layer_id: layer.attention_variant for layer in plan.shape.layers}
    roles = {layer.global_layer_id: layer.plans_tile_schedule for layer in plan.shape.layers}
    counts = {('uncompressed', True): 2, ('csa', True): 1, ('csa', False): 20,
              ('hca', True): 1, ('hca', False): 19}
    # Distinct multisets make every full-model expansion factor observable.
    signatures = {role: (role[0] + str(role[1]), K[1], K[2]) for role in counts}
    kernels = {layer: Counter({signatures[(variants[layer], roles[layer])]: 1}) for layer in variants}
    return kernels, variants, roles, counts, signatures


def test_planner_reuse_full_model_expansion_with_several_layers_per_role():
    kernels, variants, roles, counts, signatures = _role_expansion_inputs()
    assert identity.expand_attention_kernels(kernels, variants, roles, counts) == {
        signatures[role]: count for role, count in counts.items()}


@pytest.mark.parametrize('layer', [1, 6, 7])
def test_unequal_multisets_within_role_raise(layer):
    kernels, variants, roles, counts, _ = _role_expansion_inputs()
    kernels[layer][K] += 1
    with pytest.raises(ValueError, match='role.*unequal kernel multisets.*layers'):
        identity.expand_attention_kernels(kernels, variants, roles, counts)


def test_missing_full_model_role_raises():
    kernels, variants, roles, counts, _ = _role_expansion_inputs()
    del variants[2]
    with pytest.raises(ValueError, match="no reduced layers.*csa.*True"):
        identity.expand_attention_kernels(kernels, variants, roles, counts)


def test_reduced_role_absent_from_full_model_raises():
    kernels, variants, roles, counts, _ = _role_expansion_inputs()
    roles[0] = False
    with pytest.raises(ValueError, match="roles absent from the full model.*uncompressed.*False"):
        identity.expand_attention_kernels(kernels, variants, roles, counts)


def test_missing_kernels_in_any_reduced_layer_raise():
    kernels, variants, roles, counts, _ = _role_expansion_inputs()
    del kernels[10]
    with pytest.raises(ValueError, match="role.*csa.*False.*no attention kernels.*layer 10"):
        identity.expand_attention_kernels(kernels, variants, roles, counts)


def _pilot(tmp_path):
    mode='FULL_AND_PIECEWISE'
    plan_args = tuple(PLAN_ARGS[:-1])
    plan=cli.build_plan(cli.build_parser().parse_args(['plan',*plan_args,*graph_mode_flags(mode,(1,2,4,8))]))
    case=next(case for case in plan.cases if case.phase=='decode' and case.engine_index == 0)
    engine={'kind':'engine','engine_policy':'graph_replay','engine_index':0,'device_name':'fake',
        'driver_version':'fake','target_tensor_parallel_size':8,'runtime_versions':{'torch':'fake','torch_cuda':'fake'},
        'step_hold_us':0.0,'hold_cycles_per_us':None,'multi_stream_gemm_token_threshold':1024,
        'cudagraph_capture':{'cudagraph_mode':mode},
        'kv_cache_groups':plan.summary()['kv_cache_groups'],
        'layers':[dict(global_layer_id=layer.global_layer_id,attention_variant=layer.attention_variant,compress_ratio=max(1,layer.compress_ratio)) for layer in plan.shape.layers]}
    step=dict(kind='step',engine_index=0,case_id=case.case_id,step_index=0,phase='decode',cudagraph_dispatch={'cg_mode':'FULL','num_tokens':1})
    intervals=tuple(dict(kind='interval',engine_index=0,case_id=case.case_id,step_index=0,
        global_layer_id=layer.global_layer_id,scope='dsv4_sparse_attention',activity='kernel',name=K[0],grid=list(K[1]),block=list(K[2])) for layer in plan.shape.layers)
    record=ReplayRecords('synthetic',engine,(step,),(),intervals)
    config=BridgeConfig({},plan_args,(),(BridgeRun(mode,0,(case.case_id,)),),(1,2,4,8),tmp_path/'bridge')
    return config,record,case


def test_saved_pilot_full_expansion_containment_and_margin_gate(tmp_path):
    config,record,case=_pilot(tmp_path)
    assert identity.layer_role_counts(config)=={('uncompressed',True):2,('csa',True):1,
        ('csa',False):20,('hca',True):1,('hca',False):19}
    candidate=identity.NativeCandidate('decode','FULL',Counter({K:44}),2,('decode',))
    report=identity.check_identity(config,[record],[candidate])
    assert report['verdict']=='PASS' and report['steps'][0]['expected_kernel_count']==43
    assert report['layer_role_counts'] == [
        {'variant': variant, 'plans_tile_schedule': planner, 'count': count}
        for (variant, planner), count in sorted(identity.layer_role_counts(config).items())]
    assert 'variant_layer_counts' not in report
    report=identity.check_identity(config,[record],[])
    assert report['verdict']=='FAIL' and report['steps'][0]['missing_kernel_count']==43
    report=identity.check_identity(config,[record],[],[case.case_id])
    assert report['verdict']=='PASS' and report['cases'][0]['status']=='FAIL' and report['cases'][0]['margin']


@pytest.mark.parametrize('count,margin,exit_code',[(43,False,0),(42,False,1),(42,True,0)])
def test_identity_cli_reads_raw_and_native_traces_and_returns_gate(tmp_path,count,margin,exit_code):
    config,record,case=_pilot(tmp_path)
    raw=config.output_root/'FULL_AND_PIECEWISE/raw/dsv4_raw_e0.jsonl'
    raw.parent.mkdir(parents=True)
    raw.write_text(''.join(json.dumps(row)+'\n' for row in (record.engine,*record.steps,*record.intervals)))
    config_path=tmp_path/'config.json'
    config_path.write_text(json.dumps(dict(asdict(config),output_root=str(config.output_root))))
    native=tmp_path/'native'
    _native_trace(native,'decode','decode','FULL',Counter({K:count}))
    output=tmp_path/'report.json'
    args=['--config',str(config_path),'--native_trace_dirs',str(native),'--output',str(output)]
    if margin:args+=['--margin_cases',case.case_id]
    assert identity.main(args)==exit_code
    assert json.loads(output.read_text())['steps'][0]['missing_kernel_count']==max(0,43-count)
