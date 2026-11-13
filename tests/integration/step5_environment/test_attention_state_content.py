"""Regression checks for retained state, duplicate records and selector independence."""
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location('attention_content',Path(__file__).with_name('compare_attention_state.py'))
content = importlib.util.module_from_spec(spec); spec.loader.exec_module(content)


def fixture(tmp_path,monkeypatch,dsa=False):
    monkeypatch.setattr(content,'summary',lambda *args: {})
    records=[]
    sequence=iter(range(1000))
    def record(name,value):
        path=tmp_path/(str(next(sequence))+'.pt'); torch.save(value,path)
        row=dict(name=name,path=str(path),size_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 shape=list(value.shape),dtype=str(value.dtype),stride=list(value.stride()))
        records.append(row); return row
    count=10 if dsa else 2
    cache=torch.zeros((1,1,16,4),dtype=torch.bfloat16)
    initial=[record('raw_cache',cache.clone())];records.clear()
    positions=torch.arange(count);query=torch.zeros((count,1,2),dtype=torch.bfloat16)
    key=torch.zeros_like(query);value=torch.arange(1,count+1).bfloat16()[:,None,None].expand(-1,1,2).clone()
    cache[0,0,:count,2:]=value[:,0]
    output=(torch.arange(1,count+1).double()+1)[:,None,None].expand(-1,1,2)/2
    output=output.bfloat16()*(0.5 if dsa else 1)
    x=(((positions[:,None]+1)*(torch.arange(2)[None]+3))%257-128).float().div(128).bfloat16()
    for name,t in [('input',x),('positions',positions),('slots',positions),('block_table',torch.zeros((1,1),dtype=torch.int32)),
                   ('output',output),('raw_cache',cache),('query',query),('key',key),('value',value),('attention_output',output)]:record(name,t)
    if dsa:
        current=positions//8
        for name,t in [('dsa_gate',torch.zeros((count,1),dtype=torch.bfloat16)),('plan_0_current',current),
                       ('plan_0_topk_lengths',current.int()),('plan_0_selected',torch.tensor([content.selected_regions(None,int(c),512,3) for c in current]))]:record(name,t)
    case=dict(case='dsa_short' if dsa else 'swa',initial=initial,spec=dict(block_size=16),frames=[dict(previous=0,count=count,total=count,graph='eager',
         finite=True,exact_key=True,exact_value=True,untouched=True,tensors=records)])
    config=dict(hidden_size=2,sliding_window=512,sparse_config=dict(region_block_size=8,topk=512))
    return case,config


def test_earlier_duplicate_record_is_verified(tmp_path,monkeypatch):
    case,config=fixture(tmp_path,monkeypatch)
    case['frames'][0]['tensors'].insert(0,dict(case['frames'][0]['tensors'][6],sha256='invalid'))
    with pytest.raises(AssertionError): content.analyze(case,0,tmp_path,tmp_path,config)


def test_unscored_selector_is_not_reference_truth(tmp_path,monkeypatch):
    case,config=fixture(tmp_path,monkeypatch,dsa=True)
    assert len(content.analyze(case,0,tmp_path,tmp_path,config)['selection_checks'])==10
    row=case['frames'][0]['tensors'][-1];value=torch.load(row['path'],weights_only=True);value[8]=torch.tensor([-1,-1,1])
    torch.save(value,row['path']);row['size_bytes']=Path(row['path']).stat().st_size;row['sha256']=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
    with pytest.raises(AssertionError):content.analyze(case,0,tmp_path,tmp_path,config)


def test_untouched_state_is_independently_checked(tmp_path,monkeypatch):
    case,config=fixture(tmp_path,monkeypatch)
    assert content.analyze(case,0,tmp_path,tmp_path,config)['raw_write']=='PASS'
    row=case['frames'][0]['tensors'][5];value=torch.load(row['path'],weights_only=True);value[0,0,15]=1
    torch.save(value,row['path']);row['size_bytes']=Path(row['path']).stat().st_size;row['sha256']=hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()
    with pytest.raises(AssertionError):content.analyze(case,0,tmp_path,tmp_path,config)


def test_relative_errors_preserve_zero_actual():
    report=content.errors(torch.tensor([1000,1,0]),torch.tensor([1000,2,1]))
    assert report['max_amplitude_normalized_error']==0.001
    assert report['maximum_element_relative_error']==1
    assert report['relative_error_undefined_elements']==1 and report['zero_actual_nonzero_error_elements']==1


@pytest.mark.parametrize('round_products',[False,True])
def test_rope_half_split_and_product_cast(round_products):
    value=torch.tensor([[[0.7890625,-1.4921875,7.0]]],dtype=torch.bfloat16)
    cache=torch.tensor([[0.70703125,0.70703125]],dtype=torch.bfloat16)
    a=value[0,0,0].double()*cache[0,0].double()
    b=-value[0,0,1].double()*cache[0,1].double()
    c=value[0,0,1].double()*cache[0,0].double()
    d=value[0,0,0].double()*cache[0,1].double()
    if round_products:a,b,c,d=[v.bfloat16().double() for v in (a,b,c,d)]
    expected=torch.tensor([[[float((a+b).bfloat16()),float((c+d).bfloat16()),7]]],dtype=torch.bfloat16)
    assert torch.equal(content.rope(value,cache,torch.tensor([0]),round_products),expected)


def test_rms_weight_plus_one_and_norm_cast():
    value=torch.tensor([[[3,4]]],dtype=torch.bfloat16)
    expected=torch.tensor([[[3/(12.5**0.5),8/(12.5**0.5)]]],dtype=torch.bfloat16)
    assert torch.equal(content.rms(value,torch.tensor([0,1]),0),expected)


def test_dsa_stage_group_padding_and_duplicate_gate(tmp_path):
    def record(name,value):
        path=tmp_path/(str(len(list(tmp_path.iterdir())))+'.pt');torch.save(value,path)
        return dict(name=name,path=str(path),size_bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    shape=list(value.shape),dtype=str(value.dtype),stride=list(value.stride()))
    heads,groups,dim=8,4,192
    widths=[heads*dim,groups*dim,groups*dim,4096,256,256,heads]
    packed=torch.cat([torch.ones(w) if i in (0,1,3) else torch.full((w,),0.25) if i in (2,5) else torch.zeros(w) for i,w in enumerate(widths)]).bfloat16()[:,None]
    weight_values={'qkv_indexer_proj.weight':packed,'q_norm.weight':torch.zeros(dim),'k_norm.weight':torch.zeros(dim),
        'rotary_emb.cos_sin_cache':torch.cat([torch.ones(32),torch.zeros(32)])[None].bfloat16(),
        'sparse_indexer_q_norm.weight':torch.zeros(256).bfloat16(),'sparse_indexer_k_norm.weight':torch.ones(256).bfloat16(),
        'sparse_indexer_k_norm.bias':torch.full((256,),0.5).bfloat16(),
        'sparse_indexer_rotary_emb.cos_sin_cache':torch.cat([torch.ones(16),torch.zeros(16)])[None].bfloat16(),
        'sparse_indexer_w.weight':torch.full((16,1),0.25).bfloat16(),'o_proj.weight':torch.cat([torch.ones(1),torch.zeros(heads*dim-1)])[None].bfloat16()}
    q=torch.ones((1,heads,dim)).bfloat16();k=torch.ones((1,groups,dim)).bfloat16()
    pq=torch.zeros((1,groups,8,256));pq[:,:,:4]=1
    pw=torch.zeros((1,groups,8));pw[:,:,:4]=0.125
    frame_values=[('input',torch.ones(1,1).bfloat16()),('positions',torch.tensor([0])),('query',q),('key',k),('value',torch.full_like(k,0.25)),
        ('dsa_gate',torch.zeros(1,heads).bfloat16()),('dsa_proxy_query',pq.to(torch.float8_e4m3fn)),('dsa_proxy_key',torch.full((1,1,256),0.5).bfloat16()),
        ('dsa_proxy_z',torch.full((1,1,256),0.25).bfloat16()),('dsa_proxy_weights',pw),('attention_output',torch.ones_like(q)),
        ('query',q.clone()),('attention_output',torch.full_like(q,0.5)),('output',torch.full((1,1),0.5).bfloat16())]
    case=dict(case='dsa_short',weights=[record(n,v) for n,v in weight_values.items()],frames=[dict(tensors=[record(n,v) for n,v in frame_values])])
    config=dict(rms_norm_eps=1e-5,sparse_config=dict(proxy_dim=256,sparse_indexer_num_heads=16,sparse_indexer_num_k_heads=1))
    report=content.stages(case,0,tmp_path,tmp_path,config)
    assert all(row['error']['max_absolute_error']==0 for row in report['comparisons'])
    assert len([row for row in report['comparisons'] if row['stage']=='query'])==2
    assert len([row for row in report['comparisons'] if row['stage']=='dsa_proxy_query'])==2
    assert report['frame_records']==14
    row=case['frames'][0]['tensors'][2];row['sha256']='invalid'
    with pytest.raises(AssertionError):content.stages(case,0,tmp_path,tmp_path,config)
