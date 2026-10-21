#!/usr/bin/env python3
"""Analyze retained native attention state with independent CPU mathematical references."""
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import torch


def tensor(record, cloud, original):
    path = (cloud/Path(record['path']).relative_to(original)).resolve()
    assert path.is_relative_to(cloud.resolve()) and path.stat().st_size == record['size_bytes']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(piece)
    assert digest.hexdigest() == record['sha256']
    value = torch.load(path, map_location='cpu', weights_only=True)
    assert list(value.shape) == record['shape'] and str(value.dtype) == record['dtype']
    return value


def errors(actual, expected):
    actual, expected = actual.double(), expected.double()
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    delta = (actual[finite]-expected[finite]).abs()
    maximum = actual[finite].abs().max().item() if finite.any() else 0
    nonzero = finite & (actual != 0)
    relative = (actual[nonzero]-expected[nonzero]).abs()/actual[nonzero].abs()
    return dict(elements=actual.numel(), finite_elements=int(finite.sum()),
        nonfinite_match=bool(torch.equal(actual[~finite], expected[~finite])),
        max_absolute_error=delta.max().item() if delta.numel() else 0,
        mean_absolute_error=delta.mean().item() if delta.numel() else 0,
        max_amplitude_normalized_error=None if maximum == 0 else delta.max().item()/maximum,
        maximum_element_relative_error=relative.max().item() if relative.numel() else None,
        mean_element_relative_error=relative.mean().item() if relative.numel() else None,
        relative_error_undefined_elements=int((finite & (actual == 0)).sum()),
        zero_actual_nonzero_error_elements=int((finite & (actual == 0) & (expected != 0)).sum()),
        unequal_elements=int((actual != expected).sum()))


def e4m3(value):
    codes = torch.arange(127, dtype=torch.int64)
    exponents, mantissas = codes//8, codes%8
    values = torch.where(exponents == 0, mantissas.double()*2**-9,
        (1+mantissas.double()/8)*torch.exp2(exponents.double()-7))
    magnitude = value.double().abs().clamp(max=values[-1])
    upper = torch.searchsorted(values, magnitude).clamp(max=126)
    lower = (upper-1).clamp(min=0)
    choose = ((values[upper]-magnitude) < (magnitude-values[lower])) | (
        ((values[upper]-magnitude) == (magnitude-values[lower])) & (upper%2 == 0))
    return values[torch.where(choose, upper, lower)]*value.sign().double()


def summary(case, index, cloud, original, config):
    initial = {r['name']: tensor(r, cloud, original) for r in case['initial']}
    mean = initial['mean'].view(torch.float8_e4m3fn).double()
    numerator, denominator, maximum = (initial[n].double() for n in ('numerator','denominator','maximum'))
    last = initial['last_slot'].clone()
    region, block = config['sparse_config']['region_block_size'], case['spec']['block_size']
    for frame in case['frames'][:index+1]:
        records = {r['name']: r for r in frame['tensors']}
        key, z, slots = (tensor(records[n], cloud, original) for n in ('dsa_proxy_key','dsa_proxy_z','slots'))
        for row, slot in enumerate(slots.tolist()):
            page, offset = divmod(slot, block)
            if offset%region == 0:
                numerator[page].zero_(); denominator[page].zero_(); maximum[page].fill_(-float('inf'))
            else:
                assert last[page] == slot-1
            logits, value = z[row].reshape(-1).double(), key[row].reshape(-1).double()
            updated = torch.maximum(maximum[page], logits.max())
            scale = torch.where(denominator[page] > 0, torch.exp(maximum[page]-updated), 0)
            weight = torch.exp(logits-updated)
            numerator[page] = numerator[page]*scale+weight*value
            denominator[page] = denominator[page]*scale+weight
            maximum[page] = updated
            mean[page, offset//region] = e4m3((numerator[page]/denominator[page].clamp(min=1e-20)).bfloat16())
            last[page] = -1 if offset%region == region-1 else slot
    records = {r['name']: r for r in case['frames'][index]['tensors']}
    observed = {n: tensor(records[n], cloud, original) for n in ('mean','numerator','denominator','maximum','last_slot')}
    assert torch.equal(observed['last_slot'], last)
    return {n: errors(observed[n].view(torch.float8_e4m3fn) if n == 'mean' else observed[n], expected)
        for n, expected in [('mean',mean),('numerator',numerator),('denominator',denominator),('maximum',maximum)]}


def integrity(case, cloud, original):
    count, size, started = 0, 0, time.perf_counter()
    for record in case['pre_weights']+case['weights']+case['initial']:
        value = tensor(record,cloud,original)
        count += 1; size += record['size_bytes']
        del value
    return dict(case=case['case'],verified_records=count,verified_bytes=size,status='PASS',
                wall_seconds=time.perf_counter()-started,host_peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def scores_from_summaries(values, captures, positions, table, block, region, groups):
    means = values['mean'].view(torch.float8_e4m3fn).double()
    query, weights = values['dsa_proxy_query'].double(), values['dsa_proxy_weights'].double()
    report = []
    for capture_index, scores in enumerate(captures):
        for row in range(scores.shape[0]):
            token, group = (0,row) if positions.numel() == 1 else (row,capture_index)
            live = int(positions[token])//region+1
            ids = torch.arange(live)
            physical = table[0,(ids*region)//block].long()
            key = means[physical,ids%(block//region)]
            expected = ((query[token,group]@key.T).clamp(min=0)*weights[token,group,:,None]).sum(dim=0)
            report.append(dict(capture=capture_index,row=row,live_regions=live,
                               error=errors(scores[row,:live],expected)))
    return report


def selected_regions(scores, current, topk, width):
    history = list(range(current))
    if scores is not None and current > topk:
        history.sort(key=lambda i: (-float(scores[i]), i))
        history = sorted(history[:topk])
    return [-1]*(width-len(history)-1)+history+[current]


def analyze(case, index, cloud, original, config):
    started = time.perf_counter(); torch.set_num_threads(4)
    frame = case['frames'][index]
    # Recursive prefill snapshots precede the outer gated result; verify every record.
    snapshots = [(r, tensor(r,cloud,original)) for r in frame['tensors']]
    values = {r['name']: value for r,value in snapshots}
    x, positions, slots, table, cache, query = (values[n] for n in ('input','positions','slots','block_table','raw_cache','query'))
    block, region = case['spec']['block_size'], config['sparse_config']['region_block_size']
    assert torch.equal(positions, torch.arange(frame['previous'],frame['total']))
    expected_slots = table[0, positions//block].long()*block+positions%block
    assert torch.equal(slots, expected_slots)
    expected_input = (((positions[:,None]+1)*(torch.arange(config['hidden_size'])[None]+3))%257-128).float().div(128).bfloat16()
    assert torch.equal(x, expected_input)
    keys, vals = cache.transpose(1,2).split(query.shape[-1], dim=-1)
    assert torch.equal(keys[slots//block,slots%block],values['key'].reshape(keys[slots//block,slots%block].shape))
    assert torch.equal(vals[slots//block,slots%block],values['value'].reshape(vals[slots//block,slots%block].shape))
    prior_records = case['frames'][index-1]['tensors'] if index else case['initial']
    prior = tensor(next(r for r in prior_records if r['name'] == 'raw_cache'),cloud,original)
    written = torch.zeros(cache.shape[0:3:2],dtype=torch.bool)
    written[slots//block,slots%block] = True
    assert torch.equal(cache.transpose(1,2)[~written],prior.transpose(1,2)[~written])
    assert frame['finite'] and frame['exact_key'] and frame['exact_value'] and frame['untouched']
    assert frame['previous'] == (case['frames'][index-1]['total'] if index else 0)
    heads, groups, dim = query.shape[1], cache.shape[1], query.shape[-1]
    selected_captures = [v for r,v in snapshots if r['name'] == 'selected']
    score_captures = [v for r,v in snapshots if r['name'] == 'scores']
    current_captures = [v for r,v in snapshots if r['name'] == 'current']
    lengths_captures = [v for r,v in snapshots if r['name'] == 'topk_lengths']
    expected_current = positions//region
    for current in current_captures+lengths_captures:
        assert torch.equal(current.long(),expected_current.repeat_interleave(groups) if frame['count'] == 1 else expected_current)
    selected = values.get('selected')
    selection_checks = []
    if selected is not None:
        current = values['current']; topk = config['sparse_config']['topk']
        for capture_index, capture in enumerate(selected_captures):
            for row in range(capture.shape[0]):
                score = score_captures[capture_index][row] if score_captures else None
                expected = selected_regions(score, int(current[row]), topk, capture.shape[1])
                assert capture[row].tolist() == expected
                selection_checks.append(dict(capture=capture_index,row=row,current=int(current[row]),valid=sum(i>=0 for i in expected)))
    plan = values.get('plan_0_selected')
    if values.get('plan_0_current') is not None:
        assert torch.equal(values['plan_0_current'],expected_current)
        assert torch.equal(values['plan_0_topk_lengths'].long(),expected_current)
    if plan is not None:
        for row,current in enumerate(expected_current.tolist()):
            assert current <= config['sparse_config']['topk']
            expected_ids = selected_regions(None,current,config['sparse_config']['topk'],plan.shape[1])
            assert plan[row].tolist() == expected_ids
            selection_checks.append(dict(kind='unscored_prefill',row=row,current=current,valid=current+1))
    expected = torch.empty_like(query, dtype=torch.float64)
    for row, position in enumerate(positions.tolist()):
        for group in range(groups):
            q = query[row,group*heads//groups:(group+1)*heads//groups].double()
            if case['case'] == 'swa':
                tokens = torch.arange(max(0,position-config['sliding_window']+1),position+1)
            elif selected is not None and frame['count'] == 1:
                ids = selected[group]; ids = ids[ids>=0]
                tokens = (ids[:,None]*region+torch.arange(region)[None]).flatten()
                tokens = tokens[tokens<=position]
            elif selected_captures:
                assert len(selected_captures) == groups
                ids = selected_captures[group][row]; ids = ids[ids>=0]
                tokens = (ids[:,None]*region+torch.arange(region)[None]).flatten()
                tokens = tokens[tokens<=position]
            else:
                assert plan is not None, 'Prefill requires row/group selection records'
                ids = plan[row]; ids = ids[ids>=0]
                tokens = (ids[:,None]*region+torch.arange(region)[None]).flatten()
                tokens = tokens[tokens<=position]
            physical = table[0,tokens//block].long()
            k, v = keys[physical,tokens%block,group].double(), vals[physical,tokens%block,group].double()
            weights = torch.softmax(q@k.T/(dim**0.5),dim=-1)
            expected[row,group*heads//groups:(group+1)*heads//groups] = weights@v
    if case['case'] != 'swa':
        gate = values['dsa_gate']
        sigmoid = torch.sigmoid(gate.double()).to(gate.dtype).double()
        expected = (expected.bfloat16().double()*sigmoid.unsqueeze(-1)).bfloat16().double()
    report = dict(case=case['case'], step=index, total=frame['total'], graph=frame['graph'],
        raw_write='PASS', input_and_slots='PASS', selection_checks=selection_checks,
        attention_given_native_preparation=errors(values['attention_output'],expected),
        numerical_guarantee='UNVERIFIED: mathematical reference is not a compiled-kernel error bound',
        tensor_integrity='PASS', verified_records=len(snapshots),
        native_and_serialized_strides=[dict(name=r['name'],path=r['path'],native=r['stride'],serialized=list(value.stride())) for r,value in snapshots])
    if case['case'] != 'swa':
        report['summary_given_native_proxy'] = summary(case,index,cloud,original,config)
        if score_captures:
            report['scores_given_native_proxy_and_summaries'] = scores_from_summaries(values,score_captures,positions,table,block,region,groups)
    report.update(wall_seconds=time.perf_counter()-started,host_peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        python=platform.python_version(),torch=torch.__version__)
    return report


def rms(value, weight, eps):
    value = value.double()
    return (value*torch.rsqrt(value.square().mean(dim=-1,keepdim=True)+eps)*(weight.double()+1)).bfloat16()


def rope(value, cache, positions, round_products=False):
    rotary = cache.shape[-1]; half = rotary//2
    phase = cache[positions.long()].double().unsqueeze(1)
    x = value.double(); low, high = x[...,:half], x[...,half:rotary]
    first = torch.cat((low,high),dim=-1)*phase[...,:half].repeat(1,1,2)
    second = torch.cat((-high,low),dim=-1)*phase[...,half:].repeat(1,1,2)
    if round_products:
        first, second = first.bfloat16().double(), second.bfloat16().double()
    return torch.cat(((first+second).bfloat16(),value[...,rotary:]),dim=-1)


def stages(case, index, cloud, original, config):
    started = time.perf_counter(); torch.set_num_threads(4)
    snapshots = [(r,tensor(r,cloud,original)) for r in case['frames'][index]['tensors']]
    values = {r['name']: value for r,value in snapshots}
    weights = {r['name']: tensor(r,cloud,original) for r in case['weights']}
    x, positions = values['input'], values['positions']
    dsa = case['case'] != 'swa'
    packed_name = 'qkv_indexer_proj.weight' if dsa else 'qkvg_proj.weight'
    projection = (x.double()@weights[packed_name].double().T).bfloat16()
    heads, dim, groups = values['query'].shape[1], values['query'].shape[2], values['key'].shape[1]
    sparse = config['sparse_config']
    proxy_dim = sparse['proxy_dim']
    index_heads = sparse['sparse_indexer_num_heads']
    key_heads = sparse['sparse_indexer_num_k_heads']
    widths = [heads*dim,groups*dim,groups*dim]+(
        [index_heads*proxy_dim,key_heads*proxy_dim,key_heads*proxy_dim,heads] if dsa else [heads])
    pieces = projection.split(widths,dim=-1)
    q, k, v = [piece.reshape(x.shape[0],-1,dim) for piece in pieces[:3]]
    eps = config['rms_norm_eps']
    references = dict(query=rope(rms(q,weights['q_norm.weight'],eps),weights['rotary_emb.cos_sin_cache'],positions),
        key=rope(rms(k,weights['k_norm.weight'],eps),weights['rotary_emb.cos_sin_cache'],positions),value=v)
    reports = []
    def compare(name, expected, arithmetic):
        for ordinal,(record,actual) in enumerate(snapshots):
            if record['name'] == name:
                assert actual.shape == expected.shape
                reports.append(dict(stage=name,record_ordinal=ordinal,actual_sha256=record['sha256'],
                    reference_arithmetic=arithmetic,error=errors(actual,expected)))
    for name,expected in references.items():
        compare(name,expected,'FP64 projection -> BF16 -> FP64 RMS(weight+1) -> BF16 -> retained BF16 RoPE cache -> BF16; V projection only')
    gate = pieces[-1]
    if dsa:
        compare('dsa_gate',gate,'FP64 projection -> BF16')
        iq, ik, iz = pieces[3:6]
        real_heads = index_heads//groups
        assert index_heads%groups == 0 and key_heads == 1
        padded_heads = max(real_heads,8)
        iq = rms(iq.reshape(x.shape[0],index_heads,proxy_dim),weights['sparse_indexer_q_norm.weight'],eps)
        ik = ik.reshape(x.shape[0],key_heads,proxy_dim).double()
        centered = ik-ik.mean(dim=-1,keepdim=True)
        ik = (centered*torch.rsqrt(centered.square().mean(dim=-1,keepdim=True)+eps)*
              weights['sparse_indexer_k_norm.weight'].double()+weights['sparse_indexer_k_norm.bias'].double()).bfloat16()
        iw = (x.double()@weights['sparse_indexer_w.weight'].double().T).bfloat16().double()
        padded_w = torch.zeros((x.shape[0],groups,padded_heads),dtype=torch.float64)
        padded_w[:,:,:real_heads] = iw.reshape(x.shape[0],groups,real_heads)/real_heads**0.5
        compare('dsa_proxy_weights',padded_w,'FP64 indexer weight projection -> BF16 -> FP32 * 1/sqrt(real heads per provider); zero padding heads')
        compare('dsa_proxy_z',iz.reshape(x.shape[0],key_heads,proxy_dim),'FP64 projection -> BF16; no norm/RoPE')
        for native_rope in (False,True):
            cache = weights['sparse_indexer_rotary_emb.cos_sin_cache']
            rq, rk = rope(iq,cache,positions,native_rope), rope(ik,cache,positions,native_rope)
            assert rq.abs().max() <= 448, 'This E4M3 reference requires representable finite proxy inputs'
            padded_q = torch.zeros((x.shape[0],groups,padded_heads,proxy_dim),dtype=torch.float64)
            padded_q[:,:,:real_heads] = e4m3(rq).reshape(x.shape[0],groups,real_heads,proxy_dim)
            label = f'Conditional indexer native_rope={native_rope}; FP64 projection/norm -> BF16, retained cache, BF16 -> E4M3; not historical mode inference'
            compare('dsa_proxy_query',padded_q,label); compare('dsa_proxy_key',rk,label)
    attention = [(ordinal,value) for ordinal,(record,value) in enumerate(snapshots) if record['name']=='attention_output']
    if dsa:
        output_input = attention[-1][1]
        if len(attention) > 1:
            sigmoid = torch.sigmoid(values['dsa_gate'].double()).bfloat16().double()
            gated = (attention[0][1].double()*sigmoid.unsqueeze(-1)).bfloat16()
            reports.append(dict(stage='gate_given_native_ungated_attention_and_gate',record_ordinal=attention[-1][0],
                reference_arithmetic='FP64 sigmoid -> BF16 -> multiply native BF16 attention -> BF16',error=errors(output_input,gated)))
        gate_disposition = 'Prefill first ungated/last gated; decode has only gated output, no separate gate-stage target'
    else:
        sigmoid = torch.sigmoid(gate.double()).bfloat16().double()
        output_input = (attention[-1][1].double()*sigmoid.unsqueeze(-1)).bfloat16()
        gate_disposition = 'Reconstructed projection gate conditioned on native ungated attention; native gate/gated seam not retained'
    output = (output_input.reshape(x.shape[0],-1).double()@weights['o_proj.weight'].double().T).bfloat16()
    compare('output',output,'FP64 output projection -> BF16; '+gate_disposition)
    return dict(case=case['case'],step=index,positions=positions.tolist(),tensor_integrity='PASS',
        frame_records=len(snapshots),weight_records=len(weights),comparisons=reports,
        reference_inputs=[dict(ordinal=i,name=r['name'],sha256=r['sha256']) for i,(r,_) in enumerate(snapshots)],
        weight_inputs=[dict(name=r['name'],sha256=r['sha256']) for r in case['weights']],
        attribution='Combined projection->preparation only: native raw packed projection is not retained',
        source_citations=['model.py:550-684,1132-1144,1408-1424,1632-1677','ops/qkv_indexer_norm_rope.py:51-90',
                          'ops/indexer_norm_rope.py:119-188','ops/attention_gate.py:30-53'],
        numerical_guarantee='UNVERIFIED: FP64 mathematical reference is not exact transcendental arithmetic or a compiled-kernel error bound',
        wall_seconds=time.perf_counter()-started,host_peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        python=platform.python_version(),torch=torch.__version__)


if __name__ == '__main__':
    case,index,cloud,original,config = json.loads(sys.stdin.read())
    if isinstance(index,str) and index.startswith('stages_'):
        result = stages(case,int(index.split('_')[1]),Path(cloud),Path(original),config)
    else:
        result = integrity(case,Path(cloud),Path(original)) if index == 'integrity' else analyze(case,index,Path(cloud),Path(original),config)
    print(json.dumps(result),flush=True)
