"""Hypothetical routing policy counterexamples, not a native implementation."""

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

TASK = Path(__file__).resolve().parent
ROOT = Path('/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm')


def cyclic(positions):
    positions = torch.as_tensor(positions, dtype=torch.int64)
    return (positions[:,None]*8+torch.arange(8))%352


def histogram(ids):
    return torch.bincount(ids.flatten(),minlength=352)


def main():
    full=cyclic(range(6144))
    reset_split=torch.cat([cyclic(range(6128)),cyclic(range(16))])
    stable_split=torch.cat([cyclic(range(6128)),cyclic(range(6128,6144))])
    assert not torch.equal(full,reset_split)
    assert torch.equal(full,stable_split)
    assert full[6128].tolist()==list(range(96,104))
    assert reset_split[6128].tolist()==list(range(8))
    chunk_sizes=[2048,2048,2032,16]
    assert sum(chunk_sizes)==6144
    reset_chunked=torch.cat([cyclic(range(n)) for n in chunk_sizes])
    offset=0
    stable_chunks=[]
    for n in chunk_sizes:
        stable_chunks.append(cyclic(range(offset,offset+n)))
        offset+=n
    assert torch.equal(full,torch.cat(stable_chunks))
    assert not torch.equal(full,reset_chunked)
    q1_stable=cyclic(range(6144,6148))
    q1_reset=torch.cat([cyclic([0]) for _ in range(4)])
    assert q1_stable[:,0].tolist()==[224,232,240,248]
    assert q1_reset[:,0].tolist()==[0,0,0,0]

    # Existing MRv2 dummy InputBatch.make_dummy carries all-zero positions;
    # real graph padding positions may be stale and must be masked explicitly.
    positions=[6128,6129]
    pad_variants=[[],[0],[999,0,7]]
    for padding in pad_variants:
        assert torch.equal(cyclic(positions+padding)[:2],cyclic(positions))
    real=cyclic([6144])
    dummy=cyclic([0])
    aggregate=torch.cat([real,dummy])
    h=histogram(aggregate)
    assert torch.equal(aggregate[0],real[0])
    # Balanced two-token Hamilton should be experts0..15, one each.
    expected=histogram(cyclic([0,1]))
    assert not torch.equal(h,expected)
    repeated_q1=torch.cat([cyclic([6144]),cyclic([6144])])
    assert histogram(repeated_q1).max().item()==2
    assert not torch.equal(histogram(repeated_q1),expected)

    # Extract actual native make_dummy, intercept allocation with CPU tensors.
    source=ROOT/'v1/worker/gpu/input_batch.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='InputBatch')
    definitions=[n for n in cls.body if isinstance(n,(ast.AnnAssign,ast.FunctionDef))
        and (isinstance(n,ast.AnnAssign) or n.name=='make_dummy')]
    selected=ast.ClassDef(name='InputBatch',bases=[],keywords=[],body=definitions,
        decorator_list=[ast.Name(id='dataclass',ctx=ast.Load())])
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),selected],type_ignores=[])
    ns=dict(torch=torch,np=np,dataclass=dataclass,random_uuid=lambda:'cpu_probe')
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),ns)
    from types import SimpleNamespace as NS
    for tokens,requests in [(1,1),(4,1),(16,16)]:
        buffers=NS(device=torch.device('cpu'),input_ids=torch.zeros(tokens,dtype=torch.int64),
            positions=torch.ones(tokens,dtype=torch.int64),query_start_loc=torch.zeros(requests+1,dtype=torch.int32),
            seq_lens=torch.zeros(requests,dtype=torch.int32),is_padding=torch.zeros(tokens,dtype=torch.bool),
            idx_mapping=torch.zeros(requests,dtype=torch.int32))
        row=ns['InputBatch'].make_dummy(requests,tokens,buffers)
        assert row.positions.tolist()==[0]*tokens
        assert row.num_computed_tokens_np.tolist()==[0]*requests

    receipt=dict(task_id='D29-NATIVE-BALANCED',status='PASS_SEMANTIC_COUNTEREXAMPLES',
        interpreter=sys.executable,python=sys.version,torch=torch.__version__,
        full6144_vs_split=dict(tail_position=6128,tail_offset=96,
            full_first_tail_ids=full[6128].tolist(),reset_first_tail_ids=reset_split[6128].tolist(),
            reset_mismatched_token_rows=int((full!=reset_split).any(-1).sum()),
            stable_position_exact_ids=True),
        prefix_chunking=dict(sizes=chunk_sizes,reset_mismatched_token_rows=int((full!=reset_chunked).any(-1).sum()),stable_position_exact_ids=True),
        q1_continuation=dict(positions=list(range(6144,6148)),
            stable_first_experts=q1_stable[:,0].tolist(),reset_first_experts=q1_reset[:,0].tolist()),
        padding=dict(real_ids_invariant=True,
            actual_native_dummy_positions_zero=True,
            real_q1_6144_plus_dummy0_experts=torch.nonzero(h).flatten().tolist(),
            hamilton_two_token_expected_experts=list(range(16)),
            stable_position_aggregate_not_hamilton=True,
            two_same_position_requests_count_max=2,two_same_position_requests_not_exact_balanced=True),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        limits=['Math/CPU seam checks only; no installed routing strategy or native patch',
            'Stable expert IDs are necessary but do not prove GPU cross-shape numerical/state agreement',
            'No GPU readback, extra collective, platform operation or overhead measurement'])
    with (TASK/'semantic_receipt.json').open('x') as stream:json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
