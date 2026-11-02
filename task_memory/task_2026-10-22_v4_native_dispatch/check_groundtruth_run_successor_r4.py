#!/usr/bin/env python3
"""Hash-bound R3 successor; replace only the incomplete routing-owner census."""
from __future__ import annotations

import hashlib
from pathlib import Path

R3_PATH = Path(__file__).with_name('check_groundtruth_run_successor_r3.py')
R3_SHA256 = 'ca8ddcea83c852c27295c8a73453746d755c51055e703bd94df48e8f1af7d917'
RUNNER_CLASS = 'vllm.model_executor.layers.fused_moe.runner.moe_runner.MoERunner'
CHILD_CLASS = 'vllm.model_executor.layers.fused_moe.routed_experts.RoutedExperts'
CHILD_SUFFIX = '.routed_experts'
BACKEND_RECORDS = (
    dict(quant_method='Mxfp4MoEMethod', backend='MARLIN', experts='MarlinExperts',
         prepare_finalize='MoEPrepareAndFinalizeNaiveDPEPModular', input_dtype='None'),
    dict(quant_method='Fp8MoEMethod', backend='TRITON', experts='TritonExperts',
         prepare_finalize='MoEPrepareAndFinalizeNaiveDPEPModular', block_shape=[128, 128]),
)


def independent_routing_owners(inventory):
    initialization = inventory['initialization']
    modules, backends = initialization['modules'], initialization['backends']
    if type(modules) is not dict or type(backends) is not dict:
        raise ValueError('Independent routing module/backend inventories must be maps')
    owners = {name for name, cls in modules.items() if cls == RUNNER_CLASS}
    children = {name for name, cls in modules.items()
                if cls == CHILD_CLASS or name.endswith(CHILD_SUFFIX)}
    consumers = {name for name, row in backends.items()
                 if name.endswith(CHILD_SUFFIX) or name in children
                 or 'experts' in row or 'prepare_finalize' in row
                 or row.get('quant_method') in {r['quant_method'] for r in BACKEND_RECORDS}}
    expected = {name + CHILD_SUFFIX for name in owners}
    if (not owners or children != expected or consumers != expected
            or any(modules[name] != CHILD_CLASS for name in children)
            or any(backends[name] not in BACKEND_RECORDS for name in consumers)):
        raise ValueError('Independent routing owner/child/backend completeness or consumer mismatch')
    return owners


def routing_successor_source(path=R3_PATH):
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != R3_SHA256:
        raise ValueError('Frozen R3 dependency hash mismatch')
    text = content.decode()
    old = "    routers = {name.removesuffix('.routed_experts') for name in inventory['moe_backends']}\n"
    if text.count(old) != 1:
        raise ValueError('Frozen R3 routing-owner replacement boundary is not unique')
    return text.replace(old, '    routers = independent_routing_owners(inventory)\n')


exec(compile(routing_successor_source(), __file__ + ':frozen_r3_with_independent_owners', 'exec'), globals())
