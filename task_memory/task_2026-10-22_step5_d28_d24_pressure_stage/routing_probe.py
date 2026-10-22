"""Read-only CPU check of the frozen native strategy assignments, not GPU qualification."""

import ast
from abc import ABC, abstractmethod
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from typing import Any

import torch

SOURCE = Path('/data/ycfeng/tmp/step5_profiling_20261020/image_src/vllm')
TASK = Path(__file__).resolve().parent


def main():
    relative = 'model_executor/layers/fused_moe/router/routing_simulator_router.py'
    path = SOURCE / relative
    tree = ast.parse(path.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    namespace = dict(ABC=ABC, abstractmethod=abstractmethod, Any=Any, torch=torch,
        BaseRouter=object, logger=NS(warning_once=lambda *args: None))
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
        ast.alias(name='annotations')], level=0), *classes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    simulator = namespace['RoutingSimulator']
    assert simulator.get_available_strategies() == ['uniform_random', 'normal_routing']
    counts = []
    for tokens in (8, 44, 128, 6144):
        torch.manual_seed(0)
        weights, ids = simulator.simulate_routing(torch.zeros(tokens, 1),
            torch.zeros(tokens, 352), 'uniform_random', 8, torch.int32)
        per_expert = torch.bincount(ids.flatten().long(), minlength=352)
        assert weights.eq(1).all() and ids.shape == (tokens, 8)
        counts.append(dict(tokens=tokens, assignments=tokens * 8,
            min_count=int(per_expert.min()), max_count=int(per_expert.max()),
            counts=per_expert.tolist(), exactly_balanced=bool(per_expert.min() == per_expert.max())))
    assert not any(row['exactly_balanced'] for row in counts)
    with __import__('pytest').raises(ValueError, match='Unknown routing strategy'):
        simulator.simulate_routing(torch.zeros(44, 1), torch.zeros(44, 352), 'balanced', 8)
    files = (relative, 'model_executor/layers/fused_moe/router/router_factory.py',
        'model_executor/layers/fused_moe/layer.py', 'models/step5/nvidia/model.py',
        'models/step5/nvidia/mega_moe.py', 'envs.py')
    receipt = dict(task_id='D28-D24-PRESSURE-STAGE', status='BLOCKED_NATIVE_BALANCED_UNSUPPORTED',
        scope='read_only_native_CPU_strategy_method_body', interpreter=sys.executable,
        python=sys.version, torch=torch.__version__, source_root=str(SOURCE),
        source_sha256={name: hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in files},
        accepted_strategy_names=simulator.get_available_strategies(), expert_count=352, top_k=8,
        assignments=counts, native_setting_applied=False, source_modified=False,
        limits=['CPU executed frozen native strategy class bodies; not a loaded full engine or GPU router',
            'Factory route selected before biased sigmoid router; random weights become all ones, not normalized native sigmoid scale3',
            'Step5 normal factory and mega branch both construct factory routers; GPU/monolithic dispatch still requires live verification',
            'No built-in balanced assignment; custom registration is a new implementation needing parent design/approval'],
        network_platform_auth_cloud_gpu_operations=0)
    with (TASK/'native_routing_receipt.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({key: receipt[key] for key in ('status', 'accepted_strategy_names', 'expert_count', 'top_k')}))
    print(json.dumps([{key: row[key] for key in ('tokens', 'assignments', 'min_count', 'max_count', 'exactly_balanced')} for row in counts]))


if __name__ == '__main__':
    main()
