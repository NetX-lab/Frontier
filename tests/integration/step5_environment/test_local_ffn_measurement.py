"""Check independent FFN shape selection without importing native GPU packages."""
import json
import ast
from types import SimpleNamespace
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import local_ffn_measurement as module

TASK = Path(__file__).resolve().parents[3]/'task_memory/task_2026-10-15_step5_preview_h800'


@pytest.mark.parametrize('case,queries', [('dense', [7, 8, 9, 3]),
    ('moe_fp8', [17]), ('moe_excluded', [17])])
def test_queries_and_seeds_come_from_preregistered_local_cases(case, queries):
    plan = json.loads((TASK/'local_scope_sampling_01.json').read_text())
    selected = module.samples(plan, case)
    assert [row['query'] for row in selected] == queries
    assert all(row in plan['cases'] and row['measurement_status'] == 'unmeasured' for row in selected)
    assert selected[-1]['partition'] == ('holdout' if case == 'dense' else 'sample')


def test_unsupported_owner_is_rejected():
    with pytest.raises(ValueError):
        module.samples({}, 'unqualified')


@pytest.mark.parametrize('warmups,repeats', [(0, 2), (1, 1)])
def test_invalid_schedule_rejected_before_native_import(tmp_path, warmups, repeats):
    with pytest.raises(ValueError):
        module.measure(TASK, tmp_path/'new', None, None, 'dense', warmups=warmups, repeats=repeats)


@pytest.mark.parametrize('tp,group_size', [(2, 1), (1, 2)])
def test_actual_topology_must_match_plan(tp, group_size):
    native = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=tp,
        data_parallel_size=1, pipeline_parallel_size=1, enable_expert_parallel=False))
    with pytest.raises(ValueError):
        module.validate_topology(native, [SimpleNamespace(world_size=group_size, rank=0)])


def test_both_dense_dispatches_are_recorded():
    def gemm():
        pass
    method = type('UnquantizedLinearMethod', (), {'_gemm_impl': staticmethod(gemm)})()
    weight = SimpleNamespace(dtype='torch.bfloat16', shape=(4, 8), stride=lambda: (8, 1))
    layer = SimpleNamespace(quant_method=method, weight=weight)
    result = module.linear_dispatch(SimpleNamespace(gate_up_proj=layer, down_proj=layer))
    assert set(result) == {'gate_up_proj', 'down_proj'}
    assert all(row['gemm'].endswith('gemm') for row in result.values())


def test_native_ffn_method_owns_combine():
    source = TASK/'native_source_export/observed/vllm/models/step5/nvidia/model.py'
    cls = next(node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == 'Step5DecoderLayer')
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_forward_ffn')
    namespace = {}
    exec('from __future__ import annotations\n'+ast.unparse(fn), namespace)
    class MoE:
        fuse_all_reduce = False
        def __call__(self, x, **kwargs):
            return 7, 11
    target = SimpleNamespace(use_moe=True, moe=MoE(), _add_for_residual=lambda a, b: a+b)
    assert namespace['_forward_ffn'](target, None) == 18
