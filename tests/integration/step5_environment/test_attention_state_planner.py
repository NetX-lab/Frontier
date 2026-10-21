"""Check the real planner factory expression under the harness device scope."""
import ast
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).with_name('check_attention_state.py')
OWNER = SCRIPT.parents[3]/'task_memory/task_2026-10-15_step5_preview_h800/native_source_export/observed/vllm/models/step5/nvidia/ops/sparse_prefill_pages.py'


def test_native_metadata_planner_uses_cpu_factory_scope(monkeypatch):
    native = ast.parse(OWNER.read_text())
    expression = next(node for node in ast.walk(native) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == 'pin_memory'
        and isinstance(node.func.value, ast.Call) and isinstance(node.func.value.func, ast.Attribute)
        and node.func.value.func.attr == 'arange')
    def pin(tensor):
        assert tensor.device.type == 'cpu', 'Native pinned planner tensors must be CPU tensors'
        return tensor
    monkeypatch.setattr(torch.Tensor, 'pin_memory', pin)
    class Builder:
        def build(self, *args, **kwargs):
            return eval(compile(ast.Expression(expression), str(OWNER), 'eval'), {'torch': torch, 'rows': 2})
    tree = ast.parse(SCRIPT.read_text())
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == 'builder'
        and node.value.func.attr == 'build')
    enclosing = next((node for node in ast.walk(tree) if isinstance(node, ast.With)
        and call in node.body), None)
    statement = enclosing if enclosing is not None else call
    scope = {'torch': torch, 'builder': Builder(), 'common': object()}
    with torch.device('meta'):
        exec(compile(ast.Module(body=[statement], type_ignores=[]), str(SCRIPT), 'exec'), scope)
    assert scope['metadata'].device.type == 'cpu'
    assert scope['metadata'].tolist() == [0, 1, 2]


def test_capture_keeps_external_metadata_buffers_alive():
    import gc
    from types import SimpleNamespace
    import weakref

    tree = ast.parse(SCRIPT.read_text())
    branch = next(node for node in ast.walk(tree) if isinstance(node, ast.If)
        and ast.unparse(node.test) == 'step == len(chunks)')
    bindings = [node for node in branch.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Tuple) or isinstance(target, ast.Name)
                and target.id.startswith('graph_') for target in node.targets)]
    starts, lengths = torch.tensor([0, 1]), torch.tensor([515])
    common = SimpleNamespace(query_start_loc=starts, seq_lens=lengths)
    metadata = SimpleNamespace(query_start_loc=starts, seq_lens=lengths,
                               scheduler_metadata=torch.tensor([17]))
    references = [weakref.ref(value) for value in
                  (starts, lengths, metadata.scheduler_metadata)]
    scope = dict(common=common, metadata=metadata, positions=torch.tensor([514]),
                 x=torch.zeros(1, 2), slots=torch.tensor([29]))
    exec(compile(ast.Module(body=bindings, type_ignores=[]), str(SCRIPT), 'exec'), scope)
    # Replay replaces the per-iteration objects; the captured graph stores raw pointers.
    scope.pop('common'); scope.pop('metadata')
    del starts, lengths, common, metadata
    gc.collect()
    assert all(reference() is not None for reference in references), \
        'Captured external metadata must remain owned across changed-input replays'
