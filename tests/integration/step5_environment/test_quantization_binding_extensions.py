"""Check the recorded native extension names without importing GPU packages."""
import ast
from pathlib import Path


def test_loaded_extension_identity_coverage():
    path = Path(__file__).with_name('inspect_quantization_binding.py')
    tree = ast.parse(path.read_text())
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == 'name'
                and isinstance(node.iter, ast.Tuple))
    names = {node.value for node in loop.iter.elts}
    assert names == {'vllm._C_stable_libtorch', 'vllm.third_party.deep_gemm',
                     'vllm.third_party.deep_gemm._C', 'vllm.third_party.deep_gemm._C_extension'}
