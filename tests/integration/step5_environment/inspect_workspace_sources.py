#!/usr/bin/env python3
"""Read the imported MoE workspace owner and modular allocation entry points."""
import ast
import importlib.metadata
import importlib.util
from pathlib import Path
import sys

SOURCES = {
    'vllm/model_executor/layers/fused_moe/modular_kernel.py': 'Actual modular expert workspace allocation calls.',
}

if __name__ == '__main__':
    path = Path(__file__).with_name('inspect_runtime_sources.py')
    spec = importlib.util.spec_from_file_location('runtime_sources', path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    if sys.argv[1] == 'run':
        package = importlib.metadata.distribution('vllm')
        owner = Path(package.locate_file(next(iter(SOURCES))))
        imports = [node.module for node in ast.walk(ast.parse(owner.read_bytes()))
                   if isinstance(node, ast.ImportFrom) and node.module
                   and 'workspace' in node.module]
        assert imports, 'No workspace import in the installed modular kernel owner'
        for module in imports:
            SOURCES[module.replace('.', '/') + '.py'] = 'Workspace owner imported by the installed modular kernel.'
    collector.SOURCES = SOURCES
    collector.__file__ = __file__
    collector.main()
