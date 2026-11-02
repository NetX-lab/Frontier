"""CPU AST check of the pinned native config decorator and profiler validators."""
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import typing
from pydantic import ConfigDict, Field, model_validator
from pydantic.dataclasses import dataclass
from typing_extensions import dataclass_transform, Self

root = Path(__file__).resolve().parent
native = Path('/data/ycfeng/Frontier/.real-engine/vLLM-BS/.worktrees/glm-h5-clean-2cf0a69')
package = Path('/data/ycfeng/Frontier/.worktrees/v4-native-profile04-run/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_04')
engine = json.loads((package / 'engine.json').read_text())
namespace = dict(vars(typing))
namespace.update(os=os, ConfigDict=ConfigDict, dataclass=dataclass, PydanticField=Field,
                 dataclass_transform=dataclass_transform, Field=Field,
                 model_validator=model_validator, Self=Self,
                 ConfigT=typing.TypeVar('ConfigT'),
                 logger=SimpleNamespace(warning=lambda *a, **k: None),
                 safe_hash=hashlib.sha256, __name__='cpu_native_config_verification')
tree = ast.parse((native / 'vllm/config/utils.py').read_text())
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
         and node.name == 'config' and len(node.body) > 1]
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'native_config_decorator', 'exec'), namespace)
tree = ast.parse((native / 'vllm/config/profiler.py').read_text())
nodes = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
         and not (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                  and target.id == 'logger' for target in node.targets))]
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'native_profiler_config', 'exec'), namespace)
config = namespace['ProfilerConfig'](**engine['profiler_config'])
assert config.max_iterations == 64 and config.ignore_frontend and config.torch_profiler_use_gzip
assert config.torch_profiler_dump_cuda_time_total is False
assert namespace['ProfilerConfig']().torch_profiler_dump_cuda_time_total is True
assert (config.delay_iterations, config.warmup_iterations, config.wait_iterations) == (0, 0, 0)
assert not any((config.torch_profiler_with_stack, config.torch_profiler_record_shapes,
                config.torch_profiler_with_flops, config.torch_profiler_with_memory))
role_configs = {}
for role, count in {'prefill': 512, 'decode': 64}.items():
    validated = namespace['ProfilerConfig'](**(engine['profiler_config'] | {'max_iterations': count}))
    assert validated.max_iterations == count and validated.torch_profiler_dump_cuda_time_total is False
    role_configs[role] = count
rejected = 0
for bad in (dict(max_iterations=-1), dict(unknown_config_key=1), dict(profiler='invalid')):
    try:
        namespace['ProfilerConfig'](**(engine['profiler_config'] | bad))
    except Exception:
        rejected += 1
assert rejected == 3
receipt = dict(status='PASS_ACTUAL_NATIVE_DECORATOR_CLASS_VALIDATORS_AST',
               profiler_config=engine['profiler_config'], negative_controls=rejected, role_configs=role_configs,
               non_native_standins=['logger', 'safe_hash'], python=sys.version)
(root / 'native_profiler.receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
