"""Test doubles for executing pinned native method bodies without a vLLM wheel.

This fallback is evidence about the harness only; it never fulfills the required
installed-vLLM check. Source method bodies are compiled without edits. Models,
GPU execution, constructors and process-group setup are deliberately absent.
"""
import ast
import dataclasses
import enum
import queue
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np


def module(name):
    if name in sys.modules:
        return sys.modules[name]
    value = ModuleType(name)
    value.__path__ = []
    sys.modules[name] = value
    if '.' in name:
        parent, child = name.rsplit('.', 1)
        setattr(module(parent), child, value)
    return value


def methods(path, class_name, names, namespace):
    source = ast.parse(path.read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name==class_name)
    selected = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    tree = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def install(checkout):
    core_path = checkout / 'vllm/v1/engine/core.py'
    vllm = module('vllm')
    vllm.__version__ = 'SOURCE_ONLY_NO_INSTALLED_VLLM'
    vllm.__file__ = str(checkout / 'vllm/__init__.py')
    module('vllm.config').ParallelConfig = type('ParallelConfig', (), {'sync_dp_state':staticmethod(lambda *a,**k: (False,False))})
    request_type = enum.Enum('EngineCoreRequestType', 'ADD ABORT UTILITY START_DP_WAVE WAKEUP EXECUTOR_FAILED')
    module('vllm.v1.engine').EngineCoreRequestType = request_type
    pause = enum.Enum('PauseState', 'UNPAUSED')
    module('vllm.v1.core.sched.scheduler').PauseState = pause
    module('vllm.v1.core.sched.async_scheduler').AsyncScheduler = type('AsyncScheduler', (), {})
    module('vllm.v1.request').RequestStatus = enum.Enum('RequestStatus', 'WAITING_FOR_REMOTE_KVS')
    shutdown = enum.Enum('EngineShutdownState', 'RUNNING REQUESTED SHUTTING_DOWN')
    @dataclasses.dataclass
    class Outputs:
        wave_complete: int = None
        start_wave: int = None
    logger = SimpleNamespace(debug=lambda *a,**k:None, warning=lambda *a,**k:None,
        isEnabledFor=lambda _:False)
    namespace = dict(queue=queue, logger=logger, DEBUG=10, PauseState=pause,
        EngineCoreRequestType=request_type, EngineCoreOutputs=Outputs,
        EngineShutdownState=shutdown, ParallelConfig=module('vllm.config').ParallelConfig,
        fault_tolerant_wrapper=lambda fn:fn, Request=object, Any=object,
        POOLING_TASKS=(), time=__import__('time'))
    base = type('EngineCore', (), methods(core_path, 'EngineCore', ['add_request'], namespace))
    namespace['EngineCore'] = base
    proc_names = ['has_work','is_running','_process_input_queue','_process_engine_step',
        '_handle_client_request','_reject_add_in_shutdown']
    namespace['__name__'] = 'vllm.v1.engine.core'
    # zero-argument super requires a __class__ closure: compile original class
    # nodes with only the driven methods and their original base relationship.
    tree = ast.parse(core_path.read_text())
    proc_node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EngineCoreProc')
    proc_node.body = [n for n in proc_node.body if isinstance(n,ast.FunctionDef) and n.name in proc_names]
    dp_node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DPEngineCoreProc')
    dp_node.body = [n for n in dp_node.body if isinstance(n,ast.FunctionDef) and n.name in
        ['add_request','_handle_client_request','_maybe_publish_request_counts','run_busy_loop','_has_global_unfinished_reqs']]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[proc_node,dp_node],type_ignores=[])),str(core_path),'exec'),namespace)
    core = module('vllm.v1.engine.core')
    core.DPEngineCoreProc = namespace['DPEngineCoreProc']
    core.EngineShutdownState = shutdown
    core.__file__ = str(core_path)
    class Tensor(np.ndarray):
        def copy_(self, other):
            np.copyto(self, other)
            return self
    torch = module('torch')
    torch.int32 = np.int32
    torch.Tensor = Tensor
    torch.tensor = lambda values,dtype=None: np.array(values,dtype=dtype).view(Tensor)
    torch.zeros = lambda *shape,dtype=None,device=None: np.zeros(shape,dtype=dtype).view(Tensor)
    torch.all, torch.equal = np.all, np.array_equal
    torch.distributed = module('torch.distributed')
    torch.distributed.all_reduce = lambda *a,**k:None
    mode = enum.IntEnum('CUDAGraphMode', {'NONE':0})
    module('vllm.config.compilation').CUDAGraphMode = mode
    @dataclasses.dataclass(frozen=True)
    class Descriptor:
        cg_mode: object
        num_tokens: int
        num_reqs: int
        num_active_loras: int = 0
    cuda = module('vllm.v1.worker.gpu.cudagraph_utils')
    cuda.BatchExecutionDescriptor, cuda.CudaGraphManager = Descriptor, object
    module('vllm.distributed.parallel_state').get_dp_group = lambda:None
    dp_utils = module('vllm.v1.worker.gpu.dp_utils')
    dp_utils.__file__ = str(checkout/'vllm/v1/worker/gpu/dp_utils.py')
    exec(compile(Path(dp_utils.__file__).read_text(), dp_utils.__file__, 'exec'), vars(dp_utils))
    return core
