#!/usr/bin/env python3
"""Collect installed owners for native Step5 dummy initialization and FFN execution."""
import importlib.util
from pathlib import Path


SOURCES = {
    'vllm/distributed/parallel_state.py': 'Real TP/DP/PP and distributed initialization entry points.',
    'vllm/model_executor/model_loader/base_loader.py': 'Native device/dtype and model load lifecycle.',
    'vllm/model_executor/model_loader/reload/__init__.py': 'Native reload lifecycle exports.',
    'vllm/model_executor/model_loader/reload/layerwise.py': 'Native layer materialization and online finalization.',
    'vllm/model_executor/model_loader/reload/meta.py': 'Native tensor reload metadata and registration.',
    'vllm/model_executor/model_loader/reload/types.py': 'Native LayerReloadingInfo contract.',
    'vllm/model_executor/model_loader/reload/utils.py': 'Native layer tensor collection for dummy initialization.',
    'vllm/model_executor/layers/fused_moe/routed_experts.py': 'Complete expert tensor construction and native quant prefix.',
    'vllm/model_executor/layers/fused_moe/runner/moe_runner.py': 'Native router/expert execution and forward context.',
    'vllm/model_executor/layers/fused_moe/config.py': 'Native expert topology and padding config.',
    'vllm/model_executor/layers/fused_moe/expert_map_manager.py': 'Actual expert allocation at TP1 and EP placement.',
    'vllm/model_executor/layers/fused_moe/oracle/fp8.py': 'Actual FP8 backend conversion, kernel selection and workspaces.',
    'vllm/model_executor/layers/fused_moe/__init__.py': 'Locate the native unquantized MoE method owner.',
    'vllm/model_executor/layers/quantization/utils/fp8_utils.py': 'FP8 weight/scale construction and dummy initialization values.',
}


if __name__ == '__main__':
    path = Path(__file__).with_name('inspect_runtime_sources.py')
    spec = importlib.util.spec_from_file_location('runtime_sources', path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    collector.SOURCES = SOURCES
    collector.__file__ = __file__
    collector.main()
