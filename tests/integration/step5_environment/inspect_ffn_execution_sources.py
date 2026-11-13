#!/usr/bin/env python3
"""Collect the exact native owners for FFN forward context, routing and kernels."""
import importlib.util
from pathlib import Path


SOURCES = {
    'vllm/forward_context.py': 'Native forward context and MoE workspace scope.',
    'vllm/model_executor/layers/fused_moe/unquantized_fused_moe_method.py': 'Native excluded expert weight processing, backend and forward.',
    'vllm/model_executor/layers/fused_moe/router/router_factory.py': 'Actual Step5 sigmoid correction-bias routing class selection.',
    'vllm/model_executor/layers/fused_moe/router/fused_moe_router.py': 'Native router call signature and top-k output.',
    'vllm/model_executor/layers/fused_moe/activation.py': 'Native SwiGLU clamp and cast equations.',
    'vllm/model_executor/layers/fused_moe/experts/triton_moe.py': 'H800 default FP8 block expert execution and scratch ownership.',
}


if __name__ == '__main__':
    path = Path(__file__).with_name('inspect_runtime_sources.py')
    spec = importlib.util.spec_from_file_location('runtime_sources', path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    collector.SOURCES = SOURCES
    collector.__file__ = __file__
    collector.main()
