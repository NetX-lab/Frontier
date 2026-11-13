#!/usr/bin/env python3
"""Read remaining installed routing, workspace and attention interface owners."""
import importlib.util
from pathlib import Path

SOURCES = {
    'vllm/utils/workspace.py': 'Actual MoE workspace initialization and allocation context.',
    'vllm/model_executor/layers/fused_moe/oracle/unquantized.py': 'Actual H800 excluded expert backend and weight format.',
    'vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py': 'Native sigmoid correction-bias top-k math and callback.',
    'vllm/model_executor/layers/fused_moe/fused_moe.py': 'Native fused expert accumulation and FP8 quantization math.',
    'vllm/model_executor/layers/activation.py': 'Actual dense/shared SwiGLU clamp and BF16 casts.',
    'vllm/v1/attention/backend.py': 'Native common attention metadata constructor.',
    'vllm/v1/attention/backends/fa_utils.py': 'Native H800 FlashAttention version resolver.',
    'vllm/model_executor/layers/indexer_topk.py': 'Native H800 per-row selector dispatch.',
}

if __name__ == '__main__':
    path = Path(__file__).with_name('inspect_runtime_sources.py')
    spec = importlib.util.spec_from_file_location('runtime_sources', path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    collector.SOURCES = SOURCES
    collector.__file__ = __file__
    collector.main()
