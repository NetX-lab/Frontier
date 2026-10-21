#!/usr/bin/env python3
"""Read installed numerical dispatch owners without executing their operators."""
import importlib.util
from pathlib import Path
import sys

SOURCES = {
    'vllm/model_executor/layers/fused_moe/utils.py': 'Actual MoE activation quantization call parameters.',
    'vllm/utils/deep_gemm.py': 'Native UE8M0 default resolution and environment dependencies.',
    'vllm/model_executor/layers/fused_moe/all2all_utils.py': 'Actual Prepare/Finalize selection and imported implementations.',
    'vllm/model_executor/layers/utils.py': 'Unquantized GEMM implementation dispatch.',
    'vllm/model_executor/layers/quantization/utils/w8a8_utils.py': 'W8A8 quantization helpers used by FP8 utilities.',
    'vllm/model_executor/custom_op.py': 'Native activation custom-op forward selection.',
    'vllm/model_executor/determinism/batch_invariant.py': 'Reachable alternate GEMM dispatch condition.',
    'vllm/envs.py': 'Defaults and environment dependencies controlling numerical dispatch.',
    'vllm/_custom_ops.py': 'Compiled quantizer/router/reduction registration wrappers.',
}

if __name__ == '__main__':
    path = Path(__file__).with_name('inspect_runtime_sources.py')
    spec = importlib.util.spec_from_file_location('runtime_sources', path)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    task = Path(sys.argv[sys.argv.index('--task-root')+1]).resolve()
    output = Path(sys.argv[sys.argv.index('--output-dir')+1]).resolve()
    launcher = collector.load(task/'native_source_export/submit_export.py')
    assert task == launcher.inspection.TASK_ROOT
    assert output.is_relative_to(task) and output != task and not output.exists()
    collector.SOURCES = SOURCES
    collector.__file__ = __file__
    collector.main()
