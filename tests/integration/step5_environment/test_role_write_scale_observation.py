"""Check source-bound native writer and quantizer metadata without GPU claims."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from test_role_model_input_observation import (
    CENSUS_SOURCE, census, census_runner, events, native, observation, record, work,
)

TASK = Path(__file__).parents[3]/'task_memory/task_2026-10-15_step5_preview_h800'
SITES = ('kv_fused', 'kv_sparse_fused', 'kv_update', 'moe_quantize', 'silu_quantize')
WRAPPER_SOURCE = '''
writer_failure = None
quant_dtype = object()

def fused_qknorm_rope_cache(qkv, positions, slot_mapping, key_cache, value_cache,
    num_q_heads, num_kv_heads, head_dim, rotary_dim, kv_cache_dtype, k_scale, v_scale):
    if writer_failure is not None:
        raise writer_failure
    return qkv, key_cache, value_cache

def qkv_indexer_norm_rope(query, key, weights, positions, *, main_inputs,
    decode_z=None, decode_slots=None):
    if writer_failure is not None:
        raise writer_failure
    return query, key, weights, main_inputs[0], main_inputs[5], main_inputs[6]

def unified_kv_cache_update(key, value, layer_name):
    if writer_failure is not None:
        raise writer_failure
    return key

def moe_kernel_quantize_input(A, A_scale, quant_dtype, per_act_token_quant,
    block_shape=None, is_scale_swizzled=True, quantization_emulation=False):
    return Tensor(11000), Tensor(11001)

def silu_and_mul_per_block_quant(input, group_size, quant_dtype, scale_ub=None,
    is_scale_transposed=False):
    return Tensor(12000), Tensor(12001)
'''


@pytest.fixture
def write_scale(census, monkeypatch):
    source = CENSUS_SOURCE.replace(
        "    def forward(self, positions, hidden_states):\n        return hidden_states\n",
        """    writer = 'fused'
    def forward(self, positions, hidden_states):
        slots, key, value = Tensor(10001), Tensor(10002), Tensor(10003)
        if self.writer == 'fused':
            fused_qknorm_rope_cache(hidden_states, positions, slots, key, value,
                self.num_heads, self.num_kv_heads, self.head_dim, 64, 'auto', None, None)
        elif self.writer == 'sparse':
            qkv_indexer_norm_rope(hidden_states, key, value, positions,
                main_inputs=(hidden_states, None, None, None, slots, key, value),
                decode_z=None, decode_slots=None)
        else:
            unified_kv_cache_update(key, value, 'layer')
        return hidden_states
""", 1)
    source = source.replace(
        "        def _base_w13_fn():\n            return invoke_fused_moe_triton_kernel(hidden_states, w1, output,\n                a1q_scale, self.w1_scale,",
        """        a1q, scale1 = moe_kernel_quantize_input(hidden_states, a1q_scale,
            quant_dtype, self.per_act_token_quant, self.block_shape)
        def _base_w13_fn():
            return invoke_fused_moe_triton_kernel(a1q, w1, output,
                scale1, self.w1_scale,""", 1)
    source = source.replace(
        "        def _base_w2_fn():\n            return invoke_fused_moe_triton_kernel(hidden_states, w2, output,\n                a2_scale, self.w2_scale,",
        """        a2q, scale2 = silu_and_mul_per_block_quant(hidden_states, 128, quant_dtype)
        def _base_w2_fn():
            return invoke_fused_moe_triton_kernel(a2q, w2, output,
                scale2, self.w2_scale,""", 1)
    source += WRAPPER_SOURCE
    census.path.write_text(source)
    exec(compile(source, str(census.path), 'exec', dont_inherit=True), census.module.__dict__)
    monkeypatch.setattr(observation, 'CENSUS_SOURCES', {
        key: (census.module.__name__, hashlib.sha256(census.path.read_bytes()).hexdigest())
        for key in observation.CENSUS_SOURCES})
    calls = dict(observation.CENSUS_CALLS)
    # The original census fixture selects only methods present in its source.
    for key in SITES:
        calls[key] = ORIGINAL_CALLS[key]
    monkeypatch.setattr(observation, 'CENSUS_CALLS', calls)
    monkeypatch.setattr(observation, 'CENSUS_LAYER_LINE', next(i for i, line in
        enumerate(source.splitlines(), 1) if '# CENSUS_LAYER' in line))
    return census


ORIGINAL_CALLS = {key: observation.CENSUS_CALLS[key] for key in SITES}


@pytest.mark.parametrize('writer,site', [
    ('fused', 'kv_fused'), ('sparse', 'kv_sparse_fused'), ('generic', 'kv_update'),
])
def test_native_writer_metadata_keeps_actual_layer_parent_and_unknown_completion(write_scale, writer, site):
    observed, _, _ = census_runner(write_scale)
    for layer in observed.model.layers:
        layer.self_attn.writer = writer
    result = observed.execute_model(work())
    assert result is observed.vllm_config.model_inputs['inputs_embeds']
    calls = [row for row in events(observed, 'native_call') if row['site'] == site]
    assert [row['layer_id'] for row in calls] == [0, 1]
    assert all(row['owner_binding'] == 'NATIVE_FUNCTION_ANCESTOR' and
               row['outcome'] == 'RETURNED_CPU_CALL_NOT_GPU_COMPLETION' and
               row['gpu_completion'] == 'UNMEASURED' and row['metadata_complete'] for row in calls)
    by_event = {row['event']: row for row in events(observed, 'native_call')}
    assert all(by_event[row['parent_call']]['site'] == 'attention' for row in calls)
    assert record(observed)['call_census']['physical_kernel_census'] == 'UNMEASURED'


def test_fp8_producer_outputs_are_the_actual_gemm_argument_references(write_scale):
    observed, _, _ = census_runner(write_scale)
    observed.execute_model(work())
    calls = events(observed, 'native_call')
    producers = [row for row in calls if row['site'] in ('moe_quantize', 'silu_quantize')]
    gemms = [row for row in calls if row['site'] == 'triton_gemm']
    assert [row['site'] for row in producers] == ['moe_quantize', 'silu_quantize']
    assert [row['gemm'] for row in gemms] == ['GEMM1', 'GEMM2']
    assert [row['arguments']['A_scale'] for row in gemms] == [row['result'][1] for row in producers]
    assert [row['arguments']['A'] for row in gemms] == [row['result'][0] for row in producers]
    assert all(row['arguments']['quant_dtype'] == str(write_scale.module.quant_dtype)
               and row['metadata_complete'] and row['return_metadata_complete'] for row in producers)
    assert record(observed)['call_census']['transient_scale_content'] == 'UNMEASURED'
    json.dumps(record(observed), allow_nan=False)


@pytest.mark.parametrize('failure', [ValueError('native writer'), KeyboardInterrupt('native writer')])
def test_writer_failure_preserves_native_exception_and_restores_hooks(write_scale, failure):
    observed, _, _ = census_runner(write_scale)
    write_scale.module.writer_failure = failure
    before = sys.getprofile(), sys.gettrace()
    with pytest.raises(type(failure)) as caught:
        observed.execute_model(work())
    assert caught.value is failure and observed.execution_calls == 1
    assert (sys.getprofile(), sys.gettrace()) == before
    row = next(row for row in events(observed, 'native_call') if row['site'] == 'kv_fused')
    assert row['outcome'] == 'UNVERIFIED_NATIVE_UNWIND' and row['result'] is None


def test_full_replay_does_not_invent_writer_or_scale_python_delivery(write_scale):
    observed, graph, _ = census_runner(write_scale, 'FULL')
    observed.execute_model(work())
    assert graph.calls == 1
    assert not any(row['site'] in SITES for row in events(observed, 'native_call'))
    assert record(observed)['call_census']['transient_scale_content'] == 'UNMEASURED'


def test_capture_writer_and_scale_calls_reference_original_capture_binding(write_scale):
    observed, _, rows = census_runner(write_scale, 'FULL')
    observed.capture_model()
    capture = rows[-1]
    binding = next(row for row in capture['events'] if row['kind'] == 'capture_forward'
                   and row['native_capture_region'])
    calls = [row for row in capture['events'] if row['kind'] == 'native_call' and row['site'] in SITES]
    assert len(calls) == 4 and all(row['capture_binding_event'] == binding['event'] for row in calls)
    assert capture['call_census']['transient_scale_content'] == 'UNMEASURED'


@pytest.mark.parametrize('site', SITES)
def test_retained_writer_quantizer_signatures_and_sources_match_pins(site):
    key, owner, arguments, _ = ORIGINAL_CALLS[site]
    import check_role_execution as caller
    module, digest = observation.CENSUS_SOURCES[key]
    roots = dict(caller.CENSUS_SOURCE_ROOTS,
                 attention_update='native_runtime_sources/observed', qkv_indexer='native_source_export/observed')
    path = TASK/roots[key]/(module.replace('.', '/')+'.py')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    tree = ast.parse(path.read_bytes())
    for part in owner:
        tree = next(node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                    and node.name == part)
    parameters = {arg.arg for arg in [*tree.args.posonlyargs, *tree.args.args, *tree.args.kwonlyargs]}
    assert set(arguments) <= parameters
