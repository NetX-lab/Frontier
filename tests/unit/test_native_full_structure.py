"""Check read-only CUTLASS consumer links at the native model-inventory seam."""
from types import SimpleNamespace

import torch

from tests.comparison.calibration.native_full_structure import cutlass_state


def routed_module():
    module = torch.nn.Module()
    module.w13_weight = torch.nn.Parameter(torch.ones(2, 256, 128).to(torch.float8_e4m3fn))
    module.w2_weight = torch.nn.Parameter(torch.ones(2, 128, 128).to(torch.float8_e4m3fn))
    module.w13_weight_scale_inv = torch.nn.Parameter(torch.ones(2, 2, 1))
    module.w2_weight_scale_inv = torch.nn.Parameter(torch.ones(2, 1, 1))
    module.w13_input_scale = module.w2_input_scale = None
    module.weight_block_size = [128, 128]
    quant = SimpleNamespace(w1_scale=module.w13_weight_scale_inv,
        w2_scale=module.w2_weight_scale_inv, a1_scale=None, a2_scale=None,
        block_shape=[128, 128], quant_dtype=torch.float8_e4m3fn,
        weight_quant_dtype=torch.float8_e4m3fn, is_block_quantized=True,
        use_fp8_w8a8=True, w1_bias=None, w2_bias=None,
        per_act_token_quant=False, per_out_ch_quant=False)
    moe = SimpleNamespace(hidden_dim=128, intermediate_size_per_partition=128,
        num_local_experts=2, w13_num_shards=2, in_dtype=torch.bfloat16,
        activation='SILU', has_bias=False)
    module.moe_config = moe
    experts = SimpleNamespace(moe_config=moe, quant_config=quant, w1_scale=quant.w1_scale,
        w2_scale=quant.w2_scale, a1_scale=None, a2_scale=None,
        num_experts=2, out_dtype=torch.bfloat16, expects_unquantized_inputs=True,
        use_deepseek_fp8_block_scale=True, gemm1_alpha=None, gemm1_beta=None,
        gemm1_clamp_limit=None, w1_bias=None, w2_bias=None,
        tp_rank=0, tp_size=1, ep_rank=0, ep_size=1, use_dp=False)
    module.quant_method = SimpleNamespace(fp8_backend=SimpleNamespace(name='FLASHINFER_CUTLASS'),
        moe_quant_config=quant, moe=moe, weight_scale_name='weight_scale_inv',
        block_quant=True, weight_block_size=[128, 128],
        moe_kernel=SimpleNamespace(fused_experts=experts))
    return module


def test_cutlass_snapshot_retains_native_aliases_without_validating_apply_outputs():
    module = routed_module()
    pointers = [p.data_ptr() for p in module.parameters()]
    row = cutlass_state(module)
    assert all(row['structural_checks'].values())
    assert row['weights']['w13']['shape'] == [2, 256, 128]
    assert row['scales']['w1']['shape'] == [2, 2, 1]
    assert row['activation_call'] == 'unverified; constructor state is not apply arguments'
    assert pointers == [p.data_ptr() for p in module.parameters()]
    assert module.w13_weight_scale_inv.flatten().tolist() == [1.0] * 4


def test_cutlass_consumer_alias_dtype_and_block_mismatches_are_visible():
    module = routed_module()
    quant = module.quant_method.moe_quant_config
    quant.w1_scale = quant.w1_scale.clone()
    assert cutlass_state(module)['structural_checks']['scale_aliases'] is False
    quant.w1_scale = module.w13_weight_scale_inv
    module.w2_weight = torch.nn.Parameter(module.w2_weight.float())
    assert cutlass_state(module)['structural_checks']['weight_layout'] is False
    module.w2_weight = torch.nn.Parameter(module.w2_weight.to(torch.float8_e4m3fn))
    quant.block_shape = [64, 128]
    assert cutlass_state(module)['structural_checks']['block_domain'] is False


def test_cutlass_call_failures_limits_and_retrieval(monkeypatch):
    import json
    import pytest
    from tests.comparison.calibration.native_full_structure import (
        CutlassCallInventory, FullModelInventory)
    module = routed_module()
    arguments = dict(input=torch.ones(3, 128), output=torch.empty(3, 128),
        fc1_expert_weights=module.w13_weight, fc2_expert_weights=module.w2_weight,
        quant_scales=[module.w13_weight_scale_inv, module.w2_weight_scale_inv],
        token_selected_experts=torch.zeros(3, 2, dtype=torch.int32),
        token_final_scales=torch.ones(3, 2),
        input_sf=None, tp_size=1, tp_rank=0, ep_size=1, ep_rank=0,
        activation_type='Swiglu', use_deepseek_fp8_block_scale=True,
        use_mxfp8_act_scaling=False, use_w4_group_scaling=False,
        output_dtype=torch.bfloat16)
    for fault, byte_limit in [('normal', 8192), ('native', 8192),
                              ('metadata', 8192), ('byte_limit', 1800)]:
        error, calls = RuntimeError('native failure'), []

        def native(**kwargs):
            calls.append(kwargs)
            if fault == 'native':
                raise error
            return module.w2_weight

        observer = CutlassCallInventory({'records': 1, 'bytes': byte_limit})
        namespace = SimpleNamespace(flashinfer_cutlass_fused_moe=native)
        observer.install(namespace,
            [('layer.3', *arguments['quant_scales'])], {'rank': 0},
            {'layer.3': 'owned'})
        supplied = dict(arguments)
        if fault == 'metadata':
            supplied['input'] = object()
        for _ in range(2):
            if fault == 'native':
                with pytest.raises(RuntimeError) as caught:
                    namespace.flashinfer_cutlass_fused_moe(**supplied)
                assert caught.value is error
            else:
                returned = namespace.flashinfer_cutlass_fused_moe(**supplied)
                assert returned is module.w2_weight
        assert len(calls) == 2 and all(
            c[k] is v for c in calls for k, v in supplied.items())
        monkeypatch.setenv('FRONTIER_CUTLASS_CALL_LIMITS', json.dumps(observer.limits))
        worker = SimpleNamespace(_cutlass_calls=observer)
        encoded = FullModelInventory.full_model_inventory(worker)
        assert len(encoded.encode()) <= byte_limit
        rows = json.loads(encoded)
        assert len(rows['records']) == (1 if fault in ('normal', 'native') else 0)
        expected_drops = 2 if fault == 'byte_limit' else 0 if fault == 'metadata' else 1
        assert rows['dropped'] == expected_drops
        assert rows['errors'] == (2 if fault == 'metadata' else 0)
        assert rows['retained_bytes'] <= byte_limit and rows['run_verified'] is False
        if rows['records']:
            row = rows['records'][0]
            assert row['layer'] == 'layer.3' and row['input_sf'] is None
            assert row['tensors']['input']['shape'] == [3, 128]
            assert row['tensors']['token_selected_experts']['dtype'] == 'torch.int32'
            assert row['device_completion'] == 'unobserved'
            assert row['native_outcome'] == (
                'native_exception' if fault == 'native' else 'host_return')
    with pytest.raises(ValueError):
        CutlassCallInventory({'records': 1, 'bytes': 1})


def test_cutlass_calls_retain_owned_existing_host_context(monkeypatch):
    import json
    import sys
    from tests.comparison.calibration.native_full_structure import CutlassCallInventory
    module = routed_module()
    arguments = dict(input=torch.ones(3, 128), output=torch.empty(3, 128),
        fc1_expert_weights=module.w13_weight, fc2_expert_weights=module.w2_weight,
        quant_scales=[module.w13_weight_scale_inv, module.w2_weight_scale_inv],
        token_selected_experts=torch.zeros(3, 2, dtype=torch.int32),
        token_final_scales=torch.ones(3, 2), input_sf=None, tp_size=1, tp_rank=0,
        ep_size=1, ep_rank=0, activation_type='Swiglu', output_dtype=torch.bfloat16,
        use_deepseek_fp8_block_scale=True, use_mxfp8_act_scaling=False,
        use_w4_group_scaling=False)
    selected = SimpleNamespace(num_reqs=2, num_actual_tokens=11, num_decodes=1,
        num_prefills=1, num_decode_tokens=3, prefill_max_seq_len=384)
    context = SimpleNamespace(attn_metadata={'unrelated': object(), 'owned': selected},
        dp_metadata=SimpleNamespace(local_sizes=[5, 6]),
        cudagraph_runtime_mode=SimpleNamespace(name='PIECEWISE'),
        batch_descriptor=SimpleNamespace(num_tokens=16, num_reqs=2, uniform=False,
            has_lora=False, num_active_loras=0))
    monkeypatch.setitem(sys.modules, 'vllm.forward_context',
        SimpleNamespace(_forward_context=context))
    observer = CutlassCallInventory({'records': 4, 'bytes': 16384})
    namespace = SimpleNamespace(flashinfer_cutlass_fused_moe=lambda **kw: module)
    observer.install(namespace, [('layer.3', *arguments['quant_scales'])],
        {'rank': 0, 'world_size': 16}, {'layer.3': 'owned'})
    assert namespace.flashinfer_cutlass_fused_moe(**arguments) is module
    row = json.loads(observer.snapshot())['records'][0]['forward_context']
    assert row['attention_layer'] == 'owned'
    assert row['attention'] == dict(num_reqs=2, num_actual_tokens=11, num_decodes=1,
        num_prefills=1, num_decode_tokens=3, prefill_max_seq_len=384)
    assert row['dp_local_sizes'] == [5, 6] and row['graph_mode'] == 'PIECEWISE'
    assert row['batch_descriptor']['num_tokens'] == 16
    assert context.dp_metadata.local_sizes == [5, 6]
    for raw, sizes in [(None, None), ([{'owned': selected}], [5, 6]),
                       ({'owned': selected}, torch.ones(2)),
                       ({'owned': selected}, [1] * 17)]:
        context.attn_metadata, context.dp_metadata.local_sizes = raw, sizes
        observer = CutlassCallInventory({'records': 1, 'bytes': 8192})
        namespace = SimpleNamespace(flashinfer_cutlass_fused_moe=lambda **kw: module)
        observer.install(namespace, [('layer.3', *arguments['quant_scales'])],
            {'rank': 0, 'world_size': 16}, {'layer.3': 'owned'})
        assert namespace.flashinfer_cutlass_fused_moe(**arguments) is module
        report = json.loads(observer.snapshot())
        facts = report['records'][0]['forward_context']
        assert facts['attention_status'] == (
            'observed' if type(raw) is dict else 'unavailable')
        assert facts['dp_local_sizes'] == (
            'unsupported' if sizes is not None and not isinstance(sizes, list)
            or isinstance(sizes, list) and len(sizes) > 16 else sizes)
        assert report['errors'] == 0
    selected.num_actual_tokens = torch.ones(1)
    context.dp_metadata.local_sizes = None
    observer = CutlassCallInventory({'records': 1, 'bytes': 8192})
    namespace = SimpleNamespace(flashinfer_cutlass_fused_moe=lambda **kw: module)
    observer.install(namespace, [('layer.3', *arguments['quant_scales'])],
        {'rank': 0, 'world_size': 16}, {'layer.3': 'owned'})
    assert namespace.flashinfer_cutlass_fused_moe(**arguments) is module
    facts = json.loads(observer.snapshot())['records'][0]['forward_context']
    assert facts['attention']['num_actual_tokens'] == 'unsupported'
    assert torch.equal(selected.num_actual_tokens, torch.ones(1))
    context.attn_metadata = {'owned': object()}
    observer = CutlassCallInventory({'records': 1, 'bytes': 8192})
    namespace = SimpleNamespace(flashinfer_cutlass_fused_moe=lambda **kw: module)
    observer.install(namespace, [('layer.3', *arguments['quant_scales'])],
        {'rank': 0, 'world_size': 16}, {'layer.3': 'owned'})
    assert namespace.flashinfer_cutlass_fused_moe(**arguments) is module
    report = json.loads(observer.snapshot())
    assert report['errors'] == 1
    assert report['records'][0]['forward_context'] == {'status': 'error'}


def test_cutlass_inventory_missing_attention_does_not_disable_calls():
    import ast
    import inspect
    import json
    from tests.comparison.calibration.native_full_structure import (
        CutlassCallInventory, FullModelInventory)
    native = ast.parse(inspect.getsource(FullModelInventory))
    body = native.body[0].body[0].body[-2].body[0].body
    start = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == 'owned_attention')
    selected = routed_module()
    decoder_type = type('DeepseekV2DecoderLayer', (), {})
    missing, owned = decoder_type(), decoder_type()
    for decoder in (missing, owned):
        decoder.mlp = SimpleNamespace(experts=SimpleNamespace(routed_experts=selected))
    missing.self_attn = SimpleNamespace()
    owned.self_attn = SimpleNamespace(mla_attn=SimpleNamespace(
        mla_attn=SimpleNamespace(layer_name='actual.attention')))
    for decoders in ([missing], [missing, owned]):
        model = SimpleNamespace(modules=lambda: decoders,
            named_modules=lambda: [('selected', selected)])
        scope = {'self': SimpleNamespace(get_model=lambda: model)}
        block = ast.Module(body=body[start:-1], type_ignores=[])
        exec(compile(block, '<native-inventory-association>', 'exec'), scope)
        assert scope['attention_layers'] == (
            {} if len(decoders) == 1 else {'selected': 'actual.attention'})
        namespace = SimpleNamespace(flashinfer_cutlass_fused_moe=lambda **kw: selected)
        observer = CutlassCallInventory({'records': 1, 'bytes': 8192})
        observer.install(namespace, [], {'rank': 0, 'world_size': 16},
            scope['attention_layers'])
        assert json.loads(observer.snapshot())['installed'] is True


def test_other_backends_remain_unverified_without_cutlass_consumer_reads():
    module = SimpleNamespace(quant_method=SimpleNamespace(
        fp8_backend=SimpleNamespace(name='TRITON')))
    assert cutlass_state(module) == {'status': 'unverified', 'backend': 'TRITON'}
