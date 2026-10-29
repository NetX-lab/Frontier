"""A3's sole expected ledger: design section 4, one lane across 43 layers."""

from collections import Counter
from types import SimpleNamespace

import pytest

from frontier.entities import DummyForwardBatch
from frontier.entities.batch import DecodeCudaGraphMetadata, KVCacheBlockZeroing, KVBlockTableWrites
from frontier.execution_time_predictor.deepseek_v4 import serving_context, source_sp_token_widths, timing_operator_times
from frontier.moe_ep_workload import materialize_balanced_ep_workload
from frontier.types import ClusterType
from test_dsv4_source_execution import predictor
from test_dsv4_runner_consumer import real_forward

# Each row is (owning phase, real count, dummy count). The dispatch endpoint
# contains the three activation/weight/ID allgathers of design section 4.
EXPECTED = {
    'kv_cache_block_zero': ('pre_dp', 1, 0),
    'dsv4_block_table_write': ('pre_dp', 1, 0),
    'dsv4_input_preparation': ('attention', 1, 1),
    'dsv4_embedding': ('attention', 1, 1),
    'dsv4_embedding_allreduce': ('attention', 1, 1),
    'dsv4_mhc_first_pre': ('attention', 1, 1),
    'dsv4_mhc_ffn_post_attn_pre': ('attention', 42, 42),
    'dsv4_sp_attention_allgather': ('attention', 43, 43),
    'dsv4_sp_attention_reduce_scatter': ('attention', 43, 43),
    'dsv4_fused_qkv_projection': ('attention', 43, 43),
    'dsv4_fused_qkv_rmsnorm': ('attention', 43, 43),
    'dsv4_q_projection': ('attention', 43, 43),
    'dsv4_fused_q_kv_prepare': ('attention', 43, 43),
    'dsv4_sparse_attention': ('attention', 43, 43),
    'dsv4_output_inv_rope_quant': ('attention', 43, 43),
    'dsv4_grouped_output_projection': ('attention', 43, 43),
    'dsv4_output_projection': ('attention', 43, 43),
    'dsv4_compressor_projection': ('attention', 41, 41),
    'dsv4_compressor_save_partial_states': ('attention', 41, 41),
    'dsv4_compressor_fused_compress_store': ('attention', 41, 41),
    'dsv4_index_weights_projection': ('attention', 21, 21),
    'dsv4_index_compressor_projection': ('attention', 21, 21),
    'dsv4_index_compressor_save_partial_states': ('attention', 21, 21),
    'dsv4_index_compressor_fused_compress_store': ('attention', 21, 21),
    'dsv4_index_query_projection': ('attention', 21, 21),
    'dsv4_index_fused_query_prepare': ('attention', 21, 21),
    'dsv4_index_scores': ('attention', 21, 21),
    'dsv4_index_topk': ('attention', 21, 21),
    'dsv4_mhc_attn_post_ffn_pre': ('pre_dispatch', 43, 43),
    'share_expert_up_proj': ('pre_dispatch', 43, 43),
    'share_expert_act': ('pre_dispatch', 43, 43),
    'share_expert_down_proj': ('pre_dispatch', 43, 43),
    'moe_gating_linear': ('pre_dispatch', 43, 43),
    'moe_gating_routing_topk': ('pre_dispatch', 43, 43),
    'dsv4_ep_dispatch_allgather': ('dispatch', 43, 43),
    'moe_grouped_gemm': ('routed_compute', 43, 43),
    'dsv4_ep_output_reduce_scatter': ('combine', 43, 43),
    'dsv4_mhc_final_post': ('post_combine', 1, 1),
    'dsv4_sp_final_mhc_allgather': ('post_combine', 1, 1),
    'dsv4_mhc_head': ('post_combine', 1, 1),
    'dsv4_final_norm': ('post_combine', 1, 1),
    'dsv4_lm_head': ('post_combine', 1, 0),
    'dsv4_logits_gather': ('post_combine', 1, 0),
    'dsv4_sampling': ('post_combine', 1, 0),
}


@pytest.mark.parametrize('kind', ['decode', 'first_tail', 'prefill', 'dummy_FULL', 'dummy_PIECEWISE', 'dummy_NONE'])
def test_forward_ledger_and_phase_ownership(predictor, kind):
    p = predictor
    p._replica_config.attn_dp, p._replica_config.moe_expert_parallel_size = 2, 16
    p._replica_config.router_topk = 6
    p._replica_config.total_expert_num = 256
    p._cc_backend = SimpleNamespace(predict_allreduce=lambda **kw: 2.0,
                                    predict_allgather=lambda **kw: 2.0,
                                    predict_reduce_scatter=lambda **kw: 2.0)
    dummy = kind.startswith('dummy_')
    p._cluster_type = ClusterType.PREFILL if kind == 'prefill' else ClusterType.DECODE
    if dummy:
        mode = kind.removeprefix('dummy_')
        b = DummyForwardBatch(0, 0, 0)
        b._cluster_type = p._cluster_type
        if mode != 'NONE':
            b.decode_cuda_graph_metadata = DecodeCudaGraphMetadata(
                'full_and_piecewise', mode, True, False, 1, 8, 1, 8)
    else:
        b = real_forward(p._cluster_type, first_tail=kind == 'first_tail')
        b.cache_block_zeroing = KVCacheBlockZeroing((1,), 1002240)
        b.kv_block_table_writes = KVBlockTableWrites((), ((b.requests[0].id, (
            ('compressed', 24), ('swa_even', 96), ('swa_odd', 96),
            ('csa_state', 1536), ('hca_state', 768))),))
    widths = source_sp_token_widths(p, {0: b, 1: b})
    work = materialize_balanced_ep_workload(
        routing_token_count=widths.gathered_tokens, router_topk=6,
        total_expert_num=256, moe_expert_parallel_size=16)
    observed, costs, phases = Counter(), Counter(), {}
    for index in range(43):
        record = p._predict_dsv4_structural_execution(
            b, index, 0, lane_workload=work.lane(0), sp_token_widths=widths,
            **serving_context(p, b, index))
        observed.update(record.op_times.keys())
        costs.update(record.op_times)
        for phase in ('pre_dp', 'attention', 'pre_dispatch', 'dispatch', 'routed_compute', 'combine', 'post_combine'):
            for _, name, _ in timing_operator_times(record, phase):
                assert name not in phases or phases[name] == phase
                phases[name] = phase
    expected = {name: count[2 if dummy else 1] for name, count in EXPECTED.items()}
    # NONE dummies take the native short-context indexer branch.
    if kind == 'dummy_NONE':
        for name in ('dsv4_index_query_projection', 'dsv4_index_fused_query_prepare', 'dsv4_index_scores', 'dsv4_index_topk'):
            expected[name] = 0
        expected['dsv4_index_short_context_fill'] = 21
    # The sentinel duration is two milliseconds per scope invocation.
    assert dict(observed) == {name: count for name, count in expected.items() if count}
    assert dict(costs) == {name: count * 2.0 for name, count in expected.items() if count}
    for name in observed:
        owner = EXPECTED[name][0] if name in EXPECTED else 'attention'
        assert phases[name] == owner
