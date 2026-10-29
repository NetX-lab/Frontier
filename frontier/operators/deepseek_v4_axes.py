"""Row axes of native DeepSeek-V4 standalone scopes; no profiling runtime imports."""

from types import MappingProxyType

DEEPSEEK_V4_OPERATOR_AXES = MappingProxyType({
    "dsv4_mhc_first_pre": "sp_local_rows",
    "dsv4_mhc_ffn_post_attn_pre": "sp_local_rows",
    "dsv4_mhc_attn_post_ffn_pre": "sp_local_rows",
    "dsv4_mhc_final_post": "sp_local_rows",
    "dsv4_embedding": "descriptor_rows",
    "dsv4_mhc_head": "descriptor_rows",
    "dsv4_final_norm": "descriptor_rows",
    "dsv4_lm_head": "logits_rows",
    "share_expert_up_proj": "sp_local_rows",
    "share_expert_act": "sp_local_rows",
    "share_expert_down_proj": "sp_local_rows",
})

DEEPSEEK_V4_STANDALONE_AXES = ("sp_local_rows", "descriptor_rows", "logits_rows")
