"""
Interactive confirmation utility for profiling modules.

This module provides a unified confirmation interface before profiling execution,
displaying key parameters in a clean tabular format.
"""

import sys
from typing import Any, Dict, List, Optional, Tuple, Sequence

from frontier.model_architectures import get_model_architecture_profile
from frontier.profiling.common.parallel_config import ParallelConfig


# Constants for ASCII table formatting
TABLE_WIDTH = 70
SECTION_HEADER_CHAR = "-"
TABLE_BORDER_CHAR = "="


def _format_list(items: List[Any], max_display: int = 8) -> str:
    """Format a list for display, truncating if too long."""
    if not items:
        return "[]"
    if len(items) <= max_display:
        return str(items)
    return f"{items[:max_display]}... ({len(items)} total)"


def _format_value(value: Any) -> str:
    """Format a value for display."""
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        return _format_list(value)
    return str(value)


def _build_ascii_table(
    title: str,
    sections: List[Tuple[str, List[Tuple[str, str]]]],
) -> str:
    """Build an ASCII table for display.

    Args:
        title: The table title
        sections: List of (section_name, [(key, value), ...]) tuples

    Returns:
        Formatted ASCII table string
    """
    lines = []

    # Title
    lines.append(TABLE_BORDER_CHAR * TABLE_WIDTH)
    lines.append(title.center(TABLE_WIDTH))
    lines.append(TABLE_BORDER_CHAR * TABLE_WIDTH)

    for section_name, rows in sections:
        lines.append(f"\n[{section_name}]")
        lines.append(SECTION_HEADER_CHAR * TABLE_WIDTH)
        for key, value in rows:
            # Handle multi-line values (indented continuation)
            if "\n" in value:
                value_lines = value.split("\n")
                lines.append(f"  {key:<30} : {value_lines[0]}")
                for continuation in value_lines[1:]:
                    lines.append(f"  {'':<30}   {continuation}")
            else:
                lines.append(f"  {key:<30} : {value}")

    lines.append(TABLE_BORDER_CHAR * TABLE_WIDTH)
    return "\n".join(lines)


def confirm_profiling_execution(
    module_name: str,
    config_sections: List[Tuple[str, List[Tuple[str, str]]]],
    skip_confirmation: bool = False,
) -> bool:
    """
    Display profiling configuration and request user confirmation.

    Args:
        module_name: Name of the profiling module (e.g., "Linear Op", "Attention", "MoE")
        config_sections: List of (section_name, [(key, value), ...]) tuples
        skip_confirmation: If True, skip the confirmation prompt

    Returns:
        True if user confirms (or skip_confirmation is True), False otherwise
    """
    title = f"{module_name} Profiling Configuration"

    # Display configuration
    print(_build_ascii_table(title, config_sections))

    # Skip confirmation if requested
    if skip_confirmation:
        print("\n[--yes flag set, proceeding without confirmation]\n")
        return True

    # Request confirmation
    print("\nProceed with profiling? [y/N]: ", end="", flush=True)

    try:
        response = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n\nAborted by user.")
        return False

    if response in ("y", "yes"):
        print("\nStarting profiling...\n")
        return True
    else:
        print("\nProfiling aborted by user.")
        return False


# ============================================================================
# Linear Op module configuration builder
# ============================================================================

def build_linear_op_config_sections(
    args,
    model_config,
    num_tokens_count: int,
    precision_str: str,
    torch_dtype,
) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """Build configuration sections for linear_op profiling.

    Args:
        args: Parsed command-line arguments
        model_config: ModelConfig instance for the model being profiled
        num_tokens_count: Number of token counts to profile
        precision_str: String representation of precision (e.g., "BF16")
        torch_dtype: torch.dtype being used

    Returns:
        List of (section_name, [(key, value), ...]) tuples
    """
    # Execution mode
    if args.disable_ray:
        exec_mode = f"Non-Ray (multiprocessing), {args.num_gpus} GPUs"
    else:
        exec_mode = f"Ray, {args.num_gpus} GPUs"

    # TP sizes
    attn_tp = args.attn_tp if args.attn_tp else args.num_tensor_parallel_workers
    ffn_tp = args.ffn_tp if args.ffn_tp else args.num_tensor_parallel_workers

    # Format precision with dtype
    precision_display = f"{precision_str} ({torch_dtype})"

    # Build operations by TP sharding section
    attn_tp_str = str(attn_tp).replace(" ", "")
    ffn_tp_str = str(ffn_tp).replace(" ", "")
    replicated_tp_str = "[1]"
    replicated_disabled = getattr(args, "disable_replicated", False)
    replicated_note = "SKIP (--disable_replicated)" if replicated_disabled else f"TP={replicated_tp_str} (Replicated)"

    attn_lines = [
        f"    - input_layernorm            : {precision_str}, {replicated_note}",
        f"    - attn_pre_proj (QKV)        : {precision_str}, TP={attn_tp_str} (Sharded)",
        f"    - attn_rope                  : {precision_str}, TP={attn_tp_str} (Sharded)",
        f"    - attn_post_proj             : {precision_str}, TP={attn_tp_str} (Sharded)",
    ]

    architecture_profile = get_model_architecture_profile(model_config)
    for op_name in architecture_profile.predictor_attention_extra_ops:
        attn_lines.append(
            f"    - {op_name:<28}: {precision_str}, TP={attn_tp_str} (Sharded)"
        )
    for op_name in architecture_profile.linear_attention.replicated_ops:
        attn_lines.append(f"    - {op_name:<28}: {precision_str}, {replicated_note}")
    for op_name in architecture_profile.linear_attention.additional_sharded_ops:
        if op_name in architecture_profile.predictor_attention_extra_ops:
            continue
        attn_lines.append(
            f"    - {op_name:<28}: {precision_str}, TP={attn_tp_str} (Sharded)"
        )

    is_moe = getattr(args, 'is_moe', False)
    moe_skip_note = "SKIP (--is_moe, profiled by MoE module)"
    ffn_lines = [
        f"    - post_attention_layernorm   : {precision_str}, {replicated_note}",
    ]
    if is_moe:
        ffn_lines.extend([
            f"    - mlp_up_proj                : {moe_skip_note}",
            f"    - mlp_act                    : {moe_skip_note}",
            f"    - mlp_down_proj              : {moe_skip_note}",
        ])
    else:
        ffn_lines.extend([
            f"    - mlp_up_proj                : {precision_str}, TP={ffn_tp_str} (Sharded)",
            f"    - mlp_act                    : {precision_str}, TP={ffn_tp_str} (Sharded)",
            f"    - mlp_down_proj              : {precision_str}, TP={ffn_tp_str} (Sharded)",
        ])

    supports_share_expert = (
        model_config.supports_share_expert()
        if hasattr(model_config, "supports_share_expert")
        else False
    )
    if getattr(model_config, "is_moe", False) and supports_share_expert:
        ffn_lines.extend(
            [
                f"    - share_expert_up_proj       : {precision_str}, TP={ffn_tp_str} (Sharded)",
                f"    - share_expert_act           : {precision_str}, TP={ffn_tp_str} (Sharded)",
                f"    - share_expert_down_proj     : {precision_str}, TP={ffn_tp_str} (Sharded)",
            ]
        )

    common_lines = [
        f"    - add                        : {precision_str}, {replicated_note}",
        f"    - emb (embedding)            : {precision_str}, {replicated_note}",
    ]

    ops_content = (
        f"Attention Ops (attn_tp={attn_tp_str}):\n"
        f"{chr(10).join(attn_lines)}\n"
        f"\n"
        f"  FFN Ops (ffn_tp={ffn_tp_str}):\n"
        f"{chr(10).join(ffn_lines)}\n"
        f"\n"
        f"  Common Ops:\n"
        f"{chr(10).join(common_lines)}"
    )

    # Calculate total combinations
    unique_tps = set(attn_tp) | set(ffn_tp)
    if not replicated_disabled:
        unique_tps.add(1)
    total_combos = len(unique_tps) * num_tokens_count

    sections = [
        ("Execution", [
            ("Mode", exec_mode),
            ("Output Directory", args.output_dir),
            ("Profile Method", args.profile_method),
            ("Disable Replicated Ops", _format_value(replicated_disabled)),
        ]),
        (f"Model: {model_config.name}", [
            ("Embedding Dim", str(model_config.embedding_dim)),
            ("MLP Hidden Dim", str(model_config.mlp_hidden_dim)),
            ("Num Q Heads", str(model_config.num_q_heads)),
            ("Num KV Heads", str(model_config.num_kv_heads)),
            ("Is MoE", _format_value(getattr(model_config, 'is_moe', False))),
        ]),
        ("Precision & Quantization", [
            ("Precision (dtype)", precision_display),
            ("FP8 Quantization", _format_value(args.use_fp8)),
            ("Block Shape", _format_value(args.block_shape)),
        ]),
        ("Operations by TP Sharding", [
            ("", ops_content),
        ]),
        ("Profiling Matrix", [
            ("Attention TP Sizes", _format_value(attn_tp)),
            ("FFN TP Sizes", _format_value(ffn_tp)),
            ("Replicated TP Size", "N/A (disabled)" if replicated_disabled else "[1]"),
            ("Max Tokens", str(args.max_tokens)),
            ("Token Count Samples", str(num_tokens_count)),
            ("Total Combinations", f"~{total_combos} ({len(unique_tps)} TP configs x {num_tokens_count} token counts)"),
        ]),
    ]

    return sections


# ============================================================================
# Attention module configuration builder
# ============================================================================

def build_attention_config_sections(
    args,
    model_config,
    input_combinations_count: int,
    mixed_combinations_count: int,
    precision_str: str,
    torch_dtype,
    true_mixed_combinations_count: int = 0,
) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """Build configuration sections for attention profiling.

    Args:
        args: Parsed command-line arguments
        model_config: ModelConfig instance
        input_combinations_count: Number of standard attention input combinations
        mixed_combinations_count: Number of mixed prefill input combinations
        true_mixed_combinations_count: Number of true mixed prefill+decode combinations
        precision_str: String representation of precision
        torch_dtype: torch.dtype being used

    Returns:
        List of (section_name, [(key, value), ...]) tuples
    """
    # Execution mode
    if args.disable_ray:
        exec_mode = f"Non-Ray (multiprocessing), {args.num_gpus} GPUs"
    else:
        exec_mode = f"Ray, {args.num_gpus} GPUs"

    # Format precision with dtype
    precision_display = f"{precision_str} ({torch_dtype})"

    # Build per-TP configuration
    tp_sizes = args.num_tensor_parallel_workers
    head_dim = model_config.get_head_dim()

    per_tp_lines = []
    for tp in tp_sizes:
        parallel_config = ParallelConfig(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
        )
        q_heads = model_config.get_num_q_heads(parallel_config)
        kv_heads = model_config.get_num_kv_heads(parallel_config)
        per_tp_lines.append(f"TP={tp}: Q_heads={q_heads}, KV_heads={kv_heads}, head_dim={head_dim}")
    per_tp_content = "\n".join(per_tp_lines)

    # Calculate total configurations
    total_inputs = (
        input_combinations_count
        + mixed_combinations_count
        + true_mixed_combinations_count
    )
    total_configs = len(tp_sizes) * total_inputs

    sections = [
        ("Execution", [
            ("Mode", exec_mode),
            ("Output Directory", args.output_dir),
            ("Profile Method", args.profile_method),
            ("Attention Backend", str(args.attention_backend)),
        ]),
        (f"Model: {model_config.name}", [
            ("Embedding Dim", str(model_config.embedding_dim)),
            ("Head Dim", str(head_dim)),
            ("Num Q Heads", str(model_config.num_q_heads)),
            ("Num KV Heads", str(model_config.num_kv_heads)),
        ]),
        ("Precision & Quantization", [
            ("Precision (dtype)", precision_display),
            ("FP8 Quantization", _format_value(args.use_fp8)),
            ("Block Shape", _format_value(args.block_shape)),
        ]),
        ("Attention Parameters by TP", [
            ("TP Sizes to Profile", _format_value(tp_sizes)),
            ("Per-TP Configuration", per_tp_content),
        ]),
        ("Profiling Range", [
            ("Max Model Length", str(args.max_model_len)),
            ("Max Sequence Length", str(args.max_seq_len)),
            ("Batch Size Range", f"{args.min_batch_size} - {args.max_batch_size}"),
            ("Block Size", str(args.block_size)),
            ("Profile Only Prefill", _format_value(args.profile_only_prefill)),
            ("Profile Only Decode", _format_value(args.profile_only_decode)),
            ("Enable Mixed Prefill", _format_value(args.enable_mixed_prefill)),
        ]),
        ("Workload Summary", [
            ("Standard Input Combinations", f"{input_combinations_count:,}"),
            ("Mixed Input Combinations", f"{mixed_combinations_count:,}"),
            ("True Mixed Input Combinations", f"{true_mixed_combinations_count:,}"),
            ("Total Configurations", f"~{total_configs:,} ({len(tp_sizes)} TP x {total_inputs:,} inputs)"),
        ]),
    ]

    return sections


# ============================================================================
# MoE module configuration builder
# ============================================================================

def build_moe_config_sections(
    args,
    model_config,
    num_tokens_count: int,
    use_vllm_kernel: bool,
    precision_str: str,
    torch_dtype,
) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """Build configuration sections for MoE profiling.

    Args:
        args: Parsed command-line arguments
        model_config: ModelConfig instance
        num_tokens_count: Number of token counts to profile
        use_vllm_kernel: Whether vLLM kernel is being used
        precision_str: String representation of precision
        torch_dtype: torch.dtype being used

    Returns:
        List of (section_name, [(key, value), ...]) tuples
    """
    # Execution mode
    if args.disable_ray:
        exec_mode = f"Non-Ray (multiprocessing), {args.num_gpus} GPUs"
    else:
        exec_mode = f"Ray, {args.num_gpus} GPUs"

    # Format precision with dtype
    precision_display = f"{precision_str} ({torch_dtype})"

    # Build operations by parallelism section
    tp_sizes = args.num_tensor_parallel_workers
    ep_sizes = args.expert_parallel_sizes
    num_experts = model_config.num_experts

    ops_content = (
        f"Operations (all at precision {precision_str}):\n"
        f"    - moe_gating_linear          : replicated (TP=1)\n"
        f"    - moe_gating_routing_topk    : replicated (TP=1)\n"
        f"    - moe_shuffling              : replicated (TP=1)\n"
        f"    - moe_grouped_gemm           : TP-sharded (TP=moe_tp)\n"
        f"\n"
        f"  Per-EP Expert Distribution:"
    )
    for ep in ep_sizes:
        experts_per_device = num_experts // ep
        ops_content += f"\n    EP={ep}: {experts_per_device} experts/device"

    # Calculate total configurations
    if args.enable_load_imbalance:
        test_cases_per_config = len(args.load_distributions) * args.num_samples_per_distribution
    else:
        test_cases_per_config = 1

    total_configs = (
        len(tp_sizes) *
        len(ep_sizes) *
        num_tokens_count *
        test_cases_per_config
    )

    config_breakdown = f"({len(tp_sizes)} TP x {len(ep_sizes)} EP x {num_tokens_count} tokens"
    if args.enable_load_imbalance:
        config_breakdown += f" x {len(args.load_distributions)} distributions x {args.num_samples_per_distribution} samples"
    config_breakdown += ")"

    sections = [
        ("Execution", [
            ("Mode", exec_mode),
            ("Device", args.device),
            ("Output Directory", args.output_dir),
            ("Profile Method", args.profile_method),
            ("Use vLLM Kernel", _format_value(use_vllm_kernel)),
        ]),
        (f"Model: {model_config.name}", [
            ("Num Experts", str(num_experts)),
            ("Experts per Token (TopK)", str(model_config.num_experts_per_tok)),
            ("Expert Hidden Dim", str(model_config.mlp_hidden_dim)),
            ("Is MoE", _format_value(model_config.is_moe)),
        ]),
        ("Precision & Quantization", [
            ("Precision (dtype)", precision_display),
            ("FP8 Quantization", _format_value(args.use_fp8)),
            ("Per-Channel Quant", _format_value(getattr(args, 'per_channel_quant', False))),
            ("Block Shape", _format_value(args.block_shape)),
        ]),
        ("MoE Operations by Parallelism", [
            ("TP Sizes", _format_value(tp_sizes)),
            ("EP Sizes", _format_value(ep_sizes)),
            ("", ops_content),
        ]),
        ("Load Distribution", [
            ("Enable Load Imbalance", _format_value(args.enable_load_imbalance)),
            ("Distributions", _format_value(args.load_distributions) if args.enable_load_imbalance else "N/A"),
            ("Samples per Distribution", str(args.num_samples_per_distribution) if args.enable_load_imbalance else "N/A"),
        ]),
        ("Profiling Matrix", [
            ("Max Tokens", str(args.max_tokens)),
            ("Token Count Samples", str(num_tokens_count)),
            ("Total Configurations", f"~{total_configs:,}\n    {config_breakdown}"),
        ]),
    ]

    return sections
