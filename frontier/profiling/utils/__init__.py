import binascii
import enum
import importlib
import os
import re
from itertools import product
from math import floor
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from frontier.profiling.attention.attention_input import AttentionInput
from frontier.profiling.collectives.collectives_input import CollectivesInput
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.common.parallel_config import ParallelConfig
from frontier.profiling.utils.confirmation import (
    confirm_profiling_execution,
    build_linear_op_config_sections,
    build_attention_config_sections,
    build_moe_config_sections,
)
from frontier.types import MeasurementType

if TYPE_CHECKING:
    import torch


class ProfileMethod(enum.Enum):
    CUDA = "cuda"
    KERNEL_ONLY = "kernel_only"
    CUDA_EVENT = "cuda_event"
    KINETO = "kineto"
    PERF_COUNTER = "perf_counter"
    RECORD_FUNCTION = "record_function"


EXPORTABLE_PROFILE_METHOD_CHOICES = [
    ProfileMethod.CUDA.value,
    ProfileMethod.CUDA_EVENT.value,
    ProfileMethod.KERNEL_ONLY.value,
    ProfileMethod.RECORD_FUNCTION.value,
]

_MAX_MIXED_ATTENTION_BATCH_SIZE = 128
_TRUE_MIXED_PREFILL_CHUNK_ANCHORS = (64, 128, 256, 512, 1024)
_TRUE_MIXED_DECODE_KV_ANCHORS = (128, 256, 512, 1024, 2048)


def normalize_profile_method(profile_method: str) -> str:
    normalized = str(profile_method).strip().lower()
    if normalized in {ProfileMethod.CUDA.value, ProfileMethod.CUDA_EVENT.value}:
        return ProfileMethod.CUDA_EVENT.value
    if normalized in {
        ProfileMethod.KERNEL_ONLY.value,
        ProfileMethod.RECORD_FUNCTION.value,
    }:
        return ProfileMethod.RECORD_FUNCTION.value
    return normalized


def profile_method_to_measurement_type(profile_method: str) -> MeasurementType:
    normalized = normalize_profile_method(profile_method)
    if normalized == ProfileMethod.CUDA_EVENT.value:
        return MeasurementType.CUDA_EVENT
    if normalized == ProfileMethod.RECORD_FUNCTION.value:
        return MeasurementType.KERNEL_ONLY
    raise ValueError(
        "Only cuda_event and record_function profiling methods can be exported to predictor-training CSVs "
        "(aliases: cuda and kernel_only). "
        f"Got profile_method={profile_method!r}."
    )


def build_profiling_output_path(
    *,
    output_root: str | os.PathLike[str],
    profiling_type: str,
    hardware: str,
    model_name: str,
    op_name: str,
) -> Path:
    """Build the canonical profiling CSV path.

    Schema:
        data/profiling/<type>/<hardware>/<model_name>/<op_name>.csv
    """
    for field_name, field_value in (
        ("profiling_type", profiling_type),
        ("hardware", hardware),
        ("model_name", model_name),
        ("op_name", op_name),
    ):
        if str(field_value).strip() == "":
            raise ValueError(f"{field_name} must be a non-empty string.")

    return (
        Path(output_root)
        / str(profiling_type).strip()
        / str(hardware).strip()
        / str(model_name).strip()
        / f"{str(op_name).strip()}.csv"
    )


def build_profile_method_output_path(
    *,
    output_root: str | os.PathLike[str],
    profiling_type: str,
    hardware: str,
    model_name: str,
    op_name: str,
    profile_method: str,
) -> Path:
    """Build the canonical output path for a profiling method.

    CUDA-event data keeps the primary op filename used by simulator defaults,
    for example `linear_op.csv`. Kernel-only data uses the simulator's
    dedicated `*_kernel_only.csv` input-file convention.
    """
    measurement_type = profile_method_to_measurement_type(profile_method)
    output_op_name = str(op_name).strip()
    if measurement_type.value == "KERNEL_ONLY":
        output_op_name = f"{output_op_name}_kernel_only"

    return build_profiling_output_path(
        output_root=output_root,
        profiling_type=profiling_type,
        hardware=hardware,
        model_name=model_name,
        op_name=output_op_name,
    )


def require_profiling_dependencies(
    module_name: str,
    dependencies: tuple[str, ...],
) -> None:
    """Fail fast when GPU profiling dependencies are missing."""
    missing = []
    for dependency in dependencies:
        try:
            importlib.import_module(dependency)
        except ImportError:
            missing.append(dependency)

    if not missing:
        return

    readable_module = str(module_name).strip() or "GPU"
    missing_text = ", ".join(missing)
    raise ImportError(
        f"{readable_module.capitalize()} profiling requires missing dependencies: {missing_text}. "
        "Install the dedicated profiling environment with "
        "`conda env create -f environment_profiling.yml` and "
        "`conda activate frontier-profiling`, or run the profiling command from an "
        "existing environment with vllm and flashinfer already configured. "
        "The minimal `environment.yml` intentionally excludes these GPU profiling packages."
    )


def get_num_tokens_to_profile(
    max_num_tokens: int,
    extra_num_tokens: List[int] = None,
    num_tokens_list: List[int] = None,
):
    if num_tokens_list is not None:
        if extra_num_tokens:
            raise ValueError(
                "num_tokens_list cannot be combined with extra_num_tokens; "
                "provide one exact token set."
            )
        if not num_tokens_list:
            raise ValueError("num_tokens_list must be non-empty.")

        exact_num_tokens = set()
        for num_tokens in num_tokens_list:
            token_value = int(num_tokens)
            if token_value <= 0:
                raise ValueError(
                    f"num_tokens_list must contain positive integers, got {token_value}"
                )
            if token_value > max_num_tokens:
                raise ValueError(
                    "num_tokens_list values must be <= max_num_tokens, "
                    f"got {token_value} > {max_num_tokens}"
                )
            exact_num_tokens.add(token_value)

        return sorted(exact_num_tokens, reverse=True)

    NUM_TOKENS_SPACE = (
        list([1, 2, 4])
        + list(range(8, 1024, 8))
        + list(range(1024, 2 * 1024 + 1, 16))
        + list(range(2 * 1024, 4 * 1024 + 1, 32))
        + list(range(4 * 1024, 8 * 1024 + 1, 64))
        + list(range(8 * 1024, 16 * 1024 + 1, 128))
        + list(range(16 * 1024, 32 * 1024 + 1, 256))
        + list(range(32 * 1024, 64 * 1024 + 1, 512))
        + list(range(64 * 1024, 128 * 1024 + 1, 1024))
    )
    num_tokens_to_profile = []
    for num_tokens in NUM_TOKENS_SPACE:
        if num_tokens <= max_num_tokens:
            num_tokens_to_profile.append(num_tokens)
        else:
            break
    if max_num_tokens > 0:
        num_tokens_to_profile.append(max_num_tokens)

    if extra_num_tokens:
        for num_tokens in extra_num_tokens:
            token_value = int(num_tokens)
            if token_value <= 0:
                raise ValueError(
                    f"extra_num_tokens must be positive integers, got {token_value}"
                )
            # Explicit extensions belong to the requested profiling domain.
            # The profiler owns the physical-capacity check for each workload.
            num_tokens_to_profile.append(token_value)

    # Deduplicate while preserving deterministic sort order.
    num_tokens_to_profile = sorted(set(num_tokens_to_profile), reverse=True)

    return num_tokens_to_profile


def build_profile_position_indices(
    num_tokens: int,
    max_position_embeddings: int,
) -> List[int]:
    """Build legal RoPE positions for flattened profiling workloads.

    Linear-op profiling treats ``num_tokens`` as a flattened prefill workload,
    which can exceed a model's context length. The position tensor is only used
    to exercise position-dependent kernels such as RoPE; wrapping it keeps every
    index inside the model's precomputed cache without changing the profiled
    token count or operator shapes.
    """
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")
    if max_position_embeddings <= 0:
        raise ValueError(
            "max_position_embeddings must be positive, "
            f"got {max_position_embeddings}"
        )
    return [index % max_position_embeddings for index in range(num_tokens)]


def get_attention_batch_sizes_to_profile(
    min_batch_size: int,
    max_batch_size: int,
    batch_size_list: List[int] = None,
):
    if batch_size_list is not None:
        if not batch_size_list:
            raise ValueError("batch_size_list must be non-empty.")

        normalized = []
        seen = set()
        for value in batch_size_list:
            batch_size = int(value)
            if batch_size < min_batch_size or batch_size > max_batch_size:
                raise ValueError(
                    "batch_size_list values must be within min_batch_size and max_batch_size."
                )
            if batch_size in seen:
                continue
            seen.add(batch_size)
            normalized.append(batch_size)
        return sorted(normalized)

    BATCH_SIZE_SPACE = list(range(1, 128 + 1, 1)) + list(range(128, 1024 + 1, 8))
    batch_sizes_to_profile = list(
        filter(
            lambda x: (x >= min_batch_size and x <= max_batch_size), BATCH_SIZE_SPACE
        )
    )
    for boundary in (min_batch_size, max_batch_size):
        if boundary > 0 and min_batch_size <= boundary <= max_batch_size:
            batch_sizes_to_profile.append(boundary)
    return sorted(set(batch_sizes_to_profile))


def get_attention_prefill_chunk_sizes_to_profile(max_seq_len: int):
    # PREFILL_CHUNK_SIZE_SPACE = [64, 128, 256, 512, 768, 1024, 1536, 2048, 3076, 4096, 8192, 16384]
    # PREFILL_CHUNK_SIZE_SPACE = range(128, 128 * 1024, 128)
    PREFILL_CHUNK_SIZE_SPACE = (
        list(range(64, 128 + 1, 16))
        + list(range(128, 1024 + 1, 32))
        + list(range(1024, 4 * 1024 + 1, 64))
        + list(range(4 * 1024, 16 * 1024 + 1, 128))
        + list(range(16 * 1024, 64 * 1024 + 1, 256))
    )
    prefill_chunk_sizes_to_profile = []
    for prefill_chunk_size in PREFILL_CHUNK_SIZE_SPACE:
        if prefill_chunk_size <= max_seq_len:
            prefill_chunk_sizes_to_profile.append(prefill_chunk_size)
        else:
            break
    if max_seq_len > 0:
        prefill_chunk_sizes_to_profile.append(max_seq_len)
    return sorted(set(prefill_chunk_sizes_to_profile))


def get_seq_lengths_to_profile(max_seq_len: int):
    SEQ_LENGTH_SIZE_SPACE = (
        list(range(0, 1024 + 1, 32))
        + list(range(1024, 4 * 1024 + 1, 64))
        + list(range(4 * 1024, 64 * 1024 + 1, 256))
    )
    extra_raw = os.environ.get("FRONTIER_EXTRA_SEQ_LENGTHS", "").strip()
    if extra_raw:
        for token in re.split(r"[\s,]+", extra_raw):
            if not token:
                continue
            value = int(token)
            if value < 0:
                raise ValueError(
                    f"FRONTIER_EXTRA_SEQ_LENGTHS contains negative value: {value}"
                )
            SEQ_LENGTH_SIZE_SPACE.append(value)
    SEQ_LENGTH_SIZE_SPACE.append(max_seq_len)
    seq_lengths_to_profile = [
        seq_length for seq_length in SEQ_LENGTH_SIZE_SPACE if seq_length <= max_seq_len
    ]
    return sorted(set(seq_lengths_to_profile))


def get_attention_input_combinations(
    max_seq_len: int,
    min_batch_size: int,
    max_batch_size: int,
    profile_only_prefill: bool,
    profile_only_decode: bool,
    batch_size_list: List[int] = None,
    decode_kv_cache_size_list: List[int] = None,
    enable_chunked_prefill_grid_search: bool = True,
    fixed_chunked_prefill_size: int = -1,
    max_model_len: Optional[int] = None,
):
    max_model_len = _resolve_profile_max_model_len(max_seq_len, max_model_len)
    input_combinations = []

    if fixed_chunked_prefill_size == 0:
        raise ValueError(
            "fixed_chunked_prefill_size must be a positive integer or -1 to disable chunking."
        )

    def _resolve_prefill_chunk_sizes():
        # Grid search enabled: use either provided fixed size or the full sweep.
        if enable_chunked_prefill_grid_search:
            if fixed_chunked_prefill_size > 0:
                return [min(fixed_chunked_prefill_size, max_seq_len)]
            return get_attention_prefill_chunk_sizes_to_profile(max_seq_len)

        # Grid search disabled: use a single fixed chunk size (default to full length).
        if fixed_chunked_prefill_size > 0:
            return [min(fixed_chunked_prefill_size, max_seq_len)]
        return [max_seq_len]

    # Chunked Prefills
    prefill_chunk_sizes_to_profile = _resolve_prefill_chunk_sizes()
    for prefill_chunk_size in prefill_chunk_sizes_to_profile:
        num_partitions = max_seq_len // prefill_chunk_size
        kv_cache_sizes_to_profile = [
            partition_index * prefill_chunk_size
            for partition_index in range(num_partitions)
        ]
        input_combinations.extend(
            product([prefill_chunk_size], kv_cache_sizes_to_profile, [1], [True])
        )
    # Full prefills
    prefill_lengths_to_profile = get_seq_lengths_to_profile(max_seq_len)
    input_combinations.extend(product(prefill_lengths_to_profile, [0], [1], [True]))
    # Decodes
    max_decode_kv_cache_size = max_seq_len - 1
    max_runtime_decode_kv_cache_size = max_model_len - 1
    if decode_kv_cache_size_list is not None:
        kv_cache_sizes_to_profile = _normalize_positive_int_list(
            "decode_kv_cache_size_list",
            decode_kv_cache_size_list,
            minimum=0,
        )
        oversized_kv_cache_sizes = [
            kv_cache_size
            for kv_cache_size in kv_cache_sizes_to_profile
            if kv_cache_size > max_runtime_decode_kv_cache_size
        ]
        if oversized_kv_cache_sizes:
            raise ValueError(
                "decode_kv_cache_size_list values must be <= max_model_len - 1, "
                f"got {oversized_kv_cache_sizes} > {max_runtime_decode_kv_cache_size}"
            )
    else:
        kv_cache_sizes_to_profile = get_seq_lengths_to_profile(
            max_decode_kv_cache_size
        )
    batch_sizes_to_profile = get_attention_batch_sizes_to_profile(
        min_batch_size, max_batch_size, batch_size_list
    )
    input_combinations.extend(
        product([0], kv_cache_sizes_to_profile, batch_sizes_to_profile, [False])
    )

    valid_input_combinations = []
    for input_combination in input_combinations:
        prefill_chunk_size, kv_cache_size, batch_size, is_prefill = input_combination

        if is_prefill and profile_only_decode:
            continue

        if not is_prefill and profile_only_prefill:
            continue

        attention_input = AttentionInput(
            prefill_chunk_size,
            kv_cache_size,
            batch_size,
            is_prefill,
        )

        if attention_input.is_valid(
            max_seq_len=max_seq_len,
            max_model_len=max_model_len,
        ):
            valid_input_combinations.append(attention_input)
    return valid_input_combinations


"""
    For a given model and parallel config, get the maximum number of blocks that can be allocated.
    This doesn't take into account the weights and activations.
"""


def get_max_num_blocks(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    block_size: int,
    dtype: "torch.dtype",
    gpu_memory_utilization: float = 0.9,
    max_pipeline_parallel_size: int = 8,
):
    import torch

    element_size = torch.randn(1, dtype=dtype).element_size()
    block_memory_size = (
        2
        * block_size
        * model_config.get_num_kv_heads(parallel_config)
        * model_config.get_head_size()
        * element_size
    )
    assert model_config.num_layers % max_pipeline_parallel_size == 0
    block_memory_total = block_memory_size * (
        model_config.num_layers // max_pipeline_parallel_size
    )
    return floor(
        (torch.cuda.mem_get_info()[1] * gpu_memory_utilization) / (block_memory_total)
    )


def get_collectives_sizes_to_profile(max_collective_size: int):
    COLLECTIVE_SIZE_SPACE = (
        list(range(1024, 512 * 1024 + 1, 4 * 1024))
        + list(range(512 * 1024, 8 * 1024 * 1024 + 1, 16 * 1024))
        + list(range(8 * 1024 * 1024, 64 * 1024 * 1024 + 1, 64 * 1024))
        + list(range(64 * 1024 * 1024 + 1, 512 * 1024 * 1024 + 1, 265 * 1024))
    )
    collectives_size_to_profile = []
    for collectives_size in COLLECTIVE_SIZE_SPACE:
        if collectives_size <= max_collective_size:
            collectives_size_to_profile.append(collectives_size)
        else:
            break

    # Ensure the requested upper bound is profiled explicitly.
    if (
        collectives_size_to_profile
        and collectives_size_to_profile[-1] < max_collective_size
    ):
        collectives_size_to_profile.append(max_collective_size)

    return collectives_size_to_profile


def get_collectives_inputs(
    num_nodes: int,
    num_workers_per_node_combinations: List[int],
    max_collective_size: int,
    collective: str,
    total_gpus_available: int,
):
    num_workers = []

    for num_workers_per_node in num_workers_per_node_combinations:
        for _num_nodes in range(1, num_nodes + 1):
            num_workers.append(num_workers_per_node * _num_nodes)

    num_workers = list(set(num_workers))
    collectives_sizes = get_collectives_sizes_to_profile(max_collective_size)

    collectives_inputs = []

    for num_workers, num_workers_per_node, collective_size in product(
        num_workers, num_workers_per_node_combinations, collectives_sizes
    ):
        collectives_input = CollectivesInput(
            num_workers, num_workers_per_node, collective_size, collective
        )
        if not collectives_input.is_valid(total_gpus_available, num_nodes):
            continue

        collectives_inputs.append(collectives_input)

    return collectives_inputs


def get_cpu_overhead_batch_sizes_to_profile(max_batch_size: int):
    BATCH_SIZE_SPACE = list(range(8, 64 + 1, 8)) + list(range(64, 256 + 1, 16))
    batch_size_to_profile = []
    for batch_size in BATCH_SIZE_SPACE:
        if batch_size <= max_batch_size:
            batch_size_to_profile.append(batch_size)
        else:
            break
    return batch_size_to_profile


def get_mixed_prefill_input_combinations(
    max_seq_len: int,
    min_batch_size: int = 2,
    max_batch_size: int = 8,
    mode: str = "even",
    num_samples_per_config: int = 3,
    kv_cache_sizes: Optional[List[int]] = None,
    max_model_len: Optional[int] = None,
):
    """
    Generate mixed-length prefill input combinations for profiling.
    
    This function creates test cases for profiling attention performance
    with multiple sequences of potentially different lengths in a single batch.
    
    Args:
        max_seq_len: Maximum sequence length in the profiling envelope.
        max_model_len: Runtime context limit for each sequence, defaulting to
                       ``max_seq_len``.
        min_batch_size: Minimum batch size (default 2, since 1 is single-seq).
        max_batch_size: Maximum batch size (default 8).
        mode: Generation mode - "even", "random", or "both".
        num_samples_per_config: Number of random samples to generate per configuration 
                                (only used in random mode, default 3).
    
    Returns:
        List of MixedAttentionInput objects for profiling.
    
    Modes:
        - "even": All sequences in a batch have the same length (baseline).
        - "random": Sequences have randomly distributed lengths.
        - "both": Generate both even and random combinations.
    """
    from frontier.profiling.attention.mixed_attention_input import MixedAttentionInput

    max_model_len = _resolve_profile_max_model_len(max_seq_len, max_model_len)
    if min_batch_size < 2:
        raise ValueError("min_batch_size must be >= 2 for mixed prefill profiling.")
    if max_batch_size < min_batch_size:
        raise ValueError("max_batch_size must be >= min_batch_size.")
    if max_batch_size > _MAX_MIXED_ATTENTION_BATCH_SIZE:
        raise ValueError(
            "max_batch_size must be <= "
            f"{_MAX_MIXED_ATTENTION_BATCH_SIZE} for mixed prefill profiling."
        )
    if num_samples_per_config <= 0:
        raise ValueError("num_samples_per_config must be > 0.")
    if mode not in {"even", "random", "both"}:
        raise ValueError(
            f"mode must be one of even, random, or both, got {mode!r}."
        )

    explicit_kv_cache_sizes = kv_cache_sizes is not None
    normalized_kv_cache_sizes = _normalize_positive_int_list(
        "kv_cache_sizes",
        [0] if kv_cache_sizes is None else kv_cache_sizes,
        minimum=0,
    )
    if normalized_kv_cache_sizes is None:
        raise RuntimeError("kv_cache_sizes normalization unexpectedly returned None")
    kv_cache_sizes = normalized_kv_cache_sizes
    explicit_kv_size_set = set(kv_cache_sizes if explicit_kv_cache_sizes else ())
    input_combinations = []
    seen_inputs = set()

    def _append_unique(seq_lens, kv_cache_size, input_mode):
        input_key = (tuple(seq_lens), kv_cache_size, input_mode)
        if input_key in seen_inputs:
            return
        candidate = MixedAttentionInput(
            seq_lens=list(seq_lens),
            kv_cache_size=kv_cache_size,
            mode=input_mode,
        )
        if not candidate.is_valid(
            max_seq_len=max_model_len,
            max_batch_size=_MAX_MIXED_ATTENTION_BATCH_SIZE,
        ):
            raise RuntimeError(f"Generated invalid mixed prefill input: {candidate}")
        seen_inputs.add(input_key)
        input_combinations.append(candidate)

    modes_to_generate = []
    
    if mode == "both":
        modes_to_generate = ["even", "random"]
    else:
        modes_to_generate = [mode]
    
    # Sequence length candidates (aligned to 8 for FlashAttention efficiency)
    seq_length_candidates = (
        list(range(64, 128 + 1, 16))
        + list(range(128, 1024 + 1, 32))
        + list(range(1024, 4 * 1024 + 1, 64))
        + list(range(4 * 1024, min(16 * 1024, max_seq_len) + 1, 256))
    )
    seq_length_candidates.append(max_seq_len)
    for kv_cache_size in kv_cache_sizes:
        endpoint_seq_len = min(max_seq_len, max_model_len - kv_cache_size)
        if endpoint_seq_len > 0:
            seq_length_candidates.append(endpoint_seq_len)
    seq_length_candidates = sorted(set(seq_length_candidates))

    # Filter to only include lengths <= max_seq_len
    seq_length_candidates = [s for s in seq_length_candidates if s <= max_seq_len]
    
    # Batch sizes to test
    batch_sizes = list(range(min_batch_size, max_batch_size + 1))
    
    for current_mode in modes_to_generate:
        if current_mode == "even":
            # Even mode: all sequences have the same length
            for batch_size in batch_sizes:
                for kv_cache_size in kv_cache_sizes:
                    for seq_len in seq_length_candidates:
                        if seq_len + kv_cache_size > max_model_len:
                            continue
                        seq_lens = [seq_len] * batch_size
                        _append_unique(seq_lens, kv_cache_size, "even")
        
        elif current_mode == "random":
            # Random mode: sequences have varied lengths
            import numpy as np
            
            for batch_size in batch_sizes:
                # Define length ranges for random generation
                length_ranges = [
                    (64, 512),
                    (128, 1024),
                    (512, 2048),
                    (1024, 4096),
                    (2048, 8192),
                    (4096, 16384),
                ]
                
                for min_len, max_len in length_ranges:
                    # Skip if the range is beyond the profiling envelope.
                    if min_len > max_seq_len:
                        continue
                    
                    # Clip max_len to max_seq_len
                    effective_max_len = min(max_len, max_seq_len)
                    
                    if effective_max_len <= min_len:
                        continue
                    
                    # Generate multiple random samples for each range
                    for kv_cache_size in kv_cache_sizes:
                        if min_len + kv_cache_size > max_model_len:
                            continue
                        effective_max_len_with_cache = min(
                            effective_max_len,
                            max_model_len - kv_cache_size,
                        )
                        if effective_max_len_with_cache <= min_len:
                            continue

                        for seed_idx in range(num_samples_per_config):
                            # Use a deterministic seed for reproducibility
                            seed = seed_idx * 10000 + batch_size * 100 + min_len
                            np.random.seed(seed)

                            # Generate random sequence lengths
                            seq_lens = np.random.randint(
                                min_len,
                                effective_max_len_with_cache + 1,
                                size=batch_size
                            ).tolist()

                            # Round to multiples of 8 for FlashAttention efficiency
                            seq_lens = [((s + 7) // 8) * 8 for s in seq_lens]

                            # Ensure all are within bounds after rounding
                            seq_lens = [
                                min(s, effective_max_len_with_cache) for s in seq_lens
                            ]

                            _append_unique(seq_lens, kv_cache_size, "random")

                # Keep one deterministic heterogeneous shape at the context
                # boundary for every requested KV cache size.
                for kv_cache_size in kv_cache_sizes:
                    endpoint_seq_len = min(max_seq_len, max_model_len - kv_cache_size)
                    if endpoint_seq_len > 0:
                        boundary_seq_lens = [endpoint_seq_len] + [
                            max(1, endpoint_seq_len // 2)
                        ] * (batch_size - 1)
                        _append_unique(boundary_seq_lens, kv_cache_size, "random")

    valid_kv_sizes = {
        item.kv_cache_size
        for item in input_combinations
    }
    missing_explicit_kv = sorted(explicit_kv_size_set - valid_kv_sizes)
    if missing_explicit_kv:
        raise ValueError(
            "kv_cache_sizes values have no runtime-legal pairing under "
            f"max_model_len={max_model_len}: {missing_explicit_kv}."
        )

    if not input_combinations:
        raise ValueError("Mixed prefill profiling produced no valid input combinations.")

    return input_combinations


def _build_balanced_seq_lens(batch_size: int, total_tokens: int) -> List[int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if total_tokens < batch_size:
        raise ValueError(
            f"total_tokens ({total_tokens}) must be >= batch_size ({batch_size})."
        )

    base = total_tokens // batch_size
    remainder = total_tokens % batch_size
    seq_lens = [base + 1] * remainder + [base] * (batch_size - remainder)
    if len(seq_lens) != batch_size:
        raise RuntimeError("Balanced shape generation produced invalid batch size.")
    return seq_lens


def _build_skewed_seq_lens(
    batch_size: int,
    total_tokens: int,
    max_seq_len: int,
) -> List[int]:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive.")
    if total_tokens > batch_size * max_seq_len:
        raise ValueError(
            f"total_tokens ({total_tokens}) exceeds capacity ({batch_size * max_seq_len}) "
            f"for batch_size={batch_size}, max_seq_len={max_seq_len}."
        )

    # Build a deterministic front-loaded shape by filling earlier sequences first.
    seq_lens = [1] * batch_size
    remaining = total_tokens - batch_size
    for idx in range(batch_size):
        if remaining == 0:
            break
        headroom = max_seq_len - seq_lens[idx]
        allocated = min(headroom, remaining)
        seq_lens[idx] += allocated
        remaining -= allocated

    if remaining != 0:
        raise RuntimeError(
            "Skewed shape generation failed to consume all tokens under memory constraints."
        )
    return seq_lens


def _validate_online_grid_seq_lens(
    seq_lens: List[int],
    batch_size: int,
    total_tokens: int,
    max_model_len: int,
    kv_cache_size: int = 0,
) -> None:
    if len(seq_lens) != batch_size:
        raise RuntimeError(
            f"Invalid seq_lens length: expected {batch_size}, got {len(seq_lens)}."
        )
    if sum(seq_lens) != total_tokens:
        raise RuntimeError(
            f"Invalid seq_lens total: expected {total_tokens}, got {sum(seq_lens)}."
        )
    if min(seq_lens) <= 0:
        raise RuntimeError("All seq_lens values must be positive.")
    if max(seq_lens) > max_model_len:
        raise RuntimeError(
            f"Invalid seq_lens max: {max(seq_lens)} exceeds max_model_len={max_model_len}."
        )
    if kv_cache_size < 0:
        raise RuntimeError("kv_cache_size must be non-negative.")
    if max(seq_lens) + kv_cache_size > max_model_len:
        raise RuntimeError(
            "Invalid seq_lens + kv_cache_size: "
            f"max(seq_lens) + kv_cache_size = {max(seq_lens) + kv_cache_size} "
            f"exceeds max_model_len={max_model_len}."
        )


def _normalize_positive_int_list(
    name: str,
    values: Optional[List[int]],
    minimum: int,
) -> Optional[List[int]]:
    if values is None:
        return None
    if not values:
        raise ValueError(f"{name} must be non-empty.")

    normalized = set()
    for value in values:
        int_value = int(value)
        if int_value < minimum:
            raise ValueError(
                f"{name} values must be >= {minimum}, got {int_value}."
            )
        normalized.add(int_value)
    return sorted(normalized)


def _resolve_profile_max_model_len(
    max_seq_len: int,
    max_model_len: Optional[int],
) -> int:
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive.")
    resolved_max_model_len = (
        max_seq_len if max_model_len is None else int(max_model_len)
    )
    if resolved_max_model_len <= 0:
        raise ValueError("max_model_len must be positive.")
    if resolved_max_model_len < max_seq_len:
        raise ValueError(
            "max_model_len must be >= max_seq_len, got "
            f"max_model_len={resolved_max_model_len}, max_seq_len={max_seq_len}."
        )
    return resolved_max_model_len


def _derive_axis_with_headroom(
    *,
    anchors: tuple[int, ...],
    config_max: int,
    physical_max: int,
    minimum: int,
) -> List[int]:
    """Build an automatic axis through the first legal point above config_max."""
    values = [
        anchor
        for anchor in sorted(anchors)
        if minimum <= anchor <= config_max
    ]
    if config_max >= minimum:
        values.append(config_max)

    next_anchor = next(
        (anchor for anchor in sorted(anchors) if anchor > config_max),
        None,
    )
    headroom_value = (
        next_anchor if next_anchor is not None and next_anchor <= physical_max else physical_max
    )
    if headroom_value > config_max and headroom_value >= minimum:
        values.append(headroom_value)
    return sorted(set(values))


def get_online_grid_mixed_prefill_input_combinations(
    max_seq_len: int,
    min_batch_size: int,
    max_batch_size: int,
    min_total_tokens: int,
    max_total_tokens: int,
    shapes_per_point: int = 2,
    kv_cache_sizes: Optional[List[int]] = None,
    batch_size_list: Optional[List[int]] = None,
    total_tokens_list: Optional[List[int]] = None,
    max_model_len: Optional[int] = None,
):
    """
    Generate deterministic mixed prefill inputs for online serving coverage.

    Grid definition:
      - batch_size in [min_batch_size, max_batch_size]
      - total_tokens in [min_total_tokens, max_total_tokens]
      - per point shapes: balanced (+ optionally skewed)
    """
    from frontier.profiling.attention.mixed_attention_input import MixedAttentionInput

    max_model_len = _resolve_profile_max_model_len(max_seq_len, max_model_len)
    if min_batch_size < 2:
        raise ValueError("min_batch_size must be >= 2.")
    if max_batch_size < min_batch_size:
        raise ValueError("max_batch_size must be >= min_batch_size.")
    if max_batch_size > _MAX_MIXED_ATTENTION_BATCH_SIZE:
        raise ValueError(
            "max_batch_size must be <= "
            f"{_MAX_MIXED_ATTENTION_BATCH_SIZE} for mixed prefill profiling."
        )
    if min_total_tokens <= 0:
        raise ValueError("min_total_tokens must be positive.")
    if max_total_tokens < min_total_tokens:
        raise ValueError("max_total_tokens must be >= min_total_tokens.")
    if shapes_per_point not in (1, 2):
        raise ValueError("shapes_per_point must be 1 or 2.")
    explicit_kv_cache_sizes = kv_cache_sizes is not None
    kv_cache_sizes = _normalize_positive_int_list(
        "kv_cache_sizes",
        [0] if kv_cache_sizes is None else kv_cache_sizes,
        minimum=0,
    )

    batch_size_extensions = _normalize_positive_int_list(
        "batch_size_list",
        batch_size_list,
        minimum=2,
    )
    batch_sizes = set(range(min_batch_size, max_batch_size + 1))
    if batch_size_extensions is not None:
        oversized_batch_sizes = [
            batch_size
            for batch_size in batch_size_extensions
            if batch_size > _MAX_MIXED_ATTENTION_BATCH_SIZE
        ]
        if oversized_batch_sizes:
            raise ValueError(
                "batch_size_list values must be <= "
                f"{_MAX_MIXED_ATTENTION_BATCH_SIZE}, got {oversized_batch_sizes}."
            )
        batch_sizes.update(batch_size_extensions)
    batch_sizes = sorted(batch_sizes)

    total_token_extensions = _normalize_positive_int_list(
        "total_tokens_list",
        total_tokens_list,
        minimum=1,
    )
    total_tokens_values = set(range(min_total_tokens, max_total_tokens + 1))
    if total_token_extensions is not None:
        total_tokens_values.update(total_token_extensions)
    total_tokens_values = sorted(total_tokens_values)
    explicit_batch_sizes = set(batch_size_extensions or ())
    explicit_total_tokens = set(total_token_extensions or ())
    explicit_kv_sizes = set(kv_cache_sizes if explicit_kv_cache_sizes else ())

    input_combinations = []
    valid_batch_sizes = set()
    valid_total_tokens = set()
    valid_kv_sizes = set()
    for batch_size in batch_sizes:
        for total_tokens in total_tokens_values:
            point_is_explicit = (
                batch_size in explicit_batch_sizes
                and total_tokens in explicit_total_tokens
            )
            if total_tokens < batch_size:
                if point_is_explicit:
                    raise ValueError(
                        f"Invalid grid point: total_tokens={total_tokens} < "
                        f"batch_size={batch_size}."
                    )
                continue
            if total_tokens > batch_size * max_model_len:
                if point_is_explicit:
                    raise ValueError(
                        f"Invalid grid point: total_tokens={total_tokens} exceeds "
                        f"batch_capacity={batch_size * max_model_len}."
                    )
                continue

            balanced_seq_lens = _build_balanced_seq_lens(
                batch_size=batch_size,
                total_tokens=total_tokens,
            )
            skewed_seq_lens = None
            try:
                if shapes_per_point == 2:
                    skewed_seq_lens = _build_skewed_seq_lens(
                        batch_size=batch_size,
                        total_tokens=total_tokens,
                        max_seq_len=max_model_len,
                    )
                    if skewed_seq_lens == balanced_seq_lens:
                        raise RuntimeError(
                            "Balanced and skewed seq_lens are identical; cannot satisfy "
                            "shapes_per_point=2."
                        )
            except (RuntimeError, ValueError) as exc:
                if point_is_explicit:
                    raise ValueError(
                        f"Invalid explicit online_grid point "
                        f"(batch_size={batch_size}, total_tokens={total_tokens}): {exc}"
                    ) from exc
                continue

            point_inputs = []
            valid_point_kv_sizes = set()
            for kv_cache_size in kv_cache_sizes:
                kv_has_valid_shape = False
                try:
                    _validate_online_grid_seq_lens(
                        seq_lens=balanced_seq_lens,
                        batch_size=batch_size,
                        total_tokens=total_tokens,
                        max_model_len=max_model_len,
                        kv_cache_size=kv_cache_size,
                    )
                    point_inputs.append(
                        MixedAttentionInput(
                            seq_lens=balanced_seq_lens,
                            kv_cache_size=kv_cache_size,
                            mode="online_grid_balanced",
                        )
                    )
                    kv_has_valid_shape = True
                except RuntimeError as exc:
                    if point_is_explicit:
                        raise ValueError(
                            f"Invalid explicit online_grid point "
                            f"(batch_size={batch_size}, total_tokens={total_tokens}, "
                            f"kv_cache_size={kv_cache_size}): {exc}"
                        ) from exc

                if skewed_seq_lens is not None:
                    try:
                        _validate_online_grid_seq_lens(
                            seq_lens=skewed_seq_lens,
                            batch_size=batch_size,
                            total_tokens=total_tokens,
                            max_model_len=max_model_len,
                            kv_cache_size=kv_cache_size,
                        )
                        point_inputs.append(
                            MixedAttentionInput(
                                seq_lens=skewed_seq_lens,
                                kv_cache_size=kv_cache_size,
                                mode="online_grid_skewed",
                            )
                        )
                        kv_has_valid_shape = True
                    except RuntimeError as exc:
                        if point_is_explicit:
                            raise ValueError(
                                f"Invalid explicit online_grid point "
                                f"(batch_size={batch_size}, total_tokens={total_tokens}, "
                                f"kv_cache_size={kv_cache_size}): {exc}"
                            ) from exc
                if kv_has_valid_shape:
                    valid_point_kv_sizes.add(kv_cache_size)

            if point_inputs:
                input_combinations.extend(point_inputs)
                valid_batch_sizes.add(batch_size)
                valid_total_tokens.add(total_tokens)
                valid_kv_sizes.update(valid_point_kv_sizes)

    missing_explicit_batches = sorted(explicit_batch_sizes - valid_batch_sizes)
    if missing_explicit_batches:
        raise ValueError(
            "batch_size_list values have no physically legal pairing: "
            f"{missing_explicit_batches}."
        )
    missing_explicit_totals = sorted(explicit_total_tokens - valid_total_tokens)
    if missing_explicit_totals:
        raise ValueError(
            "total_tokens_list values have no physically legal pairing: "
            f"{missing_explicit_totals}."
        )
    missing_explicit_kv = sorted(explicit_kv_sizes - valid_kv_sizes)
    if missing_explicit_kv:
        raise ValueError(
            "kv_cache_sizes values have no physically legal pairing: "
            f"{missing_explicit_kv}."
        )

    if not input_combinations:
        raise ValueError("online_grid mixed prefill produced no valid input combinations.")

    return input_combinations


def get_true_mixed_attention_input_combinations(
    max_seq_len: int,
    prefill_batch_sizes: List[int],
    prefill_chunk_sizes: Optional[List[int]] = None,
    decode_batch_sizes: Optional[List[int]] = None,
    decode_kv_cache_sizes: Optional[List[int]] = None,
    prefill_kv_cache_size: int = 0,
    max_model_len: Optional[int] = None,
):
    """Generate true mixed prefill+decode attention input combinations."""
    from frontier.profiling.attention.true_mixed_batch_input import TrueMixedBatchInput

    max_model_len = _resolve_profile_max_model_len(max_seq_len, max_model_len)
    if prefill_kv_cache_size < 0:
        raise ValueError("prefill_kv_cache_size must be non-negative")
    if prefill_kv_cache_size >= max_model_len:
        raise ValueError(
            "prefill_kv_cache_size must leave room for at least one prefill token "
            f"under max_model_len={max_model_len}, got {prefill_kv_cache_size}."
        )
    if not prefill_batch_sizes:
        raise ValueError("prefill_batch_sizes cannot be empty")
    if not decode_batch_sizes:
        raise ValueError("decode_batch_sizes cannot be empty")

    normalized_prefill_batch_sizes = []
    for prefill_bs in prefill_batch_sizes:
        if prefill_bs <= 0:
            raise ValueError(f"Invalid prefill batch size: {prefill_bs}")
        if prefill_bs > _MAX_MIXED_ATTENTION_BATCH_SIZE:
            raise ValueError(
                "prefill_batch_sizes values must be <= "
                f"{_MAX_MIXED_ATTENTION_BATCH_SIZE}, got {prefill_bs}."
            )
        normalized_prefill_batch_sizes.append(int(prefill_bs))
    prefill_batch_sizes = sorted(set(normalized_prefill_batch_sizes))

    def _normalize_explicit_axis(name: str, values: List[int], minimum: int):
        normalized = _normalize_positive_int_list(name, values, minimum)
        if normalized is None:
            raise RuntimeError(f"{name} normalization unexpectedly returned None")
        return normalized

    explicit_prefill_chunks = prefill_chunk_sizes is not None
    explicit_prefill_values = (
        _normalize_explicit_axis("prefill_chunk_sizes", prefill_chunk_sizes, 1)
        if explicit_prefill_chunks
        else []
    )
    explicit_prefill_value_set = set(explicit_prefill_values)
    profile_endpoint = max_seq_len - prefill_kv_cache_size
    automatic_prefill_values = _derive_axis_with_headroom(
        anchors=_TRUE_MIXED_PREFILL_CHUNK_ANCHORS,
        config_max=profile_endpoint,
        physical_max=max_model_len - prefill_kv_cache_size,
        minimum=1,
    )
    normalized_prefill_chunk_sizes = sorted(
        set(automatic_prefill_values + explicit_prefill_values)
    )

    explicit_decode_kv = decode_kv_cache_sizes is not None
    explicit_decode_values = (
        _normalize_explicit_axis("decode_kv_cache_sizes", decode_kv_cache_sizes, 0)
        if explicit_decode_kv
        else []
    )
    explicit_decode_value_set = set(explicit_decode_values)
    profile_endpoint = max_seq_len - 1
    automatic_decode_values = _derive_axis_with_headroom(
        anchors=_TRUE_MIXED_DECODE_KV_ANCHORS,
        config_max=profile_endpoint,
        physical_max=max_model_len - 1,
        minimum=0,
    )
    normalized_decode_kv_cache_sizes = sorted(
        set(automatic_decode_values + explicit_decode_values)
    )

    def _physical_prefill_chunks(values):
        valid = []
        for value in values:
            if value + prefill_kv_cache_size > max_model_len:
                if value in explicit_prefill_value_set:
                    raise ValueError(
                        "prefill_chunk_sizes values plus prefill_kv_cache_size must "
                        f"be <= max_model_len, got {value} + "
                        f"{prefill_kv_cache_size} > {max_model_len}."
                    )
                continue
            valid.append(value)
        return valid

    def _physical_decode_kv(values):
        valid = []
        for value in values:
            if value + 1 > max_model_len:
                if value in explicit_decode_value_set:
                    raise ValueError(
                        "decode_kv_cache_sizes values must be <= max_model_len - 1, "
                        f"got {value} > {max_model_len - 1}."
                    )
                continue
            valid.append(value)
        return valid

    normalized_prefill_chunk_sizes = _physical_prefill_chunks(
        normalized_prefill_chunk_sizes
    )
    normalized_decode_kv_cache_sizes = _physical_decode_kv(
        normalized_decode_kv_cache_sizes
    )
    if not normalized_prefill_chunk_sizes:
        raise ValueError(
            "True-mixed attention profiling produced no physically valid prefill "
            "chunk sizes."
        )
    if not normalized_decode_kv_cache_sizes:
        raise ValueError(
            "True-mixed attention profiling produced no physically valid decode "
            "KV cache sizes."
        )

    normalized_decode_batch_sizes = []
    for decode_bs in decode_batch_sizes:
        if decode_bs <= 0:
            raise ValueError(f"Invalid decode batch size: {decode_bs}")
        if decode_bs > _MAX_MIXED_ATTENTION_BATCH_SIZE:
            raise ValueError(
                "decode_batch_sizes values must be <= "
                f"{_MAX_MIXED_ATTENTION_BATCH_SIZE}, got {decode_bs}."
            )
        normalized_decode_batch_sizes.append(int(decode_bs))
    decode_batch_sizes = sorted(set(normalized_decode_batch_sizes))
    oversized_total_batch_sizes = [
        (prefill_bs, decode_bs)
        for prefill_bs in prefill_batch_sizes
        for decode_bs in decode_batch_sizes
        if prefill_bs + decode_bs > _MAX_MIXED_ATTENTION_BATCH_SIZE
    ]
    if oversized_total_batch_sizes:
        raise ValueError(
            "true-mixed total batch size must be <= "
            f"{_MAX_MIXED_ATTENTION_BATCH_SIZE}, got "
            f"{oversized_total_batch_sizes}."
        )

    combinations = []
    for prefill_bs in prefill_batch_sizes:
        for prefill_chunk_size in normalized_prefill_chunk_sizes:
            for decode_bs in decode_batch_sizes:
                for decode_kv_cache_size in normalized_decode_kv_cache_sizes:
                    candidate = TrueMixedBatchInput(
                        prefill_seq_lens=[prefill_chunk_size] * prefill_bs,
                        prefill_kv_cache_sizes=[prefill_kv_cache_size] * prefill_bs,
                        decode_kv_cache_sizes=[decode_kv_cache_size] * decode_bs,
                    )
                    if not candidate.is_valid(
                        max_seq_len=max_model_len,
                        max_batch_size=_MAX_MIXED_ATTENTION_BATCH_SIZE,
                    ):
                        raise RuntimeError(
                            f"Generated invalid true-mixed attention input: {candidate}"
                        )
                    combinations.append(candidate)
    if not combinations:
        raise ValueError(
            "True-mixed attention profiling produced no valid input combinations."
        )
    return combinations


def hex_to_binary(hex_identifier):
    return binascii.unhexlify(hex_identifier)
