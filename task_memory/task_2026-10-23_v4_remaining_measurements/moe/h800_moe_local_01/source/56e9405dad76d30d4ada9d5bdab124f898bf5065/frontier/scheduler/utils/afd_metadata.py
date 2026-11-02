"""Aggregation helpers for DECODE_FFN AFD metadata."""

from typing import Any, List, Tuple

from frontier.entities.batch import Batch

def aggregate_afd_metadata(
    source_batches: List[Batch] | Tuple[Batch, ...],
) -> tuple[Any | None, bool]:
    """Aggregate AFD metadata for one synthetic DECODE_FFN batch.

    DECODE_FFN groups may combine source batches from multiple attention
    Replicas.  The synthetic target batch therefore needs the sum of each
    source batch's already-padded stage tokens, rather than one arbitrary
    source metadata object.  All source batches must agree on the metadata
    configuration and on whether they represent one stage or a macro wave.
    """

    from frontier.entities.batch import AFDStageMetadata

    if type(source_batches) not in (list, tuple) or not source_batches:
        raise ValueError(
            "DECODE_FFN metadata aggregation requires a non-empty "
            "source-batch list or tuple"
        )

    metadata_entries = []
    representation_flags = []
    missing_metadata = False
    for index, source_batch in enumerate(source_batches):
        if not isinstance(source_batch, Batch):
            raise TypeError(
                "DECODE_FFN metadata aggregation requires Batch instances, "
                f"got {type(source_batch).__name__} at index {index}"
            )
        metadata = getattr(source_batch, "afd_stage_metadata", None)
        if metadata is None:
            missing_metadata = True
            continue
        if type(metadata) is not AFDStageMetadata:
            raise TypeError(
                "DECODE_FFN source afd_stage_metadata must be an exact "
                f"AFDStageMetadata, got {type(metadata).__name__}"
            )
        represents_all_stages = getattr(
            source_batch,
            "afd_stage_represents_all_stages",
            False,
        )
        if type(represents_all_stages) is not bool:
            raise TypeError(
                "DECODE_FFN source afd_stage_represents_all_stages must be "
                f"an exact bool, got {represents_all_stages!r}"
            )
        metadata_entries.append(metadata)
        representation_flags.append(represents_all_stages)

    if not metadata_entries:
        return None, False
    if missing_metadata:
        raise ValueError(
            "DECODE_FFN source batches must either all provide "
            "afd_stage_metadata or all omit it"
        )
    if len(set(representation_flags)) != 1:
        raise ValueError(
            "DECODE_FFN source batches disagree on "
            "afd_stage_represents_all_stages"
        )

    first = metadata_entries[0]
    scalar_fields = (
        "num_stages",
        "attention_use_cuda_graph",
        "ffn_use_cuda_graph",
        "attention_cudagraph_capture_sizes",
        "ffn_cudagraph_capture_sizes",
    )
    for metadata in metadata_entries[1:]:
        for field_name in scalar_fields:
            if getattr(metadata, field_name) != getattr(first, field_name):
                raise ValueError(
                    "DECODE_FFN source AFD metadata configuration mismatch: "
                    f"field={field_name!r}"
                )

    num_stages = first.num_stages
    if type(num_stages) is not int or num_stages <= 0:
        raise ValueError(
            "DECODE_FFN source AFD metadata num_stages must be an exact "
            f"positive int, got {num_stages!r}"
        )

    vector_fields = (
        ("original_stage_token_lens", False),
        ("padded_stage_token_lens", True),
        ("ffn_compute_stage_token_lens", True),
    )
    vector_totals: dict[str, List[int]] = {}
    for field_name, require_full_length in vector_fields:
        vectors = []
        for metadata in metadata_entries:
            vector = getattr(metadata, field_name)
            if type(vector) is not list:
                raise ValueError(
                    "DECODE_FFN source AFD metadata field must be an exact "
                    f"list: field={field_name!r}, got {type(vector).__name__}"
                )
            if require_full_length and len(vector) != num_stages:
                raise ValueError(
                    "DECODE_FFN source AFD metadata stage vector length "
                    f"must equal num_stages: field={field_name!r}, "
                    f"length={len(vector)}, num_stages={num_stages}"
                )
            if len(vector) > num_stages:
                raise ValueError(
                    "DECODE_FFN source AFD metadata stage vector exceeds "
                    f"num_stages: field={field_name!r}, length={len(vector)}, "
                    f"num_stages={num_stages}"
                )
            if any(type(value) is not int or value < 0 for value in vector):
                raise ValueError(
                    "DECODE_FFN source AFD metadata stage vector must contain "
                    f"exact non-negative ints: field={field_name!r}"
                )
            vectors.append(vector)

        max_length = max(len(vector) for vector in vectors)
        aggregate = [
            sum(
                vector[index] if index < len(vector) else 0
                for vector in vectors
            )
            for index in range(max_length)
        ]
        vector_totals[field_name] = aggregate

    scalar_token_fields = (
        "original_total_tokens",
        "padded_total_tokens",
        "ffn_compute_total_tokens",
    )
    scalar_token_totals = {}
    for field_name in scalar_token_fields:
        values = [getattr(metadata, field_name) for metadata in metadata_entries]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError(
                "DECODE_FFN source AFD metadata token totals must be exact "
                f"non-negative ints: field={field_name!r}"
            )
        scalar_token_totals[field_name] = sum(values)

    aggregated = AFDStageMetadata(
        num_stages=num_stages,
        original_total_tokens=scalar_token_totals["original_total_tokens"],
        padded_total_tokens=scalar_token_totals["padded_total_tokens"],
        ffn_compute_total_tokens=scalar_token_totals[
            "ffn_compute_total_tokens"
        ],
        original_stage_token_lens=vector_totals["original_stage_token_lens"],
        padded_stage_token_lens=vector_totals["padded_stage_token_lens"],
        ffn_compute_stage_token_lens=vector_totals[
            "ffn_compute_stage_token_lens"
        ],
        attention_use_cuda_graph=first.attention_use_cuda_graph,
        attention_cudagraph_capture_sizes=(
            list(first.attention_cudagraph_capture_sizes)
            if first.attention_cudagraph_capture_sizes is not None
            else None
        ),
        ffn_use_cuda_graph=first.ffn_use_cuda_graph,
        ffn_cudagraph_capture_sizes=(
            list(first.ffn_cudagraph_capture_sizes)
            if first.ffn_cudagraph_capture_sizes is not None
            else None
        ),
    )
    return aggregated, representation_flags[0]
