import logging
import os
from typing import Iterable, List


logger = logging.getLogger(__name__)


MIXED_ATTENTION_FILE_NAMES = (
    "attention_mixed.csv",
    "attention_true_mixed.csv",
    "attention_combined.csv",
)

MIXED_ATTENTION_KERNEL_ONLY_FILE_NAMES = (
    "attention_mixed_kernel_only.csv",
    "attention_true_mixed_kernel_only.csv",
    "attention_combined_kernel_only.csv",
)

# Keep the supported filename relationships in one declarative table.  Runtime
# consumers use this table to select the combined artifact before falling back
# to the standard artifact, and the kernel-only suffix never crosses into the
# CUDA-event dataset family.
_ATTENTION_FILE_VARIANTS = {
    "attention.csv": MIXED_ATTENTION_FILE_NAMES,
    "attention_kernel_only.csv": MIXED_ATTENTION_KERNEL_ONLY_FILE_NAMES,
}
_STANDARD_TO_COMBINED = {
    standard_name: variant_names[-1]
    for standard_name, variant_names in _ATTENTION_FILE_VARIANTS.items()
}
_COMBINED_TO_STANDARD = {
    combined_name: standard_name
    for standard_name, combined_name in _STANDARD_TO_COMBINED.items()
}
_COMBINED_FILE_NAMES = frozenset(_COMBINED_TO_STANDARD)

TRUE_MIXED_SUPPLEMENT_FILE_NAMES = (
    "attention_true_mixed.csv",
    "attention_true_mixed_kernel_only.csv",
)

# Any of these columns indicates that the configured attention input includes
# mixed-batch profiling metadata.
MIXED_ATTENTION_MARKER_COLUMNS = (
    "is_mixed_batch",
    "is_true_mixed_batch",
    "total_tokens",
)


def _collect_existing_mixed_attention_files(attention_file_path: str) -> List[str]:
    attention_dir = os.path.dirname(os.path.abspath(attention_file_path))
    basename = os.path.basename(attention_file_path)
    variant_names = _ATTENTION_FILE_VARIANTS.get(basename)
    if variant_names is None:
        standard_name = _COMBINED_TO_STANDARD.get(basename)
        variant_names = (
            _ATTENTION_FILE_VARIANTS[standard_name]
            if standard_name is not None
            else MIXED_ATTENTION_FILE_NAMES
        )
    existing_files = []
    for file_name in variant_names:
        candidate = os.path.join(attention_dir, file_name)
        if os.path.exists(candidate):
            existing_files.append(candidate)
    return existing_files


def resolve_attention_input_file(
    attention_file_path: str,
    *,
    require_exists: bool = True,
) -> str:
    """Resolve the effective attention CSV under the combined-first contract.

    Canonical ``attention.csv`` and ``attention_kernel_only.csv`` configuration
    paths are treated as the standard fallback names.  Their corresponding
    combined siblings are selected whenever present.  When only the standard
    file exists, the resolver emits a visible warning and returns it.  Callers
    that initialize paths before loading may set ``require_exists=False``; the
    strict loader path keeps the failure explicit when neither artifact exists.
    Explicit custom filenames remain caller-owned and are returned unchanged.
    """

    if isinstance(attention_file_path, os.PathLike):
        attention_file_path = os.fspath(attention_file_path)
    if not isinstance(attention_file_path, str):
        raise ValueError(
            "attention_file_path must be a non-empty string or path-like value"
        )
    if not attention_file_path:
        if not require_exists:
            return attention_file_path
        raise ValueError(
            "attention_file_path must be a non-empty string or path-like value"
        )

    configured_path = attention_file_path
    basename = os.path.basename(configured_path)
    directory = os.path.dirname(configured_path)

    if basename in _STANDARD_TO_COMBINED:
        combined_path = os.path.join(directory, _STANDARD_TO_COMBINED[basename])
        if os.path.exists(combined_path):
            logger.info(
                "[ATTENTION-DATA] Using combined profiling input %s for configured path %s",
                combined_path,
                configured_path,
            )
            return combined_path
        if os.path.exists(configured_path):
            logger.warning(
                "[ATTENTION-DATA] Combined profiling input %s is unavailable; "
                "falling back to standard attention input %s",
                combined_path,
                configured_path,
            )
            return configured_path
        if require_exists:
            raise FileNotFoundError(
                "Attention profiling input is missing: neither combined nor standard "
                f"file exists (combined={combined_path}, standard={configured_path})"
            )
        return configured_path

    if basename in _COMBINED_TO_STANDARD:
        standard_path = os.path.join(directory, _COMBINED_TO_STANDARD[basename])
        if os.path.exists(configured_path):
            return configured_path
        if os.path.exists(standard_path):
            logger.warning(
                "[ATTENTION-DATA] Configured combined profiling input %s is unavailable; "
                "falling back to standard attention input %s",
                configured_path,
                standard_path,
            )
            return standard_path
        if require_exists:
            raise FileNotFoundError(
                "Attention profiling input is missing: neither combined nor standard "
                f"file exists (combined={configured_path}, standard={standard_path})"
            )
        return configured_path

    if os.path.exists(configured_path) or not require_exists:
        return configured_path
    raise FileNotFoundError(f"Attention profiling input file does not exist: {configured_path}")


def enforce_mixed_attention_input_contract(
    attention_file_path: str, available_columns: Iterable[str]
) -> None:
    """
    Fail fast on attention dataset misconfiguration for mixed profiling data.

    Contract:
    - A selected combined artifact must expose mixed-batch columns.
    - A standard artifact may be used as the explicit fallback when its
      combined sibling is unavailable; the resolver emits the fallback warning.
    - For H800 true-mixed supplement publication, use attention_true_mixed*.csv
      only as the supplement source, then merge those rows into canonical
      attention.csv / attention_kernel_only.csv before training.
    """
    existing_mixed_files = _collect_existing_mixed_attention_files(attention_file_path)
    column_set = set(available_columns)
    has_mixed_columns = any(
        column in column_set for column in MIXED_ATTENTION_MARKER_COLUMNS
    )
    if has_mixed_columns:
        return

    basename = os.path.basename(attention_file_path)
    if basename not in _COMBINED_FILE_NAMES:
        # Standard fallback is intentionally supported.  The path resolver is
        # responsible for warning about the missing combined sibling; this
        # validator only rejects a file that claims to be combined but carries
        # no mixed-batch schema.
        return

    if not existing_mixed_files:
        raise ValueError(
            "Configured combined attention profiling input is missing mixed-batch "
            f"columns: attention_input_file={attention_file_path}. "
            f"required_any_column={list(MIXED_ATTENTION_MARKER_COLUMNS)}"
        )

    attention_dir = os.path.dirname(os.path.abspath(attention_file_path))
    true_mixed_supplements = [
        os.path.join(attention_dir, file_name)
        for file_name in TRUE_MIXED_SUPPLEMENT_FILE_NAMES
    ]
    raise ValueError(
        "Combined attention profiling input does not include mixed-batch columns. "
        f"attention_input_file={attention_file_path}. "
        f"mixed_files={existing_mixed_files}. "
        f"required_any_column={list(MIXED_ATTENTION_MARKER_COLUMNS)}. "
        "Use canonical attention.csv / attention_kernel_only.csv after merging "
        f"true-mixed supplement files={true_mixed_supplements}. "
        "Do not use attention_combined*.csv as the true-mixed supplement source."
    )


__all__ = [
    "MIXED_ATTENTION_FILE_NAMES",
    "MIXED_ATTENTION_KERNEL_ONLY_FILE_NAMES",
    "MIXED_ATTENTION_MARKER_COLUMNS",
    "TRUE_MIXED_SUPPLEMENT_FILE_NAMES",
    "enforce_mixed_attention_input_contract",
    "resolve_attention_input_file",
]
