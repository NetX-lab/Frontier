import os
from typing import Iterable, List


MIXED_ATTENTION_FILE_NAMES = (
    "attention_mixed.csv",
    "attention_true_mixed.csv",
    "attention_combined.csv",
)

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
    existing_files = []
    for file_name in MIXED_ATTENTION_FILE_NAMES:
        candidate = os.path.join(attention_dir, file_name)
        if os.path.exists(candidate):
            existing_files.append(candidate)
    return existing_files


def enforce_mixed_attention_input_contract(
    attention_file_path: str, available_columns: Iterable[str]
) -> None:
    """
    Fail fast on attention dataset misconfiguration for mixed profiling data.

    Contract:
    - If mixed profiling artifacts exist in the same directory (attention_mixed /
      attention_true_mixed / attention_combined), the configured attention input
      must expose mixed-batch columns. Otherwise the run would silently ignore
      mixed profiling coverage.
    - For H800 true-mixed supplement publication, use attention_true_mixed*.csv
      only as the supplement source, then merge those rows into canonical
      attention.csv / attention_kernel_only.csv before training.
    """
    existing_mixed_files = _collect_existing_mixed_attention_files(attention_file_path)
    if not existing_mixed_files:
        return

    column_set = set(available_columns)
    has_mixed_columns = any(
        column in column_set for column in MIXED_ATTENTION_MARKER_COLUMNS
    )
    if has_mixed_columns:
        return

    attention_dir = os.path.dirname(os.path.abspath(attention_file_path))
    true_mixed_supplements = [
        os.path.join(attention_dir, file_name)
        for file_name in TRUE_MIXED_SUPPLEMENT_FILE_NAMES
    ]
    raise ValueError(
        "Mixed attention profiling files detected but the configured attention input "
        "does not include mixed-batch columns. "
        f"attention_input_file={attention_file_path}. "
        f"mixed_files={existing_mixed_files}. "
        f"required_any_column={list(MIXED_ATTENTION_MARKER_COLUMNS)}. "
        "Use canonical attention.csv / attention_kernel_only.csv after merging "
        f"true-mixed supplement files={true_mixed_supplements}. "
        "Do not use attention_combined*.csv as the true-mixed supplement source."
    )
