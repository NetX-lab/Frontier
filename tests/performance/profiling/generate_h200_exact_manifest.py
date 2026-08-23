"""Generate the frozen six-model H200 profiling manifest.

The manifest records the exact module-native workload axes, canonical MoE
gating contexts, logical row counts, physical output files, and the active
H200 coverage subtraction used to decide what the GPU worker must collect.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from frontier.moe_gating_runtime import (
    DIRECT_MOE_GATING_RUNTIME_CONTEXT,
    PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT,
    should_enable_prefill_warmed_moe_gating_contract,
)
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.utils import (
    get_attention_input_combinations,
    get_num_tokens_to_profile,
    get_true_mixed_attention_input_combinations,
)
from tests.performance.profiling.validate_h200_six_model_non_dummy_e2e import (
    SUPPORTED_MODELS,
    build_model_contract,
)

DEVICE = "h200"
PROFILE_METHODS = {
    "cuda_event": "CUDA_EVENT",
    "record_function": "KERNEL_ONLY",
}
TP_SIZES = (1, 2, 4, 8)
EP_SIZES = (1, 2, 4, 8)
ATTENTION_MAX_SEQ_LEN = 128
ATTENTION_MAX_MODEL_LEN = 128
LINEAR_MAX_TOKENS = 128
MOE_MAX_TOKENS = 64
MOE_LOAD_DISTRIBUTIONS = ("uniform",)
MOE_SAMPLE_SEEDS = (0, 1, 2)
MOE_ROUTING_RUNTIME_PATH = "standard_fused_topk"


def _git_output(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _profile_filename(module: str, profile_method: str) -> str:
    suffix = "_kernel_only" if profile_method == "record_function" else ""
    return f"{module}{suffix}.csv"


def _attention_files(profile_method: str) -> dict[str, str]:
    suffix = "_kernel_only" if profile_method == "record_function" else ""
    return {
        "standard": f"attention{suffix}.csv",
        "true_mixed": f"attention_true_mixed{suffix}.csv",
        "combined_derived": f"attention_combined{suffix}.csv",
    }


def _standard_attention_tuple(row: Any) -> dict[str, Any]:
    return {
        "prefill_chunk_size": int(row.prefill_chunk_size),
        "kv_cache_size": int(row.kv_cache_size),
        "batch_size": int(row.batch_size),
        "is_prefill": bool(row.is_prefill),
    }


def _true_mixed_attention_tuple(row: Any) -> dict[str, Any]:
    return {
        "prefill_seq_lens": [int(value) for value in row.prefill_seq_lens],
        "prefill_kv_cache_sizes": [
            int(value) for value in row.prefill_kv_cache_sizes
        ],
        "decode_kv_cache_sizes": [
            int(value) for value in row.decode_kv_cache_sizes
        ],
    }


def _gating_contexts(model_config: ModelConfig) -> list[str]:
    contexts = [DIRECT_MOE_GATING_RUNTIME_CONTEXT]
    if should_enable_prefill_warmed_moe_gating_contract(
        model_config=model_config
    ):
        contexts.append(PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT)
    return contexts


def _expected_output_files(
    *,
    active_root: Path,
    model_name: str,
    is_moe: bool,
) -> list[str]:
    model_root = active_root / DEVICE / model_name
    filenames = [
        "attention.csv",
        "attention_true_mixed.csv",
        "attention_combined.csv",
        "attention_kernel_only.csv",
        "attention_true_mixed_kernel_only.csv",
        "attention_combined_kernel_only.csv",
        "linear_op.csv",
        "linear_op_kernel_only.csv",
    ]
    if is_moe:
        filenames.extend(("moe.csv", "moe_kernel_only.csv"))
    return [str(model_root / filename) for filename in filenames]


def generate_manifest(
    *,
    repo_root: Path,
    active_root: Path,
    stable_dedup_commit: str,
) -> dict[str, Any]:
    source_commit = _git_output(repo_root, "rev-parse", "HEAD")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", stable_dedup_commit, source_commit],
        cwd=repo_root,
        check=True,
    )

    standard_inputs = get_attention_input_combinations(
        max_seq_len=ATTENTION_MAX_SEQ_LEN,
        min_batch_size=1,
        max_batch_size=8,
        profile_only_prefill=False,
        profile_only_decode=False,
        batch_size_list=None,
        decode_kv_cache_size_list=None,
        enable_chunked_prefill_grid_search=True,
        fixed_chunked_prefill_size=-1,
        max_model_len=ATTENTION_MAX_MODEL_LEN,
    )
    true_mixed_inputs = get_true_mixed_attention_input_combinations(
        max_seq_len=ATTENTION_MAX_SEQ_LEN,
        max_model_len=ATTENTION_MAX_MODEL_LEN,
        prefill_batch_sizes=[1, 2, 4],
        prefill_chunk_sizes=None,
        decode_batch_sizes=[1, 2, 4, 8],
        decode_kv_cache_sizes=None,
        prefill_kv_cache_size=0,
    )
    linear_tokens = get_num_tokens_to_profile(LINEAR_MAX_TOKENS)
    moe_tokens = get_num_tokens_to_profile(MOE_MAX_TOKENS)

    standard_tuples = [_standard_attention_tuple(row) for row in standard_inputs]
    true_mixed_tuples = [
        _true_mixed_attention_tuple(row) for row in true_mixed_inputs
    ]
    if len({json.dumps(row, sort_keys=True) for row in standard_tuples}) != len(
        standard_tuples
    ):
        raise ValueError("Standard attention generator emitted duplicate workloads.")
    if len({json.dumps(row, sort_keys=True) for row in true_mixed_tuples}) != len(
        true_mixed_tuples
    ):
        raise ValueError("True-mixed attention generator emitted duplicate workloads.")

    expected_files: list[str] = []
    model_summaries: dict[str, Any] = {}
    collection_entries: list[dict[str, Any]] = []
    module_rows = {"attention": 0, "linear_op": 0, "moe": 0}

    for model_name in SUPPORTED_MODELS:
        model_config = ModelConfig.from_model_name(model_name)
        contract = build_model_contract(model_name)
        contexts = _gating_contexts(model_config) if contract.is_moe else []
        model_files = _expected_output_files(
            active_root=active_root,
            model_name=model_name,
            is_moe=contract.is_moe,
        )
        expected_files.extend(model_files)

        per_model_rows = {"attention": 0, "linear_op": 0, "moe": 0}
        for profile_method, measurement_type in PROFILE_METHODS.items():
            for tp_size in TP_SIZES:
                attention_rows = len(standard_tuples) + len(true_mixed_tuples)
                collection_entries.append(
                    {
                        "model": model_name,
                        "module": "attention",
                        "profile_method": profile_method,
                        "measurement_type": measurement_type,
                        "tp": tp_size,
                        "ep": None,
                        "routing_runtime_path": None,
                        "gating_runtime_context": None,
                        "required_partition_rows": {
                            "standard": len(standard_tuples),
                            "true_mixed": len(true_mixed_tuples),
                        },
                        "required_logical_rows": attention_rows,
                        "required_physical_rows_including_combined": (
                            attention_rows * 2
                        ),
                        "output_files": _attention_files(profile_method),
                    }
                )
                per_model_rows["attention"] += attention_rows
                module_rows["attention"] += attention_rows

                collection_entries.append(
                    {
                        "model": model_name,
                        "module": "linear_op",
                        "profile_method": profile_method,
                        "measurement_type": measurement_type,
                        "tp": tp_size,
                        "ep": None,
                        "routing_runtime_path": None,
                        "gating_runtime_context": None,
                        "required_logical_rows": len(linear_tokens),
                        "output_file": _profile_filename(
                            "linear_op", profile_method
                        ),
                    }
                )
                per_model_rows["linear_op"] += len(linear_tokens)
                module_rows["linear_op"] += len(linear_tokens)

                for context in contexts:
                    for ep_size in EP_SIZES:
                        moe_rows = (
                            len(moe_tokens)
                            * len(MOE_LOAD_DISTRIBUTIONS)
                            * len(MOE_SAMPLE_SEEDS)
                        )
                        collection_entries.append(
                            {
                                "model": model_name,
                                "module": "moe",
                                "profile_method": profile_method,
                                "measurement_type": measurement_type,
                                "tp": tp_size,
                                "ep": ep_size,
                                "routing_runtime_path": (
                                    MOE_ROUTING_RUNTIME_PATH
                                ),
                                "gating_runtime_context": context,
                                "required_logical_rows": moe_rows,
                                "output_file": _profile_filename(
                                    "moe", profile_method
                                ),
                            }
                        )
                        per_model_rows["moe"] += moe_rows
                        module_rows["moe"] += moe_rows

        model_summaries[model_name] = {
            "kind": "moe" if contract.is_moe else "dense",
            "model_type": str(model_config.model_type),
            "model_arch": str(model_config.model_arch),
            "model_architecture_profile": (
                model_config.get_model_architecture_profile().profile_id
            ),
            "profiling_precision": contract.profiling_precision,
            "quant_signature": contract.quant_signature,
            "linear_target_column_count": len(contract.linear_target_columns),
            "moe_target_column_count": len(contract.moe_target_columns),
            "moe_experts": int(model_config.num_experts) if contract.is_moe else None,
            "gating_runtime_contexts": contexts,
            "required_logical_rows": per_model_rows,
            "required_logical_row_total": sum(per_model_rows.values()),
            "expected_output_files": model_files,
        }

    matching_device_dirs = sorted(
        str(path)
        for path in active_root.iterdir()
        if path.is_dir() and path.name.lower() == DEVICE
    ) if active_root.is_dir() else []
    existing_files = sorted(path for path in expected_files if Path(path).is_file())
    missing_files = sorted(path for path in expected_files if not Path(path).is_file())
    if existing_files:
        raise ValueError(
            "Existing canonical H200 files require row-level coverage subtraction "
            "before this all-missing manifest can be frozen: "
            f"{existing_files}"
        )

    total_logical_rows = sum(module_rows.values())
    expected_module_rows = {
        "attention": 3_408,
        "linear_op": 912,
        "moe": 6_336,
    }
    if module_rows != expected_module_rows:
        raise ValueError(
            f"Unexpected H200 module row totals: {module_rows}; "
            f"expected {expected_module_rows}."
        )
    if total_logical_rows != 10_656:
        raise ValueError(
            f"Unexpected H200 logical row total: {total_logical_rows}; "
            "expected 10656."
        )
    if len(collection_entries) != 288:
        raise ValueError(
            f"Unexpected collection entry count: {len(collection_entries)}; "
            "expected 288."
        )

    attention_physical_rows = module_rows["attention"] * 2
    physical_rows = (
        attention_physical_rows + module_rows["linear_op"] + module_rows["moe"]
    )
    return {
        "schema_version": 3,
        "generated_date": "2026-08-23",
        "status": "frozen_ready_for_h200_collection",
        "source_commit": source_commit,
        "stable_attention_dedup_commit": stable_dedup_commit,
        "supersedes": "h200_exact_manifest_candidates_v2.json",
        "scope": {
            "device": DEVICE,
            "models": list(SUPPORTED_MODELS),
            "profile_methods": PROFILE_METHODS,
            "tp_sizes": list(TP_SIZES),
            "attention": {
                "max_seq_len": ATTENTION_MAX_SEQ_LEN,
                "max_model_len": ATTENTION_MAX_MODEL_LEN,
                "min_batch_size": 1,
                "max_batch_size": 8,
                "batch_size_list": None,
                "decode_kv_cache_size_list": None,
                "enable_chunked_prefill_grid_search": True,
                "fixed_chunked_prefill_size": -1,
                "enable_true_mixed": True,
                "standard_workloads": standard_tuples,
                "true_mixed_workloads": true_mixed_tuples,
            },
            "linear_op": {
                "max_tokens": LINEAR_MAX_TOKENS,
                "num_tokens": linear_tokens,
            },
            "moe": {
                "max_tokens": MOE_MAX_TOKENS,
                "num_tokens": moe_tokens,
                "expert_parallel_sizes": list(EP_SIZES),
                "load_distributions": list(MOE_LOAD_DISTRIBUTIONS),
                "sample_seeds": list(MOE_SAMPLE_SEEDS),
                "routing_runtime_paths": [MOE_ROUTING_RUNTIME_PATH],
                "canonical_gating_runtime_contexts": [
                    DIRECT_MOE_GATING_RUNTIME_CONTEXT,
                    PREFILL_WARMED_MOE_GATING_RUNTIME_CONTEXT,
                ],
            },
        },
        "counting_contract": {
            "attention_standard_rows_per_model_method_tp": len(standard_tuples),
            "attention_true_mixed_rows_per_model_method_tp": len(
                true_mixed_tuples
            ),
            "attention_logical_rows_per_model_method_tp": (
                len(standard_tuples) + len(true_mixed_tuples)
            ),
            "attention_combined_file_relationship": (
                "Derived row copy of standard plus true-mixed partitions; "
                "excluded from logical row totals."
            ),
            "module_logical_rows": module_rows,
            "total_logical_rows": total_logical_rows,
            "total_physical_rows_including_attention_combined": physical_rows,
            "collection_entry_count": len(collection_entries),
        },
        "models": model_summaries,
        "collection_entries": collection_entries,
        "canonical_coverage_audit": {
            "active_root": str(active_root),
            "matching_device_directories": matching_device_dirs,
            "expected_file_count": len(expected_files),
            "existing_expected_file_count": len(existing_files),
            "missing_expected_file_count": len(missing_files),
            "existing_expected_files": existing_files,
            "missing_expected_files": missing_files,
            "existing_logical_rows": 0,
            "missing_logical_rows": total_logical_rows,
            "subtraction_result": (
                "No canonical H200 profiling files exist; every frozen workload "
                "is missing and must be collected."
            ),
        },
        "evidence": [
            {
                "artifact": "frontier/profiling/utils/__init__.py",
                "observed": (
                    f"Current module-native generators emit {len(standard_tuples)} "
                    "unique standard attention workloads, "
                    f"{len(true_mixed_tuples)} unique true-mixed workloads, "
                    f"{len(linear_tokens)} linear token points, and "
                    f"{len(moe_tokens)} MoE token points."
                ),
            },
            {
                "artifact": "frontier/moe_gating_runtime.py",
                "observed": (
                    "New collection uses only canonical direct and "
                    "prefill_warmed labels; legacy aliases remain consumer-only."
                ),
            },
            {
                "artifact": str(active_root / DEVICE),
                "observed": (
                    "Canonical coverage audit found zero existing expected H200 "
                    "CSV files."
                ),
            },
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the frozen six-model H200 profiling manifest."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Manifest JSON output path.",
    )
    parser.add_argument(
        "--active-root",
        type=Path,
        default=Path("data/profiling/compute"),
        help="Canonical compute profiling root.",
    )
    parser.add_argument(
        "--stable-dedup-commit",
        default="08b70647d737368ee47b7147a6084c01f7cba60f",
        help="Required stable attention dedup commit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    active_root = (
        args.active_root
        if args.active_root.is_absolute()
        else (repo_root / args.active_root)
    ).resolve()
    output_path = (
        args.output if args.output.is_absolute() else (repo_root / args.output)
    ).resolve()
    manifest = generate_manifest(
        repo_root=repo_root,
        active_root=active_root,
        stable_dedup_commit=args.stable_dedup_commit,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "source_commit": manifest["source_commit"],
                "module_logical_rows": manifest["counting_contract"][
                    "module_logical_rows"
                ],
                "total_logical_rows": manifest["counting_contract"][
                    "total_logical_rows"
                ],
                "collection_entry_count": manifest["counting_contract"][
                    "collection_entry_count"
                ],
                "existing_expected_file_count": manifest[
                    "canonical_coverage_audit"
                ]["existing_expected_file_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
