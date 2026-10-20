from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data" / "profiling" / "compute"
REQUIRED_METADATA_COLUMNS = {
    "profiling_precision",
    "measurement_type",
    "model_arch",
    "model_architecture_profile",
    "quant_signature",
}


DEVICE_EXPECTATIONS = {
    "rtx_pro_6000": {
        "allowed_profiles": {"generic"},
        "directory_profiles": {},
    },
    "h800": {
        "allowed_profiles": {"generic"},
        "directory_profiles": {
            "Step2Mini-tiny": "step2_mini",
            "step-moe-noquant-small": "step3_text",
            "zai-org/GLM-5.3": "glm_moe_dsa",
        },
    },
}


def _csv_files_for_device(device: str) -> list[Path]:
    device_root = DATA_ROOT / device
    assert device_root.exists(), f"Missing profiling dataset directory: {device_root}"
    files = sorted(device_root.rglob("*.csv"))
    assert files, f"No profiling CSV files found under {device_root}"
    return files


def test_release_profiling_datasets_have_required_metadata_columns() -> None:
    missing: list[str] = []
    empty_profile_values: list[str] = []

    for device in DEVICE_EXPECTATIONS:
        for csv_path in _csv_files_for_device(device):
            header = set(pd.read_csv(csv_path, nrows=0).columns)
            missing_columns = sorted(REQUIRED_METADATA_COLUMNS - header)
            if missing_columns:
                missing.append(
                    f"{csv_path.relative_to(REPO_ROOT)} missing {','.join(missing_columns)}"
                )
                continue

            profiles = pd.read_csv(csv_path, usecols=["model_architecture_profile"])[
                "model_architecture_profile"
            ]
            normalized_profiles = profiles.fillna("").astype(str).str.strip()
            if normalized_profiles.empty or any(not value for value in normalized_profiles):
                empty_profile_values.append(str(csv_path.relative_to(REPO_ROOT)))

    assert not missing, "Profiling CSV metadata columns are incomplete:\n" + "\n".join(missing)
    assert not empty_profile_values, (
        "Profiling CSVs contain empty model_architecture_profile values:\n"
        + "\n".join(empty_profile_values)
    )


def test_release_profiling_dataset_profile_values_are_model_consistent() -> None:
    violations: list[str] = []

    for device, expectation in DEVICE_EXPECTATIONS.items():
        directory_profiles = expectation["directory_profiles"]
        allowed_profiles = expectation["allowed_profiles"] | set(directory_profiles.values())
        for csv_path in _csv_files_for_device(device):
            if "model_architecture_profile" not in pd.read_csv(csv_path, nrows=0).columns:
                violations.append(f"{csv_path.relative_to(REPO_ROOT)} missing model_architecture_profile")
                continue

            observed_profiles = sorted(
                set(
                    pd.read_csv(csv_path, usecols=["model_architecture_profile"])[
                        "model_architecture_profile"
                    ]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
            )
            unexpected = sorted(set(observed_profiles) - allowed_profiles)
            if unexpected:
                violations.append(
                    f"{csv_path.relative_to(REPO_ROOT)} unexpected profiles {unexpected}; "
                    f"allowed={sorted(allowed_profiles)}"
                )

            model_dir = csv_path.parent.relative_to(DATA_ROOT / device).as_posix()
            expected_profile = directory_profiles.get(model_dir)
            if expected_profile is not None and observed_profiles != [expected_profile]:
                violations.append(
                    f"{csv_path.relative_to(REPO_ROOT)} expected profile {expected_profile!r}; "
                    f"observed={observed_profiles}"
                )

    assert not violations, "Profiling CSV model_architecture_profile values are inconsistent:\n" + "\n".join(
        violations
    )


def test_non_generic_h800_profiles_pass_predictor_metadata_validation() -> None:
    from frontier.config.model_config import BaseModelConfig
    from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
        SklearnExecutionTimePredictor,
    )

    class _ConcretePredictor(SklearnExecutionTimePredictor):
        def _get_estimator(self):
            raise AssertionError("not used")

        def _get_grid_search_params(self):
            raise AssertionError("not used")

    cases = [
        (
            "Step2Mini-tiny",
            "step2_mini",
            [
                "attention_combined.csv",
                "attention_combined_kernel_only.csv",
            ],
        ),
        (
            "step-moe-noquant-small",
            "step3_text",
            [
                "attention_combined.csv",
                "attention_combined_kernel_only.csv",
            ],
        ),
        (
            "zai-org/GLM-5.3",
            "glm_moe_dsa",
            ["linear_op.csv", "linear_op_kernel_only.csv", "moe.csv", "moe_kernel_only.csv"],
        ),
    ]

    for model_name, expected_profile, filenames in cases:
        predictor = object.__new__(_ConcretePredictor)
        predictor._model_config = BaseModelConfig.create_from_name(model_name)
        assert predictor._model_config.get_model_architecture_profile().profile_id == expected_profile

        for filename in filenames:
            csv_path = DATA_ROOT / "h800" / model_name / filename
            metadata = predictor._get_profiling_metadata(pd.read_csv(csv_path), str(csv_path))

            assert metadata.model_architecture_profile == expected_profile


def test_glm_h800_moe_tables_match_runtime_contract() -> None:
    import yaml

    from frontier.config.model_config import BaseModelConfig
    from frontier.moe_routing_runtime import resolve_moe_gating_routing_runtime_path

    model_root = DATA_ROOT / "h800" / "zai-org/GLM-5.3"
    config = yaml.safe_load((model_root / "moe_config.yaml").read_text())
    profile = BaseModelConfig.create_from_name("zai-org/GLM-5.3").get_model_architecture_profile()
    runtime_path = resolve_moe_gating_routing_runtime_path("balanced", profile)
    assert config["routing_runtime_path"] == runtime_path

    contract = {
        "routing_runtime_path": runtime_path,
        "routing_assignment_policy": "grouped_topk_bias",
        "routing_weight_policy": "sigmoid_renorm_scaled",
        "moe_native_backend": "FLASHINFER_CUTLASS",
        "source_revision": config["source_revision"],
    }
    for filename, measurement in (
        ("moe.csv", "CUDA_EVENT"),
        ("moe_kernel_only.csv", "KERNEL_ONLY"),
    ):
        expected = {**contract, "measurement_type": measurement}
        frame = pd.read_csv(model_root / filename, usecols=list(expected))
        assert not frame.empty
        for column, value in expected.items():
            assert frame[column].notna().all(), (filename, column)
            assert set(frame[column]) == {value}, (filename, column)


def test_glm_h800_dsa_extension_preserves_row_keys_and_runtime_identity() -> None:
    import hashlib

    from frontier.profiling.common.runtime_identity import IDENTITY_COLUMNS

    model_root = DATA_ROOT / "h800" / "zai-org/GLM-5.3"
    manifest = {
        filename: checksum
        for checksum, filename in (
            line.split() for line in (model_root / "SHA256SUMS").read_text().splitlines()
        )
    }
    source_rows = {
        "71cde5ebdee3a8eb5eec3b176b4affed478171fd": 1942,
        "0660717b4d402101ed04ca08f7ba60ccf1f6a116": 1766,
    }
    keys = [
        "measurement_type", "case_id", "attention_variant", "prefill_seq_lens",
        "prefill_kv_cache_sizes", "decode_kv_cache_sizes", "max_num_batched_tokens",
    ]
    for filename, measurement in (
        ("dsa_attention.csv", "CUDA_EVENT"),
        ("dsa_attention_kernel_only.csv", "KERNEL_ONLY"),
    ):
        path = model_root / filename
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[filename]
        frame = pd.read_csv(path)
        assert frame["source_revision"].value_counts().to_dict() == source_rows
        assert set(frame["measurement_type"]) == {measurement}
        assert set(frame["model_architecture_profile"]) == {"glm_moe_dsa"}
        assert frame[list(IDENTITY_COLUMNS)].notna().all().all()
        metadata = [
            name for name in frame if name.startswith(("mla_metadata.", "indexer_metadata."))
        ]
        assert not frame.duplicated(subset=keys + metadata).any()
        shared = frame["attention_variant"] == "shared_index"
        assert frame.loc[shared, "time_stats.attn_dsa_indexer.median"].isna().all()
        assert frame.loc[~shared, "time_stats.attn_dsa_indexer.median"].gt(0).all()
        assert frame["time_stats.attn_dsa_mla.median"].gt(0).all()
