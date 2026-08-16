from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from frontier.profiling.attention.main import _required_max_num_blocks
from frontier.profiling.attention.provenance import validate_attention_run_sidecar


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "performance"
    / "measure_cache_save_held_out_points.py"
)
FINALIZER_PATH = (
    Path(__file__).resolve().parents[1]
    / "performance"
    / "finalize_cache_save_high_token_measurement.py"
)


def _load_measurement_module():
    spec = importlib.util.spec_from_file_location(
        "measure_cache_save_held_out_points",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load measurement script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_finalizer_module():
    if not FINALIZER_PATH.is_file():
        pytest.fail(f"Missing cache-save measurement finalizer: {FINALIZER_PATH}")
    spec = importlib.util.spec_from_file_location(
        "finalize_cache_save_high_token_measurement",
        FINALIZER_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load finalizer script: {FINALIZER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_measurement_profile_identity_is_complete():
    module = _load_measurement_module()
    model_config = module.ModelConfig.from_model_name(module.MODEL_NAME)

    assert module._measurement_profile_identity(  # pylint: disable=protected-access
        model_config=model_config,
        precision="BF16",
    ) == {
        "attention_backend": "FLASHINFER",
        "precision": "BF16",
        "quant_signature": "none",
        "model_architecture_profile": "generic",
    }


def test_high_token_training_point_set_is_bounded_and_keeps_held_out_key_out():
    module = _load_measurement_module()

    profile_inputs = module._profile_inputs(module.HIGH_TOKEN_TRAINING_POINT_SET)
    observed = [
        (
            scenario,
            int(profile_input.to_dict()["total_tokens"]),
            int(profile_input.to_dict()["kv_cache_size"]),
            int(profile_input.to_dict()["batch_size"]),
        )
        for scenario, profile_input in profile_inputs
    ]

    assert observed == [
        ("batch4_seq3072", 12288, 0, 4),
        ("batch4_seq3584", 14336, 0, 4),
        ("batch4_seq3840", 15360, 0, 4),
        ("batch4_seq4032", 16128, 0, 4),
    ]
    assert (16380, 0, 4) not in {
        (total_tokens, kv_cache_size, batch_size)
        for _, total_tokens, kv_cache_size, batch_size in observed
    }
    assert module._expected_keys(module.HIGH_TOKEN_TRAINING_POINT_SET) == (
        (12288, 0, 4),
        (14336, 0, 4),
        (15360, 0, 4),
        (16128, 0, 4),
    )
    assert (
        _required_max_num_blocks(
            [profile_input for _, profile_input in profile_inputs],
            block_size=module.BLOCK_SIZE,
        )
        == 1008
    )
    assert (
        module._measurement_schema_version(module.HIGH_TOKEN_TRAINING_POINT_SET)
        == "frontier.attention.cache_save_high_token_training_measurement/v1"
    )
    assert (
        module._measurement_output_stem(module.HIGH_TOKEN_TRAINING_POINT_SET)
        == "attention_cache_save_high_token_training"
    )


def test_mixed_prefill_tail_point_set_is_minimal_and_keeps_held_out_key_out():
    module = _load_measurement_module()

    profile_inputs = module._profile_inputs(
        module.MIXED_PREFILL_TAIL_TRAINING_POINT_SET
    )
    observed = [
        (
            scenario,
            int(profile_input.to_dict()["total_tokens"]),
            int(profile_input.to_dict()["kv_cache_size"]),
            int(profile_input.to_dict()["batch_size"]),
        )
        for scenario, profile_input in profile_inputs
    ]

    assert observed == [
        ("batch4_seq4056", 16224, 0, 4),
        ("batch4_seq4072", 16288, 0, 4),
        ("batch4_seq4088", 16352, 0, 4),
    ]
    assert (16380, 0, 4) not in {
        (total_tokens, kv_cache_size, batch_size)
        for _, total_tokens, kv_cache_size, batch_size in observed
    }
    assert module._expected_keys(
        module.MIXED_PREFILL_TAIL_TRAINING_POINT_SET
    ) == (
        (16224, 0, 4),
        (16288, 0, 4),
        (16352, 0, 4),
    )
    assert (
        _required_max_num_blocks(
            [profile_input for _, profile_input in profile_inputs],
            block_size=module.BLOCK_SIZE,
        )
        == 1024
    )
    assert (
        module._measurement_schema_version(
            module.MIXED_PREFILL_TAIL_TRAINING_POINT_SET
        )
        == "frontier.attention.mixed_prefill_tail_training_measurement/v1"
    )
    assert (
        module._measurement_output_stem(
            module.MIXED_PREFILL_TAIL_TRAINING_POINT_SET
        )
        == "attention_mixed_prefill_tail_training"
    )


def _write_measurement_fixture(
    root: Path,
    *,
    schema_version: str = (
        "frontier.attention.cache_save_high_token_training_measurement/v1"
    ),
    point_set: str = "high-token-training",
    output_stem: str = "attention_cache_save_high_token_training",
) -> tuple[Path, Path]:
    rows = []
    allocations = {}
    requested_tuples = []
    for tp_size in (1, 2, 4, 8):
        physical = 2000 * tp_size
        allocations[str(tp_size)] = {
            "physical": physical,
            "requested": 1008,
            "required": 1008,
            "selected": 1008,
            "workspace_bytes": 4096,
        }
        requested_tuples.append([tp_size, 12288, 0, 4])
        rows.append(
            {
                "num_tensor_parallel_workers": tp_size,
                "prefill_chunk_size": 0,
                "kv_cache_size": 0,
                "batch_size": 4,
                "seq_lens": "[3072,3072,3072,3072]",
                "is_prefill": True,
                "is_mixed_batch": True,
                "total_tokens": 12288,
                "mode": "even",
                "scenario": "batch4_seq3072",
                "physical_max_num_blocks": physical,
                "requested_max_num_blocks": 1008,
                "required_max_num_blocks": 1008,
                "selected_max_num_blocks": 1008,
                "allocated_max_num_blocks": 1008,
                "allocated_kv_token_capacity": 16128,
                "block_size": 16,
                "backend_workspace_reservation_bytes": 4096,
                "profile_input_grid_max_seq_len": 4095,
                "is_native_profile_allocation": True,
                "measurement_type": "CUDA_EVENT",
                "attention_backend": "FLASHINFER",
                "profiling_precision": "BF16",
                "quant_signature": "none",
                "model_architecture_profile": "generic",
                "time_stats.attn_kv_cache_save.count": 5,
                "time_stats.attn_kv_cache_save.median": 0.1,
            }
        )
    csv_path = root / f"{output_stem}.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    requested_digest = hashlib.sha256(
        json.dumps(
            requested_tuples,
            separators=(",", ":"),
            sort_keys=False,
        ).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": schema_version,
        "point_set": point_set,
        "run_id": "unit-high-token",
        "command": ["measure_cache_save_held_out_points.py"],
        "model": "Qwen3-30B-A3B-tiny",
        "device": "h800",
        "tp_sizes": [1, 2, 4, 8],
        "measurement_type": "CUDA_EVENT",
        "profile_method": "cuda_event",
        "attention_backend": "FLASHINFER",
        "precision": "BF16",
        "quant_signature": "none",
        "model_architecture_profile": "generic",
        "max_model_len": 4096,
        "profile_max_seq_len": 4095,
        "block_size": 16,
        "requested_tuples": requested_tuples,
        "requested_tuple_sha256": requested_digest,
        "allocations_by_tp": allocations,
        "csv": csv_path.name,
        "csv_rows": len(rows),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "environment": {"hostname": "unit"},
    }
    sidecar_path = root / f"{output_stem}.provenance.json"
    sidecar_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return csv_path, sidecar_path


def test_high_token_measurement_finalizer_publishes_native_run_sidecar(tmp_path):
    module = _load_finalizer_module()
    csv_path, measurement_sidecar = _write_measurement_fixture(tmp_path)
    output_dir = tmp_path / "formal"

    published = module.finalize_measurement(
        measurement_csv=csv_path,
        measurement_sidecar=measurement_sidecar,
        output_dir=output_dir,
    )

    canonical = Path(published["canonical"])
    alias = Path(published["alias"])
    sidecar = Path(published["sidecar"])
    run_csv = Path(published["run_csv"])
    assert canonical.read_bytes() == alias.read_bytes() == run_csv.read_bytes()
    output_rows = pd.read_csv(canonical)
    assert len(output_rows) == 4
    assert "scenario" not in output_rows.columns
    assert output_rows["is_mixed_batch"].all()
    assert not output_rows["is_true_mixed_batch"].any()
    validate_attention_run_sidecar(csv_path=canonical, sidecar_path=sidecar)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["tensor_parallel_sizes"] == [1, 2, 4, 8]
    assert sorted(payload["allocation_by_tp"]) == ["1", "2", "4", "8"]
    assert payload["source_measurement_csv_sha256"] == hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()
    assert payload["source_measurement_sidecar_sha256"] == hashlib.sha256(
        measurement_sidecar.read_bytes()
    ).hexdigest()
    assert payload["profiling_precision"] == "BF16"
    assert payload["quant_signature"] == "none"
    assert payload["model_architecture_profile"] == "generic"
    assert payload["attention_backend"] == "FLASHINFER"


def test_tail_measurement_finalizer_publishes_native_run_sidecar(tmp_path):
    module = _load_finalizer_module()
    csv_path, measurement_sidecar = _write_measurement_fixture(
        tmp_path,
        schema_version=(
            "frontier.attention.mixed_prefill_tail_training_measurement/v1"
        ),
        point_set="mixed-prefill-tail-training",
        output_stem="attention_mixed_prefill_tail_training",
    )
    output_dir = tmp_path / "formal"

    published = module.finalize_measurement(
        measurement_csv=csv_path,
        measurement_sidecar=measurement_sidecar,
        output_dir=output_dir,
    )

    canonical = Path(published["canonical"])
    sidecar = Path(published["sidecar"])
    assert len(pd.read_csv(canonical)) == 4
    validate_attention_run_sidecar(csv_path=canonical, sidecar_path=sidecar)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["source_measurement_schema_version"] == (
        "frontier.attention.mixed_prefill_tail_training_measurement/v1"
    )


def test_high_token_measurement_finalizer_rejects_source_digest_drift(tmp_path):
    module = _load_finalizer_module()
    csv_path, measurement_sidecar = _write_measurement_fixture(tmp_path)
    payload = json.loads(measurement_sidecar.read_text(encoding="utf-8"))
    payload["csv_sha256"] = "0" * 64
    measurement_sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "formal"

    with pytest.raises(ValueError, match="measurement csv_sha256 mismatch"):
        module.finalize_measurement(
            measurement_csv=csv_path,
            measurement_sidecar=measurement_sidecar,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("sidecar_field", "csv_field", "conflicting_value"),
    [
        ("precision", "profiling_precision", "FP16"),
        ("quant_signature", "quant_signature", "fp8_w8a8"),
        (
            "model_architecture_profile",
            "model_architecture_profile",
            "step3_text",
        ),
        ("attention_backend", "attention_backend", "FLASH_ATTN"),
    ],
)
def test_high_token_measurement_finalizer_rejects_profile_identity_drift(
    tmp_path,
    sidecar_field: str,
    csv_field: str,
    conflicting_value: str,
):
    module = _load_finalizer_module()
    csv_path, measurement_sidecar = _write_measurement_fixture(tmp_path)
    payload = json.loads(measurement_sidecar.read_text(encoding="utf-8"))
    payload[sidecar_field] = conflicting_value
    measurement_sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "formal"

    with pytest.raises(
        ValueError,
        match=rf"measurement profile identity mismatch.*{csv_field}",
    ):
        module.finalize_measurement(
            measurement_csv=csv_path,
            measurement_sidecar=measurement_sidecar,
            output_dir=output_dir,
        )

    assert not output_dir.exists()
