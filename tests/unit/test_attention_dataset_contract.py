from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.execution_time_predictor.attention_dataset_contract import (
    enforce_mixed_attention_input_contract,
    resolve_attention_input_file,
)
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)
from frontier.config.model_config import BaseModelConfig
from frontier.profiling.common.model_config import ModelConfig
from frontier.profiling.linear_op.profiling_plan import build_profiling_plan
from frontier.profiling.linear_op import main as linear_op_main
from frontier.training.attention_trainer import AttentionTrainer
from frontier.types import ClusterType, MeasurementType


class _ConcreteSklearnExecutionTimePredictor(SklearnExecutionTimePredictor):
    def _get_estimator(self):
        return None

    def _get_grid_search_params(self):
        return {}


def test_mixed_attention_contract_warns_for_standard_fallback(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    attention_file = tmp_path / "attention.csv"
    attention_file.write_text("standard\n", encoding="utf-8")
    (tmp_path / "attention_true_mixed.csv").write_text("header\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        resolved = resolve_attention_input_file(str(attention_file))
        enforce_mixed_attention_input_contract(
            str(attention_file),
            available_columns=("is_decode",),
        )

    assert resolved == str(attention_file)
    assert "falling back" in caplog.text
    assert "attention_combined.csv" in caplog.text


def test_resolve_attention_input_prefers_combined_sibling(tmp_path: Path) -> None:
    standard = tmp_path / "attention.csv"
    combined = tmp_path / "attention_combined.csv"
    standard.write_text("standard\n", encoding="utf-8")
    combined.write_text("combined\n", encoding="utf-8")

    assert resolve_attention_input_file(str(standard)) == str(combined)


def test_resolve_attention_input_warns_and_falls_back_to_standard(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    standard = tmp_path / "attention.csv"
    standard.write_text("standard\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        resolved = resolve_attention_input_file(str(standard))

    assert resolved == str(standard)
    assert "falling back" in caplog.text
    assert "attention_combined.csv" in caplog.text


def test_resolve_attention_input_fails_when_combined_and_standard_are_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="combined.*standard"):
        resolve_attention_input_file(str(tmp_path / "attention.csv"))


def test_resolve_attention_input_preserves_empty_non_strict_initialization_path() -> None:
    """Initialization may defer an intentionally disabled attention family."""

    assert resolve_attention_input_file("", require_exists=False) == ""


def test_resolve_attention_input_rejects_empty_strict_loader_path() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        resolve_attention_input_file("", require_exists=True)


def _empty_attention_consumer_config() -> SimpleNamespace:
    return SimpleNamespace(
        linear_op_input_file="linear.csv",
        mlp_input_file="",
        atten_input_file="",
        moe_input_file="moe.csv",
        linear_op_kernel_only_input_file="linear_kernel.csv",
        atten_kernel_only_input_file="",
        moe_kernel_only_input_file="moe_kernel.csv",
        all_reduce_input_file="all.csv",
        send_recv_input_file="send.csv",
        cpu_overhead_input_file="cpu.csv",
    )


def _empty_attention_replica_config() -> SimpleNamespace:
    model_config = SimpleNamespace(get_name=lambda: "fixture")
    return SimpleNamespace(
        device="h200",
        network_device="ib",
        model_config=model_config,
    )


def test_predictor_initialization_keeps_empty_attention_paths() -> None:
    predictor = object.__new__(_ConcreteSklearnExecutionTimePredictor)
    predictor._config = _empty_attention_consumer_config()
    predictor._replica_config = _empty_attention_replica_config()
    predictor._model_config = predictor._replica_config.model_config

    assert predictor._get_input_files(MeasurementType.CUDA_EVENT)[1] == ""
    assert predictor._get_input_files(MeasurementType.KERNEL_ONLY)[1] == ""


def test_shared_manager_initialization_keeps_empty_attention_path() -> None:
    manager = object.__new__(ExecutionTimePredictionModelManager)

    files = manager._resolve_measurement_input_files_for_config(
        _empty_attention_replica_config(),
        _empty_attention_consumer_config(),
        MeasurementType.CUDA_EVENT,
    )

    assert files[1] == ""


def test_predictor_training_paths_prefer_combined_attention_inputs(
    tmp_path: Path,
) -> None:
    eager_standard = tmp_path / "attention.csv"
    eager_combined = tmp_path / "attention_combined.csv"
    kernel_standard = tmp_path / "attention_kernel_only.csv"
    kernel_combined = tmp_path / "attention_combined_kernel_only.csv"
    for path in (eager_standard, eager_combined, kernel_standard, kernel_combined):
        path.write_text("placeholder\n", encoding="utf-8")

    predictor = object.__new__(_ConcreteSklearnExecutionTimePredictor)
    predictor._initialize_file_paths(
        {
            "compute_input_file": str(tmp_path / "linear.csv"),
            "attention_input_file": str(eager_standard),
            "moe_input_file": str(tmp_path / "moe.csv"),
            "compute_kernel_only_input_file": str(tmp_path / "linear_kernel.csv"),
            "attention_kernel_only_input_file": str(kernel_standard),
            "moe_kernel_only_input_file": str(tmp_path / "moe_kernel.csv"),
        }
    )

    assert predictor._attention_input_file_eager == str(eager_combined)
    assert predictor._attention_input_file_kernel_only == str(kernel_combined)
    assert predictor._attention_input_file == str(eager_combined)


def test_shared_manager_training_paths_prefer_combined_attention_inputs(
    tmp_path: Path,
) -> None:
    eager_standard = tmp_path / "attention.csv"
    eager_combined = tmp_path / "attention_combined.csv"
    kernel_standard = tmp_path / "attention_kernel_only.csv"
    kernel_combined = tmp_path / "attention_combined_kernel_only.csv"
    for path in (eager_standard, eager_combined, kernel_standard, kernel_combined):
        path.write_text("placeholder\n", encoding="utf-8")

    model_config = SimpleNamespace(get_name=lambda: "fixture")
    replica_config = SimpleNamespace(
        device="h200",
        network_device="ib",
        model_config=model_config,
    )
    predictor_config = SimpleNamespace(
        linear_op_input_file=str(tmp_path / "linear.csv"),
        mlp_input_file="",
        atten_input_file=str(eager_standard),
        moe_input_file=str(tmp_path / "moe.csv"),
        all_reduce_input_file=str(tmp_path / "all_reduce.csv"),
        send_recv_input_file=str(tmp_path / "send_recv.csv"),
        cpu_overhead_input_file=str(tmp_path / "cpu.csv"),
        cpu_overhead_kernel_only_input_file=str(tmp_path / "cpu_kernel.csv"),
        pp_stage_boundary_input_file=str(tmp_path / "pp_boundary.csv"),
        pp_receiver_head_input_file=str(tmp_path / "pp_receiver.csv"),
        pp_producer_send_path_input_file=str(tmp_path / "pp_send.csv"),
        pp_prefill_consumer_active_input_file=str(tmp_path / "pp_active.csv"),
        linear_op_kernel_only_input_file=str(tmp_path / "linear_kernel.csv"),
        atten_kernel_only_input_file=str(kernel_standard),
        moe_kernel_only_input_file=str(tmp_path / "moe_kernel.csv"),
    )
    manager = object.__new__(ExecutionTimePredictionModelManager)
    manager._cluster_configs = {
        ClusterType.MONOLITHIC: SimpleNamespace(
            replica_config=replica_config,
            execution_time_predictor_config=predictor_config,
        )
    }

    paths = manager.get_training_file_paths(ClusterType.MONOLITHIC)

    assert paths["attention_input_file"] == str(eager_combined)
    assert paths["attention_kernel_only_input_file"] == str(kernel_combined)


def test_resolve_attention_input_keeps_kernel_only_suffix_isolated(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    standard = tmp_path / "attention_kernel_only.csv"
    eager_combined = tmp_path / "attention_combined.csv"
    standard.write_text("kernel\n", encoding="utf-8")
    eager_combined.write_text("eager\n", encoding="utf-8")

    with caplog.at_level("WARNING"):
        resolved = resolve_attention_input_file(str(standard))

    assert resolved == str(standard)
    assert "attention_combined_kernel_only.csv" in caplog.text


def test_attention_trainer_cache_identity_includes_profile_and_attention_shape() -> None:
    config_a = BaseModelConfig.create_from_name("step3-moe-noquant")
    config_b = BaseModelConfig.create_from_name("step3-moe-noquant")
    config_b.dense_mlp_hidden_dim = 12288

    trainer_a = AttentionTrainer.__new__(AttentionTrainer)
    trainer_a.model_config = config_a
    trainer_a.model_name = "step3-moe-noquant"
    trainer_a.device = "h200"
    trainer_a.tensor_parallel_size = 8
    trainer_a.block_size = 16

    trainer_b = AttentionTrainer.__new__(AttentionTrainer)
    trainer_b.model_config = config_b
    trainer_b.model_name = "step3-moe-noquant"
    trainer_b.device = "h200"
    trainer_b.tensor_parallel_size = 8
    trainer_b.block_size = 16

    assert trainer_a._get_model_hash_identity("attn_prefill") != trainer_b._get_model_hash_identity(
        "attn_prefill"
    )

    trainer_b.tensor_parallel_size = 4
    assert trainer_a._get_model_hash_identity("attn_prefill") != trainer_b._get_model_hash_identity(
        "attn_prefill"
    )


def test_attention_compute_loader_keeps_typed_rows_across_mixed_ffn_widths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attention rows do not become invalid when a mixed model has two FFN widths."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[1],
        is_moe=False,
    )
    attention_contract = plan["typed_operator_contracts"]["attn_pre_proj"]

    rows = []
    for width, median in ((5120, 1.0), (18432, 2.0)):
        row = {
            "n_head": config.num_q_heads,
            "n_kv_head": config.num_kv_heads,
            "n_embd": config.embedding_dim,
            "n_expanded_embd": width,
            "use_gated_mlp": config.use_gated_mlp,
            "vocab_size": config.vocab_size,
            "num_tensor_parallel_workers": 8,
            "use_qk_norm": config.use_qk_norm,
            "time_stats.attn_pre_proj.median": median,
            "typed_operator_contracts": {
                "attn_pre_proj": attention_contract,
            },
        }
        rows.append(row)

    input_file = tmp_path / "linear_op.csv"
    frame = pd.DataFrame(rows)
    serialized = linear_op_main._serialize_linear_op_output(frame)
    serialized.to_csv(input_file, index=False)

    trainer = AttentionTrainer.__new__(AttentionTrainer)
    trainer.train_compute_models = True
    trainer.compute_dataset_path = str(input_file)
    trainer.model_name = "step3-moe-noquant"
    trainer.tensor_parallel_size = 8
    trainer.model_config = SimpleNamespace(
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        embedding_dim=config.embedding_dim,
        mlp_hidden_dim=config.mlp_hidden_dim,
        use_gated_mlp=config.use_gated_mlp,
        vocab_size=config.vocab_size,
        use_qk_norm=config.use_qk_norm,
        get_model_architecture_profile=lambda: config.get_model_architecture_profile(),
    )
    monkeypatch.setattr(trainer, "_set_dataset_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer, "_verify_compute_dataset_columns", lambda _df: None)

    loaded = trainer._load_dataset()

    assert loaded["n_expanded_embd"].tolist() == [5120, 18432]


def test_attention_compute_loader_rejects_malformed_typed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A declared typed column must contain canonical JSON for every row."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    input_file = tmp_path / "linear_op.csv"
    pd.DataFrame(
        [
            {
                "n_head": config.num_q_heads,
                "n_kv_head": config.num_kv_heads,
                "n_embd": config.embedding_dim,
                "n_expanded_embd": config.mlp_hidden_dim,
                "use_gated_mlp": config.use_gated_mlp,
                "vocab_size": config.vocab_size,
                "num_tensor_parallel_workers": 8,
                "use_qk_norm": config.use_qk_norm,
                "time_stats.attn_pre_proj.median": 1.0,
                "typed_operator_contracts": "not-json",
            }
        ]
    ).to_csv(input_file, index=False)

    trainer = AttentionTrainer.__new__(AttentionTrainer)
    trainer.train_compute_models = True
    trainer.compute_dataset_path = str(input_file)
    trainer.model_name = "step3-moe-noquant"
    trainer.tensor_parallel_size = 8
    trainer.model_config = SimpleNamespace(
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        embedding_dim=config.embedding_dim,
        mlp_hidden_dim=config.mlp_hidden_dim,
        use_gated_mlp=config.use_gated_mlp,
        vocab_size=config.vocab_size,
        use_qk_norm=config.use_qk_norm,
        get_model_architecture_profile=lambda: config.get_model_architecture_profile(),
    )
    monkeypatch.setattr(trainer, "_set_dataset_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer, "_verify_compute_dataset_columns", lambda _df: None)

    with pytest.raises(ValueError, match="canonical JSON"):
        trainer._load_dataset()


def test_attention_compute_loader_rejects_missing_typed_attention_operator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A measured attention target must have an explicit typed owner."""

    config = ModelConfig.from_model_name("step3-moe-noquant")
    plan = build_profiling_plan(
        model_config=config,
        tp_size=8,
        attn_tp=[8],
        ffn_tp=[8],
        moe_tp=[1],
        is_moe=False,
    )
    input_file = tmp_path / "linear_op.csv"
    pd.DataFrame(
        [
            {
                "n_head": config.num_q_heads,
                "n_kv_head": config.num_kv_heads,
                "n_embd": config.embedding_dim,
                "n_expanded_embd": config.mlp_hidden_dim,
                "use_gated_mlp": config.use_gated_mlp,
                "vocab_size": config.vocab_size,
                "num_tensor_parallel_workers": 8,
                "use_qk_norm": config.use_qk_norm,
                "time_stats.attn_pre_proj.median": 1.0,
                "time_stats.attn_post_proj.median": 2.0,
                "typed_operator_contracts": {
                    "attn_pre_proj": plan["typed_operator_contracts"]["attn_pre_proj"],
                },
            }
        ]
    ).pipe(linear_op_main._serialize_linear_op_output).to_csv(
        input_file,
        index=False,
    )

    trainer = AttentionTrainer.__new__(AttentionTrainer)
    trainer.train_compute_models = True
    trainer.compute_dataset_path = str(input_file)
    trainer.model_name = "step3-moe-noquant"
    trainer.tensor_parallel_size = 8
    trainer.model_config = SimpleNamespace(
        num_q_heads=config.num_q_heads,
        num_kv_heads=config.num_kv_heads,
        embedding_dim=config.embedding_dim,
        mlp_hidden_dim=config.mlp_hidden_dim,
        use_gated_mlp=config.use_gated_mlp,
        vocab_size=config.vocab_size,
        use_qk_norm=config.use_qk_norm,
        get_model_architecture_profile=lambda: config.get_model_architecture_profile(),
    )
    monkeypatch.setattr(trainer, "_set_dataset_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer, "_verify_compute_dataset_columns", lambda _df: None)

    with pytest.raises(ValueError, match="attn_post_proj"):
        trainer._load_dataset()


def test_attention_layer_trainer_rejects_combined_file_without_mixed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone layer training must enforce the selected combined schema."""

    config = BaseModelConfig.create_from_name("llama3.1-8b")
    combined_file = tmp_path / "attention_combined.csv"
    pd.DataFrame(
        [
            {
                "n_embd": config.embedding_dim,
                "n_q_head": config.num_q_heads,
                "n_kv_head": config.num_kv_heads,
                "block_size": 16,
                "num_tensor_parallel_workers": 8,
                "prefill_chunk_size": 4,
                "batch_size": 1,
                "is_prefill": True,
                "time_stats.attn_prefill.median": 1.0,
            }
        ]
    ).to_csv(combined_file, index=False)

    trainer = AttentionTrainer.__new__(AttentionTrainer)
    trainer.layer_dataset_path = str(combined_file)
    trainer.model_config = config
    trainer.block_size = 16
    trainer.tensor_parallel_size = 8
    monkeypatch.setattr(trainer, "_set_dataset_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer, "_verify_layer_dataset_columns", lambda _df: None)

    with pytest.raises(ValueError, match="Combined attention profiling input"):
        trainer._load_layer_dataset()
