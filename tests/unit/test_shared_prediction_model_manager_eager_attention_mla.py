"""Shared-manager latent-MLA attention-core training contract tests.

Mirrors ``test_shared_prediction_model_manager_eager_attention_decode.py`` (dense
path) and ``test_mla_predictor_training_integration.py`` (monolithic MLA path).

The shared ``ExecutionTimePredictionModelManager`` is always constructed at
``simulator.py:153`` and drives attention-core training for every architecture,
including PD-disaggregation. Before the A2 fix it hard-codes
``DENSE_ATTENTION_FAMILY`` for the attention-core block, so feeding it a latent-MLA
profile fails fast at ``shared_prediction_model_manager.py:1256`` with
``Missing columns for attn_kv_cache_save training: ['total_tokens', 'kv_cache_size']``
and the six ``attn_mla_*`` models are never produced. These tests lock the fixed
behaviour: the manager routes MLA profiles through a family-aware branch and trains
exactly the six latent-MLA attention-core operators.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from frontier.attention.families import LATENT_MLA_ATTENTION_FAMILY
from frontier.attention.profiling_mapping import (
    get_enabled_predictor_median_columns,
    get_enabled_predictor_metric_names,
    get_enabled_shared_predictor_feature_columns,
)
from frontier.execution_time_predictor.shared_prediction_model_manager import (
    ExecutionTimePredictionModelManager,
)
from frontier.profiling.attention.vllm_mla_profile_importer import (
    build_frontier_mla_profile_dataframe,
)
from frontier.types import ClusterType, MeasurementType
from tests.unit.mla_h800_fixture import h800_mla_mixed_rows


def _mla_model_config() -> SimpleNamespace:
    """DeepSeek-V2-Lite style latent-MLA model config (hermetic).

    Field set mirrors the monolithic ``_build_mla_predictor`` model config so
    ``bind_attention_family`` resolves to the latent family and the structural
    MLA filter sees the runtime latent layout (kv heads = 1, head size = 576).
    """
    return SimpleNamespace(
        use_mla=True,
        model_arch="deepseek_v2",
        num_q_heads=128,
        num_kv_heads=128,
        embedding_dim=73728,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        kv_lora_rank=512,
        v_head_dim=128,
        use_qk_norm=False,
        get_model_arch=lambda: "deepseek_v2",
        get_qk_head_dim=lambda: 192,
        get_runtime_num_kv_heads=lambda: 1,
        get_runtime_head_size=lambda: 576,
        get_quant_signature=lambda: "none",
    )


def _make_mla_replica_config() -> SimpleNamespace:
    return SimpleNamespace(
        device="h800",
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        attn_tensor_parallel_size=1,
        speculative_decoding_config=None,
        model_config=_mla_model_config(),
    )


def _make_manager_without_init() -> ExecutionTimePredictionModelManager:
    manager = ExecutionTimePredictionModelManager.__new__(
        ExecutionTimePredictionModelManager
    )
    manager._active_measurement_type = MeasurementType.CUDA_EVENT
    manager._attention_tp_warning_cache = set()
    return manager


def _imported_h800_mla_df() -> pd.DataFrame:
    """13 H800 vLLM MLA rows -> 5 grouped Frontier profile rows (6 sparse targets)."""
    return build_frontier_mla_profile_dataframe(
        h800_mla_mixed_rows(),
        model_name="deepseek-ai/DeepSeek-V2-Lite",
        model_arch="deepseek_v2",
        precision="bf16",
        quant_signature="none",
        measurement_type="CUDA_EVENT",
        num_tensor_parallel_workers=1,
        max_model_len=163840,
    )


def _fake_linear_op_df(*_args, **_kwargs) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_tokens": [1, 2],
            "time_stats.input_layernorm.median": [0.1, 0.2],
            "time_stats.attn_pre_proj.median": [0.1, 0.2],
            "time_stats.attn_post_proj.median": [0.1, 0.2],
            "time_stats.attn_rope.median": [0.1, 0.2],
        }
    )


def test_shared_manager_trains_six_mla_attention_core_ops(tmp_path) -> None:
    """A2 GREEN: a latent-MLA profile trains the six ``attn_mla_*`` operators.

    Pre-fix this raises ``Missing columns for attn_kv_cache_save training`` at
    ``shared_prediction_model_manager.py:1256`` (the promoted RED smoke).
    """
    manager = _make_manager_without_init()
    replica_config = _make_mla_replica_config()

    linear_ops_file = tmp_path / "linear_op.csv"
    attn_file = tmp_path / "attention.csv"
    linear_ops_file.write_text("", encoding="utf-8")
    attn_file.write_text("", encoding="utf-8")

    mla_df = _imported_h800_mla_df()
    manager._load_linear_op_df = _fake_linear_op_df  # type: ignore[attr-defined]
    manager._load_attention_df = lambda *_a, **_k: mla_df.copy()  # type: ignore[attr-defined]
    # Passthrough: the dense derive would KeyError on MLA columns pre-fix; post-fix
    # the family-aware branch bypasses it and calls the MLA derived-features path.
    manager._get_attention_df_with_derived_features = lambda df: df  # type: ignore[attr-defined]

    trained: dict[str, dict[str, object]] = {}

    def _fake_train_single_model(*, model_name: str, df: pd.DataFrame, feature_cols, target_col, **_kwargs):
        trained[model_name] = {
            "feature_cols": tuple(feature_cols),
            "target_col": target_col,
            "num_rows": len(df),
        }
        model = SimpleNamespace()
        model._frontier_feature_names = list(feature_cols)
        return model

    manager._train_single_model = _fake_train_single_model  # type: ignore[attr-defined]

    models = manager._train_attn_models_for_cluster(
        cluster_type=ClusterType.DECODE,
        replica_config=replica_config,
        execution_time_predictor_config=SimpleNamespace(),
        replica_scheduler_config=SimpleNamespace(block_size=64),
        linear_ops_file=str(linear_ops_file),
        attn_file=str(attn_file),
        trained_model_signatures=set(),
    )

    expected_ops = list(get_enabled_predictor_metric_names(LATENT_MLA_ATTENTION_FAMILY))
    expected_features = get_enabled_shared_predictor_feature_columns(
        LATENT_MLA_ATTENTION_FAMILY
    )
    expected_targets = dict(
        zip(
            expected_ops,
            get_enabled_predictor_median_columns(LATENT_MLA_ATTENTION_FAMILY),
        )
    )

    for op_name in expected_ops:
        assert op_name in models, f"{op_name} missing from returned models"
        assert op_name in trained
        assert trained[op_name]["feature_cols"] == tuple(expected_features[op_name])
        assert trained[op_name]["target_col"] == expected_targets[op_name]
        # Exact profiling-row memoization paired with the on-demand consumer.
        assert getattr(models[op_name], "_frontier_exact_lookup")

    # Sparse-by-target row counts: kv_cache_save observed in all 3 batches; the
    # remaining five operators in 2 batches each (mirrors the monolithic path).
    assert {op_name: trained[op_name]["num_rows"] for op_name in expected_ops} == {
        "attn_mla_kv_cache_save": 3,
        "attn_mla_prefill_kv_up_proj": 2,
        "attn_mla_prefill": 2,
        "attn_mla_decode_q_latent_proj": 2,
        "attn_mla_decode": 2,
        "attn_mla_v_up_proj": 2,
    }

    # The dense attention-core operators must NOT be trained for an MLA profile.
    for dense_op in ("attn_kv_cache_save", "attn_prefill", "attn_decode"):
        assert dense_op not in models
        assert dense_op not in trained


def test_shared_manager_load_attention_df_filters_mla_structural_rows(tmp_path) -> None:
    """A2 GREEN: ``_load_attention_df`` routes MLA profiles through the structural filter."""
    manager = _make_manager_without_init()
    replica_config = _make_mla_replica_config()

    mla_df = _imported_h800_mla_df()
    mismatched_df = mla_df.copy()
    mismatched_df["kv_lora_rank"] = 256
    input_file = tmp_path / "attention.csv"
    pd.concat([mismatched_df, mla_df], ignore_index=True).to_csv(input_file, index=False)

    filtered = manager._load_attention_df(
        str(input_file),
        replica_config,
        SimpleNamespace(block_size=64),
        cluster_type=ClusterType.DECODE,
    )

    assert len(filtered) == len(mla_df)
    assert (filtered["kv_lora_rank"].astype(int) == 512).all()
    assert (filtered["n_kv_head"].astype(int) == 1).all()
    assert (filtered["head_size"].astype(int) == 576).all()
