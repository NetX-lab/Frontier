"""Round-trip actual GDN producer rows through CSV, training and prediction."""

from types import SimpleNamespace

import pandas as pd
import pytest

from frontier.attention.gdn.features import GDNBatchFeatures
from frontier.entities import Batch, Request
from frontier.execution_time_predictor.gdn_predictor import GDNPredictor
from frontier.profiling.gdn.inputs import GDNProfileInput
from frontier.profiling.gdn.vllm_wrapper import VllmQwen35GDNWrapper
from tests.unit.test_gdn_artifact_boundary import _trainer
from tests.unit.test_gdn_hybrid_e2e_increment14ab import _qwen35_fixture_config


def producer_row(workload):
    """Replace only native execution/timing boundaries; execute row construction."""
    noop = lambda *args, **kwargs: None
    wrapper = object.__new__(VllmQwen35GDNWrapper)
    wrapper.frontier_model_config = _qwen35_fixture_config()
    wrapper.max_batch_size = 16
    wrapper.max_model_len = 4096
    wrapper.tensor_parallel_size = 1
    wrapper.profile_method = "device_event"
    wrapper.device_name = "cpu"
    wrapper.runtime_stack_signature = "cpu_contract"
    wrapper.use_aiter_dispatch = False
    wrapper.layer = SimpleNamespace(
        get_state_dtype=lambda: ("bfloat16", "float32"),
        gdn_prefill_backend="test", gdn_decode_kernel="test",
        enable_packed_recurrent_decode=False, gqa_interleaved_layout=False,
    )
    wrapper.torch = SimpleNamespace(
        bfloat16="bfloat16", cuda=SimpleNamespace(synchronize=noop),
        randn=lambda *args, **kwargs: SimpleNamespace(dtype="torch.bfloat16"),
    )
    wrapper.timer_stats_store = SimpleNamespace(
        clear_stats=noop, get_times=lambda: {
            name: [value, value] for name, value in (
                ("gdn_input_projections", .11), (f"gdn_core_{workload.phase}", .22),
                ("gdn_output_projection", .33), ("gdn_layer_e2e", .66),
            )
        },
    )
    for method in ("_build_metadata", "_prepare_initial_state", "_restore_state",
                   "_run_e2e", "_run_decomposed", "_validate_decomposition"):
        setattr(wrapper, method, noop)
    return wrapper.profile(workload, warmup_iterations=0, profile_iterations=2)


@pytest.mark.parametrize("lengths", [(2, 4), (1, 7, 16), (4, 4)])
def test_producer_csv_trainer_feature_roundtrip(tmp_path, lengths):
    workload = GDNProfileInput(lengths, (0,) * len(lengths), "prefill")
    row = producer_row(workload)
    assert row["query_len_cv"] == pytest.approx(workload.query_len_cv)
    rows = [row, producer_row(GDNProfileInput((1,), (16,), "decode"))]
    dataframe = pd.DataFrame(rows)
    dataframe = pd.json_normalize(dataframe["time_stats"]).add_prefix("time_stats.").join(
        dataframe.drop(columns=["time_stats"])
    )
    source = tmp_path / "gdn.csv"
    dataframe.to_csv(source, index=False)
    imported = pd.read_csv(source, keep_default_na=False).iloc[0]
    batch = Batch(0, [Request(0, n, 2) for n in lengths], list(lengths), is_moe=True)
    expected = GDNBatchFeatures.from_batch(batch)
    assert GDNBatchFeatures.from_row(imported).exact_key() == expected.exact_key()
    output = tmp_path / "models"
    models = _trainer(source, output).train()
    assert expected.exact_key() in models["gdn_core_prefill_prefill"]._frontier_gdn_exact_lookup
    assert GDNPredictor.from_directory(output).predict_operator_times(expected)["gdn_core_prefill"] == .22


def test_imported_ragged_row_derives_and_validates_dispersion():
    row = producer_row(GDNProfileInput((2, 4), (0, 0), "prefill"))
    row.pop("query_len_cv")
    assert GDNBatchFeatures.from_row(row).query_len_cv == pytest.approx(1 / 3)
    row["query_len_cv"] = 0
    with pytest.raises(ValueError, match="disagrees"):
        GDNBatchFeatures.from_row(row)
    row.pop("query_len_cv")
    row.pop("query_lens")
    with pytest.raises(ValueError, match="Ragged GDN rows require"):
        GDNBatchFeatures.from_row(row)
