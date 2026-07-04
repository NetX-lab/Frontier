from types import SimpleNamespace

from frontier.metrics import metrics_store
from frontier.metrics.metrics_store import MetricsStore


class _FailingFigure:
    def write_image(self, _path: str) -> None:
        raise RuntimeError("image renderer unavailable")


def test_store_bar_plot_logs_image_export_failure_without_masking_original_error(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(metrics_store.px, "bar", lambda **_kwargs: _FailingFigure())

    store = SimpleNamespace(_config=SimpleNamespace(store_plots=True))

    MetricsStore._store_bar_plot(
        store,
        str(tmp_path),
        "test_plot",
        "operation",
        "latency_ms",
        {"attn_prefill": 1.0},
    )
