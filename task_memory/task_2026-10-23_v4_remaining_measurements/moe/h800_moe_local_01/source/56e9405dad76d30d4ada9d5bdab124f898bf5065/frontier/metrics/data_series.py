from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px

from frontier.logger import init_logger
from frontier.metrics.wandb_utils import get_wandb

logger = init_logger(__name__)
_PLOT_IMAGE_EXPORT_WARNING_LOGGED = False
wandb = get_wandb()


class DataSeries:
    def __init__(
        self,
        x_name: str,
        y_name: str,
        subsamples: Optional[int] = None,
        save_table_to_wandb: bool = True,
        save_plots: bool = True,
    ) -> None:
        # metrics are a data series of two-dimensional (x, y) datapoints
        self._data_series = []
        # column names of x, y datatpoints for data collection
        self._x_name = x_name
        self._y_name = y_name

        # most recently collected y datapoint for incremental updates
        # to aid incremental updates to y datapoints
        self._last_data_y = 0

        self._subsamples = subsamples
        self._save_table_to_wandb = save_table_to_wandb
        self._save_plots = save_plots

    def consolidate(
        self,
    ):
        res = defaultdict(list)
        for x, y in self._data_series:
            res[x].append(y)
        self._data_series = [(x, sum(y) / len(y)) for x, y in res.items()]

        # sort by x
        self._data_series = sorted(self._data_series, key=lambda x: x[0])
        self._last_data_y = self._data_series[-1][1] if len(self._data_series) else 0

    def __len__(self):
        return len(self._data_series)

    @property
    def _metric_name(self) -> str:
        return self._y_name

    # add a new x, y datapoint
    def put(self, data_x: float, data_y: float) -> None:
        self._last_data_y = data_y
        self._data_series.append((data_x, data_y))

    # get most recently collected y datapoint
    def _peek_y(self):
        return self._last_data_y

    # convert list of x, y datapoints to a pandas dataframe
    def _to_df(self):
        return pd.DataFrame(self._data_series, columns=[self._x_name, self._y_name])

    # add a new x, y datapoint as an incremental (delta) update to
    # recently collected y datapoint
    def put_delta(self, data_x: float, data_y_delta: float) -> None:
        last_data_y = self._peek_y()
        data_y = last_data_y + data_y_delta
        self.put(data_x, data_y)

    def print_series_stats(
        self, df: pd.DataFrame, plot_name: str, x_name: str = None, y_name: str = None
    ) -> None:
        if len(self._data_series) == 0:
            return
        if x_name is None:
            x_name = self._x_name

        if y_name is None:
            y_name = self._y_name

        # Convert column to numeric, coercing errors (empty strings, etc.) to NaN
        numeric_series = pd.to_numeric(df[y_name], errors='coerce')
        
        # Skip if all values are NaN after conversion
        if numeric_series.isna().all():
            logger.debug(f"{plot_name}: {y_name} - no valid numeric data to compute stats")
            return

        logger.debug(
            f"{plot_name}: {y_name} stats:"
            f" min: {numeric_series.min()},"
            f" max: {numeric_series.max()},"
            f" mean: {numeric_series.mean()},"
        )
        if wandb is not None and wandb.run:
            wandb.log(
                {
                    f"{plot_name}_min": numeric_series.min(),
                    f"{plot_name}_max": numeric_series.max(),
                    f"{plot_name}_mean": numeric_series.mean(),
                },
                step=0,
            )

    def print_distribution_stats(
        self, df: pd.DataFrame, plot_name: str, y_name: str = None
    ) -> None:
        if len(self._data_series) == 0:
            return

        if y_name is None:
            y_name = self._y_name

        # Convert column to numeric, coercing errors (empty strings, etc.) to NaN
        numeric_series = pd.to_numeric(df[y_name], errors='coerce')
        
        # Skip if all values are NaN after conversion
        if numeric_series.isna().all():
            logger.debug(f"{plot_name}: {y_name} - no valid numeric data to compute stats")
            return

        logger.debug(
            f"{plot_name}: {y_name} stats:"
            f" min: {numeric_series.min()},"
            f" max: {numeric_series.max()},"
            f" mean: {numeric_series.mean()},"
            f" median: {numeric_series.median()},"
            f" 95th percentile: {numeric_series.quantile(0.95)},"
            f" 99th percentile: {numeric_series.quantile(0.99)}"
            f" 99.9th percentile: {numeric_series.quantile(0.999)}"
        )
        if wandb is not None and wandb.run:
            wandb.log(
                {
                    f"{plot_name}_min": numeric_series.min(),
                    f"{plot_name}_max": numeric_series.max(),
                    f"{plot_name}_mean": numeric_series.mean(),
                    f"{plot_name}_median": numeric_series.median(),
                    f"{plot_name}_95th_percentile": numeric_series.quantile(0.95),
                    f"{plot_name}_99th_percentile": numeric_series.quantile(0.99),
                    f"{plot_name}_99.9th_percentile": numeric_series.quantile(0.999),
                },
                step=0,
            )

    def _should_subsample(self, length: int) -> bool:
        return isinstance(self._subsamples, int) and length > self._subsamples

    def _save_df(self, df: pd.DataFrame, path: str, plot_name: str) -> None:
        df.to_csv(f"{path}/{plot_name}.csv", index=False)
        if wandb is not None and wandb.run and self._save_table_to_wandb:
            wand_table = wandb.Table(dataframe=df)
            wandb.log({f"{plot_name}_table": wand_table}, step=0)

    def _safe_write_image(self, fig: object, output_path: str, plot_name: str) -> None:
        global _PLOT_IMAGE_EXPORT_WARNING_LOGGED

        try:
            fig.write_image(output_path)
        except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch
            if not _PLOT_IMAGE_EXPORT_WARNING_LOGGED:
                logger.warning(
                    "Plot image export is unavailable; continuing without PNG outputs. "
                    "plot_name=%s output_path=%s error=%s",
                    plot_name,
                    output_path,
                    exc,
                )
                _PLOT_IMAGE_EXPORT_WARNING_LOGGED = True

    def save_df(self, path: str, plot_name: str) -> None:
        df = self._to_df()
        self._save_df(df, path, plot_name)

    def plot_step(
        self,
        path: str,
        plot_name: str,
        y_axis_label: str = None,
        start_time: float = 0,
        y_cumsum: bool = True,
    ) -> None:

        if len(self._data_series) == 0:
            return

        if y_axis_label is None:
            y_axis_label = self._y_name

        df = self._to_df()
        df[self._x_name] -= start_time

        if y_cumsum:
            df[self._y_name] = df[self._y_name].cumsum()

        self.print_series_stats(df, plot_name)

        # subsample
        if self._should_subsample(len(df)):
            # pick self._subsamples from the dataframe
            # however, if we make the difference between indices constant
            # we might pick spurious periodic patterns
            # so we pick the indices with a constant difference in the indices with a random offset
            # this is to avoid picking spurious periodic patterns
            indices = np.arange(0, len(df), len(df) // self._subsamples)
            offsets = np.random.randint(0, 5)
            indices = (indices + offsets) % len(df)
            df = df.iloc[indices]

        if wandb is not None and wandb.run:
            wandb_df = df.copy()
            # rename the self._y_name column to y_axis_label
            wandb_df = wandb_df.rename(columns={self._y_name: y_axis_label})

            wandb.log(
                {
                    f"{plot_name}_step": wandb.plot.line(
                        wandb.Table(dataframe=wandb_df),
                        self._x_name,
                        y_axis_label,
                        title=plot_name,
                    )
                },
                step=0,
            )

        if self._save_plots:
            fig = px.line(
                df,
                x=self._x_name,
                y=self._y_name,
                markers=True,
                labels={"x": y_axis_label},
            )
            fig.update_traces(marker=dict(color="red", size=2))
        self._save_df(df, path, plot_name)
        if self._save_plots:
            self._safe_write_image(fig, f"{path}/{plot_name}.png", plot_name)

    def plot_cdf(self, path: str, plot_name: str, y_axis_label: str = None) -> None:
        if len(self._data_series) == 0:
            return

        if y_axis_label is None:
            y_axis_label = self._y_name

        df = self._to_df()

        self.print_distribution_stats(df, plot_name)

        df["cdf"] = df[self._y_name].rank(method="first", pct=True)
        # sort by cdf
        df = df.sort_values(by=["cdf"])

        # subsample
        if self._should_subsample(len(df)):
            df = df.iloc[:: len(df) // self._subsamples]

        if wandb is not None and wandb.run:
            wandb_df = df.copy()
            # rename the self._y_name column to y_axis_label
            wandb_df = wandb_df.rename(columns={self._y_name: y_axis_label})

            wandb.log(
                {
                    f"{plot_name}_cdf": wandb.plot.line(
                        wandb.Table(dataframe=wandb_df),
                        "cdf",
                        y_axis_label,
                        title=plot_name,
                    )
                },
                step=0,
            )

        if self._save_plots:
            fig = px.line(
                df, x=self._y_name, y="cdf", markers=True, labels={"x": y_axis_label}
            )
            fig.update_traces(marker=dict(color="red", size=2))
        self._save_df(df, path, plot_name)
        if self._save_plots:
            self._safe_write_image(fig, f"{path}/{plot_name}.png", plot_name)

    def plot_histogram(self, path: str, plot_name: str) -> None:
        if len(self._data_series) == 0:
            return

        df = self._to_df()
        
        # Convert y column to numeric, coercing errors to NaN
        df[self._y_name] = pd.to_numeric(df[self._y_name], errors='coerce')
        
        # Drop rows with NaN values in y column
        df = df.dropna(subset=[self._y_name])
        
        # Skip if no valid data remains
        if len(df) == 0:
            logger.debug(f"{plot_name}: no valid numeric data for histogram")
            return

        self.print_distribution_stats(df, plot_name)

        if wandb is not None and wandb.run:
            # wandb histogram is highly inaccurate so we need to generate the histogram
            # ourselves and then use wandb bar chart
            histogram_df = (
                df[self._y_name].value_counts(bins=25, sort=False).sort_index()
            )
            histogram_df = histogram_df.reset_index()
            histogram_df.columns = ["Bins", "count"]
            histogram_df["Bins"] = histogram_df["Bins"].apply(lambda x: x.mid)
            histogram_df = histogram_df.sort_values(by=["Bins"])
            # convert to percentage
            histogram_df["Percentage"] = histogram_df["count"] * 100 / len(df)
            # drop bins with less than 0.1% of the total count
            histogram_df = histogram_df[histogram_df["Percentage"] > 0.1]

            wandb.log(
                {
                    f"{plot_name}_histogram": wandb.plot.bar(
                        wandb.Table(dataframe=histogram_df),
                        "Bins",
                        "Percentage",  # wandb plots are horizontal
                        title=plot_name,
                    )
                },
                step=0,
            )

        if self._save_plots:
            fig = px.histogram(df, x=self._y_name, nbins=25)
            self._safe_write_image(fig, f"{path}/{plot_name}.png", plot_name)

        self._save_df(df, path, plot_name)

    def plot_differential(self, path: str, plot_name: str) -> None:
        if len(self._data_series) == 0:
            return

        df = self._to_df()

        differential_col_name = f"{self._y_name}/{self._x_name}"

        # if there are repeated values of x, then we cannot compute a differential
        # in that, case aggreagate the y values for the same x value
        df = df.groupby(self._x_name).max().reset_index().sort_values(self._x_name)

        # compute differential
        df[differential_col_name] = df[self._y_name].diff() / df[self._x_name].diff()

        df = df.dropna()

        self.print_distribution_stats(df, plot_name, differential_col_name)

        # subsample
        if self._should_subsample(len(df)):
            df = df.iloc[:: len(df) // self._subsamples]

        if wandb is not None and wandb.run:
            wandb.log(
                {
                    f"{plot_name}_differential": wandb.plot.line(
                        wandb.Table(dataframe=df),
                        self._x_name,
                        differential_col_name,
                        title=plot_name,
                    )
                },
                step=0,
            )

        if self._save_plots:
            fig = px.line(df, x=self._x_name, y=differential_col_name, markers=True)
            fig.update_traces(marker=dict(color="red", size=2))

        self._save_df(df, path, plot_name)
        if self._save_plots:
            self._safe_write_image(fig, f"{path}/{plot_name}.png", plot_name)
