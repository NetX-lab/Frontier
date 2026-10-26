"""Collective device-time curves from vLLM serving process groups.

Payloads are the largest rank's input bytes, not wire traffic. Callers own
payload geometry; unequal rank rows of TP and EP groups are holdouts.

DP-group gathers and scatters are priced from each rank's row count. Their
curves are kept per row layout and CUDA-graph mode, because the measured
FP8 and BF16 dispatch layouts and graph replays differ at equal bytes, and
unequal rows run grouped NCCL kernels (AllGatherV, per-root Reduce) keyed by
`_unequal_rows_bytes`.
"""

import csv
import math
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Optional, Tuple

import numpy as np

from frontier.cc_backend.base_cc_backend import BaseCCBackend
from frontier.cc_backend.cc_backend_config import MeasuredCCBackendConfig
from frontier.cc_backend.cc_backend_factory import CCBackendFactory
from frontier.logger import init_logger
from frontier.types import CCBackendType, ClusterType

logger = init_logger(__name__)
# (op, group, group_size); DP curves add (row_bytes, cuda_graph, unequal_rows).
CurveKey = Tuple[Any, ...]

# The API domain selects the measured collective, not a model-name branch.
_COLLECTIVES = {
    ("allreduce", "ATTN_TP"): ("all_reduce", "tp"),
    ("allgather", "ATTN_TP"): ("all_gather", "tp"),
    ("reduce_scatter", "ATTN_TP"): ("reduce_scatter", "tp"),
    ("allgather", "EP"): ("all_gatherv", "ep"),
    ("allgather", "MOE_EP"): ("all_gatherv", "ep"),
    ("reduce_scatter", "EP"): ("reduce_scatterv", "ep"),
    ("reduce_scatter", "MOE_EP"): ("reduce_scatterv", "ep"),
    ("allgather", "ATTN_DP"): ("all_gatherv", "dp"),
    ("reduce_scatter", "ATTN_DP"): ("reduce_scatterv", "dp"),
}


class MeasuredCCOutOfRangeError(ValueError):
    """A collective request has no measured coverage; never extrapolate."""

    def __init__(
        self,
        curve_key: CurveKey,
        requested_bytes: int,
        covered_range: Optional[Tuple[int, int]],
        reason: str = "no measured coverage",
    ):
        self.curve_key = curve_key
        self.requested_bytes = requested_bytes
        self.covered_range = covered_range
        self.reason = reason
        super().__init__(
            f"{reason}: key={curve_key}, requested_bytes={requested_bytes}, "
            f"covered_range={covered_range}"
        )


def _unequal_rows_bytes(op: str, rows: Tuple[int, ...], row_bytes: int) -> int:
    """Bytes that set the time of a grouped DP kernel over unequal rows.

    AllGatherV moves every rank's rows to every rank. Per-root Reduce is bound
    by the busiest link, which carries all rows except the smallest rank's.
    """
    if op == "all_gatherv":
        return sum(rows) * row_bytes
    return (sum(rows) - min(rows)) * row_bytes


def _interpolate(size: int, lower: int, upper: int, low_us: float, high_us: float):
    weight = math.log(size / lower) / math.log(upper / lower)
    return math.exp(math.log(low_us) + weight * math.log(high_us / low_us))


@dataclass(frozen=True)
class _Curve:
    sizes: Tuple[int, ...]
    times_us: Tuple[float, ...]
    calls: Tuple[str, ...]

    def predict_us(self, size: int) -> float:
        index = bisect_left(self.sizes, size)
        if self.sizes[index] == size:
            return self.times_us[index]
        return _interpolate(
            size, self.sizes[index - 1], self.sizes[index],
            self.times_us[index - 1], self.times_us[index],
        )

    def describe(self):
        errors = [
            abs(_interpolate(
                self.sizes[i], self.sizes[i - 1], self.sizes[i + 1],
                self.times_us[i - 1], self.times_us[i + 1],
            ) / self.times_us[i] - 1)
            for i in range(1, len(self.sizes) - 1)
        ]
        return {
            "point_count": len(self.sizes),
            "byte_range": (self.sizes[0], self.sizes[-1]),
            "time_range_us": (min(self.times_us), max(self.times_us)),
            "calls": self.calls,
            "loo_absolute_relative_error": {
                "point_count": len(errors),
                "median": median(errors) if errors else None,
                "p90": float(np.percentile(errors, 90)) if errors else None,
                "max": max(errors) if errors else None,
            },
        }


class MeasuredCCBackend(BaseCCBackend):
    """Pool eager/graph device medians and interpolate only inside their range."""

    def __init__(
        self, config: MeasuredCCBackendConfig, cluster_type: ClusterType,
        device_type: str, network_device: str, num_devices: int,
    ) -> None:
        super().__init__(config, cluster_type, device_type, network_device, num_devices)
        self._source_path = str(Path(config.summary_csv_path.format(
            profiling_data_dir=config.profiling_data_dir, NETWORK_DEVICE=network_device,
        )).resolve())
        self._curves, self._dp_curves = self._load_curves()
        logger.info("Measured collective coverage: %s", self.describe_curves())

    def _load_curves(self):
        samples = defaultdict(lambda: defaultdict(list))
        dp_samples = defaultdict(lambda: defaultdict(list))
        calls = defaultdict(set)
        meanings = {}
        required = {
            "scenario", "call", "op", "group", "group_size", "tokens_per_dp_rank",
            "cuda_graph", "max_input_bytes", "max_output_bytes", "median_us",
        }
        with open(self._source_path, newline="") as stream:
            reader = csv.DictReader(stream)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{self._source_path}: missing columns {sorted(missing)}")
            rows = list(reader)
        # Serving runs every DP rail at once; fewer rails share less bandwidth.
        serving_rails = defaultdict(int)
        for row in rows:
            if row.get("concurrent_dp_rails"):
                rails = int(row["concurrent_dp_rails"])
                serving_rails[row["group"]] = max(serving_rails[row["group"]], rails)
        for line, row in enumerate(rows, 2):
            # CPU synchronization is not device time.
            # Logits include a layout copy and are priced with hidden-state curves.
            if (row["op"] == "cpu_all_reduce" or row["call"] == "logits_all_gather"
                    or (row.get("concurrent_dp_rails")
                        and int(row["concurrent_dp_rails"]) < serving_rails[row["group"]])):
                continue
            key = (row["op"], row["group"], int(row["group_size"]))
            tokens = tuple(int(value) for value in row["tokens_per_dp_rank"].split("/"))
            unequal = min(tokens) != max(tokens)
            if unequal and key[1] != "dp":
                continue
            if key[:2] not in _COLLECTIVES.values():
                raise ValueError(f"{self._source_path}:{line}: unsupported curve {key}")
            call = row["call"]
            meaning = key[:2]
            if call in meanings and meanings[call] != meaning:
                raise ValueError(
                    f"{self._source_path}:{line}: ambiguous call {call!r}: "
                    f"{meanings[call]} vs {meaning}"
                )
            meanings[call] = meaning
            size, time_us = int(row["max_input_bytes"]), float(row["median_us"])
            if key[2] < 2 or size <= 0 or not math.isfinite(time_us) or time_us <= 0:
                raise ValueError(f"{self._source_path}:{line}: invalid measured row {key}")
            if key[1] == "dp":
                row_bytes = self._dp_row_bytes(row, key, tokens, size, line)
                key = (*key, row_bytes, row["cuda_graph"] == "True", unequal)
                if unequal:
                    size = _unequal_rows_bytes(key[0], tokens, row_bytes)
                dp_samples[key][size].append(time_us)
            else:
                samples[key][size].append(time_us)
            calls[key].add(call)
        if not samples and not dp_samples:
            raise ValueError(f"{self._source_path}: no device collective curves")
        return self._fit(samples, calls), self._fit(dp_samples, calls)

    def _dp_row_bytes(self, row, key, tokens, size, line):
        """Recover the bytes per row from a DP row's input and output sizes."""
        gather = key[0] == "all_gatherv"
        row_bytes, remainder = divmod(size, max(tokens) if gather else sum(tokens))
        output_rows = sum(tokens) if gather else max(tokens)
        if (len(tokens) != key[2] or min(tokens) <= 0 or remainder
                or row["cuda_graph"] not in ("True", "False")
                or int(row["max_output_bytes"]) != output_rows * row_bytes):
            raise ValueError(
                f"{self._source_path}:{line}: inconsistent DP rows {tokens}, "
                f"input/output bytes or graph mode for {key}"
            )
        return row_bytes

    @staticmethod
    def _fit(samples, calls):
        curves = {}
        for key, points in sorted(samples.items()):
            sizes = tuple(sorted(points))
            times = tuple(median(points[size]) for size in sizes)
            curves[key] = _Curve(sizes, times, tuple(sorted(calls[key])))
            for i in range(1, len(sizes)):
                if times[i] < times[i - 1]:
                    logger.warning(
                        "Non-monotone measured collective %s: bytes=%s, us=%s",
                        key, sizes[i - 1:i + 1], times[i - 1:i + 1],
                    )
        return curves

    def describe_curves(self):
        """Return loaded coverage and LOO diagnostics, not an accuracy guarantee."""
        return [
            dict(curve.describe(), key=key, source_files=(self._source_path,))
            for curves in (self._curves, self._dp_curves)
            for key, curve in sorted(curves.items())
        ]

    @staticmethod
    def _lookup(curves, key, data_size_bytes):
        curve = curves.get(key)
        coverage = (curve.sizes[0], curve.sizes[-1]) if curve else None
        if curve is None or not coverage[0] <= data_size_bytes <= coverage[1]:
            raise MeasuredCCOutOfRangeError(key, data_size_bytes, coverage)
        return curve.predict_us(data_size_bytes) / 1000.0

    def _predict(self, operation, data_size_bytes, num_devices, comm_domain):
        measured = _COLLECTIVES.get((operation, comm_domain), (operation, comm_domain))
        return self._lookup(self._curves, (*measured, num_devices), data_size_bytes)

    def predict_dp_rows(self, operation, rows, row_bytes, cuda_graph, comm_domain):
        """Price a DP-group gather or scatter from each rank's row count.

        Equal rows use the largest rank's input bytes: its own rows for a
        gather, every rank's rows for a scatter.
        """
        rows = tuple(rows)
        measured = _COLLECTIVES.get((operation, comm_domain), (operation, comm_domain))
        unequal = min(rows) != max(rows)
        if unequal:
            size = _unequal_rows_bytes(measured[0], rows, row_bytes)
        else:
            size = (rows[0] if measured[0] == "all_gatherv" else sum(rows)) * row_bytes
        key = (*measured, len(rows), row_bytes, cuda_graph, unequal)
        return self._lookup(self._dp_curves, key, size)

    def predict_allreduce(self, data_size_bytes, num_devices, cluster_type=None, comm_domain=None):
        return self._predict("allreduce", data_size_bytes, num_devices, comm_domain)

    def predict_allgather(self, data_size_bytes, num_devices, cluster_type=None, comm_domain=None):
        return self._predict("allgather", data_size_bytes, num_devices, comm_domain)

    def predict_reduce_scatter(self, data_size_bytes, num_devices, cluster_type=None, comm_domain=None):
        return self._predict("reduce_scatter", data_size_bytes, num_devices, comm_domain)

    def predict_all_to_all(self, data_size_bytes, num_devices, cluster_type=None, comm_domain=None):
        return self._predict("all_to_all", data_size_bytes, num_devices, comm_domain)

    def predict_broadcast(self, data_size_bytes, num_devices, cluster_type=None, comm_domain=None):
        return self._predict("broadcast", data_size_bytes, num_devices, comm_domain)

    def predict_send_recv(self, data_size_bytes, cluster_type=None, comm_domain=None):
        return self._predict("send_recv", data_size_bytes, 2, comm_domain)


CCBackendFactory.register(CCBackendType.MEASURED, MeasuredCCBackend)
