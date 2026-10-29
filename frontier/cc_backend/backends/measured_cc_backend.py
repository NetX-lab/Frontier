"""Collective device-time curves from vLLM serving process groups.

Payloads are the largest rank's input bytes, not wire traffic. Callers own
payload geometry and kernel selection: unequal EP combine rows use exact
per-root Reduce measurements, never the uniform reduce-scatter curve.
"""

import csv
import math
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Optional, Tuple

import numpy as np

from frontier.cc_backend.base_cc_backend import BaseCCBackend
from frontier.cc_backend.cc_backend_config import MeasuredCCBackendConfig
from frontier.cc_backend.cc_backend_factory import CCBackendFactory
from frontier.config.parallel_semantics import sequence_parallel_moe_token_widths
from frontier.config.precision_type import PrecisionType
from frontier.operators.spec import CollectiveImplementation
from frontier.logger import init_logger
from frontier.types import CCBackendType, ClusterType

logger = init_logger(__name__)
CurveKey = Tuple[str, str, int]

# Alias, domain and native implementation select the measured curve, never an operator name.
_COLLECTIVE_CURVES = {
    ("allreduce", "ATTN_TP", CollectiveImplementation.GROUP_COORDINATOR_DIM0): ("all_reduce", "tp"),
    ("allgather", "ATTN_TP", CollectiveImplementation.GROUP_COORDINATOR_DIM0): ("all_gather", "tp"),
    ("reduce_scatter", "ATTN_TP", CollectiveImplementation.GROUP_COORDINATOR_DIM0): ("reduce_scatter", "tp"),
    **{(alias, domain, CollectiveImplementation.GROUP_COORDINATOR_DIM0): (op, "ep")
       for domain in ("EP", "MOE_EP")
       for alias, op in (("allgather", "all_gatherv"), ("reduce_scatter", "reduce_scatterv"))},
    ("allgather", "ATTN_TP", CollectiveImplementation.VLLM_SEQUENCE_PARALLEL): ("sp_all_gather", "tp"),
    ("reduce_scatter", "ATTN_TP", CollectiveImplementation.VLLM_SEQUENCE_PARALLEL): ("sp_reduce_scatter", "tp"),
    ("allgather", "ATTN_TP", CollectiveImplementation.GROUP_COORDINATOR_LAST_DIM): ("logits_all_gather", "tp"),
}
_ROW_CURVES = {
    ("all_reduce", "tp"): ("all_reduce", "tp"),
    ("all_gather", "tp"): ("all_gather", "tp"),
    ("reduce_scatter", "tp"): ("reduce_scatter", "tp"),
    ("all_gatherv", "ep"): ("all_gatherv", "ep"),
    ("reduce_scatterv", "ep"): ("reduce_scatterv", "ep"),
    ("sp_all_gather", "tp"): ("sp_all_gather", "tp"),
    ("sp_reduce_scatter", "tp"): ("sp_reduce_scatter", "tp"),
    # Only this call needs an override: op cannot distinguish the dim -1
    # all-gather plus layout copy from GroupCoordinator's dim 0 all-gather.
    ("logits_all_gather", None): ("logits_all_gather", "tp"),
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
        self._curves = self._load_curves()
        logger.info("Measured collective coverage: %s; holdout rows: %s",
                    self.describe_curves(), len(self._holdouts))

    def _load_curves(self):
        samples = defaultdict(lambda: defaultdict(list))
        unequal_samples = defaultdict(list)
        calls = defaultdict(set)
        meanings = {}
        self._holdouts = []
        required = {
            "scenario", "call", "op", "group", "group_size",
            "max_input_bytes", "median_us", "tokens_per_dp_rank",
        }
        with open(self._source_path, newline="") as stream:
            reader = csv.DictReader(stream)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{self._source_path}: missing columns {sorted(missing)}")
            for line, row in enumerate(reader, 2):
                if row["group"] == "dp":
                    continue
                single_active = "single_active" in row["scenario"]
                measured = _ROW_CURVES.get((row["call"], None),
                                          _ROW_CURVES.get((row["op"], row["group"])))
                key = (*(measured or (row["op"], row["group"])), int(row["group_size"]))
                if measured is None:
                    raise ValueError(f"{self._source_path}:{line}: unsupported curve {key}")
                size, time_us = int(row["max_input_bytes"]), float(row["median_us"])
                if key[2] < 2 or size <= 0 or not math.isfinite(time_us) or time_us <= 0:
                    raise ValueError(f"{self._source_path}:{line}: invalid measured row {key}")
                tokens = tuple(int(value) for value in row["tokens_per_dp_rank"].split("/")
                               if value)
                uniform = bool(tokens) and len(set(tokens)) == 1
                if single_active or not uniform:
                    if key[:2] == ("reduce_scatterv", "ep"):
                        rows = self._retained_combine_rows(row, key, size, line)
                        if rows is not None:
                            unequal_samples[(key, rows, size)].append(time_us)
                            continue
                    self._holdouts.append((key, dict(row)))
                    continue
                call = row["call"]
                meaning = key[:2]
                if call in meanings and meanings[call] != meaning:
                    raise ValueError(
                        f"{self._source_path}:{line}: ambiguous call {call!r}: "
                        f"{meanings[call]} vs {meaning}"
                    )
                meanings[call] = meaning
                samples[key][size].append(time_us)
                calls[key].add(call)
        if not samples:
            raise ValueError(f"{self._source_path}: no device collective curves")
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
        self._unequal_combine_times = {key: median(times) for key, times in unequal_samples.items()}
        return curves

    def _retained_combine_rows(self, row, key, size, line):
        """Recover the measured SP vector using the shared padding semantics."""
        location = f"{self._source_path}:{line}: invalid retained SP combine"
        required = {"tokens_per_dp_rank", "max_output_bytes", "kernels"}
        if any(not row.get(field) for field in required):
            raise ValueError(f"{location}: missing shape or kernel evidence")
        tokens = tuple(int(value) for value in row["tokens_per_dp_rank"].split("/"))
        tp, remainder = divmod(key[2], len(tokens))
        if len(tokens) < 2 or remainder or tp <= 1 or min(tokens) <= 0:
            raise ValueError(f"{location}: inconsistent DP/TP shape")
        widths = sequence_parallel_moe_token_widths(tokens, tp)
        if max(widths.local_tokens) == min(widths.local_tokens):
            return None
        # cuda_communicator.py selects PyNCCL's grouped per-root ncclReduce.
        kernels = [name for name in row["kernels"].split(";") if name.startswith("ncclDevKernel_")]
        if not kernels or any(not name.startswith("ncclDevKernel_Reduce_Sum_bf16_") for name in kernels):
            raise ValueError(f"{location}: kernels do not prove per-root BF16 Reduce")
        row_bytes, remainder = divmod(size, widths.gathered_tokens)
        if (remainder or row_bytes % int(PrecisionType.BF16.bytes_per_element)
                or int(row["max_output_bytes"]) != max(widths.local_tokens) * row_bytes):
            raise ValueError(f"{location}: inconsistent SP rows and input/output bytes")
        return widths.local_tokens

    def describe_curves(self):
        """Return loaded coverage and LOO diagnostics, not an accuracy guarantee."""
        return [
            dict(curve.describe(), key=key, source_files=(self._source_path,))
            for key, curve in sorted(self._curves.items())
        ]

    def describe_holdouts(self):
        """Compare held-out device rows with uniform curves, without fitting them."""
        result = []
        for key, row in self._holdouts:
            size, measured = int(row["max_input_bytes"]), float(row["median_us"])
            curve = self._curves.get(key)
            predicted = (curve.predict_us(size) if curve is not None
                         and curve.sizes[0] <= size <= curve.sizes[-1] else None)
            result.append(dict(
                key=key, call=row["call"], group_size=key[2],
                scenario=row["scenario"], tokens=row["tokens_per_dp_rank"],
                bytes=size, measured_us=measured, curve_us=predicted,
                relative_error=predicted / measured - 1 if predicted is not None else None,
            ))
        return result

    def _curve_key(self, alias, implementation, num_devices, comm_domain):
        measured = _COLLECTIVE_CURVES.get((alias, comm_domain, implementation))
        if alias == "alltoall":
            alias = "all_to_all"
        return (*measured, num_devices) if measured else (alias, comm_domain, num_devices)

    def predict_operator_collective(
        self, alias, implementation, data_size_bytes, num_devices, comm_domain, sp_rows=None,
    ):
        """Resolve implementation and price uniform or exact ordered SP work."""
        key = self._curve_key(alias, implementation, num_devices, comm_domain)
        if key[:2] == ("reduce_scatterv", "ep") and sp_rows is not None:
            rows = tuple(sp_rows)
            if max(rows) != min(rows):
                return self._predict_unequal(key, data_size_bytes, rows)
        curve = self._curves.get(key)
        coverage = (curve.sizes[0], curve.sizes[-1]) if curve else None
        if curve is None or not coverage[0] <= data_size_bytes <= coverage[1]:
            raise MeasuredCCOutOfRangeError(key, data_size_bytes, coverage)
        return curve.predict_us(data_size_bytes) / 1000.0

    def _predict(self, operation, data_size_bytes, num_devices, comm_domain):
        return self.predict_operator_collective(
            operation, CollectiveImplementation.GROUP_COORDINATOR_DIM0,
            data_size_bytes, num_devices, comm_domain,
        )

    def _predict_unequal(self, key, data_size_bytes, rows):
        time_us = self._unequal_combine_times.get((key, rows, data_size_bytes))
        if time_us is None:
            curve = self._curves.get(key)
            coverage = (curve.sizes[0], curve.sizes[-1]) if curve else None
            raise MeasuredCCOutOfRangeError(
                key, data_size_bytes, coverage,
                reason=f"unequal SP rows {rows} have no exact per-root Reduce measurement",
            )
        return time_us / 1000.0

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
