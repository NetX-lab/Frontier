"""Measured host time of the Step KV side-storage reset before a DP forward.

A native Step engine resets the side storage of newly allocated sparse blocks
before its forward. The reset converts GPU block IDs to host lists, so the host
first waits for the previous step's device work and then runs the remaining
resets, the DP collective and the input preparation while the GPU has no
forward to run. The profile holds that exposed host time per role, measured by
the native host-step probe. It is consumed at the DP forward-start agreement.
"""

from __future__ import annotations

from bisect import bisect_left
import json
from math import isfinite
from numbers import Real
from pathlib import Path

from frontier.config import global_vars
from frontier.types import ClusterType

SCHEMA = "kv_block_zeroing_host_time.v1"
ROLE_BY_CLUSTER_TYPE = {ClusterType.PREFILL: "prefill", ClusterType.DECODE: "decode"}


def _time_ms(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError(f"KV block zeroing host profile {what} must be a finite non-negative time")
    return float(value)


class KVBlockZeroingHostTime:
    """One role's measured host time, scalar or keyed by the batch's request count."""

    def __init__(self, role: str, time_ms: float | None, table: tuple[tuple[int, float], ...]):
        self._role = role
        self._time_ms = time_ms
        self._table = table

    @classmethod
    def from_file(cls, path, *, replica_config, cluster_type, predictor_config) -> "KVBlockZeroingHostTime":
        """Load the profile and check that it was measured on the run's scope."""
        if not predictor_config.skip_cpu_overhead_modeling:
            raise ValueError(
                "KV block zeroing host profile already contains the input preparation of a "
                "resetting step; it requires skip_cpu_overhead_modeling"
            )
        role = ROLE_BY_CLUSTER_TYPE.get(cluster_type)
        if role is None:
            raise ValueError(f"KV block zeroing host profile supports PREFILL and DECODE, not {cluster_type}")
        with Path(path).open() as stream:
            profile = json.load(stream)
        try:
            if profile["schema"] != SCHEMA:
                raise ValueError(f"Unsupported KV block zeroing host profile schema {profile['schema']!r}")
            if not isinstance(profile["provenance"], dict) or not profile["provenance"]:
                raise ValueError("KV block zeroing host profile requires its provenance")
            expected = {
                "model_architecture_profile": replica_config.model_config.get_model_architecture_profile().profile_id,
                "device": replica_config.device,
                "decode_cuda_graph_mode": global_vars.get_decode_cuda_graph_mode(),
                "topology": {
                    "attention_tp": replica_config.attn_tensor_parallel_size,
                    "attention_dp": replica_config.attn_dp,
                    "moe_ep": replica_config.moe_expert_parallel_size,
                    "pipeline_stages": replica_config.num_pipeline_stages,
                },
            }
            for field, value in expected.items():
                if profile[field] != value:
                    raise ValueError(
                        f"KV block zeroing host profile {field} {profile[field]!r} "
                        f"does not match the run's {value!r}"
                    )
            if role not in profile["roles"]:
                raise ValueError(f"KV block zeroing host profile has no {role} entry")
            entry = profile["roles"][role]
            if ("time_ms" in entry) == ("by_num_requests" in entry):
                raise ValueError(f"KV block zeroing host profile {role} entry needs time_ms or by_num_requests")
            if "time_ms" in entry:
                return cls(role, _time_ms(entry["time_ms"], f"{role} time_ms"), ())
            table = tuple(
                (count, _time_ms(time_ms, f"{role} time at {count} requests"))
                for count, time_ms in entry["by_num_requests"]
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"Incomplete KV block zeroing host profile: {error}") from error
        counts = [count for count, _ in table]
        if (
            len(table) < 2
            or any(type(count) is not int or count <= 0 for count in counts)
            or counts != sorted(set(counts))
        ):
            raise ValueError(
                f"KV block zeroing host profile {role} table needs at least two strictly "
                "increasing positive request counts"
            )
        return cls(role, None, table)

    def time_ms(self, num_requests: int) -> float:
        if self._time_ms is not None:
            return self._time_ms
        counts = [count for count, _ in self._table]
        if not counts[0] <= num_requests <= counts[-1]:
            raise ValueError(
                f"KV block zeroing host profile {self._role} table covers {counts[0]}-{counts[-1]} "
                f"requests, not {num_requests}"
            )
        index = bisect_left(counts, num_requests)
        count, time_ms = self._table[index]
        if count == num_requests:
            return time_ms
        lower_count, lower_time = self._table[index - 1]
        share = (num_requests - lower_count) / (count - lower_count)
        return lower_time + share * (time_ms - lower_time)
