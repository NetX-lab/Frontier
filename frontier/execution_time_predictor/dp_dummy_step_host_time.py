"""Measured native group cadence of all-dummy attention-DP forwards.

The cadence includes host input preparation and CPU dispatch agreement across
all engines and TP ranks of a DP group. It paces the hosts of all-dummy DP
forwards of the PREFILL role while their device work queues independently.
"""

from __future__ import annotations

import json
from math import isfinite
from numbers import Real
from pathlib import Path

from frontier.config import global_vars
from frontier.types import ClusterType

SCHEMA = "dp_dummy_step_host_time.v1"


class DPDummyStepHostTime:
    """One PREFILL DP group's measured host cadence in milliseconds."""

    def __init__(self, group_cadence_ms: float):
        self.group_cadence_ms = group_cadence_ms

    @classmethod
    def from_file(cls, path, *, replica_config, cluster_type, predictor_config) -> DPDummyStepHostTime | None:
        """Load the profile and check that it was measured on the run's scope."""
        if not predictor_config.skip_cpu_overhead_modeling:
            raise ValueError(
                "DP dummy-step host profile already contains host input preparation; "
                "it requires skip_cpu_overhead_modeling"
            )
        if cluster_type == ClusterType.DECODE:
            return None
        if cluster_type != ClusterType.PREFILL:
            raise ValueError(f"DP dummy-step host profile supports PREFILL, not {cluster_type}")
        with Path(path).open() as stream:
            profile = json.load(stream)
        try:
            if profile["schema"] != SCHEMA:
                raise ValueError(f"Unsupported DP dummy-step host profile schema {profile['schema']!r}")
            if not isinstance(profile["provenance"], dict) or not profile["provenance"]:
                raise ValueError("DP dummy-step host profile requires its provenance")
            if profile["admission"] not in ("admitted", "exploratory"):
                raise ValueError("DP dummy-step host profile admission must be admitted or exploratory")
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
                        f"DP dummy-step host profile {field} {profile[field]!r} "
                        f"does not match the run's {value!r}"
                    )
            topology = profile["topology"]
            if topology["pipeline_stages"] != 1 or topology["attention_dp"] <= 1:
                raise ValueError("DP dummy-step host profile requires pipeline_stages == 1 and attention_dp > 1")
            roles = profile["roles"]
            if not isinstance(roles, dict) or set(roles) != {"prefill"}:
                raise ValueError("DP dummy-step host profile roles must contain exactly prefill")
            cadence = roles["prefill"]["group_cadence_ms"]
            if isinstance(cadence, bool) or not isinstance(cadence, Real) or not isfinite(cadence) or cadence <= 0:
                raise ValueError("DP dummy-step host profile group_cadence_ms must be a finite positive time")
        except (KeyError, TypeError) as error:
            raise ValueError(f"Incomplete DP dummy-step host profile: {error}") from error
        return cls(float(cadence))
