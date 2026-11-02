"""Diagnostic DeepSeek-V4 step costs measured on one native vLLM PD deployment.

The table prices one whole forward step of one attention-DP lane from the
composition of its batch. It is a diagnostic input, not a cost model: it carries
no formal-gate admission and is never stored in the default profiling path
(docs/adr/0002). A diagnostic driver selects it by pointing the CUDA_EVENT
attention input at the table's directory.
"""

import json
import os

from frontier.attention.deepseek_v4 import DeepSeekV4CoverageError
from frontier.execution_time_predictor.measurement_input_paths import resolve_measurement_input_paths
from frontier.types import MeasurementType

SCHEMA = "dsv4_step_cost.v1"
QUALIFICATION = "diagnostic_step_cost"
FILE_NAME = "dsv4_step_cost.json"
TOPOLOGY_FIELDS = (
    "attn_tensor_parallel_size", "attn_dp", "moe_tensor_parallel_size",
    "moe_expert_parallel_size", "num_pipeline_stages",
)


def _require(condition, message):
    if not condition:
        raise DeepSeekV4CoverageError(f"V4 step-cost table rejected: {message}")


class DeepSeekV4StepCostTable:
    """Measured lane step times keyed by role and batch composition."""

    def __init__(self, path, roles):
        self.path = path
        self._roles = roles

    def step_time_ms(self, role, batch):
        """Look up the step of one lane batch.

        A request still consuming prompt tokens counts as a prefill request; on
        a KV-receiving decode lane this is the first step after the transfer.
        """
        prefill_tokens = [
            tokens for request, tokens, done in zip(batch.requests, batch.num_tokens, batch.num_context_tokens)
            if done < request.num_prefill_tokens
        ]
        k, k_tokens = len(prefill_tokens), sum(prefill_tokens)
        m = len(batch.requests) - k
        for entry in self._roles.get(role, ()):
            decode = entry["num_decode_requests"]
            if (entry["num_prefill_requests"], entry["num_prefill_tokens"]) == (k, k_tokens) \
                    and decode["min"] <= m <= decode["max"]:
                return entry["time_ms"]
        raise DeepSeekV4CoverageError(
            f"{self.path} has no {role} step with {k} prefill requests of {k_tokens} tokens "
            f"and {m} decode requests"
        )


def load_step_cost_table(predictor_config, replica_config):
    """Bind the table of this replica; None when its V4 steps are not table-priced."""
    model_config = replica_config.model_config
    if predictor_config.enable_dummy_mode or \
            model_config.get_model_architecture_profile().profile_id != "deepseek_v4":
        return None
    attention = resolve_measurement_input_paths(
        predictor_config, MeasurementType.CUDA_EVENT, device=replica_config.device,
        model=model_config.get_name(), network_device=replica_config.network_device,
    ).attention
    path = os.path.join(os.path.dirname(attention), FILE_NAME)
    if not os.path.exists(path):
        return None
    with open(path) as source:
        table = json.load(source)
    _require(table["schema_version"] == SCHEMA, f"{path} schema {table['schema_version']!r}")
    _require(table["qualification"] == QUALIFICATION and table["formal_gate_admission"] is False,
             f"{path} must be a diagnostic input without formal-gate admission")
    _require(table["model_name"] == model_config.get_name() and table["device"] == replica_config.device,
             f"{path} was measured for {table['model_name']} on {table['device']}")
    configured = {name: getattr(replica_config, name) for name in TOPOLOGY_FIELDS}
    _require(table["topology"] == configured,
             f"{path} topology {table['topology']} differs from the replica {configured}")
    return DeepSeekV4StepCostTable(path, table["roles"])
