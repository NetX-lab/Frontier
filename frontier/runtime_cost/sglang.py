"""Opt-in, profile-provider-driven SGLang decoder cost composition.

This adapter produces native ExecutionTime objects without loading or mutating
the generic sklearn predictors. It is a static one-node decode contract, not a
serving scheduler, profile fitter, or permission to extrapolate to EP/multi-node.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
from collections import Counter
import hashlib
import json
import math
from typing import Callable

from frontier.config.precision_type import PrecisionType
from frontier.operators.sglang_family import (
    SGLANG_RUNTIME_CONTRACT, runtime_operator_name,
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def capture_stack_signature(manifest):
    return "sha256:" + _digest({
        "versions": {k: v for k, v in manifest["versions"].items() if not k.startswith("frontier_")},
        "server_args": manifest["server_args"], "environment": manifest["environment"]})


@dataclass(frozen=True)
class RuntimeIdentity:
    model_fingerprint: str
    device: str
    tensor_parallel_size: int
    runtime_stack_signature: str
    contract: str = SGLANG_RUNTIME_CONTRACT
    expert_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    nodes: int = 1
    data_parallel_size: int = 1

    def __post_init__(self):
        if self.contract != SGLANG_RUNTIME_CONTRACT:
            raise ValueError("Unknown SGLang runtime cost contract")
        if (not isinstance(self.model_fingerprint, str) or len(self.model_fingerprint) != 64
                or any(c not in "0123456789abcdef" for c in self.model_fingerprint)):
            raise ValueError("Invalid model fingerprint")
        for name in ("device", "runtime_stack_signature"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Runtime identity requires {name}")
        for name in ("tensor_parallel_size", "expert_parallel_size", "pipeline_parallel_size",
                     "nodes", "data_parallel_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"Invalid topology field {name}")
        if (self.expert_parallel_size, self.pipeline_parallel_size, self.nodes, self.data_parallel_size) != (1, 1, 1, 1):
            raise ValueError("SGLang cost composition supports one-node TP-only execution")
        if self.tensor_parallel_size == 1:
            raise ValueError("This collective-bearing contract requires TP > 1")

    @classmethod
    def for_model(cls, model_config, *, device, tensor_parallel_size, runtime_stack_signature):
        # Deliberately includes checkpoint identity, dimensions, quantization and
        # layer schedule. Equal architecture-profile labels are not sufficient.
        if model_config.get_gdn_config() is None:
            raise ValueError("SGLang runtime identity requires a hybrid GDN model")
        shape = {name: getattr(model_config, name, None) for name in (
            "name", "model_type", "num_layers", "embedding_dim", "mlp_hidden_dim",
            "num_q_heads", "num_kv_heads", "head_dim", "num_experts", "num_experts_per_tok",
            "share_expert_dim", "dtype", "use_qk_norm", "attn_output_gate", "rms_norm_eps",
            "rope_theta", "rope_scaling", "partial_rotary_factor", "vocab_size",
            "use_bias", "use_qkv_bias", "use_gated_mlp",
        )}
        if shape["name"] is None:
            shape["name"] = model_config.get_name()
        dtype = getattr(model_config, "dtype", getattr(model_config, "torch_dtype", None))
        shape["dtype"] = PrecisionType.from_torch_dtype(str(dtype)).name
        shape["head_dim"] = model_config.get_head_dim()
        shape.update(architecture=model_config.get_model_architecture_profile().profile_id,
                     quantization=model_config.get_quant_signature(),
                     gdn=asdict(model_config.get_gdn_config()),
                     layers=[model_config.is_gdn_layer(i) for i in range(model_config.num_layers)])
        return cls(_digest(shape), device, tensor_parallel_size, runtime_stack_signature)


@dataclass(frozen=True)
class DecodeWorkload:
    """Physical context lengths include the current query token, like SGLang seq_lens.

Padding context lengths must be supplied explicitly, never guessed from logical
requests. This object has no observed timing or output-logit fields.
"""

    logical_size: int
    physical_context_lens: tuple[int, ...]
    graph_mode: str = "FULL"

    def __post_init__(self):
        object.__setattr__(self, "physical_context_lens", tuple(self.physical_context_lens))
        if (type(self.logical_size) is not int or not 0 < self.logical_size <= self.physical_size
                or any(type(n) is not int or n < 1 for n in self.physical_context_lens)):
            raise ValueError("Decode workload requires logical size and every physical context length")
        if self.graph_mode != "FULL":
            raise ValueError("Runtime cost v1 supports full-graph non-speculative decode only")

    @property
    def physical_size(self):
        return len(self.physical_context_lens)


@dataclass(frozen=True)
class CostQuery:
    identity: RuntimeIdentity
    layer_id: int
    layer_kind: str
    component: str
    workload: DecodeWorkload
    residual_from_layer: int | None = None
    physical_expert_counts: tuple[int, ...] = ()

    def __post_init__(self):
        if (not isinstance(self.identity, RuntimeIdentity) or not isinstance(self.workload, DecodeWorkload)
                or type(self.layer_id) is not int or self.layer_id < 0
                or self.layer_kind not in {"gdn", "attention"}):
            raise ValueError("Invalid runtime cost query identity/layer/workload")
        runtime_operator_name(self.component)
        if ((self.component.startswith("gdn_") and self.layer_kind != "gdn")
                or (self.component.startswith("attn_") and self.layer_kind != "attention")):
            raise ValueError("Cost query mixer does not match layer kind")
        residual = (self.layer_id - 1 if self.component == "input_layernorm" and self.layer_id > 0
                    else self.layer_id if self.component == "post_attention_layernorm" else None)
        if self.residual_from_layer != residual or isinstance(self.residual_from_layer, bool):
            raise ValueError("Invalid fused residual ownership")
        object.__setattr__(self, "physical_expert_counts", tuple(self.physical_expert_counts))
        routed = self.component in {"moe_sorting", "moe_experts_quant_gemm_combine"}
        if (bool(self.physical_expert_counts) != routed or any(type(n) is not int or n < 0
                for n in self.physical_expert_counts) or (routed and sum(self.physical_expert_counts) == 0)):
            raise ValueError("Routed scopes require physical expert counts; other scopes must omit them")

    @property
    def operator(self):
        return runtime_operator_name(self.component)

    @property
    def key(self):
        # Full workload identity is intentionally conservative: no hidden
        # layer tying, padding erasure, context pooling or route extrapolation.
        return _digest(asdict(self))

    @property
    def is_collective(self):
        return self.component in {"attention_tp_allreduce", "mlp_tp_allreduce"}


@dataclass(frozen=True)
class CostEstimate:
    query_key: str
    milliseconds: float
    source: str
    measurement_type: str
    basis: str

    def __post_init__(self):
        if (not isinstance(self.query_key, str) or len(self.query_key) != 64
                or any(c not in "0123456789abcdef" for c in self.query_key)):
            raise ValueError("Estimate requires the exact query key")
        if (isinstance(self.milliseconds, bool) or not isinstance(self.milliseconds, (int, float))
                or not math.isfinite(self.milliseconds) or self.milliseconds <= 0):
            raise ValueError("Runtime costs must be finite and strictly positive")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("Runtime cost requires profile provenance")
        if self.measurement_type not in {"CUDA_EVENT", "KERNEL_ONLY", "HIP_GRAPH_EVENT", "HIP_GRAPH_REPLAY"}:
            raise ValueError("Unknown runtime measurement family")
        if self.basis not in {"isolated_compute", "calibrated_collective", "in_situ_diagnostic"}:
            raise ValueError("Unknown cost evidence basis")
        if self.measurement_type == "HIP_GRAPH_EVENT" and self.basis != "in_situ_diagnostic":
            raise ValueError("HIP graph scope events are diagnostic, not isolated kernel profiles")


@dataclass(frozen=True)
class RuntimeCostPrediction:
    execution: object
    layer_executions: tuple
    queries: tuple[CostQuery, ...]
    estimates: tuple[CostEstimate, ...]
    diagnostic: bool

    @property
    def decoder_ms(self):
        return self.execution.model_time_ms

    def summary(self):
        return {"contract": SGLANG_RUNTIME_CONTRACT, "decoder_ms": self.decoder_ms,
                "layers": len(self.layer_executions), "operators": len(self.queries),
                "diagnostic": self.diagnostic, "routing_conditioned": True,
                "full_forward_parity_admitted": False,
                "boundary": "Decoder input norm through final MLP reduction; excludes embedding, "
                            "final residual/norm, logits and host/scheduler work.",
                "measurement_types": sorted({e.measurement_type for e in self.estimates}),
                "sources": sorted({e.source for e in self.estimates})}


def _native_execution(op_times, *, layers):
    from frontier.entities.execution_time import ExecutionTime
    return ExecutionTime(
        num_layers_per_pipeline_stage=layers, is_moe=True, op_times=op_times,
        attention_rope_execution_time=0., attention_kv_cache_save_execution_time=0.,
        attention_decode_execution_time=0., attention_prefill_execution_time=0.,
        attention_layer_pre_proj_execution_time=0., attention_layer_post_proj_execution_time=0.,
        attn_norm_time=0., mlp_norm_time=0., add_time=0.,
        tensor_parallel_communication_time=0., pipeline_parallel_communication_time=0.,
        expert_parallel_communication_time=0., moe_gating_time=0., moe_shuffling_time=0.,
        schedule_time=0., sampler_e2e_time=0., prepare_inputs_e2e_time=0.,
        process_model_outputs_time=0., ray_comm_time=0.,
        # Mark both physical TP domains explicitly; no legacy allreduce fallback.
        attn_tensor_parallel_allreduce_time=0., moe_tensor_parallel_allreduce_time=0.,
        share_expert_tensor_parallel_allreduce_time=0.,
    )


class SGLangCostContract:
    """Explicitly construct this adapter to use runtime-specific cost ownership.

The provider receives only workload queries, never target observations. It must
return matching profile evidence or fail. All decoder layers are required; no
first-layer-times-N approximation or partial-stage residual ownership is allowed.
"""

    def __init__(self, model_config, identity: RuntimeIdentity):
        profile = model_config.get_model_architecture_profile()
        if (profile.profile_id != "qwen3_5_moe" or not model_config.is_moe
                or not model_config.get_gdn_config() or not model_config.attn_output_gate
                or not model_config.use_qk_norm or not model_config.share_expert_dim
                or set(model_config.get_moe_layer_ids()) != set(range(model_config.num_layers))):
            raise ValueError("Runtime contract requires the gated Qwen hybrid separate-shared MoE architecture")
        expected = RuntimeIdentity.for_model(model_config, device=identity.device,
            tensor_parallel_size=identity.tensor_parallel_size,
            runtime_stack_signature=identity.runtime_stack_signature)
        if expected != identity:
            raise ValueError("Runtime identity does not match active model dimensions/quantization")
        self.model_config, self.identity = deepcopy(model_config), identity

    def queries(self, workload: DecodeWorkload, routing_records) -> tuple[CostQuery, ...]:
        records = tuple(routing_records)
        expected_layers = set(range(self.model_config.num_layers))
        if (len(records) != len(expected_layers) or {r.layer_id for r in records} != expected_layers
                or len({r.batch_id for r in records}) != 1
                or len({r.capture_id for r in records}) != 1):
            raise ValueError("Cost planning requires complete unique layers from one routing batch/graph")
        routes = {r.layer_id: r for r in records}
        queries = []
        for layer in range(self.model_config.num_layers):
            route = routes[layer]
            if (route.rank != 0 or route.logical_size != workload.logical_size
                    or route.physical_size != workload.physical_size
                    or route.num_experts != self.model_config.num_experts
                    or route.top_k != self.model_config.num_experts_per_tok):
                raise ValueError("Routing/model/physical workload identity mismatch")
            if (route.logical_positive_weight_slots != sum(route.logical_expert_counts)
                    or route.padding_positive_weight_slots != sum(route.padding_expert_counts)):
                raise ValueError("Zero-weight valid selections need an explicit pruning-aware cost contract")
            # Native workload validation and fixed-width physical counts. Rank 0
            # may be used only after the separate all-rank routing audit passes.
            lane = route.to_lane_workload(include_padding=True)
            kind = "gdn" if self.model_config.is_gdn_layer(layer) else "attention"
            mixer = ("gdn_input_projections", "gdn_core_decode", "gdn_output_projection") if kind == "gdn" else (
                "attn_pre_proj_qknorm", "attn_rope", "attn_kv_cache_write", "attn_decode", "attn_post_proj_gate")
            components = ("input_layernorm", *mixer, "attention_tp_allreduce", "post_attention_layernorm",
                "shared_expert_gate_up", "shared_expert_activation", "shared_expert_down",
                "moe_router_linear", "moe_routing_topk", "moe_sorting", "moe_experts_quant_gemm_combine",
                "shared_gate_sigmoid_mul_add", "mlp_tp_allreduce")
            for component in components:
                queries.append(CostQuery(self.identity, layer, kind, component, workload,
                    residual_from_layer=(layer - 1 if component == "input_layernorm" and layer > 0
                                         else layer if component == "post_attention_layernorm" else None),
                    physical_expert_counts=(tuple(lane.local_token_counts) if component in {
                        "moe_sorting", "moe_experts_quant_gemm_combine"} else ())))
        return tuple(queries)

    def predict(self, workload: DecodeWorkload, routing_records,
                provider: Callable[[CostQuery], CostEstimate], *, allow_diagnostic=False) -> RuntimeCostPrediction:
        if type(allow_diagnostic) is not bool:
            raise ValueError("allow_diagnostic must be a bool")
        queries = self.queries(workload, routing_records)
        estimates, per_layer = [], [dict() for _ in range(self.model_config.num_layers)]
        for query in queries:
            estimate = provider(query)
            if not isinstance(estimate, CostEstimate) or estimate.query_key != query.key:
                raise ValueError(f"Missing/mismatched estimate for {query.operator}, layer {query.layer_id}")
            if estimate.basis == "in_situ_diagnostic":
                if not allow_diagnostic:
                    raise ValueError("In-situ timing requires explicit diagnostic opt-in")
            elif estimate.basis != ("calibrated_collective" if query.is_collective else "isolated_compute"):
                raise ValueError("Compute and independently calibrated collective evidence cannot be interchanged")
            if estimate.measurement_type == "CUDA_EVENT":
                raise ValueError("Eager CUDA_EVENT costs cannot price the full-graph decode contract")
            per_layer[query.layer_id][query.operator] = estimate.milliseconds
            estimates.append(estimate)
        layers = tuple(_native_execution(values, layers=1) for values in per_layer)
        # ExecutionTime's established per-layer carrier convention is preserved;
        # average AFTER evaluating each actual layer, then aggregate exactly once.
        average = {}
        for values in per_layer:
            for name, value in values.items():
                average[name] = average.get(name, 0.) + value / len(layers)
        execution = _native_execution(average, layers=len(layers))
        expected = math.fsum(e.milliseconds for e in estimates)
        if not math.isclose(execution.model_time_ms, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Native runtime composition lost or duplicated an operator cost")
        return RuntimeCostPrediction(execution, layers, queries, tuple(estimates),
            diagnostic=any(e.basis == "in_situ_diagnostic" for e in estimates))


class ExactRuntimeCostTable:
    """Strict profile exchange/coverage table. No automatic fitting/extrapolation.

Every row includes its query and provenance. The identity and query key are
checked before lookup. A future fitted provider must enforce its own independent
training/holdout split and feature support; this table never hides missing rows.
"""

    def __init__(self, payload, *, identity: RuntimeIdentity):
        if (not isinstance(payload, dict) or set(payload) != {"schema_version", "identity", "rows"}
                or type(payload["schema_version"]) is not int or payload["schema_version"] != 1
                or not isinstance(payload["rows"], list)
                or RuntimeIdentity(**payload["identity"]) != identity):
            raise ValueError("Runtime profile schema/identity mismatch")
        self.identity, self._estimates = identity, {}
        for row in payload["rows"]:
            if set(row) != {"query", "estimate"}:
                raise ValueError("Profile rows require query and estimate")
            query = dict(row["query"])
            query["identity"] = RuntimeIdentity(**query["identity"])
            query["workload"] = DecodeWorkload(**query["workload"])
            query["physical_expert_counts"] = tuple(query.get("physical_expert_counts", ()))
            query = CostQuery(**query)
            estimate = CostEstimate(**row["estimate"])
            if query.identity != identity or estimate.query_key != query.key or query.key in self._estimates:
                raise ValueError("Duplicate, mismatched or cross-runtime cost row")
            self._estimates[query.key] = estimate

    def __call__(self, query):
        if query.identity != self.identity:
            raise ValueError("Cost query belongs to another runtime")
        try:
            return self._estimates[query.key]
        except KeyError as exc:
            raise ValueError(f"No exact cost profile for layer {query.layer_id} / {query.operator}") from exc

    def audit(self, queries):
        queries = tuple(queries)
        if not queries or any(q.identity != self.identity for q in queries):
            raise ValueError("Coverage audit requires queries from the same runtime")
        missing = [{"layer_id": q.layer_id, "operator": q.operator, "query_key": q.key}
                   for q in queries if q.key not in self._estimates]
        return {"complete": not missing, "queries": len(queries), "missing": missing,
                "missing_by_operator": dict(sorted(Counter(row["operator"] for row in missing).items())),
                "note": "Exact row coverage only, not calibration-quality or full-forward admission."}
