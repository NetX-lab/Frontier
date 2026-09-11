"""Bounded, partial SGLang primitive calibration; no full-model timing fit.

The only feature is physical token count, and the identity fixes all dimensions
and runtime settings. Validation samples never enter fitting. Independent
weight/input buffers make graph-length sensitivity a required quality check.
"""

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import math
from statistics import median

from frontier.profiling.runtime.sglang_primitives import MEASUREMENT_TYPE, PRIMITIVES
from frontier.profiling.runtime.sglang_dense import DENSE_PRIMITIVES, validate_dense_spec
from frontier.profiling.runtime.sglang_gdn import GDN_PRIMITIVES, validate_gdn_core_spec
from frontier.profiling.runtime.sglang_attention import (
    ATTENTION_PRIMITIVES, validate_attention_decode_spec, validate_attention_workload,
)
from frontier.profiling.runtime.sglang_moe import (
    MOE_ROUTING_PRIMITIVES, validate_moe_routing_spec,
)
from frontier.runtime_cost.sglang import CostEstimate, RuntimeIdentity, _digest

PROVENANCE_FIELDS = ("versions", "hardware", "extra_collective_environment", "method", "producer_source_sha256")


def _provenance(payload):
    return {key: payload[key] for key in PROVENANCE_FIELDS}


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def aggregate_profiles(payloads, *, split, identity):
    """Require complete aligned rank cohorts; aggregate max rank per repetition."""
    payloads = deepcopy(list(payloads))
    world = identity.tensor_parallel_size
    if split not in {"calibration", "validation"} or len(payloads) != world:
        raise ValueError("Complete calibration/validation rank cohort required")
    if (any(type(p.get("rank")) is not int for p in payloads)
            or {p["rank"] for p in payloads} != set(range(world))):
        raise ValueError("Duplicate or missing ranks")
    payloads.sort(key=lambda p: p["rank"])
    groups, keys_per_rank = defaultdict(dict), []
    for payload in payloads:
        if (type(payload.get("schema_version")) is not int or payload["schema_version"] != 1
                or payload.get("split") != split or payload.get("world_size") != world
                or RuntimeIdentity(**payload["identity"]) != identity):
            raise ValueError("Primitive profile schema/split/runtime identity mismatch")
        if any(not payload.get(key) or payload[key] != payloads[0].get(key)
               for key in PROVENANCE_FIELDS if key != "extra_collective_environment"):
            raise ValueError("Incomplete or mismatched producer provenance")
        if (not isinstance(payload.get("extra_collective_environment"), dict)
                or payload["extra_collective_environment"] != payloads[0]["extra_collective_environment"]):
            raise ValueError("Collective environment differs across ranks")
        keys = set()
        for row in payload["rows"]:
            name, size, count = (row[k] for k in ("primitive", "physical_size", "invocations_per_graph"))
            if (name not in PRIMITIVES or type(size) is not int or size < 1
                    or type(count) is not int or count < 2 or row["rank"] != payload["rank"]
                    or row["correctness_checked"] is not True or row["measurement_type"] != MEASUREMENT_TYPE
                    or not isinstance(row["backend"], str) or not row["backend"].strip()
                    or not isinstance(row["samples_ms"], list) or len(row["samples_ms"]) < 20
                    or not all(_positive(v) for v in row["samples_ms"])):
                raise ValueError("Invalid primitive profile row or fewer than 20 repetitions")
            if name == "tp_allreduce" and row["backend"] not in {"ca", "qr"}:
                raise ValueError("RCCL fallback is not custom-reduction calibration")
            if name in DENSE_PRIMITIVES:
                validate_dense_spec(row.get("dense_spec"), world, name)
            if name in GDN_PRIMITIVES:
                validate_gdn_core_spec(row.get("gdn_core_spec"), world)
                if (type(row.get("logical_size")) is not int
                        or not 1 <= row["logical_size"] <= size):
                    raise ValueError("Invalid GDN core logical/physical workload")
            if name in ATTENTION_PRIMITIVES:
                validate_attention_decode_spec(row.get("attention_decode_spec"), world)
                workload = validate_attention_workload(
                    size, row.get("logical_size"), row.get("physical_context_lens"),
                    row["attention_decode_spec"])
                if row.get("attention_workload") != workload:
                    raise ValueError("Attention workload summary disagrees with physical contexts")
            if name in MOE_ROUTING_PRIMITIVES:
                validate_moe_routing_spec(row.get("moe_routing_spec"))
            if (row.get("kernel_trace_kind") != "representative_graph" or row.get("kernel_trace_invocations") != 2
                    or (payload["rank"] == 0 and (not row["kernel_names"] or not row.get("kernel_count")
                        or row["kernel_count"] % 2))):
                raise ValueError("Representative rank-0 graph kernel evidence is required")
            key = (name, size, count)
            if key in keys:
                raise ValueError("Duplicate primitive row")
            keys.add(key)
            groups[key][payload["rank"]] = row
        keys_per_rank.append(keys)
    if not groups or any(keys != keys_per_rank[0] for keys in keys_per_rank):
        raise ValueError("Primitive coverage differs across ranks")
    result = []
    for (name, size, count), ranks in sorted(groups.items()):
        rows = [ranks[rank] for rank in range(world)]
        if len({len(r["samples_ms"]) for r in rows}) != 1 or len({r["backend"] for r in rows}) != 1:
            raise ValueError("Rank timing alignment/backend mismatch")
        if len({_digest(r.get("dense_spec")) for r in rows}) != 1:
            raise ValueError("Dense primitive shape contract differs across ranks")
        if len({_digest(r.get("gdn_core_spec")) for r in rows}) != 1:
            raise ValueError("GDN core shape contract differs across ranks")
        if len({_digest(r.get("attention_decode_spec")) for r in rows}) != 1:
            raise ValueError("Attention decode shape contract differs across ranks")
        if len({_digest(r.get("moe_routing_spec")) for r in rows}) != 1:
            raise ValueError("MoE routing shape contract differs across ranks")
        if len({r.get("logical_size") for r in rows}) != 1:
            raise ValueError("Stateful logical workload differs across ranks")
        if len({_digest(r.get("physical_context_lens")) for r in rows}) != 1:
            raise ValueError("Attention physical contexts differ across ranks")
        times = [max(values) for values in zip(*(r["samples_ms"] for r in rows))]
        center = median(times)
        result.append({"primitive": name, "physical_size": size, "invocations_per_graph": count,
            "backend": rows[0]["backend"], "median_ms": center,
            "p10_ms": _percentile(times, .1), "p90_ms": _percentile(times, .9),
            "relative_spread": (_percentile(times, .9) - _percentile(times, .1)) / center,
            "rank_medians_ms": [median(r["samples_ms"]) for r in rows],
            "aligned_max_rank_samples_ms": times, "kernel_names": rows[0]["kernel_names"]})
        if name in DENSE_PRIMITIVES:
            result[-1]["dense_spec"] = rows[0]["dense_spec"]
        if name in GDN_PRIMITIVES:
            result[-1]["gdn_core_spec"] = rows[0]["gdn_core_spec"]
            result[-1]["logical_size"] = rows[0]["logical_size"]
        if name in ATTENTION_PRIMITIVES:
            result[-1]["attention_decode_spec"] = rows[0]["attention_decode_spec"]
            result[-1]["attention_workload"] = rows[0]["attention_workload"]
            result[-1]["logical_size"] = rows[0]["logical_size"]
            result[-1]["physical_context_lens"] = rows[0]["physical_context_lens"]
        if name in MOE_ROUTING_PRIMITIVES:
            result[-1]["moe_routing_spec"] = rows[0]["moe_routing_spec"]
    return result, "sha256:" + _digest(payloads)


def primitive_for_query(query):
    if query.component in (DENSE_PRIMITIVES + GDN_PRIMITIVES + ATTENTION_PRIMITIVES
                           + MOE_ROUTING_PRIMITIVES):
        return query.component
    if query.component == "input_layernorm":
        return "gemma_norm" if query.residual_from_layer is None else "gemma_residual_norm"
    return {"post_attention_layernorm": "gemma_residual_norm",
        "shared_gate_sigmoid_mul_add": "shared_gate", "attn_post_proj_gate": "attention_post",
        "attention_tp_allreduce": "tp_allreduce", "mlp_tp_allreduce": "tp_allreduce"}.get(query.component)


class PrimitiveCalibration:
    """Partial cost provider: exact identity, two endpoints, no extrapolation.

This interpolates isolated graph replay cost, NOT pure kernel duration or
full-forward performance. Graph-length/spread checks can reject a primitive.
Held-out data is accessible only to the separate validation report method.
"""

    def __init__(self, payloads, *, identity, quality_limit=.10):
        if not _positive(quality_limit) or quality_limit > .25:
            raise ValueError("Quality limit must be in (0, 0.25]")
        payloads = list(payloads)
        self.identity, self.quality_limit = identity, quality_limit
        self.rows, self.source = aggregate_profiles(payloads, split="calibration", identity=identity)
        self.provenance = deepcopy(_provenance(payloads[0]))
        self.models, self.quality = {}, {}
        for name in sorted({r["primitive"] for r in self.rows}):
            rows = [r for r in self.rows if r["primitive"] == name]
            sizes = sorted({r["physical_size"] for r in rows})
            counts = sorted({r["invocations_per_graph"] for r in rows})
            if len(sizes) != 2 or len(counts) < 2 or len(rows) != len(sizes) * len(counts):
                raise ValueError("Calibration needs two sizes and at least two graph lengths per primitive")
            if len({r["backend"] for r in rows}) != 1:
                raise ValueError("Interpolation cannot cross primitive backend selections")
            if len({_digest(r.get("dense_spec")) for r in rows}) != 1:
                raise ValueError("Interpolation cannot cross dense shape contracts")
            if len({_digest(r.get("gdn_core_spec")) for r in rows}) != 1:
                raise ValueError("Interpolation cannot cross GDN core shape contracts")
            if len({_digest(r.get("attention_decode_spec")) for r in rows}) != 1:
                raise ValueError("Interpolation cannot cross attention shape contracts")
            if len({_digest(r.get("moe_routing_spec")) for r in rows}) != 1:
                raise ValueError("Interpolation cannot cross MoE routing shape contracts")
            padding_counts = {r["physical_size"] - r["logical_size"]
                              for r in rows if name in GDN_PRIMITIVES + ATTENTION_PRIMITIVES}
            if name in GDN_PRIMITIVES + ATTENTION_PRIMITIVES and len(padding_counts) != 1:
                raise ValueError("Stateful interpolation requires a fixed padding count")
            attention_workloads = [r["attention_workload"] for r in rows
                                   if name in ATTENTION_PRIMITIVES]
            if attention_workloads and (len({r["active_context_length"] for r in attention_workloads}) != 1
                    or len({r["padding_context_length"] for r in attention_workloads}) != 1):
                raise ValueError("Attention interpolation cannot cross context lengths")
            selected = [r for r in rows if r["invocations_per_graph"] == counts[-1]]
            sensitivity = max((max(r["median_ms"] for r in rows if r["physical_size"] == size)
                - min(r["median_ms"] for r in rows if r["physical_size"] == size))
                / next(r["median_ms"] for r in selected if r["physical_size"] == size) for size in sizes)
            spread = max(r["relative_spread"] for r in rows)
            self.quality[name] = {"admitted": max(sensitivity, spread) <= quality_limit,
                "graph_length_sensitivity": sensitivity, "max_relative_spread": spread,
                "limit": quality_limit, "calibration_sizes": sizes, "graph_lengths": counts,
                "selected_graph_length": counts[-1]}
            self.models[name] = {"sizes": sizes, "times": [r["median_ms"] for r in selected],
                "backend": rows[0]["backend"], "count": counts[-1], "dense_spec": rows[0].get("dense_spec"),
                "gdn_core_spec": rows[0].get("gdn_core_spec"),
                "attention_decode_spec": rows[0].get("attention_decode_spec"),
                "moe_routing_spec": rows[0].get("moe_routing_spec"),
                "attention_workload": attention_workloads[0] if attention_workloads else None,
                "padding_count": next(iter(padding_counts)) if padding_counts else None}

    def primitive_ms(self, name, size):
        if name not in self.models:
            raise ValueError(f"No primitive calibration for {name}")
        if type(size) is not int:
            raise ValueError("Physical size must be an integer")
        model = self.models[name]
        lo, hi = model["sizes"]
        if not lo <= size <= hi:
            raise ValueError("Primitive extrapolation is not permitted")
        a, b = model["times"]
        return a + (b - a) * (size - lo) / (hi - lo)

    def __call__(self, query):
        if query.identity != self.identity:
            raise ValueError("Primitive cost query belongs to another runtime")
        name = primitive_for_query(query)
        if name not in self.quality or not self.quality[name]["admitted"]:
            raise ValueError(f"Missing or quality-rejected primitive calibration for {query.component}")
        if (name in GDN_PRIMITIVES
                and query.workload.physical_size - query.workload.logical_size
                    != self.models[name]["padding_count"]):
            raise ValueError("GDN core query padding differs from calibrated ownership")
        if name in ATTENTION_PRIMITIVES:
            model = self.models[name]
            workload = validate_attention_workload(
                query.workload.physical_size, query.workload.logical_size,
                query.workload.physical_context_lens, model["attention_decode_spec"])
            expected = model["attention_workload"]
            if any(workload[k] != expected[k] for k in (
                    "padding_count", "active_context_length", "padding_context_length")):
                raise ValueError("Attention query workload differs from calibrated context ownership")
        return CostEstimate(query.key, self.primitive_ms(name, query.workload.physical_size),
            self.source + "/" + name, MEASUREMENT_TYPE,
            "calibrated_collective" if query.is_collective else "isolated_compute")

    def validate(self, payloads):
        payloads = list(payloads)
        rows, source = aggregate_profiles(payloads, split="validation", identity=self.identity)
        if _provenance(payloads[0]) != self.provenance:
            raise ValueError("Validation producer/hardware/environment provenance changed")
        expected = set(self.models)
        if {r["primitive"] for r in rows} != expected:
            raise ValueError("Validation must cover every calibrated primitive")
        results = []
        for name, model in self.models.items():
            relevant = [r for r in rows if r["primitive"] == name]
            for size in sorted({r["physical_size"] for r in relevant}):
                if size in model["sizes"]:
                    raise ValueError("Validation size overlaps calibration")
                candidates = [r for r in relevant if r["physical_size"] == size]
                if {r["invocations_per_graph"] for r in candidates} != set(self.quality[name]["graph_lengths"]):
                    raise ValueError("Validation graph lengths differ from calibration")
                if any(r["backend"] != model["backend"] for r in candidates):
                    raise ValueError("Validation primitive backend changed")
                if any(r.get("dense_spec") != model["dense_spec"] for r in candidates):
                    raise ValueError("Validation dense shape contract changed")
                if any(r.get("gdn_core_spec") != model["gdn_core_spec"] for r in candidates):
                    raise ValueError("Validation GDN core shape contract changed")
                if any(r.get("attention_decode_spec") != model["attention_decode_spec"] for r in candidates):
                    raise ValueError("Validation attention shape contract changed")
                if any(r.get("moe_routing_spec") != model["moe_routing_spec"] for r in candidates):
                    raise ValueError("Validation MoE routing shape contract changed")
                if (name in GDN_PRIMITIVES
                        and any(r["physical_size"] - r["logical_size"] != model["padding_count"]
                                for r in candidates)):
                    raise ValueError("Validation GDN core padding ownership changed")
                if name in ATTENTION_PRIMITIVES:
                    expected = model["attention_workload"]
                    if any(any(r["attention_workload"][k] != expected[k] for k in (
                            "padding_count", "active_context_length", "padding_context_length"))
                           for r in candidates):
                        raise ValueError("Validation attention context ownership changed")
                observed = next(r["median_ms"] for r in candidates if r["invocations_per_graph"] == model["count"])
                prediction = self.primitive_ms(name, size)
                error = abs(prediction - observed) / observed
                sensitivity = (max(r["median_ms"] for r in candidates) - min(r["median_ms"] for r in candidates)) / observed
                spread = max(r["relative_spread"] for r in candidates)
                results.append({"primitive": name, "physical_size": size, "predicted_ms": prediction,
                    "observed_ms": observed, "absolute_relative_error": error,
                    "graph_length_sensitivity": sensitivity, "max_relative_spread": spread,
                    "passed": self.quality[name]["admitted"] and max(error, sensitivity, spread) <= self.quality_limit})
        return {"identity": asdict(self.identity), "calibration_source": self.source,
            "validation_source": source, "measurement_type": MEASUREMENT_TYPE,
            "calibration_quality": deepcopy(self.quality), "validation": results,
            "all_primitives_passed": all(r["passed"] for r in results),
            "full_forward_parity_admitted": False,
            "note": "Two-endpoint interpolation, trained only on calibration physical sizes. "
                    "Isolated synthetic streaming-weight graphs; not checkpoint, full-forward, "
                    "serving or multi-node validation. Max rank per aligned replay is a local-duration "
                    "aggregate, not a cross-device clock-synchronized makespan."}
