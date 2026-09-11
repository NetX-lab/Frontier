"""Strict exact-query costs for route-conditioned SGLang MoE primitives."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import math
from statistics import median

from frontier.profiling.runtime.sglang_moe import (
    MOE_ROUTED_PRIMITIVES, validate_moe_route_workload, validate_moe_routed_spec,
)
from frontier.profiling.runtime.sglang_primitives import MEASUREMENT_TYPE
from frontier.runtime_cost.sglang import CostEstimate, RuntimeIdentity, _digest


PROVENANCE_FIELDS = (
    "versions", "hardware", "extra_collective_environment", "method",
    "producer_source_sha256",
)


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _percentile(values, fraction):
    values = sorted(values)
    index = (len(values) - 1) * fraction
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def aggregate_routed_profiles(payloads, *, identity):
    """Aggregate aligned exact-query samples by max rank for each replay."""
    payloads = deepcopy(list(payloads))
    world = identity.tensor_parallel_size
    if len(payloads) != world:
        raise ValueError("Complete routed-profile rank cohort required")
    if ({payload.get("rank") for payload in payloads} != set(range(world))
            or any(type(payload.get("rank")) is not int for payload in payloads)):
        raise ValueError("Duplicate or missing routed-profile ranks")
    payloads.sort(key=lambda payload: payload["rank"])
    groups, keys_per_rank = defaultdict(dict), []
    for payload in payloads:
        if (payload.get("schema_version") != 1
                or payload.get("profile_kind") != "exact_routed_query"
                or payload.get("world_size") != world
                or RuntimeIdentity(**payload["identity"]) != identity):
            raise ValueError("Routed profile schema/runtime identity mismatch")
        if any(not payload.get(field) or payload[field] != payloads[0].get(field)
               for field in PROVENANCE_FIELDS
               if field != "extra_collective_environment"):
            raise ValueError("Incomplete or mismatched routed-profile provenance")
        if (not isinstance(payload.get("extra_collective_environment"), dict)
                or payload["extra_collective_environment"]
                    != payloads[0]["extra_collective_environment"]):
            raise ValueError("Routed-profile collective environment differs across ranks")
        keys = set()
        for row in payload.get("rows", ()):
            key = (row.get("query_key"), row.get("invocations_per_graph"))
            spec = row.get("moe_routed_spec")
            validate_moe_routed_spec(spec, world)
            workload = validate_moe_route_workload(
                row.get("physical_size"), row.get("physical_expert_counts"), spec)
            primitive = row.get("primitive")
            expected_backend = (spec["sorting_backend"] if primitive == "moe_sorting"
                                else spec["experts_backend"])
            if (primitive not in MOE_ROUTED_PRIMITIVES
                    or not isinstance(key[0], str) or len(key[0]) != 64
                    or any(char not in "0123456789abcdef" for char in key[0])
                    or type(key[1]) is not int or key[1] < 2
                    or type(row.get("layer_id")) is not int or row["layer_id"] < 0
                    or row.get("rank") != payload["rank"]
                    or row.get("backend") != expected_backend
                    or row.get("measurement_type") != MEASUREMENT_TYPE
                    or row.get("correctness_checked") is not True
                    or row.get("moe_route_workload") != workload
                    or row.get("kernel_trace_kind") != "representative_graph"
                    or row.get("kernel_trace_invocations") != 2
                    or not isinstance(row.get("samples_ms"), list)
                    or len(row["samples_ms"]) < 20
                    or not all(_positive(value) for value in row["samples_ms"])
                    or (payload["rank"] == 0
                        and (not row.get("kernel_names") or not row.get("kernel_count")
                             or row["kernel_count"] % 2
                             or (primitive == "moe_sorting"
                                 and any("aiter::opus_moe_sorting_entry" not in name
                                         for name in row["kernel_names"]))
                             or (primitive == "moe_experts_quant_gemm_combine"
                                 and (not any("mfma_moe1_" in name
                                              for name in row["kernel_names"])
                                      or not any("mfma_moe2_" in name
                                                 for name in row["kernel_names"])))))):
                raise ValueError("Invalid exact routed-profile row")
            if key in keys:
                raise ValueError("Duplicate exact routed-profile row")
            keys.add(key)
            groups[key][payload["rank"]] = row
        keys_per_rank.append(keys)
    if not groups or any(keys != keys_per_rank[0] for keys in keys_per_rank):
        raise ValueError("Exact routed-profile coverage differs across ranks")

    result = []
    for (query_key, count), ranks in sorted(groups.items()):
        rows = [ranks[rank] for rank in range(world)]
        aligned = {field: {_digest(row.get(field)) for row in rows} for field in (
            "physical_expert_counts", "moe_routed_spec", "moe_route_workload")}
        if (any(len(values) != 1 for values in aligned.values())
                or len({row["layer_id"] for row in rows}) != 1
                or len({row["backend"] for row in rows}) != 1
                or len({len(row["samples_ms"]) for row in rows}) != 1):
            raise ValueError("Exact routed workload/backend differs across ranks")
        samples = [max(values) for values in zip(*(
            row["samples_ms"] for row in rows))]
        center = median(samples)
        if len({row["primitive"] for row in rows}) != 1:
            raise ValueError("Exact routed primitive differs across ranks")
        result.append({
            "primitive": rows[0]["primitive"],
            "query_key": query_key,
            "layer_id": rows[0]["layer_id"],
            "physical_size": rows[0]["physical_size"],
            "physical_expert_counts": rows[0]["physical_expert_counts"],
            "invocations_per_graph": count,
            "backend": rows[0]["backend"],
            "median_ms": center,
            "p10_ms": _percentile(samples, .1),
            "p90_ms": _percentile(samples, .9),
            "relative_spread": (
                _percentile(samples, .9) - _percentile(samples, .1)) / center,
            "rank_medians_ms": [median(row["samples_ms"]) for row in rows],
            "aligned_max_rank_samples_ms": samples,
            "kernel_names": rows[0]["kernel_names"],
            "moe_routed_spec": rows[0]["moe_routed_spec"],
            "moe_route_workload": rows[0]["moe_route_workload"],
        })
    return result, "sha256:" + _digest(payloads)


class ExactRoutedCalibration:
    """Direct exact-query provider with replay-quality admission, not a fit."""

    def __init__(self, payloads, *, identity, quality_limit=.10):
        if not _positive(quality_limit) or quality_limit > .25:
            raise ValueError("Quality limit must be in (0, 0.25]")
        self.identity = identity
        self.rows, self.source = aggregate_routed_profiles(payloads, identity=identity)
        self.quality_limit = quality_limit
        self.models, self.quality = {}, {}
        by_query = defaultdict(list)
        for row in self.rows:
            by_query[row["query_key"]].append(row)
        for query_key, rows in by_query.items():
            counts = sorted(row["invocations_per_graph"] for row in rows)
            if len(counts) < 2 or len(counts) != len(set(counts)):
                raise ValueError("Exact routed query needs at least two graph lengths")
            if (len({row["layer_id"] for row in rows}) != 1
                    or len({_digest(row["physical_expert_counts"]) for row in rows}) != 1
                    or len({_digest(row["moe_routed_spec"]) for row in rows}) != 1
                    or len({row["backend"] for row in rows}) != 1):
                raise ValueError("Graph lengths changed the exact routed query")
            selected = next(
                row for row in rows if row["invocations_per_graph"] == counts[-1])
            sensitivity = (max(row["median_ms"] for row in rows)
                           - min(row["median_ms"] for row in rows)) / selected["median_ms"]
            spread = max(row["relative_spread"] for row in rows)
            admitted = max(sensitivity, spread) <= quality_limit
            self.quality[query_key] = {
                "layer_id": selected["layer_id"],
                "admitted": admitted,
                "graph_length_sensitivity": sensitivity,
                "max_relative_spread": spread,
                "limit": quality_limit,
                "graph_lengths": counts,
                "selected_graph_length": counts[-1],
            }
            self.models[query_key] = selected

    def __call__(self, query):
        if query.identity != self.identity or query.component not in MOE_ROUTED_PRIMITIVES:
            raise ValueError("Exact routed cost query belongs to another runtime/boundary")
        if query.key not in self.models or not self.quality[query.key]["admitted"]:
            raise ValueError("Missing or quality-rejected exact routed query")
        row = self.models[query.key]
        if (query.component != row["primitive"] or query.layer_id != row["layer_id"]
                or list(query.physical_expert_counts) != row["physical_expert_counts"]
                or query.workload.physical_size != row["physical_size"]):
            raise ValueError("Exact routed query payload differs from profile")
        return CostEstimate(
            query.key, row["median_ms"], self.source + "/" + row["primitive"],
            MEASUREMENT_TYPE, "isolated_compute")

    def report(self):
        rows = list(self.quality.values())
        return {
            "identity": asdict(self.identity),
            "profile_source": self.source,
            "measurement_type": MEASUREMENT_TYPE,
            "queries": len(rows),
            "all_queries_passed": bool(rows) and all(row["admitted"] for row in rows),
            "maximum_graph_length_sensitivity": max(
                row["graph_length_sensitivity"] for row in rows),
            "maximum_relative_spread": max(row["max_relative_spread"] for row in rows),
            "full_forward_parity_admitted": False,
            "note": "Direct exact-route measurements; no interpolation across route histograms, "
                    "checkpoint load, full-forward timing, or target correction fit.",
        }
