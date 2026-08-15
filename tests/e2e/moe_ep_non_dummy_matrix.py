#!/usr/bin/env python3
"""Real-data, non-dummy E2E matrix for the MoE EP rank-staggering contract.

The harness deliberately invokes the checked-in architecture example wrappers.  It
does not change simulator semantics, invent profile rows, or convert a failed run
to a pass.  Generation is deterministic so the same manifest can be replayed on a
read-only baseline worktree.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ARCHITECTURE_CASE_COUNTS = {
    # The vLLM reference currently covers co-location and PDD.  PD-AF is kept
    # as a smaller Frontier-only structural sample until a vLLM PD-AF runtime
    # exists for a meaningful numerical comparison.
    "co-location": 50,
    "pd-disaggregation": 50,
    "pd-af-disaggregation": 10,
}
MODEL_ORDER = ("dense", "moe", "mixed")
PD_AF_VARIANT_INDICES = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11)
MODEL_SPECS: Mapping[str, Mapping[str, Any]] = {
    "dense": {
        "model_name": "llama2_7b_dense_example",
        "total_experts": 1,
        "router_topk": 1,
    },
    "moe": {
        "model_name": "Phi-tiny-MoE-instruct",
        "device": "h800",
        "total_experts": 16,
        "router_topk": 2,
    },
    "moe_standard": {
        "model_name": "qwen3-a3b-30b-moe",
        "device": "a800",
        "total_experts": 128,
        "router_topk": 8,
    },
    "mixed": {
        "model_name": "step-moe-noquant-small",
        "device": "h800",
        "total_experts": 24,
        "router_topk": 3,
    },
}
ROUTING_DISTRIBUTIONS = ("balanced", "random", "skewed", "zipf")
WORKLOADS: Mapping[str, tuple[int, int, int]] = {
    "prefill-heavy": (8, 1, 1),
    "decode-heavy": (2, 4, 1),
    "mixed": (4, 4, 2),
    "zero-routed": (2, 2, 1),
}
VARIANTS = (
    ("balanced", "prefill-heavy"),
    ("random", "prefill-heavy"),
    ("skewed", "prefill-heavy"),
    ("zipf", "prefill-heavy"),
    ("balanced", "decode-heavy"),
    ("random", "decode-heavy"),
    ("skewed", "decode-heavy"),
    ("zipf", "decode-heavy"),
    ("balanced", "mixed"),
    ("random", "mixed"),
    ("skewed", "zero-routed"),
    ("zipf", "zero-routed"),
)
REQUIRED_PROFILE_METADATA_COLUMNS = (
    "profiling_precision",
    "model_arch",
    "model_architecture_profile",
    "quant_signature",
    "measurement_type",
)


_EP_WORKLOAD_LINE_RE = re.compile(
    r"\[EP-WORKLOAD\]\[(?P<cluster>[^\]]+)\]\s+"
    r"batch_id=(?P<batch_id>-?\d+),\s+"
    r"layer_id=(?P<layer_id>-?\d+),\s+"
    r"ep_id=(?P<ep_id>-?\d+),\s+"
    r"moe_ep_size=(?P<moe_ep_size>\d+),\s+"
    r"per_expert_tokens=(?P<per_expert_tokens>\{.*\}),\s+"
    r"lane_compute_ms=(?P<lane_compute_ms>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s+"
    r"lane_comm_ms=(?P<lane_comm_ms>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
)
_EP_BARRIER_LINE_RE = re.compile(
    r"\[EP-BARRIER\]\[(?P<cluster>[^\]]+)\]\s+"
    r"batch_id=(?P<batch_id>-?\d+),\s+"
    r"layer_id=(?P<layer_id>-?\d+),\s+"
    r"phase=(?P<phase>dispatch|combine),\s+"
    r"expected_ep_ids=(?P<expected_ep_ids>\[[^\]]*\]),\s+"
    r"arrived_ep_ids=(?P<arrived_ep_ids>\[[^\]]*\]),\s+"
    r"max_lane_time_ms=(?P<max_lane_time_ms>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s+"
    r"barrier_time_ms=(?P<barrier_time_ms>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),\s+"
    r"barrier_end_time_s=(?P<barrier_end_time_s>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"\s*$"
)
_EP_CONSERVATION_LINE_RE = re.compile(
    r"\[EP-CONSERVATION\]\[(?P<cluster>[^\]]+)\]\s+"
    r"batch_id=(?P<batch_id>-?\d+),\s+"
    r"layer_id=(?P<layer_id>-?\d+),\s+"
    r"routing_token_count=(?P<routing_token_count>\d+),\s+"
    r"router_topk=(?P<router_topk>\d+),\s+"
    r"total_routed_assignments=(?P<total_routed_assignments>\d+),\s+"
    r"per_ep_routed_tokens=(?P<per_ep_routed_tokens>\{.*\})\s*$"
)


@dataclass(frozen=True)
class MatrixCase:
    case_id: str
    baseline_case_id: str
    architecture: str
    model_kind: str
    model_name: str
    device: str
    routing_distribution: str
    seed: int
    workload_kind: str
    prefill_tokens: int
    decode_tokens: int
    num_requests: int
    ep_size: int
    moe_tensor_parallel_size: int
    total_experts: int
    router_topk: int
    pipeline_stages: int
    replica_count: int
    prefill_replicas: int
    decode_replicas: int
    decode_attn_replicas: int
    decode_ffn_replicas: int
    attn_tensor_parallel_size: int
    prefill_attn_tensor_parallel_size: int
    decode_attn_tensor_parallel_size: int
    prefill_moe_tensor_parallel_size: int
    decode_moe_tensor_parallel_size: int
    prefill_moe_expert_parallel_size: int
    decode_moe_expert_parallel_size: int
    total_cards: int
    num_layers: int
    moe_layer_ids: tuple[int, ...]

    @property
    def is_moe(self) -> bool:
        return self.model_kind != "dense"

    @property
    def expects_zero_routed_lane(self) -> bool:
        return self.is_moe and self.workload_kind == "zero-routed" and self.ep_size > 1


def _model_layer_shape(model_name: str) -> tuple[int, tuple[int, ...]]:
    # Imported lazily so manifest generation remains usable from a minimal shell
    # while still failing explicitly when the simulator environment is absent.
    from frontier.config.model_config import BaseModelConfig

    config = BaseModelConfig.create_from_name(model_name)
    return int(config.num_layers), tuple(int(x) for x in config.get_moe_layer_ids())


def _case_cards(
    architecture: str,
    replica_count: int,
    ep_size: int,
    moe_tp: int,
    pipeline_stages: int,
) -> tuple[int, dict[str, int]]:
    if architecture == "co-location":
        attn_tp = moe_tp * ep_size
        effective_replicas = min(
            replica_count, max(1, 32 // (attn_tp * pipeline_stages))
        )
        return effective_replicas * attn_tp * pipeline_stages, {
            "attn_tp": attn_tp,
            "prefill_attn_tp": attn_tp,
            "decode_attn_tp": attn_tp,
            "prefill_replicas": effective_replicas,
            "decode_replicas": effective_replicas,
            "decode_attn_replicas": effective_replicas,
            "decode_ffn_replicas": effective_replicas,
        }
    if architecture == "pd-disaggregation":
        attn_tp = moe_tp * ep_size
        effective_replicas = min(
            replica_count, max(1, 32 // (2 * attn_tp * pipeline_stages))
        )
        return 2 * effective_replicas * attn_tp * pipeline_stages, {
            "attn_tp": attn_tp,
            "prefill_attn_tp": attn_tp,
            "decode_attn_tp": attn_tp,
            "prefill_replicas": effective_replicas,
            "decode_replicas": effective_replicas,
            "decode_attn_replicas": effective_replicas,
            "decode_ffn_replicas": effective_replicas,
        }
    if architecture == "pd-af-disaggregation":
        # Decode-attention is an independent role domain.  Keep it at TP=1 in
        # this matrix so the FFN EP capacity is the dimension under test.
        prefill_attn_tp = moe_tp * ep_size
        decode_attn_tp = 1
        role_replicas = min(
            replica_count,
            max(
                1,
                32
                // (
                    prefill_attn_tp * pipeline_stages
                    + decode_attn_tp
                    + moe_tp * ep_size * pipeline_stages
                ),
            ),
        )
        cards = role_replicas * (
            prefill_attn_tp * pipeline_stages
            + decode_attn_tp
            + moe_tp * ep_size * pipeline_stages
        )
        return cards, {
            "attn_tp": prefill_attn_tp,
            "prefill_attn_tp": prefill_attn_tp,
            "decode_attn_tp": decode_attn_tp,
            "prefill_replicas": role_replicas,
            "decode_replicas": role_replicas,
            "decode_attn_replicas": role_replicas,
            "decode_ffn_replicas": role_replicas,
        }
    raise ValueError(f"unsupported architecture: {architecture}")


def validate_case_parallel_semantics(case: MatrixCase) -> None:
    """Apply the same shared-domain mapping contract used by Frontier/vLLM."""

    from frontier.config.parallel_semantics import (
        FrontierParallelismMapping,
        validate_frontier_shared_parallel_domains,
    )

    if case.architecture in {"co-location", "pd-disaggregation"}:
        mapping = FrontierParallelismMapping(
            cluster_num_replicas=case.replica_count,
            attn_tensor_parallel_size=case.attn_tensor_parallel_size,
            attn_dp=1,
            moe_tensor_parallel_size=case.moe_tensor_parallel_size,
            moe_expert_parallel_size=case.ep_size,
        )
        validate_frontier_shared_parallel_domains(mapping)
        return

    # PD-AF PREFILL is a shared full-model domain and follows the same
    # invariant.  DECODE_ATTN and DECODE_FFN are independent domains; do not
    # compare their capacities across roles.  The FFN local world must still
    # be a positive TP×EP product.
    prefill_mapping = FrontierParallelismMapping(
        cluster_num_replicas=case.prefill_replicas,
        attn_tensor_parallel_size=case.prefill_attn_tensor_parallel_size,
        attn_dp=1,
        moe_tensor_parallel_size=case.prefill_moe_tensor_parallel_size,
        moe_expert_parallel_size=case.prefill_moe_expert_parallel_size,
    )
    validate_frontier_shared_parallel_domains(prefill_mapping)
    if case.decode_moe_tensor_parallel_size * case.decode_moe_expert_parallel_size <= 0:
        raise ValueError(f"invalid PD-AF DECODE_FFN local world for {case.case_id}")


def build_matrix(repo_root: Path) -> list[MatrixCase]:
    """Build the deterministic 110-case matrix and validate its topology."""

    cases: list[MatrixCase] = []
    for architecture, case_count in ARCHITECTURE_CASE_COUNTS.items():
        for ordinal in range(case_count):
            model_kind = MODEL_ORDER[ordinal % len(MODEL_ORDER)]
            spec = MODEL_SPECS[model_kind]
            variant_index = (
                PD_AF_VARIANT_INDICES[ordinal % len(PD_AF_VARIANT_INDICES)]
                if architecture == "pd-af-disaggregation"
                else ordinal % len(VARIANTS)
            )
            distribution, workload_kind = VARIANTS[variant_index]
            if model_kind == "mixed":
                # The checked-in Step profile exposes only uniform_topk rows.
                distribution = "random"
            spec = (
                MODEL_SPECS["moe_standard"]
                if model_kind == "moe" and distribution != "random"
                else MODEL_SPECS[model_kind]
            )
            num_layers, moe_layer_ids = _model_layer_shape(str(spec["model_name"]))
            is_moe = model_kind != "dense"
            model_ordinal = ordinal // len(MODEL_ORDER)
            if not is_moe:
                ep_size = 1
            elif model_kind == "mixed":
                # Step's mixed profile needs TP=4 to fit the model shard in an
                # H800.  Its real linear/attention rows stop at TP=8, so the
                # mixed matrix uses EP<=2 (TP×EP<=8).  EP=4 remains covered by
                # the Phi/Qwen MoE populations.
                ep_size = (1, 2, 2)[model_ordinal % 3]
            else:
                ep_size = (1, 2, 4)[model_ordinal % 3]
            moe_tp = 4 if model_kind == "mixed" else 1
            pipeline_stages = 1
            replica_count = (1, 2, 4)[model_ordinal % 3]
            total_cards, topology = _case_cards(
                architecture, replica_count, ep_size, moe_tp, pipeline_stages
            )
            if total_cards > 32:
                raise AssertionError(
                    f"matrix topology exceeds 32 cards: {architecture} {model_kind} "
                    f"variant={variant_index} cards={total_cards}"
                )
            prefill_tokens, decode_tokens, num_requests = WORKLOADS[workload_kind]
            case_id = (
                f"{architecture.replace('-', '_')}_{model_kind}"
                f"_n{ordinal:02d}_v{variant_index:02d}"
            )
            baseline_id = f"{architecture.replace('-', '_')}_{model_kind}_baseline"
            case = MatrixCase(
                case_id=case_id,
                baseline_case_id=baseline_id,
                architecture=architecture,
                model_kind=model_kind,
                model_name=str(spec["model_name"]),
                device=str(spec.get("device", "h800")),
                routing_distribution=distribution if is_moe else "balanced",
                seed=42 + ordinal,
                workload_kind=workload_kind,
                prefill_tokens=prefill_tokens,
                decode_tokens=decode_tokens,
                num_requests=num_requests,
                ep_size=ep_size,
                moe_tensor_parallel_size=moe_tp,
                total_experts=int(spec["total_experts"]),
                router_topk=int(spec["router_topk"]),
                pipeline_stages=pipeline_stages,
                replica_count=replica_count,
                prefill_replicas=topology["prefill_replicas"],
                decode_replicas=topology["decode_replicas"],
                decode_attn_replicas=topology["decode_attn_replicas"],
                decode_ffn_replicas=topology["decode_ffn_replicas"],
                attn_tensor_parallel_size=1 if not is_moe else topology["attn_tp"],
                prefill_attn_tensor_parallel_size=topology["prefill_attn_tp"],
                decode_attn_tensor_parallel_size=topology["decode_attn_tp"],
                prefill_moe_tensor_parallel_size=moe_tp,
                decode_moe_tensor_parallel_size=moe_tp,
                prefill_moe_expert_parallel_size=ep_size,
                decode_moe_expert_parallel_size=ep_size,
                total_cards=total_cards
                if is_moe
                else _case_cards(architecture, replica_count, 1, 1, 1)[0],
                num_layers=num_layers,
                moe_layer_ids=moe_layer_ids,
            )
            validate_case_parallel_semantics(case)
            cases.append(case)
    if len(cases) != 110:
        raise AssertionError(f"expected 110 matrix cases, got {len(cases)}")
    return cases


def _script_for_case(case: MatrixCase, repo_root: Path) -> Path:
    root = repo_root / "examples" / "architecture"
    if case.model_kind == "dense":
        names = {
            "co-location": "co-location/offline/dense_model_basic.sh",
            "pd-disaggregation": "pdd/offline/dense_model_basic.sh",
            "pd-af-disaggregation": "pd-af-disagg/offline/dense_model_basic.sh",
        }
    else:
        names = {
            "co-location": "co-location/offline/moe_model_basic.sh",
            "pd-disaggregation": "pdd/offline/moe_model_basic.sh",
            "pd-af-disaggregation": "pd-af-disagg/offline/moe_model_ep.sh",
        }
    path = root / names[case.architecture]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_shell_command(
    case: MatrixCase, repo_root: Path, output_root: Path
) -> tuple[str, dict[str, str]]:
    """Return an exact wrapper command and its environment for one case."""

    script = _script_for_case(case, repo_root)
    metrics_root = output_root / case.case_id / "metrics"
    env = {key: value for key, value in os.environ.items()}
    env.update(
        {
            "PYTHONPATH": str(repo_root),
            "PYTHON_BIN": sys.executable,
            "WANDB_DISABLED": "true",
            "VIDUR_DISABLE_WANDB": "1",
            "MODEL_NAME": case.model_name,
            "ENABLE_DUMMY_MODE": "false",
            "DECODE_CUDA_GRAPH_MODE": "none",
            "ENABLE_CHUNKED_PREFILL": "false",
            "NUM_REQUESTS": str(case.num_requests),
            "PREFILL_TOKENS": str(case.prefill_tokens),
            "DECODE_TOKENS": str(case.decode_tokens),
            "QPS": "1.0",
            "RUN_ID": case.case_id,
            "METRICS_OUTPUT_DIR": str(metrics_root),
            "MOE_ROUTING_DISTRIBUTION_TYPE": case.routing_distribution,
            "MOE_ROUTING_SEED": str(case.seed),
            "TOTAL_EXPERTS": str(case.total_experts),
            "ROUTER_TOPK": str(case.router_topk),
            "MAX_TOKENS_IN_BATCH": "64",
            "LONG_PREFILL_TOKEN_THRESHOLD": "0",
        }
    )
    if case.architecture == "co-location":
        env.update(
            {
                "NUM_REPLICAS": str(case.replica_count),
                "ATTN_TP": str(case.attn_tensor_parallel_size),
                "MOE_TP": str(case.moe_tensor_parallel_size),
                "MOE_EP": str(case.ep_size),
                "PP": str(case.pipeline_stages),
                "DEVICE": case.device,
            }
        )
    elif case.architecture == "pd-disaggregation":
        env.update(
            {
                "PREFILL_REPLICAS": str(case.prefill_replicas),
                "DECODE_REPLICAS": str(case.decode_replicas),
                "PREFILL_ATTN_TP": str(case.prefill_attn_tensor_parallel_size),
                "PREFILL_MOE_TP": str(case.prefill_moe_tensor_parallel_size),
                "PREFILL_MOE_EP": str(case.prefill_moe_expert_parallel_size),
                "DECODE_ATTN_TP": str(case.decode_attn_tensor_parallel_size),
                "DECODE_MOE_TP": str(case.decode_moe_tensor_parallel_size),
                "DECODE_MOE_EP": str(case.decode_moe_expert_parallel_size),
                "PREFILL_DEVICE": case.device,
                "DECODE_DEVICE": case.device,
                "PREFILL_PP": str(case.pipeline_stages),
                "DECODE_PP": str(case.pipeline_stages),
            }
        )
    else:
        env.update(
            {
                "PREFILL_REPLICAS": str(case.prefill_replicas),
                "DECODE_ATTN_REPLICAS": str(case.decode_attn_replicas),
                "DECODE_FFN_REPLICAS": str(case.decode_ffn_replicas),
                "PREFILL_ATTN_TP": str(case.prefill_attn_tensor_parallel_size),
                "PREFILL_MOE_TP": str(case.prefill_moe_tensor_parallel_size),
                "PREFILL_MOE_EP": str(case.prefill_moe_expert_parallel_size),
                "DECODE_ATTN_TP": str(case.decode_attn_tensor_parallel_size),
                "DECODE_FFN_MOE_TP": str(case.decode_moe_tensor_parallel_size),
                "DECODE_FFN_MOE_EP": str(case.decode_moe_expert_parallel_size),
                "PREFILL_DEVICE": case.device,
                "DECODE_ATTN_DEVICE": case.device,
                "DECODE_FFN_DEVICE": case.device,
                "PREFILL_PP": str(case.pipeline_stages),
                "DECODE_ATTN_PP": "1",
                "DECODE_FFN_PP": str(case.pipeline_stages),
            }
        )
    command_parts = ["bash", str(script), "--"]
    # The co-location MoE wrapper predates the explicit DEVICE environment
    # contract used by the dense/PDD/PD-AF wrappers.  Pass the device as a
    # regular CLI override so non-dummy profile lookup cannot silently fall
    # back to a different SKU.
    if case.architecture == "co-location" and case.model_kind != "dense":
        command_parts.extend(["--replica_config_device", case.device])
    # Keep predictor artifacts isolated from the repository's shared cache. A
    # previously interrupted non-dummy run can leave a truncated pickle there;
    # reusing it would make an otherwise valid case fail with an unrelated
    # deserialization error. The caller chooses a fresh output root for each
    # matrix campaign, so this path is deterministic and provenance-visible.
    predictor_cache_dir = (output_root / "_predictor_cache").resolve()
    command_parts.extend(["--metrics_config_cache_dir", str(predictor_cache_dir)])
    command = shlex.join(command_parts)
    return command, env


def validate_profile_inputs(case: MatrixCase, root: Path) -> list[Path]:
    """Require real profiling rows; never substitute dummy or synthetic data."""

    profile_root = root / "data" / "profiling" / "compute"
    if (root / case.device).is_dir() and not (profile_root).is_dir():
        profile_root = root
    model_dir = profile_root / case.device / case.model_name
    required = [model_dir / "attention.csv", model_dir / "linear_op.csv"]
    if case.is_moe:
        required.append(model_dir / "moe.csv")
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required non-dummy profiling files:\n" + "\n".join(str(p) for p in missing)
        )
    for path in required:
        _validate_profile_metadata(path)
    if case.is_moe:
        expected_runtime_path = (
            "uniform_topk" if case.routing_distribution == "random" else "standard_fused_topk"
        )
        available_paths = _profile_routing_runtime_paths(model_dir / "moe.csv")
        if expected_runtime_path not in available_paths:
            raise ValueError(
                f"{case.case_id} requires routing_runtime_path={expected_runtime_path!r}, "
                f"but {model_dir / 'moe.csv'} provides {sorted(available_paths)!r}"
            )
    return required


@lru_cache(maxsize=None)
def _validate_profile_metadata(path: Path) -> None:
    """Require the predictor's immutable metadata contract before a run."""

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        missing = sorted(set(REQUIRED_PROFILE_METADATA_COLUMNS) - set(fieldnames))
        if missing:
            raise ValueError(
                f"{path} missing required profiling metadata columns: {', '.join(missing)}"
            )
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} contains no profiling rows")
    empty = [
        column
        for column in REQUIRED_PROFILE_METADATA_COLUMNS
        if any(not str(row.get(column, "")).strip() for row in rows)
    ]
    if empty:
        raise ValueError(
            f"{path} contains empty required profiling metadata columns: {', '.join(empty)}"
        )


@lru_cache(maxsize=None)
def _profile_routing_runtime_paths(path: Path) -> frozenset[str]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if "routing_runtime_path" not in (reader.fieldnames or []):
            raise ValueError(f"missing routing_runtime_path column in {path}")
        return frozenset(
            str(row.get("routing_runtime_path", "")).strip()
            for row in reader
            if str(row.get("routing_runtime_path", "")).strip()
        )


def _find_metrics_dir(
    output_root: Path,
    case: MatrixCase,
    *,
    started_at_ns: int | None = None,
) -> Path:
    root = output_root / case.case_id
    candidates = sorted(root.rglob("system_metrics.json"))
    if started_at_ns is not None:
        candidates = [
            path
            for path in candidates
            if path.stat().st_mtime_ns >= started_at_ns
        ]
        if not candidates:
            raise FileNotFoundError(
                f"no fresh system_metrics.json for {case.case_id} "
                f"under {output_root}"
            )
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected exactly one system_metrics.json for {case.case_id}, found {len(candidates)}"
        )
    return candidates[0].parent


def _finite_metric_values(value: Any) -> Iterable[float]:
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _finite_metric_values(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _finite_metric_values(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"non-finite or negative metric value: {value!r}")
        yield number


def _parse_ep_workload_records(text: str) -> list[dict[str, Any]]:
    """Parse scheduler-emitted per-lane MoE workload records.

    The parser is deliberately strict: malformed records are ignored by the
    caller only as missing evidence, while a syntactically matching record is
    validated here so a negative/non-finite lane value cannot be mistaken for
    a successful workflow trace.
    """

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _EP_WORKLOAD_LINE_RE.search(line)
        if match is None:
            continue
        groups = match.groupdict()
        try:
            per_expert_tokens = ast.literal_eval(groups["per_expert_tokens"])
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                "invalid per_expert_tokens literal in EP workload trace"
            ) from exc
        if not isinstance(per_expert_tokens, dict):
            raise ValueError("EP workload per_expert_tokens must be a dict")
        normalized_tokens: dict[int, int] = {}
        for expert_id, token_count in per_expert_tokens.items():
            if (
                type(expert_id) is not int
                or type(token_count) is not int
                or token_count < 0
            ):
                raise ValueError(
                    "EP workload per_expert_tokens must map exact integer expert IDs "
                    "to non-negative integer token counts"
                )
            normalized_tokens[int(expert_id)] = int(token_count)

        lane_compute_ms = float(groups["lane_compute_ms"])
        lane_comm_ms = float(groups["lane_comm_ms"])
        ep_id = int(groups["ep_id"])
        moe_ep_size = int(groups["moe_ep_size"])
        if moe_ep_size <= 0 or ep_id < 0 or ep_id >= moe_ep_size:
            raise ValueError(
                "EP workload ep_id must be within the declared moe_ep_size"
            )
        if not math.isfinite(lane_compute_ms) or lane_compute_ms < 0:
            raise ValueError("EP workload lane_compute_ms must be finite and non-negative")
        if not math.isfinite(lane_comm_ms) or lane_comm_ms < 0:
            raise ValueError("EP workload lane_comm_ms must be finite and non-negative")

        records.append(
            {
                "cluster": groups["cluster"],
                "batch_id": int(groups["batch_id"]),
                "layer_id": int(groups["layer_id"]),
                "ep_id": ep_id,
                "moe_ep_size": moe_ep_size,
                "per_expert_tokens": normalized_tokens,
                "lane_compute_ms": lane_compute_ms,
                "lane_comm_ms": lane_comm_ms,
            }
        )
    return records


def _expected_ep_size_for_cluster(case: MatrixCase, cluster: str) -> int | None:
    """Return the EP cardinality for one physical cluster role."""

    cluster_name = str(cluster).upper()
    if case.architecture == "pd-af-disaggregation":
        if cluster_name == "PREFILL":
            return int(case.prefill_moe_expert_parallel_size)
        if cluster_name == "DECODE_FFN":
            return int(case.decode_moe_expert_parallel_size)
        return None
    if case.architecture == "pd-disaggregation":
        if cluster_name == "PREFILL":
            return int(case.prefill_moe_expert_parallel_size)
        if cluster_name == "DECODE":
            return int(case.decode_moe_expert_parallel_size)
    return int(case.ep_size)


def _parse_ep_barrier_records(text: str) -> list[dict[str, Any]]:
    """Parse completed per-layer EP barrier records from a simulator log."""

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _EP_BARRIER_LINE_RE.search(line)
        if match is None:
            continue
        groups = match.groupdict()
        try:
            expected_ep_ids = ast.literal_eval(groups["expected_ep_ids"])
            arrived_ep_ids = ast.literal_eval(groups["arrived_ep_ids"])
        except (SyntaxError, ValueError) as exc:
            raise ValueError("invalid EP barrier participant list") from exc
        if not isinstance(expected_ep_ids, list) or not isinstance(arrived_ep_ids, list):
            raise ValueError("EP barrier participant lists must be lists")
        if any(type(ep_id) is not int or ep_id < 0 for ep_id in expected_ep_ids):
            raise ValueError("EP barrier expected_ep_ids must be non-negative ints")
        if any(type(ep_id) is not int or ep_id < 0 for ep_id in arrived_ep_ids):
            raise ValueError("EP barrier arrived_ep_ids must be non-negative ints")
        if sorted(expected_ep_ids) != sorted(set(expected_ep_ids)):
            raise ValueError("EP barrier expected_ep_ids must be unique")
        if sorted(arrived_ep_ids) != sorted(set(arrived_ep_ids)):
            raise ValueError("EP barrier arrived_ep_ids must be unique")
        if sorted(expected_ep_ids) != sorted(arrived_ep_ids):
            raise ValueError("EP barrier arrived_ep_ids must equal expected_ep_ids")
        values = {
            name: float(groups[name])
            for name in ("max_lane_time_ms", "barrier_time_ms", "barrier_end_time_s")
        }
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError("EP barrier times must be finite and non-negative")
        if values["barrier_time_ms"] < values["max_lane_time_ms"]:
            raise ValueError("EP barrier time is shorter than the slowest lane")
        batch_id = int(groups["batch_id"])
        layer_id = int(groups["layer_id"])
        if batch_id < 0 or layer_id < 0:
            raise ValueError("EP barrier batch_id/layer_id must be non-negative")
        records.append(
            {
                "cluster": groups["cluster"],
                "batch_id": batch_id,
                "layer_id": layer_id,
                "phase": groups["phase"],
                "expected_ep_ids": [int(ep_id) for ep_id in expected_ep_ids],
                "arrived_ep_ids": [int(ep_id) for ep_id in arrived_ep_ids],
                **values,
            }
        )
    return records


def _parse_ep_conservation_records(text: str) -> list[dict[str, Any]]:
    """Parse one exact routing-conservation record per materialized EP wave."""

    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        match = _EP_CONSERVATION_LINE_RE.search(line)
        if match is None:
            continue
        groups = match.groupdict()
        try:
            per_ep_routed_tokens = ast.literal_eval(groups["per_ep_routed_tokens"])
        except (SyntaxError, ValueError) as exc:
            raise ValueError("invalid per_ep_routed_tokens literal") from exc
        if not isinstance(per_ep_routed_tokens, dict):
            raise ValueError("per_ep_routed_tokens must be a dict")
        normalized: dict[int, int] = {}
        for ep_id, token_count in per_ep_routed_tokens.items():
            if (
                type(ep_id) is not int
                or type(token_count) is not int
                or ep_id < 0
                or token_count < 0
            ):
                raise ValueError(
                    "per_ep_routed_tokens must map non-negative integer IDs "
                    "to non-negative integer counts"
                )
            normalized[int(ep_id)] = int(token_count)
        batch_id = int(groups["batch_id"])
        layer_id = int(groups["layer_id"])
        routing_token_count = int(groups["routing_token_count"])
        router_topk = int(groups["router_topk"])
        total_routed_assignments = int(groups["total_routed_assignments"])
        if batch_id < 0 or layer_id < 0:
            raise ValueError("conservation batch_id/layer_id must be non-negative")
        if router_topk <= 0:
            raise ValueError("conservation router_topk must be positive")
        if routing_token_count * router_topk != total_routed_assignments:
            raise ValueError(
                "conservation total does not equal routing_token_count * router_topk"
            )
        if sum(normalized.values()) != total_routed_assignments:
            raise ValueError(
                "conservation per_ep_routed_tokens total is inconsistent"
            )
        records.append(
            {
                "cluster": groups["cluster"],
                "batch_id": batch_id,
                "layer_id": layer_id,
                "routing_token_count": routing_token_count,
                "router_topk": router_topk,
                "total_routed_assignments": total_routed_assignments,
                "per_ep_routed_tokens": normalized,
            }
        )
    return records


def check_case_log(
    case: MatrixCase,
    log_path: Path,
    metrics_dir: Path,
    *,
    strict_layers: bool = False,
) -> dict[str, Any]:
    """Check workflow evidence and numeric metrics for one completed run."""

    errors: list[str] = []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "Traceback" in text:
        errors.append("Traceback")
    if "Simulation completed successfully." not in text:
        errors.append("missing success marker")
    if "Dummy Mode: false" not in text:
        errors.append("dummy mode was not explicitly disabled")
    if re.search(r"(?i)(synthetic latency|scaling factor|visibility multiplier)", text):
        errors.append("forbidden synthetic/scaling wording in log")

    layer_ids = sorted({int(match) for match in re.findall(r"layer_id=(\d+)", text)})
    if not layer_ids:
        errors.append("no layer_id trace")
    if strict_layers and case.model_kind != "dense":
        if case.model_kind == "mixed":
            # A mixed model may legitimately aggregate dense-layer work.  The
            # correctness contract is that every declared MoE layer appears in
            # the per-layer MoE trace; dense layer IDs are not a substitute.
            moe_layer_ids_seen = sorted(
                {
                    int(match)
                    for line in text.splitlines()
                    if "[MOE]" in line
                    for match in re.findall(r"layer_id=(\d+)", line)
                }
            )
            expected = list(case.moe_layer_ids)
            if moe_layer_ids_seen != expected:
                errors.append(
                    "mixed MoE layer ids are incomplete "
                    f"expected={expected} actual={moe_layer_ids_seen}"
                )
        else:
            expected = list(range(case.num_layers))
            if layer_ids != expected:
                errors.append(
                    f"layer ids are not contiguous expected={expected} actual={layer_ids}"
                )

    moe_trace_count = text.count("[MOE]")
    ep_participant_records = text.count("per_expert_tokens extracted:")
    try:
        ep_workload_records = _parse_ep_workload_records(text)
    except ValueError as exc:
        ep_workload_records = []
        errors.append(f"invalid EP workload trace: {exc}")
    try:
        ep_barrier_records = _parse_ep_barrier_records(text)
    except ValueError as exc:
        ep_barrier_records = []
        errors.append(f"invalid EP barrier trace: {exc}")
    try:
        ep_conservation_records = _parse_ep_conservation_records(text)
    except ValueError as exc:
        ep_conservation_records = []
        errors.append(f"invalid EP conservation trace: {exc}")
    if case.is_moe:
        if "moe_grouped_gemm" not in text or "moe_shuffling" not in text:
            errors.append("missing MoE grouped-gemm/shuffling trace")
        if not ep_workload_records:
            errors.append("missing EP workload trace")
        elif strict_layers:
            expected_moe_layers = set(int(layer_id) for layer_id in case.moe_layer_ids)
            records_by_wave: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
            for record in ep_workload_records:
                cluster_name = str(record["cluster"]).upper()
                records_by_wave.setdefault(
                    (
                        cluster_name,
                        int(record["batch_id"]),
                        int(record["layer_id"]),
                    ),
                    [],
                ).append(record)
            complete_layers: set[int] = set()
            for (cluster_name, _batch_id, layer_id), wave_records in records_by_wave.items():
                expected_ep_size = _expected_ep_size_for_cluster(case, cluster_name)
                if expected_ep_size is None:
                    errors.append(
                        "EP workload uses an unsupported cluster role "
                        f"architecture={case.architecture} cluster={cluster_name}"
                    )
                    continue
                expected_ep_ids = set(range(expected_ep_size))
                actual_ep_ids = {int(record["ep_id"]) for record in wave_records}
                if actual_ep_ids == expected_ep_ids:
                    complete_layers.add(layer_id)
                if any(
                    int(record["moe_ep_size"]) != expected_ep_size
                    for record in wave_records
                ):
                    errors.append(
                        "EP workload moe_ep_size mismatch "
                        f"cluster={cluster_name} layer={layer_id} "
                        f"expected={expected_ep_size}"
                    )
                if actual_ep_ids != expected_ep_ids:
                    errors.append(
                        "EP workload participants are incomplete "
                        f"cluster={cluster_name} batch_id={_batch_id} layer={layer_id} "
                        f"expected={sorted(expected_ep_ids)} actual={sorted(actual_ep_ids)}"
                    )
            for expected_layer_id in sorted(expected_moe_layers):
                if expected_layer_id not in complete_layers:
                    errors.append(
                        "EP workload has no complete participant wave "
                        f"for layer={expected_layer_id}"
                    )
            conservation_by_wave = {
                (
                    str(record["cluster"]).upper(),
                    int(record["batch_id"]),
                    int(record["layer_id"]),
                ): record
                for record in ep_conservation_records
            }
            conservation_layers_by_cluster: dict[str, set[int]] = {}
            for record in ep_conservation_records:
                cluster_name = str(record["cluster"]).upper()
                conservation_layers_by_cluster.setdefault(cluster_name, set()).add(
                    int(record["layer_id"])
                )
            for wave_key, wave_records in records_by_wave.items():
                conservation = conservation_by_wave.get(wave_key)
                if conservation is None:
                    errors.append(
                        "missing EP conservation evidence for wave "
                        f"cluster={wave_key[0]} batch_id={wave_key[1]} layer={wave_key[2]}"
                    )
                    continue
                expected_ep_size = _expected_ep_size_for_cluster(case, wave_key[0])
                expected_ep_ids = set(range(expected_ep_size or 0))
                per_ep_totals = {
                    int(record["ep_id"]): sum(record["per_expert_tokens"].values())
                    for record in wave_records
                }
                if set(per_ep_totals) != expected_ep_ids:
                    errors.append(
                        "EP conservation workload participant IDs do not match "
                        f"cluster={wave_key[0]} layer={wave_key[2]}"
                    )
                if per_ep_totals != dict(conservation["per_ep_routed_tokens"]):
                    errors.append(
                        "EP conservation per-lane totals disagree with workload "
                        f"cluster={wave_key[0]} batch_id={wave_key[1]} layer={wave_key[2]}"
                    )
                if sum(per_ep_totals.values()) != int(
                    conservation["total_routed_assignments"]
                ):
                    errors.append(
                        "EP conservation total disagrees with lane workload "
                        f"cluster={wave_key[0]} batch_id={wave_key[1]} layer={wave_key[2]}"
                    )
            workload_clusters = {
                str(record["cluster"]).upper() for record in ep_workload_records
            }
            for cluster_name in sorted(workload_clusters):
                missing_conservation_layers = sorted(
                    expected_moe_layers
                    - conservation_layers_by_cluster.get(cluster_name, set())
                )
                if missing_conservation_layers:
                    errors.append(
                        "EP conservation has no complete layer evidence "
                        f"cluster={cluster_name} layers={missing_conservation_layers}"
                    )

            barrier_layers_by_phase: dict[tuple[str, str], set[int]] = {}
            for record in ep_barrier_records:
                barrier_layers_by_phase.setdefault(
                    (str(record["cluster"]).upper(), str(record["phase"])),
                    set(),
                ).add(int(record["layer_id"]))
            required_phases = ("combine",)
            if case.architecture == "pd-af-disaggregation":
                required_phases = ("dispatch", "combine")
            for cluster_name in sorted(workload_clusters):
                for phase in required_phases:
                    missing_barrier_layers = sorted(
                        expected_moe_layers
                        - barrier_layers_by_phase.get((cluster_name, phase), set())
                    )
                    if missing_barrier_layers:
                        errors.append(
                            "missing EP barrier evidence for MoE layers "
                            f"cluster={cluster_name} phase={phase} "
                            f"layers={missing_barrier_layers}"
                        )
        if case.architecture == "pd-af-disaggregation" and ep_participant_records == 0:
            errors.append("missing DECODE_FFN EP participant maps")
        if case.expects_zero_routed_lane and not any(
            token_count == 0
            for record in ep_workload_records
            for token_count in record["per_expert_tokens"].values()
        ) and "0}" not in text and ": 0" not in text:
            errors.append("zero-routed case has no zero-token participant evidence")

    metric_path = metrics_dir / "system_metrics.json"
    numeric_metric_count = 0
    metrics: dict[str, Any] = {}
    if not metric_path.is_file():
        errors.append(f"missing metrics file: {metric_path}")
    else:
        try:
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            numeric_metric_count = sum(1 for _ in _finite_metric_values(metrics))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"invalid metrics: {exc}")

    def _stat_value(name: str) -> float | None:
        value = metrics.get(name, {})
        if isinstance(value, Mapping):
            candidate = value.get("mean")
            if isinstance(candidate, (int, float)):
                return float(candidate)
        return None

    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": "; ".join(errors),
        "layer_ids": layer_ids,
        "moe_trace_count": moe_trace_count,
        "ep_participant_records": ep_participant_records,
        "ep_workload_records": len(ep_workload_records),
        "ep_barrier_records": len(ep_barrier_records),
        "ep_conservation_records": len(ep_conservation_records),
        "numeric_metric_count": numeric_metric_count,
        "ttft_mean_ms": _stat_value("ttft_statistics"),
        "e2e_mean_ms": _stat_value("request_e2e_time_statistics"),
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _load_result_rows(path: Path) -> list[dict[str, Any]]:
    """Load an existing result ledger, failing on malformed evidence."""

    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid result ledger JSON at {path}:{line_number}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"result ledger row is not an object at {path}:{line_number}")
        rows.append(row)
    return rows


def _validate_result_ledger_provenance(
    rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    output_root: Path,
    results_path: Path,
) -> None:
    """Reject ledger rows produced by another run/output generation.

    A case ID is not a sufficient identity for evidence: two worktrees can
    produce the same ID with identical metrics but different scheduler traces.
    Every persisted row must therefore carry the canonical output and ledger
    paths used by the current invocation.
    """

    expected_repo = repo_root.resolve()
    expected_output = output_root.resolve()
    expected_results = results_path.resolve()
    for row in rows:
        case_id = row.get("case_id", "<missing>")
        row_repo = row.get("repo_root")
        row_output = row.get("output_root")
        row_results = row.get("results_path")
        row_log = row.get("log_path")
        row_metrics = row.get("metrics_path")
        if not all(
            isinstance(value, str)
            for value in (row_repo, row_output, row_results, row_log, row_metrics)
        ):
            raise ValueError(
                "result ledger row is missing canonical provenance: "
                f"case_id={case_id!r}"
            )
        if Path(row_repo).resolve() != expected_repo:
            raise ValueError(
                "result ledger repo_root provenance mismatch: "
                f"case_id={case_id!r}, row={row_repo!r}, expected={str(expected_repo)!r}"
            )
        if Path(row_output).resolve() != expected_output:
            raise ValueError(
                "result ledger output_root provenance mismatch: "
                f"case_id={case_id!r}, row={row_output!r}, expected={str(expected_output)!r}"
            )
        if Path(row_results).resolve() != expected_results:
            raise ValueError(
                "result ledger results_path provenance mismatch: "
                f"case_id={case_id!r}, row={row_results!r}, expected={str(expected_results)!r}"
            )
        if not case_id or not isinstance(case_id, str):
            raise ValueError("result ledger case_id must be a non-empty string")
        case_root = expected_output / case_id
        log_path = Path(row_log).resolve()
        if not log_path.is_relative_to(case_root):
            raise ValueError(
                "result ledger log_path is outside its canonical case directory: "
                f"case_id={case_id!r}, path={row_log!r}"
            )
        if row_metrics:
            metrics_path = Path(row_metrics).resolve()
            if not metrics_path.is_relative_to(case_root):
                raise ValueError(
                    "result ledger metrics_path is outside its canonical case directory: "
                    f"case_id={case_id!r}, path={row_metrics!r}"
                )
        elif row.get("status") == "PASS":
            raise ValueError(
                "result ledger PASS row has no canonical metrics_path: "
                f"case_id={case_id!r}"
            )


def _merge_result_rows(
    existing_rows: Sequence[Mapping[str, Any]],
    new_rows: Sequence[Mapping[str, Any]],
    *,
    expected_case_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Merge rerun rows without losing prior case evidence.

    Existing and new rows must refer only to the current manifest and must have
    unique case IDs within each input.  A rerun replaces the prior row for the
    same case; cases not selected in the current invocation remain intact.
    Output follows manifest order and omits cases that have not run yet.
    """

    expected = tuple(expected_case_ids)
    expected_set = set(expected)
    if len(expected) != len(expected_set):
        raise ValueError("manifest contains duplicate case IDs")

    merged: dict[str, dict[str, Any]] = {}
    for source_name, rows in (("existing", existing_rows), ("new", new_rows)):
        seen: set[str] = set()
        for row in rows:
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"{source_name} result row has no non-empty case_id")
            if case_id not in expected_set:
                raise ValueError(
                    f"{source_name} result row has unknown case_id={case_id!r}"
                )
            if case_id in seen:
                raise ValueError(f"{source_name} result rows repeat case_id={case_id!r}")
            seen.add(case_id)
            merged[case_id] = dict(row)
    return [merged[case_id] for case_id in expected if case_id in merged]


def write_manifest(path: Path, cases: Sequence[MatrixCase]) -> None:
    _write_jsonl(path, [asdict(case) for case in cases])


def _load_manifest(path: Path) -> list[MatrixCase]:
    cases: list[MatrixCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record["moe_layer_ids"] = tuple(record["moe_layer_ids"])
        cases.append(MatrixCase(**record))
    return cases


def run_cases(
    cases: Sequence[MatrixCase],
    repo_root: Path,
    output_root: Path,
    results_path: Path,
    *,
    start: int = 0,
    limit: int | None = None,
    timeout_seconds: int = 600,
    continue_on_failure: bool = False,
) -> list[dict[str, Any]]:
    repo_root = repo_root.resolve()
    output_root = output_root.resolve()
    results_path = results_path.resolve()
    selected = list(cases[start : (start + limit) if limit is not None else None])
    expected_case_ids = tuple(case.case_id for case in cases)
    existing_rows = _load_result_rows(results_path)
    _validate_result_ledger_provenance(
        existing_rows,
        repo_root=repo_root,
        output_root=output_root,
        results_path=results_path,
    )
    persisted = _merge_result_rows(
        existing_rows,
        (),
        expected_case_ids=expected_case_ids,
    )
    results: list[dict[str, Any]] = []
    for case in selected:
        validate_profile_inputs(case, repo_root)
        command, env = build_shell_command(case, repo_root, output_root)
        case_dir = output_root / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        log_path = case_dir / f"{case.case_id}.log"
        metadata_path = case_dir / "case_metadata.json"
        run_started_at_ns = time.time_ns()
        run_started_at_s = time.time()
        metadata_path.write_text(
            json.dumps(
                {
                    "case": asdict(case),
                    "command": command,
                    "environment": {
                        key: env[key]
                        for key in (
                            "MODEL_NAME",
                            "ENABLE_DUMMY_MODE",
                            "DECODE_CUDA_GRAPH_MODE",
                            "MOE_ROUTING_DISTRIBUTION_TYPE",
                            "MOE_ROUTING_SEED",
                            "TOTAL_EXPERTS",
                            "ROUTER_TOPK",
                        )
                        if key in env
                    },
                    "repo_root": str(repo_root),
                    "output_root": str(output_root),
                    "results_path": str(results_path),
                    "run_started_at_unix_s": run_started_at_s,
                    "trace_schema_version": 1,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        started = time.monotonic()
        with log_path.open("w", encoding="utf-8") as stream:
            stream.write(f"MATRIX_COMMAND: {command}\n")
            stream.flush()
            try:
                completed = subprocess.run(
                    command,
                    shell=True,
                    cwd=repo_root,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False,
                )
                exit_code = int(completed.returncode)
            except subprocess.TimeoutExpired:
                exit_code = 124
                stream.write(f"MATRIX_TIMEOUT after {timeout_seconds}s\n")
        elapsed = time.monotonic() - started
        run_finished_at_s = time.time()
        check: dict[str, Any]
        try:
            if not log_path.is_file() or log_path.stat().st_mtime_ns < run_started_at_ns:
                raise FileNotFoundError(
                    f"no fresh case log for {case.case_id} under {output_root}"
                )
            metrics_dir = _find_metrics_dir(
                output_root,
                case,
                started_at_ns=run_started_at_ns,
            )
            check = check_case_log(case, log_path, metrics_dir, strict_layers=True)
            metrics_path = str(metrics_dir)
        except (FileNotFoundError, OSError) as exc:
            check = {"status": "FAIL", "errors": str(exc)}
            metrics_path = ""
        status = "PASS" if exit_code == 0 and check.get("status") == "PASS" else "FAIL"
        result = {
            "case_id": case.case_id,
            "architecture": case.architecture,
            "model_kind": case.model_kind,
            "total_cards": case.total_cards,
            "exit_code": exit_code,
            "elapsed_seconds": round(elapsed, 3),
            "log_path": str(log_path),
            "metrics_path": metrics_path,
            "repo_root": str(repo_root),
            "output_root": str(output_root),
            "results_path": str(results_path),
            "run_started_at_unix_s": run_started_at_s,
            "run_finished_at_unix_s": run_finished_at_s,
            "trace_schema_version": 1,
            "status": status,
            "check": check,
        }
        results.append(result)
        persisted = _merge_result_rows(
            persisted,
            (result,),
            expected_case_ids=expected_case_ids,
        )
        _write_jsonl(results_path, persisted)
        if status != "PASS" and not continue_on_failure:
            raise RuntimeError(f"matrix case failed: {json.dumps(result, sort_keys=True)}")
    return results


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("generate", "run"), default="generate")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--task-dir",
        type=Path,
        default=Path("task_memory/task_2026-08-12_moe_ep_rank_stragger_analysis"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/ycfeng/tmp/frontier_non_dummy_matrix"),
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help=(
            "Canonical JSONL ledger path. Required when resuming a ledger; "
            "rows from another output root or worktree are rejected."
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    repo_root = args.repo_root.resolve()
    task_dir = args.task_dir if args.task_dir.is_absolute() else repo_root / args.task_dir
    manifest_path = task_dir / "moe_ep_non_dummy_matrix_manifest.jsonl"
    results_path = (
        args.results_path
        if args.results_path is not None
        else task_dir / "moe_ep_non_dummy_matrix_results.jsonl"
    )
    cases = build_matrix(repo_root)
    write_manifest(manifest_path, cases)
    print(f"manifest={manifest_path} cases={len(cases)}")
    if args.mode == "generate":
        return 0
    results = run_cases(
        cases,
        repo_root,
        args.output_root,
        results_path,
        start=args.start,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
        continue_on_failure=args.continue_on_failure,
    )
    passed = sum(result["status"] == "PASS" for result in results)
    failed = len(results) - passed
    print(f"results={results_path} passed={passed} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
