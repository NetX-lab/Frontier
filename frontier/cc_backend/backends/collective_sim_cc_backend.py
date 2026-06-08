"""Collective-sim CC backend implementation."""

from __future__ import annotations

import importlib
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from frontier.cc_backend.base_cc_backend import BaseCCBackend
from frontier.cc_backend.cc_backend_config import CollectiveSimCCBackendConfig
from frontier.cc_backend.cc_backend_factory import CCBackendFactory
from frontier.logger import init_logger
from frontier.types import CCBackendType, ClusterType

logger = init_logger(__name__)


class CollectiveSimCCBackend(BaseCCBackend):
    """Topology-aware CC backend based on collective-sim."""

    _KNOWN_DOMAINS = frozenset(
        {
            "TP",
            "CP",
            "DP",
            "EP",
            "PP",
            "ATTN_TP",
            "MOE_TP",
            "ATTN_DP",
            "MOE_EP",
        }
    )
    _DOMAIN_TO_COLLECTIVE_SIM = {
        "TP": "TP",
        "ATTN_TP": "TP",
        "MOE_TP": "TP",
        "CP": "CP",
        "DP": "DP",
        "ATTN_DP": "DP",
        "EP": "EP",
        "MOE_EP": "EP",
        # collective-sim has no PP dimension; CP is used as PP proxy.
        "PP": "CP",
    }

    def __init__(
        self,
        config: CollectiveSimCCBackendConfig,
        cluster_type: ClusterType,
        device_type: str,
        network_device: str,
        num_devices: int,
    ) -> None:
        super().__init__(config, cluster_type, device_type, network_device, num_devices)

        self._config: CollectiveSimCCBackendConfig = config
        self._cache_size = int(config.prediction_cache_size)
        if self._cache_size <= 0:
            raise ValueError(
                f"prediction_cache_size must be > 0, got {self._cache_size}"
            )
        self._prediction_cache: "OrderedDict[Tuple[str, int, str, Tuple[int, ...], Tuple[int, int, int, int]], float]" = OrderedDict()
        self._prediction_cache_requests = 0
        self._prediction_cache_hits = 0
        self._prediction_cache_instance_hits = 0
        self._prediction_cache_misses = 0

        self._repo_root = self._resolve_collective_sim_repo_root()
        self._python_root = self._repo_root / "python"
        self._collective_sim_predict = self._load_collective_sim_predictor()

        self._placement_order = self._parse_placement_order(config.placement_order)
        self._runner_out_dir = self._resolve_runner_out_dir(config)

        self._validate_cluster_parallelism_consistency()

    def _resolve_collective_sim_repo_root(self) -> Path:
        """Resolve collective-sim repo root for both main workspaces and git worktrees."""
        direct_repo_root = Path(__file__).resolve().parent / "collective-sim"
        if (direct_repo_root / "python").exists():
            return direct_repo_root

        current = Path(__file__).resolve()
        for ancestor in current.parents:
            candidate = ancestor / "frontier" / "cc_backend" / "backends" / "collective-sim"
            if (candidate / "python").exists():
                return candidate

        raise FileNotFoundError(
            "Unable to locate collective-sim repository root. "
            f"Checked direct path {direct_repo_root} and ancestor-derived "
            "frontier/cc_backend/backends/collective-sim paths."
        )

    def _load_collective_sim_predictor(self):
        """Load collective-sim predictor API lazily with explicit submodule bootstrap."""
        if not self._python_root.exists():
            raise FileNotFoundError(
                f"collective-sim python package path does not exist: {self._python_root}"
            )

        python_root_str = str(self._python_root)
        if python_root_str not in sys.path:
            sys.path.insert(0, python_root_str)

        try:
            module = importlib.import_module("collective_sim_core")
            return getattr(module, "predict_collective_time")
        except Exception as exc:  # pragma: no cover - import-path dependent
            raise RuntimeError(
                "Failed to import collective_sim_core.predict_collective_time. "
                f"python_root={self._python_root}, error={exc}"
            ) from exc

    def _resolve_runner_out_dir(self, config: CollectiveSimCCBackendConfig) -> str:
        """Resolve a writable output directory for htsim runner artifacts."""
        if config.runner_out_dir:
            out_dir = Path(config.runner_out_dir)
        else:
            out_dir = (
                Path(config.cache_dir)
                / "collective-sim"
                / self._cluster_type.name.lower()
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        if not os.access(out_dir, os.W_OK):
            raise PermissionError(f"runner_out_dir is not writable: {out_dir}")
        return str(out_dir)

    def _parse_placement_order(self, placement_order: str) -> Tuple[str, ...]:
        dims = tuple(
            dim.strip().upper() for dim in str(placement_order).split(",") if dim.strip()
        )
        if not dims:
            raise ValueError("placement_order must be a non-empty comma-separated list")

        unknown = [dim for dim in dims if dim not in {"TP", "CP", "DP", "EP"}]
        if unknown:
            raise ValueError(
                f"placement_order contains unsupported dims: {unknown}. "
                "Supported dims: TP,CP,DP,EP"
            )
        return dims

    def _validate_cluster_parallelism_consistency(self) -> None:
        cfg = self._config
        if cfg.cluster_servers <= 0:
            raise ValueError(f"cluster_servers must be > 0, got {cfg.cluster_servers}")
        if cfg.cluster_gpus_per_server <= 0:
            raise ValueError(
                "cluster_gpus_per_server must be > 0, got "
                f"{cfg.cluster_gpus_per_server}"
            )

        for field_name in ("parallel_tp", "parallel_cp", "parallel_dp", "parallel_ep"):
            value = int(getattr(cfg, field_name))
            if value <= 0:
                raise ValueError(f"{field_name} must be > 0, got {value}")

        nodes = int(cfg.cluster_servers) * int(cfg.cluster_gpus_per_server)
        parallel_nodes = (
            int(cfg.parallel_tp)
            * int(cfg.parallel_cp)
            * int(cfg.parallel_dp)
            * int(cfg.parallel_ep)
        )
        if nodes != parallel_nodes:
            raise ValueError(
                "collective-sim cluster/parallel dims mismatch: "
                f"servers*gpus_per_server={nodes}, tp*cp*dp*ep={parallel_nodes}"
            )

    def _normalize_comm_domain(self, comm_domain: Optional[str]) -> str:
        if not comm_domain:
            raise ValueError(
                "collective-sim backend requires explicit comm_domain "
                "(ATTN_TP/MOE_TP/DP/EP/PP)"
            )

        domain = comm_domain.strip().upper()
        if domain not in self._KNOWN_DOMAINS:
            raise ValueError(
                f"Unsupported comm_domain={comm_domain!r}. "
                f"Supported domains: {sorted(self._KNOWN_DOMAINS)}"
            )

        mapped = self._DOMAIN_TO_COLLECTIVE_SIM[domain]
        if mapped not in self._placement_order:
            raise ValueError(
                f"comm_domain={domain} maps to {mapped}, but {mapped} is not in "
                f"placement_order={self._placement_order}"
            )
        return domain

    def _get_static_parallel_dim_sizes(self) -> Dict[str, int]:
        cfg = self._config
        return {
            "TP": int(cfg.parallel_tp),
            "CP": int(cfg.parallel_cp),
            "DP": int(cfg.parallel_dp),
            "EP": int(cfg.parallel_ep),
        }

    def _has_runtime_parallelism_metadata(self) -> bool:
        cfg = self._config
        required = (
            cfg.runtime_num_replicas,
            cfg.runtime_num_pipeline_stages,
            cfg.runtime_attn_tensor_parallel_size,
            cfg.runtime_attn_data_parallel_size,
            cfg.runtime_moe_tensor_parallel_size,
            cfg.runtime_moe_expert_parallel_size,
        )
        return all(value is not None for value in required)

    def _get_attention_parallel_dim_sizes(self) -> Dict[str, int]:
        if not self._has_runtime_parallelism_metadata():
            return self._get_static_parallel_dim_sizes()

        cfg = self._config
        return {
            "TP": int(cfg.runtime_attn_tensor_parallel_size),
            "CP": int(cfg.runtime_num_pipeline_stages),
            "DP": int(cfg.runtime_attn_data_parallel_size) * int(cfg.runtime_num_replicas),
            "EP": 1,
        }

    def _get_moe_parallel_dim_sizes(self) -> Dict[str, int]:
        if not self._has_runtime_parallelism_metadata():
            return self._get_static_parallel_dim_sizes()

        cfg = self._config
        return {
            "TP": int(cfg.runtime_moe_tensor_parallel_size),
            "CP": int(cfg.runtime_num_pipeline_stages),
            "DP": int(cfg.runtime_num_replicas),
            "EP": int(cfg.runtime_moe_expert_parallel_size),
        }

    def _validate_selected_parallel_dim_sizes(
        self,
        parallel_dim_sizes: Dict[str, int],
        *,
        context: str,
    ) -> Dict[str, int]:
        nodes = int(self._config.cluster_servers) * int(self._config.cluster_gpus_per_server)
        parallel_nodes = (
            int(parallel_dim_sizes["TP"])
            * int(parallel_dim_sizes["CP"])
            * int(parallel_dim_sizes["DP"])
            * int(parallel_dim_sizes["EP"])
        )
        if nodes != parallel_nodes:
            raise ValueError(
                f"collective-sim per-call parallel dims mismatch for {context}: "
                f"servers*gpus_per_server={nodes}, tp*cp*dp*ep={parallel_nodes}, "
                f"parallel_dim_sizes={parallel_dim_sizes}"
            )
        return parallel_dim_sizes

    def _is_shared_domain_cluster(self) -> bool:
        return self._cluster_type in {
            ClusterType.MONOLITHIC,
            ClusterType.PREFILL,
            ClusterType.DECODE,
        }

    def _uses_shared_domain_moe_execution_world(
        self,
        *,
        normalized_domain: str,
    ) -> bool:
        if normalized_domain not in {"MOE_TP", "MOE_EP", "EP"}:
            return False
        if not self._has_runtime_parallelism_metadata():
            return False
        if not self._is_shared_domain_cluster():
            return False
        return max(
            int(self._config.runtime_moe_tensor_parallel_size or 1),
            int(self._config.runtime_moe_expert_parallel_size or 1),
        ) > 1

    def _select_parallel_dim_sizes(
        self,
        *,
        normalized_domain: str,
        participant_count: int,
    ) -> Dict[str, int]:
        attn_parallelism = self._validate_selected_parallel_dim_sizes(
            self._get_attention_parallel_dim_sizes(),
            context="attention layout",
        )
        moe_parallelism_raw: Optional[Dict[str, int]] = None
        moe_parallelism_cache: Optional[Dict[str, int]] = None

        def _get_raw_moe_parallelism() -> Dict[str, int]:
            nonlocal moe_parallelism_raw
            if moe_parallelism_raw is None:
                moe_parallelism_raw = self._get_moe_parallel_dim_sizes()
            return moe_parallelism_raw

        def _has_valid_moe_parallelism() -> bool:
            parallel_dim_sizes = _get_raw_moe_parallelism()
            nodes = int(self._config.cluster_servers) * int(self._config.cluster_gpus_per_server)
            parallel_nodes = (
                int(parallel_dim_sizes["TP"])
                * int(parallel_dim_sizes["CP"])
                * int(parallel_dim_sizes["DP"])
                * int(parallel_dim_sizes["EP"])
            )
            return nodes == parallel_nodes

        def _validated_moe_parallelism() -> Dict[str, int]:
            nonlocal moe_parallelism_cache
            if moe_parallelism_cache is None:
                moe_parallelism_cache = self._validate_selected_parallel_dim_sizes(
                    _get_raw_moe_parallelism(),
                    context="moe layout",
                )
            return moe_parallelism_cache

        if normalized_domain in {"ATTN_TP"}:
            return attn_parallelism
        if normalized_domain in {"MOE_TP"}:
            if self._uses_shared_domain_moe_execution_world(
                normalized_domain=normalized_domain,
            ):
                return attn_parallelism
            return _validated_moe_parallelism()
        if normalized_domain in {"ATTN_DP", "DP"}:
            return attn_parallelism
        if normalized_domain in {"MOE_EP", "EP"}:
            if self._uses_shared_domain_moe_execution_world(
                normalized_domain=normalized_domain,
            ):
                return attn_parallelism
            return _validated_moe_parallelism()
        if normalized_domain in {"CP", "PP"}:
            if self._cluster_type == ClusterType.DECODE_FFN:
                return _validated_moe_parallelism()
            return attn_parallelism
        if normalized_domain == "TP":
            if not self._has_runtime_parallelism_metadata():
                return attn_parallelism
            if self._cluster_type == ClusterType.DECODE_FFN:
                return _validated_moe_parallelism()
            if self._cluster_type == ClusterType.DECODE_ATTN:
                return attn_parallelism
            if not _has_valid_moe_parallelism():
                return attn_parallelism

            moe_parallelism = _validated_moe_parallelism()
            if attn_parallelism == moe_parallelism:
                return attn_parallelism

            raise ValueError(
                "Ambiguous comm_domain='TP' for a shared attention/MoE runtime layout. "
                "Use comm_domain='ATTN_TP' or 'MOE_TP' explicitly. "
                f"cluster_type={self._cluster_type}, "
                f"attention_layout={attn_parallelism}, moe_layout={moe_parallelism}, "
                f"participant_count={participant_count}"
            )
        return attn_parallelism

    def _get_effective_domain_size(
        self,
        *,
        normalized_domain: str,
        mapped_domain: str,
        parallel_dim_sizes: Dict[str, int],
    ) -> int:
        if self._uses_shared_domain_moe_execution_world(
            normalized_domain=normalized_domain,
        ):
            return int(parallel_dim_sizes["TP"]) * int(parallel_dim_sizes["DP"])
        return self._get_domain_size(
            mapped_domain=mapped_domain,
            parallel_dim_sizes=parallel_dim_sizes,
        )

    def _build_shared_domain_moe_participant_ranks(
        self,
        *,
        participant_count: int,
        parallel_dim_sizes: Dict[str, int],
    ) -> Tuple[int, ...]:
        if participant_count <= 1:
            return tuple()

        domain_size = int(parallel_dim_sizes["TP"]) * int(parallel_dim_sizes["DP"])
        if participant_count > domain_size:
            raise ValueError(
                f"Requested participant_count={participant_count} exceeds shared-domain "
                f"MoE group size={domain_size}"
            )

        coords = {"TP": 0, "CP": 0, "DP": 0, "EP": 0}
        participant_ranks = []
        for dp_index in range(int(parallel_dim_sizes["DP"])):
            for tp_index in range(int(parallel_dim_sizes["TP"])):
                participant_ranks.append(
                    self._linearize_rank(
                        {**coords, "TP": tp_index, "DP": dp_index},
                        parallel_dim_sizes,
                    )
                )
        return tuple(participant_ranks[:participant_count])

    def _get_domain_size(
        self,
        *,
        mapped_domain: str,
        parallel_dim_sizes: Dict[str, int],
    ) -> int:
        return int(parallel_dim_sizes[mapped_domain])

    def _get_rank_strides(self, parallel_dim_sizes: Dict[str, int]) -> Dict[str, int]:
        strides: Dict[str, int] = {}
        stride = 1
        for dim in self._placement_order:
            strides[dim] = stride
            stride *= int(parallel_dim_sizes[dim])
        return strides

    def _linearize_rank(
        self,
        coords: Dict[str, int],
        parallel_dim_sizes: Dict[str, int],
    ) -> int:
        rank = 0
        strides = self._get_rank_strides(parallel_dim_sizes)
        for dim in self._placement_order:
            rank += int(coords.get(dim, 0)) * int(strides[dim])
        return rank

    def _build_participant_ranks(
        self,
        *,
        normalized_domain: str,
        mapped_domain: str,
        participant_count: int,
        parallel_dim_sizes: Dict[str, int],
    ) -> Tuple[int, ...]:
        if participant_count <= 1:
            return tuple()

        if self._uses_shared_domain_moe_execution_world(
            normalized_domain=normalized_domain,
        ):
            return self._build_shared_domain_moe_participant_ranks(
                participant_count=participant_count,
                parallel_dim_sizes=parallel_dim_sizes,
            )

        domain_size = self._get_effective_domain_size(
            normalized_domain=normalized_domain,
            mapped_domain=mapped_domain,
            parallel_dim_sizes=parallel_dim_sizes,
        )
        if participant_count > domain_size:
            raise ValueError(
                f"Requested participant_count={participant_count} exceeds domain_size={domain_size} "
                f"for mapped_domain={mapped_domain}"
            )

        coords = {"TP": 0, "CP": 0, "DP": 0, "EP": 0}
        return tuple(
            self._linearize_rank(
                {**coords, mapped_domain: index},
                parallel_dim_sizes,
            )
            for index in range(participant_count)
        )

    def _build_collective_spec(
        self,
        *,
        kind: str,
        tensor_bytes: int,
        mapped_domain: str,
        participant_ranks: Tuple[int, ...],
    ) -> Dict[str, Any]:
        cfg = self._config
        collective_spec: Dict[str, Any] = {
            "kind": kind,
            "tensor_bytes": int(tensor_bytes),
            "domain_dims": [mapped_domain],
            "placement_order": list(self._placement_order),
            "exclude_intra_server": bool(cfg.collective_exclude_intra_server),
            "use_triggers": bool(cfg.collective_use_triggers),
            "allreduce_model": str(cfg.allreduce_model),
            "allgather_model": str(cfg.allgather_model),
            "reducescatter_model": str(cfg.reducescatter_model),
            "alltoall_model": str(cfg.alltoall_model),
            "alltoall_channels": int(cfg.alltoall_channels),
            "alltoall_chunk_bytes": int(cfg.alltoall_chunk_bytes),
            "alltoall_chunk_inflight_per_peer": int(
                cfg.alltoall_chunk_inflight_per_peer
            ),
            "nchannels": int(getattr(cfg, "nchannels", 8)),
            "participant_ranks": list(participant_ranks),
        }
        if kind == "p2p":
            collective_spec.update(
                {
                    "p2p_src_index": int(cfg.p2p_src_index),
                    "p2p_dst_index": int(cfg.p2p_dst_index),
                    "p2p_direction": str(cfg.p2p_direction),
                }
            )
        return collective_spec

    def _build_scenario(
        self,
        *,
        kind: str,
        tensor_bytes: int,
        mapped_domain: str,
        participant_ranks: Tuple[int, ...],
        parallel_dim_sizes: Dict[str, int],
    ) -> Dict[str, Any]:
        cfg = self._config
        scenario: Dict[str, Any] = {
            "cluster": {
                "servers": int(cfg.cluster_servers),
                "gpus_per_server": int(cfg.cluster_gpus_per_server),
            },
            "parallelism": {
                "tp": int(parallel_dim_sizes["TP"]),
                "cp": int(parallel_dim_sizes["CP"]),
                "dp": int(parallel_dim_sizes["DP"]),
                "ep": int(parallel_dim_sizes["EP"]),
            },
            "topology": {
                "kind": str(cfg.topology_kind),
                "spines": int(cfg.topology_spines),
                "servers_per_tor": int(cfg.topology_servers_per_tor),
                "paths": int(cfg.topology_paths),
                "podsize": int(cfg.topology_podsize),
                "tor_down": int(cfg.topology_tor_down),
                "tor_up": int(cfg.topology_tor_up),
                "fattree_oversub": int(cfg.topology_fattree_oversub),
                "latency_ns": int(cfg.topology_latency_ns),
                "switch_latency_ns": int(cfg.topology_switch_latency_ns),
            },
            "network": {
                "linkspeed_mbps": int(cfg.network_linkspeed_mbps),
                "mtu": int(cfg.network_mtu),
                "q": int(cfg.network_q),
                "cwnd": int(cfg.network_cwnd),
            },
            "intra_server": {
                "model": str(cfg.intra_server_model),
                "nvlink_one_way_bw_GBps": float(cfg.nvlink_one_way_bw_GBps),
                "nvlink_latency_us": float(cfg.nvlink_latency_us),
                "nvlink_efficiency": float(cfg.nvlink_efficiency),
                "nvlink_allreduce_launch_overhead_us": float(
                    cfg.nvlink_allreduce_launch_overhead_us
                ),
            },
            "runner": {
                "end_us": int(cfg.runner_end_us),
                "build": bool(cfg.runner_build),
                "progress": bool(cfg.runner_progress),
                "print_args": bool(cfg.runner_print_args),
                "stop_on_finished": bool(cfg.runner_stop_on_finished),
                "heartbeat_s": int(cfg.runner_heartbeat_s),
                "out_dir": self._runner_out_dir,
            },
            "collective": self._build_collective_spec(
                kind=kind,
                tensor_bytes=tensor_bytes,
                mapped_domain=mapped_domain,
                participant_ranks=participant_ranks,
            ),
        }

        if cfg.scenario_profile:
            scenario["scenario_profile"] = str(cfg.scenario_profile)
        return scenario

    def _get_cached_prediction(self, key: Tuple[str, int, str, Tuple[int, ...], Tuple[int, int, int, int]]) -> Optional[float]:
        if key not in self._prediction_cache:
            return None

        value = self._prediction_cache.pop(key)
        self._prediction_cache[key] = value
        return value

    def _set_cached_prediction(self, key: Tuple[str, int, str, Tuple[int, ...], Tuple[int, int, int, int]], value: float) -> None:
        self._prediction_cache[key] = value
        if len(self._prediction_cache) > self._cache_size:
            self._prediction_cache.popitem(last=False)

    def get_prediction_cache_stats(self) -> Dict[str, int]:
        """Return lightweight runtime stats for the in-memory prediction cache."""
        return {
            "requests": int(self._prediction_cache_requests),
            "hits": int(self._prediction_cache_hits),
            "instance_hits": int(self._prediction_cache_instance_hits),
            "shared_hits": 0,
            "misses": int(self._prediction_cache_misses),
            "unique_entries": int(len(self._prediction_cache)),
            "resident_entries": int(len(self._prediction_cache)),
            "cache_capacity": int(self._cache_size),
        }

    def _predict_collective_time(self, scenario: Dict[str, Any]) -> float:
        """Invoke collective-sim predictor and return predicted time in ms."""
        result = self._collective_sim_predict(scenario, repo_root=str(self._repo_root))
        predicted = float(result["predicted_time_ms"])
        if predicted < 0:
            raise ValueError(f"collective-sim returned negative time: {predicted}")
        return predicted

    def _predict(
        self,
        *,
        kind: str,
        tensor_bytes: int,
        comm_domain: Optional[str],
        participant_count: int,
    ) -> float:
        self._validate_data_size(tensor_bytes)

        normalized_domain = self._normalize_comm_domain(comm_domain)
        mapped_domain = self._DOMAIN_TO_COLLECTIVE_SIM[normalized_domain]
        parallel_dim_sizes = self._select_parallel_dim_sizes(
            normalized_domain=normalized_domain,
            participant_count=participant_count,
        )
        domain_size = self._get_effective_domain_size(
            normalized_domain=normalized_domain,
            mapped_domain=mapped_domain,
            parallel_dim_sizes=parallel_dim_sizes,
        )
        if domain_size <= 1:
            return 0.0
        if participant_count <= 1:
            return 0.0

        participant_ranks = self._build_participant_ranks(
            normalized_domain=normalized_domain,
            mapped_domain=mapped_domain,
            participant_count=participant_count,
            parallel_dim_sizes=parallel_dim_sizes,
        )
        dim_signature = (
            int(parallel_dim_sizes["TP"]),
            int(parallel_dim_sizes["CP"]),
            int(parallel_dim_sizes["DP"]),
            int(parallel_dim_sizes["EP"]),
        )
        key = (kind, int(tensor_bytes), mapped_domain, participant_ranks, dim_signature)
        self._prediction_cache_requests += 1
        cached = self._get_cached_prediction(key)
        if cached is not None:
            self._prediction_cache_hits += 1
            self._prediction_cache_instance_hits += 1
            return cached
        self._prediction_cache_misses += 1

        scenario = self._build_scenario(
            kind=kind,
            tensor_bytes=tensor_bytes,
            mapped_domain=mapped_domain,
            participant_ranks=participant_ranks,
            parallel_dim_sizes=parallel_dim_sizes,
        )
        predicted = self._predict_collective_time(scenario)
        self._set_cached_prediction(key, predicted)
        return predicted

    def estimate_intra_server_allreduce_launch_overhead_ms(
        self,
        *,
        num_devices: int,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_num_devices(num_devices, "allreduce")
        if num_devices <= 1:
            return 0.0
        if str(self._config.intra_server_model) != "nvlink_analytic":
            return 0.0
        launch_overhead_us = float(self._config.nvlink_allreduce_launch_overhead_us)
        if launch_overhead_us <= 0.0:
            return 0.0

        normalized_domain = self._normalize_comm_domain(comm_domain)
        mapped_domain = self._DOMAIN_TO_COLLECTIVE_SIM[normalized_domain]
        parallel_dim_sizes = self._select_parallel_dim_sizes(
            normalized_domain=normalized_domain,
            participant_count=num_devices,
        )
        domain_size = self._get_effective_domain_size(
            normalized_domain=normalized_domain,
            mapped_domain=mapped_domain,
            parallel_dim_sizes=parallel_dim_sizes,
        )
        if domain_size <= 1:
            return 0.0

        participant_ranks = self._build_participant_ranks(
            normalized_domain=normalized_domain,
            mapped_domain=mapped_domain,
            participant_count=num_devices,
            parallel_dim_sizes=parallel_dim_sizes,
        )
        if not participant_ranks:
            return 0.0

        gpus_per_server = max(int(self._config.cluster_gpus_per_server), 1)
        counts_per_server: Dict[int, int] = {}
        for rank in participant_ranks:
            server_idx = rank // gpus_per_server
            counts_per_server[server_idx] = counts_per_server.get(server_idx, 0) + 1
        if not counts_per_server:
            return 0.0

        ranks_in_group = len(participant_ranks)
        servers_in_group = len(counts_per_server)
        gpus_per_group_per_server = max(
            1,
            int(round(float(ranks_in_group) / float(servers_in_group))),
        )
        if gpus_per_group_per_server <= 1:
            return 0.0

        if servers_in_group <= 1:
            steps = 2 * (ranks_in_group - 1)
        else:
            steps = int(
                math.ceil(
                    2.0
                    * (float(ranks_in_group) - 1.0)
                    * (float(gpus_per_group_per_server) - 1.0)
                    / float(gpus_per_group_per_server)
                )
            )
        if steps <= 0:
            return 0.0
        return (steps * launch_overhead_us) / 1000.0

    def predict_allreduce(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_num_devices(num_devices, "allreduce")
        if num_devices <= 1:
            return 0.0
        return self._predict(
            kind="allreduce",
            tensor_bytes=data_size_bytes,
            comm_domain=comm_domain,
            participant_count=num_devices,
        )

    def predict_allgather(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_num_devices(num_devices, "allgather")
        if num_devices <= 1:
            return 0.0
        return self._predict(
            kind="allgather",
            tensor_bytes=data_size_bytes,
            comm_domain=comm_domain,
            participant_count=num_devices,
        )

    def predict_broadcast(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        raise NotImplementedError(
            "collective-sim backend does not support broadcast prediction"
        )

    def predict_send_recv(
        self,
        data_size_bytes: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        return self._predict(
            kind="p2p",
            tensor_bytes=data_size_bytes,
            comm_domain=comm_domain,
            participant_count=2,
        )

    def predict_reduce_scatter(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_num_devices(num_devices, "reduce_scatter")
        if num_devices <= 1:
            return 0.0

        tensor_bytes = data_size_bytes // num_devices
        return self._predict(
            kind="reducescatter",
            tensor_bytes=tensor_bytes,
            comm_domain=comm_domain,
            participant_count=num_devices,
        )

    def predict_all_to_all(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_num_devices(num_devices, "all_to_all")
        if num_devices <= 1:
            return 0.0
        return self._predict(
            kind="alltoall",
            tensor_bytes=data_size_bytes,
            comm_domain=comm_domain,
            participant_count=num_devices,
        )


CCBackendFactory.register(CCBackendType.COLLECTIVE_SIM, CollectiveSimCCBackend)
