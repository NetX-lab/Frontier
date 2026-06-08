"""Lightweight ASTRA-Sim analytical CC backend implementation."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Dict, Optional, Tuple

from frontier.cc_backend.base_cc_backend import BaseCCBackend
from frontier.cc_backend.cc_backend_config import AstraSimAnalyticalCCBackendConfig
from frontier.cc_backend.cc_backend_factory import CCBackendFactory
from frontier.logger import init_logger
from frontier.types import CCBackendType, ClusterType

logger = init_logger(__name__)


class AstraSimAnalyticalCCBackend(BaseCCBackend):
    """Lightweight ASTRA-Sim-inspired analytical CC backend."""

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
    _DOMAIN_TO_CANONICAL = {
        "TP": "TP",
        "ATTN_TP": "TP",
        "MOE_TP": "TP",
        "CP": "CP",
        "PP": "CP",
        "DP": "DP",
        "ATTN_DP": "DP",
        "EP": "EP",
        "MOE_EP": "EP",
    }

    def __init__(
        self,
        config: AstraSimAnalyticalCCBackendConfig,
        cluster_type: ClusterType,
        device_type: str,
        network_device: str,
        num_devices: int,
    ) -> None:
        super().__init__(config, cluster_type, device_type, network_device, num_devices)
        self._config: AstraSimAnalyticalCCBackendConfig = config
        self._cache_size = int(config.prediction_cache_size)
        if self._cache_size <= 0:
            raise ValueError(
                f"prediction_cache_size must be > 0, got {self._cache_size}"
            )
        self._placement_order = self._parse_placement_order(config.placement_order)
        self._prediction_cache: "OrderedDict[Tuple, float]" = OrderedDict()

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

    def _normalize_comm_domain(self, comm_domain: Optional[str]) -> str:
        if not comm_domain:
            raise ValueError(
                "astra_sim_analytical backend requires explicit comm_domain "
                "(ATTN_TP/MOE_TP/DP/EP/PP)"
            )

        domain = comm_domain.strip().upper()
        if domain not in self._KNOWN_DOMAINS:
            raise ValueError(
                f"Unsupported comm_domain={comm_domain!r}. "
                f"Supported domains: {sorted(self._KNOWN_DOMAINS)}"
            )
        return domain

    def _has_runtime_parallelism_metadata(self) -> bool:
        required = (
            self._config.runtime_num_replicas,
            self._config.runtime_num_pipeline_stages,
            self._config.runtime_attn_tensor_parallel_size,
            self._config.runtime_attn_data_parallel_size,
            self._config.runtime_moe_tensor_parallel_size,
            self._config.runtime_moe_expert_parallel_size,
        )
        return all(value is not None for value in required)

    def _get_attention_parallel_dim_sizes(self) -> Dict[str, int]:
        return {
            "TP": int(self._config.runtime_attn_tensor_parallel_size),
            "CP": int(self._config.runtime_num_pipeline_stages),
            "DP": int(self._config.runtime_attn_data_parallel_size)
            * int(self._config.runtime_num_replicas),
            "EP": 1,
        }

    def _get_moe_parallel_dim_sizes(self) -> Dict[str, int]:
        return {
            "TP": int(self._config.runtime_moe_tensor_parallel_size),
            "CP": int(self._config.runtime_num_pipeline_stages),
            "DP": int(self._config.runtime_num_replicas),
            "EP": int(self._config.runtime_moe_expert_parallel_size),
        }

    def _validate_selected_parallel_dim_sizes(
        self,
        parallel_dim_sizes: Dict[str, int],
        *,
        context: str,
    ) -> Dict[str, int]:
        world_size = self._get_world_size()
        parallel_nodes = (
            int(parallel_dim_sizes["TP"])
            * int(parallel_dim_sizes["CP"])
            * int(parallel_dim_sizes["DP"])
            * int(parallel_dim_sizes["EP"])
        )
        if world_size != parallel_nodes:
            raise ValueError(
                f"astra_sim_analytical per-call parallel dims mismatch for {context}: "
                f"world_size={world_size}, tp*cp*dp*ep={parallel_nodes}, "
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
        attn_parallelism_cache: Optional[Dict[str, int]] = None
        moe_parallelism_raw: Optional[Dict[str, int]] = None
        moe_parallelism_cache: Optional[Dict[str, int]] = None

        def _validated_attn_parallelism() -> Dict[str, int]:
            nonlocal attn_parallelism_cache
            if attn_parallelism_cache is None:
                attn_parallelism_cache = self._validate_selected_parallel_dim_sizes(
                    self._get_attention_parallel_dim_sizes(),
                    context="attention layout",
                )
            return attn_parallelism_cache

        def _get_raw_moe_parallelism() -> Dict[str, int]:
            nonlocal moe_parallelism_raw
            if moe_parallelism_raw is None:
                moe_parallelism_raw = self._get_moe_parallel_dim_sizes()
            return moe_parallelism_raw

        def _has_valid_moe_parallelism() -> bool:
            parallel_dim_sizes = _get_raw_moe_parallelism()
            world_size = self._get_world_size()
            parallel_nodes = (
                int(parallel_dim_sizes["TP"])
                * int(parallel_dim_sizes["CP"])
                * int(parallel_dim_sizes["DP"])
                * int(parallel_dim_sizes["EP"])
            )
            return world_size == parallel_nodes

        def _validated_moe_parallelism() -> Dict[str, int]:
            nonlocal moe_parallelism_cache
            if moe_parallelism_cache is None:
                moe_parallelism_cache = self._validate_selected_parallel_dim_sizes(
                    _get_raw_moe_parallelism(),
                    context="moe layout",
                )
            return moe_parallelism_cache

        if normalized_domain == "ATTN_TP":
            return _validated_attn_parallelism()
        if normalized_domain == "MOE_TP":
            if self._uses_shared_domain_moe_execution_world(
                normalized_domain=normalized_domain
            ):
                return _validated_attn_parallelism()
            return _validated_moe_parallelism()
        if normalized_domain in {"ATTN_DP", "DP"}:
            return _validated_attn_parallelism()
        if normalized_domain in {"MOE_EP", "EP"}:
            if self._uses_shared_domain_moe_execution_world(
                normalized_domain=normalized_domain
            ):
                return _validated_attn_parallelism()
            return _validated_moe_parallelism()
        if normalized_domain in {"CP", "PP"}:
            if self._cluster_type == ClusterType.DECODE_FFN:
                return _validated_moe_parallelism()
            return _validated_attn_parallelism()
        if normalized_domain == "TP":
            if not self._has_runtime_parallelism_metadata():
                return _validated_attn_parallelism()
            if self._cluster_type == ClusterType.DECODE_FFN:
                return _validated_moe_parallelism()
            if self._cluster_type == ClusterType.DECODE_ATTN:
                return _validated_attn_parallelism()
            if not _has_valid_moe_parallelism():
                return _validated_attn_parallelism()
            attn_parallelism = _validated_attn_parallelism()
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
        return _validated_attn_parallelism()

    def _get_domain_size(
        self,
        *,
        canonical_domain: str,
        parallel_dim_sizes: Dict[str, int],
    ) -> int:
        return int(parallel_dim_sizes[canonical_domain])

    def _get_effective_domain_size(
        self,
        *,
        normalized_domain: str,
        canonical_domain: str,
        parallel_dim_sizes: Dict[str, int],
    ) -> int:
        if self._uses_shared_domain_moe_execution_world(
            normalized_domain=normalized_domain
        ):
            return int(parallel_dim_sizes["TP"]) * int(parallel_dim_sizes["DP"])
        return self._get_domain_size(
            canonical_domain=canonical_domain,
            parallel_dim_sizes=parallel_dim_sizes,
        )

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

    def _build_shared_domain_moe_participant_ranks(
        self,
        *,
        participant_count: int,
        parallel_dim_sizes: Dict[str, int],
    ) -> Tuple[int, ...]:
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

    def _build_participant_ranks(
        self,
        *,
        normalized_domain: str,
        canonical_domain: str,
        participant_count: int,
        parallel_dim_sizes: Dict[str, int],
    ) -> Tuple[int, ...]:
        if participant_count <= 1:
            return tuple()

        if self._uses_shared_domain_moe_execution_world(
            normalized_domain=normalized_domain
        ):
            return self._build_shared_domain_moe_participant_ranks(
                participant_count=participant_count,
                parallel_dim_sizes=parallel_dim_sizes,
            )

        domain_size = self._get_effective_domain_size(
            normalized_domain=normalized_domain,
            canonical_domain=canonical_domain,
            parallel_dim_sizes=parallel_dim_sizes,
        )
        if participant_count > domain_size:
            raise ValueError(
                f"Requested participant_count={participant_count} exceeds domain_size={domain_size} "
                f"for canonical_domain={canonical_domain}"
            )

        coords = {"TP": 0, "CP": 0, "DP": 0, "EP": 0}
        return tuple(
            self._linearize_rank(
                {**coords, canonical_domain: index},
                parallel_dim_sizes,
            )
            for index in range(participant_count)
        )

    def _resolve_collective_participants(
        self,
        *,
        num_devices: int,
        comm_domain: Optional[str],
    ) -> Tuple[int, ...]:
        if num_devices <= 1:
            return tuple()

        if not self._has_runtime_parallelism_metadata():
            return tuple(range(num_devices))

        normalized_domain = self._normalize_comm_domain(comm_domain)
        canonical_domain = self._DOMAIN_TO_CANONICAL[normalized_domain]
        parallel_dim_sizes = self._select_parallel_dim_sizes(
            normalized_domain=normalized_domain,
            participant_count=num_devices,
        )
        domain_size = self._get_effective_domain_size(
            normalized_domain=normalized_domain,
            canonical_domain=canonical_domain,
            parallel_dim_sizes=parallel_dim_sizes,
        )
        if domain_size <= 1:
            return tuple()
        return self._build_participant_ranks(
            normalized_domain=normalized_domain,
            canonical_domain=canonical_domain,
            participant_count=num_devices,
            parallel_dim_sizes=parallel_dim_sizes,
        )

    def _resolve_send_recv_participants(
        self,
        *,
        comm_domain: Optional[str],
    ) -> Tuple[int, ...]:
        required_count = max(
            2,
            int(self._config.p2p_src_index) + 1,
            int(self._config.p2p_dst_index) + 1,
        )
        if not self._has_runtime_parallelism_metadata():
            return tuple(range(required_count))

        normalized_domain = self._normalize_comm_domain(comm_domain)
        canonical_domain = self._DOMAIN_TO_CANONICAL[normalized_domain]
        parallel_dim_sizes = self._select_parallel_dim_sizes(
            normalized_domain=normalized_domain,
            participant_count=required_count,
        )
        domain_size = self._get_effective_domain_size(
            normalized_domain=normalized_domain,
            canonical_domain=canonical_domain,
            parallel_dim_sizes=parallel_dim_sizes,
        )
        if domain_size <= 1:
            return tuple()
        return self._build_participant_ranks(
            normalized_domain=normalized_domain,
            canonical_domain=canonical_domain,
            participant_count=domain_size,
            parallel_dim_sizes=parallel_dim_sizes,
        )

    def _get_world_size(self) -> int:
        return int(self._config.cluster_servers) * int(self._config.cluster_gpus_per_server)

    def _validate_rank(self, rank: int) -> None:
        if rank < 0 or rank >= self._get_world_size():
            raise ValueError(
                f"rank={rank} is out of range for world_size={self._get_world_size()}"
            )

    def _rank_to_physical_address(self, rank: int) -> Tuple[int, int]:
        self._validate_rank(rank)
        gpus_per_server = max(int(self._config.cluster_gpus_per_server), 1)
        return (rank % gpus_per_server, rank // gpus_per_server)

    def _get_first_differing_dimension(
        self,
        src_address: Tuple[int, int],
        dst_address: Tuple[int, int],
    ) -> int:
        for dim, (src_component, dst_component) in enumerate(zip(src_address, dst_address)):
            if src_component != dst_component:
                return dim
        raise ValueError(
            f"src_address={src_address} and dst_address={dst_address} are identical"
        )

    def _get_dimension_topology(self, dimension: int) -> Tuple[str, int, float, float]:
        if dimension == 0:
            return (
                str(self._config.intra_server_topology),
                int(self._config.cluster_gpus_per_server),
                float(self._config.intra_server_bandwidth_gbps),
                float(self._config.intra_server_latency_us),
            )
        if dimension == 1:
            return (
                str(self._config.inter_server_topology),
                int(self._config.cluster_servers),
                float(self._config.inter_server_bandwidth_gbps),
                float(self._config.inter_server_latency_us),
            )
        raise ValueError(f"Unsupported physical dimension: {dimension}")

    def _compute_hops(
        self,
        *,
        topology_kind: str,
        topology_size: int,
        src_index: int,
        dst_index: int,
    ) -> int:
        normalized_topology = str(topology_kind).strip().lower()
        if normalized_topology == "fullyconnected":
            return 1
        if normalized_topology == "switch":
            return 2
        if normalized_topology == "ring":
            clockwise_distance = dst_index - src_index
            if clockwise_distance < 0:
                clockwise_distance += topology_size
            if not self._config.ring_bidirectional:
                return clockwise_distance
            return min(clockwise_distance, topology_size - clockwise_distance)
        raise ValueError(
            f"Unsupported ASTRA analytical topology_kind={topology_kind!r}. "
            "Supported kinds: FullyConnected, Switch, Ring"
        )

    def _bandwidth_bytes_per_ms(self, bandwidth_gbps: float) -> float:
        return (float(bandwidth_gbps) * 1e9) / (8 * 1000)

    def _send_ms(
        self,
        *,
        src_rank: int,
        dst_rank: int,
        chunk_bytes: float,
    ) -> float:
        if src_rank == dst_rank:
            return 0.0

        src_address = self._rank_to_physical_address(src_rank)
        dst_address = self._rank_to_physical_address(dst_rank)
        dimension = self._get_first_differing_dimension(src_address, dst_address)
        topology_kind, topology_size, bandwidth_gbps, latency_us = self._get_dimension_topology(
            dimension
        )
        src_component = src_address[dimension]
        dst_component = dst_address[dimension]
        hops = self._compute_hops(
            topology_kind=topology_kind,
            topology_size=max(topology_size, 1),
            src_index=src_component,
            dst_index=dst_component,
        )
        latency_ms = float(latency_us) / 1000.0
        serialization_ms = float(chunk_bytes) / self._bandwidth_bytes_per_ms(
            bandwidth_gbps
        )
        return (float(hops) * latency_ms) + serialization_ms

    def _round_latency_ms(
        self,
        *,
        participants: Tuple[int, ...],
        chunk_bytes: float,
        offset: int,
    ) -> float:
        if len(participants) <= 1:
            return 0.0
        delays = [
            self._send_ms(
                src_rank=participants[index],
                dst_rank=participants[(index + offset) % len(participants)],
                chunk_bytes=chunk_bytes,
            )
            for index in range(len(participants))
        ]
        return max(delays)

    def _predict_ring_collective(
        self,
        *,
        kind: str,
        data_size_bytes: int,
        num_devices: int,
        comm_domain: Optional[str],
    ) -> float:
        participants = self._resolve_collective_participants(
            num_devices=num_devices,
            comm_domain=comm_domain,
        )
        if len(participants) <= 1:
            return 0.0

        participant_count = len(participants)
        if kind == "allreduce":
            phase_chunk_bytes = float(data_size_bytes) / float(participant_count)
            phase_latency = self._round_latency_ms(
                participants=participants,
                chunk_bytes=phase_chunk_bytes,
                offset=1,
            )
            return (2 * (participant_count - 1)) * phase_latency
        if kind == "allgather":
            phase_latency = self._round_latency_ms(
                participants=participants,
                chunk_bytes=float(data_size_bytes),
                offset=1,
            )
            return (participant_count - 1) * phase_latency
        if kind == "reduce_scatter":
            phase_chunk_bytes = float(data_size_bytes) / float(participant_count)
            phase_latency = self._round_latency_ms(
                participants=participants,
                chunk_bytes=phase_chunk_bytes,
                offset=1,
            )
            return (participant_count - 1) * phase_latency
        if kind == "all_to_all":
            phase_chunk_bytes = float(data_size_bytes) / float(participant_count)
            return sum(
                self._round_latency_ms(
                    participants=participants,
                    chunk_bytes=phase_chunk_bytes,
                    offset=offset,
                )
                for offset in range(1, participant_count)
            )
        raise ValueError(f"Unsupported collective kind={kind!r}")

    def _get_cached_prediction(self, key: Tuple) -> Optional[float]:
        if key not in self._prediction_cache:
            return None
        value = self._prediction_cache.pop(key)
        self._prediction_cache[key] = value
        return value

    def _set_cached_prediction(self, key: Tuple, value: float) -> None:
        self._prediction_cache[key] = value
        if len(self._prediction_cache) > self._cache_size:
            self._prediction_cache.popitem(last=False)

    def _predict_with_cache(
        self,
        *,
        kind: str,
        data_size_bytes: int,
        num_devices: int,
        comm_domain: Optional[str],
    ) -> float:
        key = (kind, int(data_size_bytes), int(num_devices), str(comm_domain or ""))
        cached = self._get_cached_prediction(key)
        if cached is not None:
            return cached
        predicted = self._predict_ring_collective(
            kind=kind,
            data_size_bytes=data_size_bytes,
            num_devices=num_devices,
            comm_domain=comm_domain,
        )
        self._set_cached_prediction(key, predicted)
        return predicted

    def predict_allreduce(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "allreduce")
        if num_devices <= 1:
            return 0.0
        return self._predict_with_cache(
            kind="allreduce",
            data_size_bytes=data_size_bytes,
            num_devices=num_devices,
            comm_domain=comm_domain,
        )

    def predict_allgather(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "allgather")
        if num_devices <= 1:
            return 0.0
        return self._predict_with_cache(
            kind="allgather",
            data_size_bytes=data_size_bytes,
            num_devices=num_devices,
            comm_domain=comm_domain,
        )

    def predict_broadcast(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "broadcast")
        raise NotImplementedError(
            "broadcast is not implemented in ASTRA-Sim analytical path for Frontier v0.3"
        )

    def predict_send_recv(
        self,
        data_size_bytes: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_data_size(data_size_bytes)
        participants = self._resolve_send_recv_participants(comm_domain=comm_domain)
        if len(participants) <= 1:
            return 0.0

        src_index = int(self._config.p2p_src_index)
        dst_index = int(self._config.p2p_dst_index)
        if src_index < 0 or dst_index < 0:
            raise ValueError(
                f"p2p indices must be non-negative, got src={src_index}, dst={dst_index}"
            )
        if src_index >= len(participants) or dst_index >= len(participants):
            raise ValueError(
                f"p2p indices out of range for participant_count={len(participants)}: "
                f"src={src_index}, dst={dst_index}"
            )
        if src_index == dst_index:
            raise ValueError(
                f"p2p source and destination indices must differ, got {src_index}"
            )

        return self._send_ms(
            src_rank=participants[src_index],
            dst_rank=participants[dst_index],
            chunk_bytes=float(data_size_bytes),
        )

    def predict_reduce_scatter(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "reduce_scatter")
        if num_devices <= 1:
            return 0.0
        return self._predict_with_cache(
            kind="reduce_scatter",
            data_size_bytes=data_size_bytes,
            num_devices=num_devices,
            comm_domain=comm_domain,
        )

    def predict_all_to_all(
        self,
        data_size_bytes: int,
        num_devices: int,
        cluster_type: Optional[ClusterType] = None,
        comm_domain: Optional[str] = None,
    ) -> float:
        self._validate_data_size(data_size_bytes)
        self._validate_num_devices(num_devices, "all_to_all")
        if num_devices <= 1:
            return 0.0
        return self._predict_with_cache(
            kind="all_to_all",
            data_size_bytes=data_size_bytes,
            num_devices=num_devices,
            comm_domain=comm_domain,
        )


CCBackendFactory.register(
    CCBackendType.ASTRA_SIM_ANALYTICAL, AstraSimAnalyticalCCBackend
)
