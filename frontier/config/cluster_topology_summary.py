"""Cluster topology accounting: replica inventory, server counts, statistics.

These methods read already-resolved cluster configurations and report on them.
They neither construct nor validate configuration, which is why they sit apart
from both the field surface and the per-role builders.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from frontier.config.cluster_role_config import _get_cc_backend_configs
from frontier.config.parallel_semantics import (
    resolve_collective_sim_physical_topology,
)
from frontier.config.replica_config import ReplicaConfig


class ClusterTopologySummary:
    """Reports the replica inventory and server counts of a cluster."""

    def _collect_cluster_info(self) -> List[Tuple[str, int, ReplicaConfig]]:
        clusters_info = []

        if self._has_disaggregation_params_set():
            if self.prefill_cluster_num_replicas and self.prefill_replica_config:
                clusters_info.append(
                    (
                        "PREFILL",
                        self.prefill_cluster_num_replicas,
                        self.prefill_replica_config,
                    )
                )

            if (
                self.decode_attn_cluster_num_replicas
                and self.decode_attn_replica_config
            ):
                clusters_info.append(
                    (
                        "DECODE_ATTN",
                        self.decode_attn_cluster_num_replicas,
                        self.decode_attn_replica_config,
                    )
                )

            if self.decode_ffn_cluster_num_replicas and self.decode_ffn_replica_config:
                clusters_info.append(
                    (
                        "DECODE_FFN",
                        self.decode_ffn_cluster_num_replicas,
                        self.decode_ffn_replica_config,
                    )
                )

            if self.decode_cluster_num_replicas and self.decode_replica_config:
                clusters_info.append(
                    (
                        "DECODE",
                        self.decode_cluster_num_replicas,
                        self.decode_replica_config,
                    )
                )
        else:
            clusters_info.append(("MONOLITHIC", self.num_replicas, self.replica_config))

        return clusters_info

    def get_server_count_metadata(self, sys_arch: str) -> Dict[str, int]:
        clusters_info = self._collect_cluster_info()
        server_counts_by_cluster = {}
        _, _, _, CollectiveSimCCBackendConfig, _, _ = _get_cc_backend_configs()
        cluster_prefix_by_name = {
            "PREFILL": "prefill",
            "DECODE_ATTN": "decode_attn",
            "DECODE_FFN": "decode_ffn",
            "DECODE": "decode",
        }

        for cluster_name, num_replicas, replica_config in clusters_info:
            cluster_total_devices = int(num_replicas) * int(replica_config.world_size)
            num_devices_per_node = int(replica_config.node_config.num_devices_per_node)
            if cluster_total_devices <= 0:
                raise ValueError(
                    "cluster_total_devices must be positive when computing "
                    f"server-count metadata, got {cluster_total_devices}"
                )
            if num_devices_per_node <= 0:
                raise ValueError(
                    "num_devices_per_node must be positive when computing "
                    f"server-count metadata, got {num_devices_per_node}"
                )
            if cluster_name == "MONOLITHIC":
                cc_backend_config = self.cc_backend_config
            else:
                cc_backend_config = self._create_cc_backend_config_for_cluster(
                    cluster_prefix_by_name[cluster_name]
                )
            if isinstance(cc_backend_config, CollectiveSimCCBackendConfig):
                physical_topology = resolve_collective_sim_physical_topology(
                    cluster_total_devices=cluster_total_devices,
                    num_devices_per_node=num_devices_per_node,
                    scenario_profile=getattr(
                        cc_backend_config,
                        "scenario_profile",
                        None,
                    ),
                )
                server_counts_by_cluster[cluster_name] = int(physical_topology.servers)
                continue
            server_counts_by_cluster[cluster_name] = (
                cluster_total_devices + num_devices_per_node - 1
            ) // num_devices_per_node

        if sys_arch == "co-location":
            if "MONOLITHIC" not in server_counts_by_cluster:
                raise ValueError("Missing MONOLITHIC cluster for co-location mode.")
            return {"server_count": server_counts_by_cluster["MONOLITHIC"]}
        if sys_arch == "pd-disaggregation":
            if "PREFILL" not in server_counts_by_cluster or "DECODE" not in server_counts_by_cluster:
                raise ValueError(
                    "Missing PREFILL or DECODE cluster for pd-disaggregation mode."
                )
            return {
                "prefill_server_count": server_counts_by_cluster["PREFILL"],
                "decode_server_count": server_counts_by_cluster["DECODE"],
            }
        if sys_arch == "pd-af-disaggregation":
            required = ["PREFILL", "DECODE_ATTN", "DECODE_FFN"]
            if any(name not in server_counts_by_cluster for name in required):
                raise ValueError(
                    "Missing PREFILL, DECODE_ATTN, or DECODE_FFN cluster for pd-af-disaggregation mode."
                )
            return {
                "prefill_server_count": server_counts_by_cluster["PREFILL"],
                "decode_attn_server_count": server_counts_by_cluster["DECODE_ATTN"],
                "decode_ffn_server_count": server_counts_by_cluster["DECODE_FFN"],
            }

        raise ValueError(f"Unknown system architecture: {sys_arch}")

    def print_cluster_statistics(self, simulation_mode: str, sys_arch: str):
        """Calculate and print statistics for all clusters (called from SimulationConfig)."""
        clusters_info = self._collect_cluster_info()

        # Calculate total statistics
        self.total_clusters = len(clusters_info)
        self.cluster_world_sizes = {}

        # Calculate world_size if not already set
        if not hasattr(self, "world_size") or self.world_size is None:
            self.world_size = sum(
                num_replicas * replica_config.world_size
                for _, num_replicas, replica_config in clusters_info
            )

        # Print cluster configuration summary
        print("\n" + "=" * 70)
        print("CLUSTER CONFIGURATION SUMMARY")
        print("=" * 70)
        print("Simulation mode: ", simulation_mode)
        print("System architecture: ", sys_arch)
        print("=" * 70)
        print(f"Total Clusters: {self.total_clusters}")
        print(f"Total World Size: {self.world_size}")
        server_count_metadata = self.get_server_count_metadata(sys_arch)
        if sys_arch == "co-location":
            print(f"Server count: {server_count_metadata['server_count']}")
        elif sys_arch == "pd-disaggregation":
            print(
                f"Prefill server count: {server_count_metadata['prefill_server_count']}"
            )
            print(
                f"Decode server count: {server_count_metadata['decode_server_count']}"
            )
        elif sys_arch == "pd-af-disaggregation":
            print(
                f"Prefill server count: {server_count_metadata['prefill_server_count']}"
            )
            print(
                f"Decode-Attn server count: {server_count_metadata['decode_attn_server_count']}"
            )
            print(
                f"Decode-FFN server count: {server_count_metadata['decode_ffn_server_count']}"
            )
        print()

        for cluster_name, num_replicas, replica_config in clusters_info:
            cluster_world_size = num_replicas * replica_config.world_size
            self.cluster_world_sizes[cluster_name] = cluster_world_size

            print(f"Cluster Type: {cluster_name}")
            print(f"   Cluster World Size: {cluster_world_size}")
            print(f"   Num Replicas (Instances): {num_replicas}")
            print(f"   Replica World Size: {replica_config.world_size}")
            if (
                cluster_name == "PREFILL"
                or cluster_name == "MONOLITHIC"
                or cluster_name == "DECODE"
            ):
                print(
                    f"   Configuration: PP{replica_config.num_pipeline_stages} × (Attn_TP{replica_config.attn_tensor_parallel_size} x Attn_DP{replica_config.attn_dp}) | (MoE_TP{replica_config.moe_tensor_parallel_size} x MoE_EP{replica_config.moe_expert_parallel_size})"
                )
                print(f"   Total Expert Num: {replica_config.total_expert_num}")
                print(f"   Local Expert Num: {replica_config.local_expert_num}")
            elif cluster_name == "DECODE_ATTN":
                print(
                    f"   Configuration: PP{replica_config.num_pipeline_stages} × Attn_TP{replica_config.attn_tensor_parallel_size} x Attn_DP{replica_config.attn_dp}"
                )
            elif cluster_name == "DECODE_FFN":
                print(
                    f"   Configuration: PP{replica_config.num_pipeline_stages} × MoE_TP{replica_config.moe_tensor_parallel_size} x MoE_EP{replica_config.moe_expert_parallel_size}"
                )
                print(f"   Total Expert Num: {replica_config.total_expert_num}")
                print(f"   Local Expert Num: {replica_config.local_expert_num}")

            print("-" * 50)

        print("=" * 70 + "\n")
