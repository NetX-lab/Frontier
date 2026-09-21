"""Per-role configuration construction for a disaggregated cluster.

`ClusterConfig` owns a single flat field surface covering every role the
release supports.  The methods here turn that flat surface into the concrete
`ReplicaConfig`, execution-time predictor configuration and communication-cost
backend configuration each role needs.

They are a mixin rather than free functions so that the split is exactly a
move: the bodies, the method names and every call site are unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

from frontier.config.execution_time_predictor_config import (
    BaseExecutionTimePredictorConfig,
)
from frontier.config.replica_config import ReplicaConfig
from frontier.types import ClusterType

if TYPE_CHECKING:
    from frontier.cc_backend.cc_backend_config import BaseCCBackendConfig
    from frontier.config.cluster_config import ClusterConfig


def _get_cc_backend_configs():
    """Lazily import CC backend config classes to avoid circular imports."""
    from frontier.cc_backend.cc_backend_config import (
        BaseCCBackendConfig,
        VidurCCBackendConfig,
        AnalyticalCCBackendConfig,
        CollectiveSimCCBackendConfig,
        AiconfiguratorCCBackendConfig,
        AstraSimAnalyticalCCBackendConfig,
    )

    return (
        BaseCCBackendConfig,
        VidurCCBackendConfig,
        AnalyticalCCBackendConfig,
        CollectiveSimCCBackendConfig,
        AiconfiguratorCCBackendConfig,
        AstraSimAnalyticalCCBackendConfig,
    )


class ClusterRoleConfigBuilder:
    """Builds the per-role configurations a disaggregated cluster needs."""

    def _create_replica_config_from_fields(self, cluster_prefix: str) -> ReplicaConfig:
        """Create ReplicaConfig object from flattened fields."""
        # Get default values from main replica_config or use ReplicaConfig defaults
        main_config = self.replica_config if self.replica_config else ReplicaConfig()

        # Extract cluster-specific fields using getattr with fallback to main config
        def get_field_value(field_name: str):
            cluster_field_name = f"{cluster_prefix}_replica_config_{field_name}"
            cluster_value = getattr(self, cluster_field_name, None)
            if cluster_value is not None:
                return cluster_value
            return getattr(main_config, field_name)

        if cluster_prefix == "decode_attn":
            moe_expert_parallel_size = 0
            moe_tensor_parallel_size = 0
            total_expert_num = 0
            local_expert_num = 0
            num_pipeline_stages = 1
            router_load_balancing_type = None
            router_topk = None
            moe_routing_distribution_type = get_field_value(
                "moe_routing_distribution_type"
            )
            attn_tensor_parallel_size = get_field_value("attn_tensor_parallel_size")
            attn_dp = get_field_value("attn_dp")
        else:
            attn_tensor_parallel_size = get_field_value("attn_tensor_parallel_size")
            attn_dp = get_field_value("attn_dp")
            moe_tensor_parallel_size = get_field_value("moe_tensor_parallel_size")
            moe_expert_parallel_size = get_field_value("moe_expert_parallel_size")
            total_expert_num = get_field_value("total_expert_num")
            local_expert_num = get_field_value("local_expert_num")
            router_load_balancing_type = get_field_value("router_load_balancing_type")
            router_topk = get_field_value("router_topk")
            moe_routing_distribution_type = get_field_value(
                "moe_routing_distribution_type"
            )
            if cluster_prefix == "decode_ffn":
                num_pipeline_stages = 1
            else:
                num_pipeline_stages = get_field_value("num_pipeline_stages")

        return ReplicaConfig(
            model_name=get_field_value("model_name"),
            memory_margin_fraction=get_field_value("memory_margin_fraction"),
            num_pipeline_stages=num_pipeline_stages,
            attn_tensor_parallel_size=attn_tensor_parallel_size,
            attn_dp=attn_dp,
            moe_tensor_parallel_size=moe_tensor_parallel_size,
            moe_expert_parallel_size=moe_expert_parallel_size,
            total_expert_num=total_expert_num,
            local_expert_num=local_expert_num,
            router_load_balancing_type=router_load_balancing_type,
            router_topk=router_topk,
            moe_routing_seed=get_field_value("moe_routing_seed"),
            moe_routing_trace_path=get_field_value("moe_routing_trace_path"),
            decode_attn_initial_lane_trace_path=get_field_value(
                "decode_attn_initial_lane_trace_path"
            ),
            decode_attn_steady_state_snapshot_path=get_field_value(
                "decode_attn_steady_state_snapshot_path"
            ),
            decode_attn_steady_state_measurement_report_path=get_field_value(
                "decode_attn_steady_state_measurement_report_path"
            ),
            moe_routing_distribution_type=moe_routing_distribution_type,
            device=get_field_value("device"),
            network_device=get_field_value("network_device"),
            cluster_prefix=cluster_prefix,
            speculative_decoding_config=main_config.speculative_decoding_config,
        )

    def get_cluster_configs_for_disaggregation(
        self,
    ) -> Dict[ClusterType, "ClusterConfig"]:
        """Generate cluster configurations for disaggregated mode."""
        # Lazy import: ClusterConfig inherits this mixin, so importing it at
        # module level would close a cycle.
        from frontier.config.cluster_config import ClusterConfig

        if not self._has_disaggregation_params_set():
            return {ClusterType.MONOLITHIC: self}

        cluster_configs = {}

        # Prefill cluster
        if self.prefill_cluster_num_replicas:
            prefill_config = ClusterConfig(
                cluster_type=ClusterType.PREFILL,
                num_replicas=self.prefill_cluster_num_replicas,
                replica_config=self.prefill_replica_config
                or self._create_replica_config_copy(),
                cluster_scheduler_config=self.cluster_scheduler_config,
                replica_scheduler_config=self.replica_scheduler_config,
                execution_time_predictor_config=(
                    self._create_execution_time_predictor_config_for_cluster(
                        "prefill"
                    )
                ),
                cc_backend_config=self._create_cc_backend_config_for_cluster("prefill"),
                # Propagate cluster-specific replica scheduler config parameters
                prefill_replica_scheduler_config_type=self.prefill_replica_scheduler_config_type,
                prefill_replica_scheduler_config_batch_size_cap=self.prefill_replica_scheduler_config_batch_size_cap,
                prefill_replica_scheduler_config_max_tokens_in_batch=self.prefill_replica_scheduler_config_max_tokens_in_batch,
                prefill_replica_scheduler_config_enable_chunked_prefill=self.prefill_replica_scheduler_config_enable_chunked_prefill,
                prefill_replica_scheduler_config_long_prefill_token_threshold=self.prefill_replica_scheduler_config_long_prefill_token_threshold,
                prefill_replica_scheduler_config_num_blocks=self.prefill_replica_scheduler_config_num_blocks,
                prefill_replica_scheduler_config_block_size=self.prefill_replica_scheduler_config_block_size,
                prefill_replica_scheduler_config_watermark_blocks_fraction=self.prefill_replica_scheduler_config_watermark_blocks_fraction,
            )
            cluster_configs[ClusterType.PREFILL] = prefill_config

        # Decode Attention cluster
        if self.decode_attn_cluster_num_replicas:
            # Determine micro-batch SIZE for decode-attn
            # NOTE: Only decode_attn_micro_batch_size exists (no generic fallback)
            _da_mbs = self.decode_attn_micro_batch_size
            decode_attn_config = ClusterConfig(
                cluster_type=ClusterType.DECODE_ATTN,
                num_replicas=self.decode_attn_cluster_num_replicas,
                replica_config=self.decode_attn_replica_config
                or self._create_replica_config_copy(),
                cluster_scheduler_config=self.cluster_scheduler_config,
                replica_scheduler_config=self.replica_scheduler_config,
                execution_time_predictor_config=(
                    self._create_execution_time_predictor_config_for_cluster(
                        "decode_attn"
                    )
                ),
                cc_backend_config=self._create_cc_backend_config_for_cluster(
                    "decode_attn"
                ),
                af_pipeline_num_micro_batch=self.decode_attn_af_pipeline_num_micro_batch,
                decode_attn_micro_batch_size=_da_mbs,
                decode_attn_request_allocation_threshold=self.decode_attn_request_allocation_threshold,
                # Propagate cluster-specific replica scheduler config parameters
                decode_attn_replica_scheduler_config_type=self.decode_attn_replica_scheduler_config_type,
                decode_attn_replica_scheduler_config_batch_size_cap=self.decode_attn_replica_scheduler_config_batch_size_cap,
                decode_attn_replica_scheduler_config_max_tokens_in_batch=self.decode_attn_replica_scheduler_config_max_tokens_in_batch,
                decode_attn_replica_scheduler_config_num_blocks=self.decode_attn_replica_scheduler_config_num_blocks,
                decode_attn_replica_scheduler_config_block_size=self.decode_attn_replica_scheduler_config_block_size,
                decode_attn_replica_scheduler_config_watermark_blocks_fraction=self.decode_attn_replica_scheduler_config_watermark_blocks_fraction,
            )
            cluster_configs[ClusterType.DECODE_ATTN] = decode_attn_config

        # Decode FFN cluster
        if self.decode_ffn_cluster_num_replicas:
            decode_ffn_config = ClusterConfig(
                cluster_type=ClusterType.DECODE_FFN,
                num_replicas=self.decode_ffn_cluster_num_replicas,
                replica_config=self.decode_ffn_replica_config
                or self._create_replica_config_copy(),
                cluster_scheduler_config=self.cluster_scheduler_config,
                replica_scheduler_config=self.replica_scheduler_config,
                execution_time_predictor_config=(
                    self._create_execution_time_predictor_config_for_cluster(
                        "decode_ffn"
                    )
                ),
                cc_backend_config=self._create_cc_backend_config_for_cluster(
                    "decode_ffn"
                ),
                af_pipeline_num_micro_batch=self.decode_ffn_af_pipeline_num_micro_batch,
                # Propagate cluster-specific replica scheduler config parameters
                decode_ffn_replica_scheduler_config_type=self.decode_ffn_replica_scheduler_config_type,
                decode_ffn_replica_scheduler_config_batch_size_cap=self.decode_ffn_replica_scheduler_config_batch_size_cap,
                decode_ffn_replica_scheduler_config_max_tokens_in_batch=self.decode_ffn_replica_scheduler_config_max_tokens_in_batch,
                decode_ffn_replica_scheduler_config_num_blocks=self.decode_ffn_replica_scheduler_config_num_blocks,
                decode_ffn_replica_scheduler_config_block_size=self.decode_ffn_replica_scheduler_config_block_size,
                decode_ffn_replica_scheduler_config_watermark_blocks_fraction=self.decode_ffn_replica_scheduler_config_watermark_blocks_fraction,
                decode_attn_cluster_num_replicas=self.decode_attn_cluster_num_replicas,
            )
            # Propagate only the source Attention-Replica capacity.  AFD
            # grouping is Replica-level; it must not manufacture DP lanes.
            decode_ffn_config.decode_attn_replica_id_start_for_ffn = int(
                self.prefill_cluster_num_replicas
            )

            cluster_configs[ClusterType.DECODE_FFN] = decode_ffn_config

        # Unified Decode cluster (PD-disaggregation mode)
        if self.decode_cluster_num_replicas:
            decode_config = ClusterConfig(
                cluster_type=ClusterType.DECODE,
                num_replicas=self.decode_cluster_num_replicas,
                replica_config=self.decode_replica_config
                or self._create_replica_config_copy(),
                cluster_scheduler_config=self.cluster_scheduler_config,
                replica_scheduler_config=self.replica_scheduler_config,
                execution_time_predictor_config=(
                    self._create_execution_time_predictor_config_for_cluster("decode")
                ),
                cc_backend_config=self._create_cc_backend_config_for_cluster("decode"),
                # Propagate cluster-specific replica scheduler config parameters
                decode_replica_scheduler_config_type=self.decode_replica_scheduler_config_type,
                decode_replica_scheduler_config_batch_size_cap=self.decode_replica_scheduler_config_batch_size_cap,
                decode_replica_scheduler_config_max_tokens_in_batch=self.decode_replica_scheduler_config_max_tokens_in_batch,
                decode_replica_scheduler_config_num_blocks=self.decode_replica_scheduler_config_num_blocks,
                decode_replica_scheduler_config_block_size=self.decode_replica_scheduler_config_block_size,
                decode_replica_scheduler_config_watermark_blocks_fraction=self.decode_replica_scheduler_config_watermark_blocks_fraction,
            )
            cluster_configs[ClusterType.DECODE] = decode_config

        return cluster_configs

    def _create_execution_time_predictor_config_for_cluster(
        self, cluster_prefix: str
    ) -> BaseExecutionTimePredictorConfig:
        """Create cluster-specific execution-time predictor config overrides."""
        base_config = self.execution_time_predictor_config
        override_values = {}
        for calibration_field in (
            "attn_pre_proj_calibration_scale",
            "attn_post_proj_calibration_scale",
            "attn_decode_calibration_scale",
            "attn_kv_cache_save_calibration_scale",
            "mlp_up_proj_calibration_scale",
            "mlp_down_proj_calibration_scale",
            "decode_phase_mlp_down_proj_calibration_scale",
        ):
            override_field = (
                f"{cluster_prefix}_execution_time_predictor_config_"
                f"{calibration_field}"
            )
            override_value = getattr(self, override_field, None)
            if override_value is None:
                continue
            override_value = float(override_value)
            if override_value <= 0.0:
                raise ValueError(
                    f"ClusterConfig.{override_field} must be > 0, got={override_value!r}"
                )
            override_values[calibration_field] = override_value

        if not override_values:
            return base_config

        return replace(base_config, **override_values)

    def _create_cc_backend_config_for_cluster(
        self, cluster_prefix: str
    ) -> BaseCCBackendConfig:
        """
        Create CC backend configuration for a specific cluster.

        This method creates a cluster-specific CC backend configuration by:
        1. Checking for cluster-specific override values
        2. Falling back to the base cc_backend_config values if not overridden

        Args:
            cluster_prefix: Cluster prefix (e.g., "prefill", "decode", "decode_attn", "decode_ffn")

        Returns:
            CC backend configuration for the specified cluster
        """
        creators = self._cc_backend_creators()

        cluster_type_str = getattr(
            self, f"{cluster_prefix}_cc_backend_config_type", None
        )
        if cluster_type_str is not None:
            cluster_type_key = cluster_type_str.lower()
            for type_key, _, create in creators:
                if type_key == cluster_type_key:
                    return create(cluster_prefix)
            raise ValueError(f"Unknown CC backend type: {cluster_type_str}")

        base_config = self.cc_backend_config
        for _, config_class, create in creators:
            if isinstance(base_config, config_class):
                return create(cluster_prefix)
        raise ValueError(
            "Unsupported base CC backend config type for cluster-specific "
            f"construction: {type(base_config).__name__}"
        )

    def _cc_backend_creators(
        self,
    ) -> Tuple[Tuple[str, type, Callable[[str], "BaseCCBackendConfig"]], ...]:
        """Return the supported CC backends as (type key, config class, creator).

        One ordered table serves both selection paths: an explicit
        ``<cluster>_cc_backend_config_type`` string, and, when that is absent,
        the concrete class of the base ``cc_backend_config``.
        """
        # Lazy import to avoid circular imports
        (
            _,
            VidurCCBackendConfig,
            AnalyticalCCBackendConfig,
            CollectiveSimCCBackendConfig,
            AiconfiguratorCCBackendConfig,
            AstraSimAnalyticalCCBackendConfig,
        ) = _get_cc_backend_configs()

        return (
            (
                "analytical",
                AnalyticalCCBackendConfig,
                self._create_analytical_cc_backend_config,
            ),
            ("vidur", VidurCCBackendConfig, self._create_vidur_cc_backend_config),
            (
                "collective_sim",
                CollectiveSimCCBackendConfig,
                self._create_collective_sim_cc_backend_config,
            ),
            (
                "aiconfigurator",
                AiconfiguratorCCBackendConfig,
                self._create_aiconfigurator_cc_backend_config,
            ),
            (
                "astra_sim_analytical",
                AstraSimAnalyticalCCBackendConfig,
                self._create_astra_sim_analytical_cc_backend_config,
            ),
        )

    def _cc_backend_value_reader(
        self, cluster_prefix: str, base_config: "BaseCCBackendConfig", config_class: type
    ) -> Callable[[str, Any], Any]:
        """Return a reader resolving one backend field for a cluster.

        Resolution order is the cluster-specific flat field, then the base
        configuration when it already is of this backend type, then the default
        the caller supplies.
        """

        def get_value(field_name: str, default_value: Any) -> Any:
            cluster_value = getattr(
                self, f"{cluster_prefix}_cc_backend_config_{field_name}", None
            )
            if cluster_value is not None:
                return cluster_value
            if isinstance(base_config, config_class):
                return getattr(base_config, field_name, default_value)
            return default_value

        return get_value

    @staticmethod

    def _shared_cc_backend_fields(base_config: "BaseCCBackendConfig") -> Dict[str, Any]:
        """Return the fields every CC backend inherits from the base config."""
        return {
            "profiling_data_dir": getattr(
                base_config, "profiling_data_dir", "data/profiling/network"
            ),
            "cache_dir": getattr(base_config, "cache_dir", "cache"),
            "no_cache": getattr(base_config, "no_cache", False),
        }

    def _create_analytical_cc_backend_config(
        self, cluster_prefix: str
    ) -> AnalyticalCCBackendConfig:
        """Create analytical CC backend config with cluster-specific overrides."""
        # Lazy import to avoid circular imports
        _, _, AnalyticalCCBackendConfig, _, _, _ = _get_cc_backend_configs()

        base_config = self.cc_backend_config

        get_value = self._cc_backend_value_reader(
            cluster_prefix, base_config, AnalyticalCCBackendConfig
        )

        return AnalyticalCCBackendConfig(
            **self._shared_cc_backend_fields(base_config),
            network_bandwidth_gbps=get_value("network_bandwidth_gbps", 100.0),
            network_latency_us=get_value("network_latency_us", 1.0),
            intra_node_bandwidth_gbps=get_value("intra_node_bandwidth_gbps", 600.0),
        )

    def _create_vidur_cc_backend_config(
        self, cluster_prefix: str
    ) -> VidurCCBackendConfig:
        """Create Vidur CC backend config with cluster-specific overrides."""
        # Lazy import to avoid circular imports
        _, VidurCCBackendConfig, _, _, _, _ = _get_cc_backend_configs()

        base_config = self.cc_backend_config

        # For Vidur config, we mainly use the base config values
        # as Vidur-specific parameters are typically shared across clusters
        if isinstance(base_config, VidurCCBackendConfig):
            return VidurCCBackendConfig(
                profiling_data_dir=base_config.profiling_data_dir,
                cache_dir=base_config.cache_dir,
                no_cache=base_config.no_cache,
                all_reduce_input_file=base_config.all_reduce_input_file,
                send_recv_input_file=base_config.send_recv_input_file,
                k_fold_cv_splits=base_config.k_fold_cv_splits,
                num_training_job_threads=base_config.num_training_job_threads,
            )
        else:
            # Create default Vidur config
            return VidurCCBackendConfig()

    def _create_collective_sim_cc_backend_config(
        self, cluster_prefix: str
    ) -> "CollectiveSimCCBackendConfig":
        """Create collective-sim CC backend config with cluster-specific overrides."""
        (
            _,
            _,
            _,
            CollectiveSimCCBackendConfig,
            _,
            _,
        ) = _get_cc_backend_configs()

        from pathlib import Path

        base_config = self.cc_backend_config
        if not isinstance(base_config, CollectiveSimCCBackendConfig):
            base_config = CollectiveSimCCBackendConfig()

        get_value = self._cc_backend_value_reader(
            cluster_prefix, base_config, CollectiveSimCCBackendConfig
        )

        if base_config.runner_out_dir:
            cluster_out_dir = str(Path(base_config.runner_out_dir) / cluster_prefix)
            return replace(
                base_config,
                runner_out_dir=cluster_out_dir,
                nvlink_allreduce_launch_overhead_us=get_value(
                    "nvlink_allreduce_launch_overhead_us",
                    50.0,
                ),
            )

        return replace(
            base_config,
            nvlink_allreduce_launch_overhead_us=get_value(
                "nvlink_allreduce_launch_overhead_us",
                50.0,
            ),
        )

    def _create_aiconfigurator_cc_backend_config(
        self, cluster_prefix: str
    ) -> "AiconfiguratorCCBackendConfig":
        """Create aiconfigurator CC backend config with cluster-specific overrides."""
        (
            _,
            _,
            _,
            _,
            AiconfiguratorCCBackendConfig,
            _,
        ) = _get_cc_backend_configs()

        base_config = self.cc_backend_config

        get_value = self._cc_backend_value_reader(
            cluster_prefix, base_config, AiconfiguratorCCBackendConfig
        )

        return AiconfiguratorCCBackendConfig(
            **self._shared_cc_backend_fields(base_config),
            repo_root=get_value("repo_root", "sota-infer-engine/aiconfigurator"),
            system=get_value("system", ""),
            source_backend=get_value("source_backend", "vllm"),
            source_version=get_value("source_version", ""),
            database_mode=get_value("database_mode", "silicon"),
            tp_allreduce_impl=get_value("tp_allreduce_impl", "custom_allreduce"),
            custom_allreduce_variant=get_value("custom_allreduce_variant", None),
        )

    def _create_astra_sim_analytical_cc_backend_config(
        self, cluster_prefix: str
    ) -> "AstraSimAnalyticalCCBackendConfig":
        """Create astra-sim analytical CC backend config with cluster-specific overrides."""
        (
            _,
            _,
            _,
            _,
            _,
            AstraSimAnalyticalCCBackendConfig,
        ) = _get_cc_backend_configs()

        base_config = self.cc_backend_config

        get_value = self._cc_backend_value_reader(
            cluster_prefix, base_config, AstraSimAnalyticalCCBackendConfig
        )

        return AstraSimAnalyticalCCBackendConfig(
            **self._shared_cc_backend_fields(base_config),
            prediction_cache_size=get_value("prediction_cache_size", 4096),
            placement_order=get_value("placement_order", "TP,CP,DP,EP"),
            intra_server_topology=get_value(
                "intra_server_topology", "FullyConnected"
            ),
            inter_server_topology=get_value(
                "inter_server_topology", "FullyConnected"
            ),
            intra_server_bandwidth_gbps=get_value(
                "intra_server_bandwidth_gbps", 600.0
            ),
            intra_server_latency_us=get_value("intra_server_latency_us", 1.0),
            inter_server_bandwidth_gbps=get_value(
                "inter_server_bandwidth_gbps", 100.0
            ),
            inter_server_latency_us=get_value("inter_server_latency_us", 1.0),
            ring_bidirectional=(
                base_config.ring_bidirectional
                if isinstance(base_config, AstraSimAnalyticalCCBackendConfig)
                else True
            ),
            p2p_src_index=get_value("p2p_src_index", 0),
            p2p_dst_index=get_value("p2p_dst_index", 1),
        )

    def _create_replica_config_copy(self) -> ReplicaConfig:
        """Create a copy of the main replica config for disaggregated clusters."""
        # Note: This method now needs to be called before replica_config is cleared
        # We need to preserve the original config temporarily
        original_config = (
            self.replica_config if self.replica_config else ReplicaConfig()
        )

        return ReplicaConfig(
            model_name=original_config.model_name,
            memory_margin_fraction=original_config.memory_margin_fraction,
            num_pipeline_stages=original_config.num_pipeline_stages,
            attn_tensor_parallel_size=original_config.attn_tensor_parallel_size,
            attn_dp=original_config.attn_dp,
            moe_tensor_parallel_size=original_config.moe_tensor_parallel_size,
            moe_expert_parallel_size=original_config.moe_expert_parallel_size,
            total_expert_num=original_config.total_expert_num,
            router_load_balancing_type=original_config.router_load_balancing_type,
            router_topk=original_config.router_topk,
            moe_routing_seed=original_config.moe_routing_seed,
            moe_routing_distribution_type=original_config.moe_routing_distribution_type,
            moe_routing_trace_path=original_config.moe_routing_trace_path,
            decode_attn_initial_lane_trace_path=(
                original_config.decode_attn_initial_lane_trace_path
            ),
            decode_attn_steady_state_snapshot_path=(
                original_config.decode_attn_steady_state_snapshot_path
            ),
            decode_attn_steady_state_measurement_report_path=(
                original_config.decode_attn_steady_state_measurement_report_path
            ),
            device=original_config.device,
            network_device=original_config.network_device,
            speculative_decoding_config=original_config.speculative_decoding_config,
        )
