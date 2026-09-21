"""MoE timing for speculative-decoding MTP replay rows.

An MTP iteration replays decoder layers for the draft tokens it proposed, so
its MoE time is an aggregate over lanes rather than a single layer lookup.
"""

import math

from frontier.entities import Batch, ExecutionTime
from frontier.types import ClusterType


class MoeMtpReplay:
    """MoE time for MTP replay rows and terminal overshoot."""

    def _predict_mtp_moe_lane_phase_aggregate(
        self,
        *,
        predictor,
        batch: Batch,
        pipeline_stage: int,
        cluster_type: ClusterType,
        layer_id: int,
        num_layers: int,
    ) -> tuple[ExecutionTime, tuple[float, float, float, float, float]]:
        """Return one shared attention result and the five lane barriers.

        ``predictor`` is explicit because structural MTP may run against a
        secondary predictor owned by this parent.  The attention probe is kept
        at one layer: pipeline and CPU overhead are batch-level terms, while
        the returned physical phase barriers are the only values scaled by
        ``num_layers`` at the caller.
        """

        if type(num_layers) is not int or num_layers < 1:
            raise ValueError(f"num_layers must be a positive integer, got {num_layers!r}")

        attention_execution_time = predictor.predict_stage_execution_time(
            batch=batch,
            stage_id=pipeline_stage,
            cluster_type=cluster_type,
            num_layers=1,
            layer_id=layer_id,
            include_ffn=False,
        )
        attention_time_ms = float(attention_execution_time.model_time_ms)
        if not math.isfinite(attention_time_ms) or attention_time_ms < 0:
            raise ValueError(
                "MTP structural attention time must be finite and non-negative, "
                f"got {attention_time_ms}"
            )

        workload = predictor._materialize_layer_ep_workload(
            batch=batch,
            cluster_type=cluster_type,
            layer_id=layer_id,
        )
        participant_ep_ids = tuple(workload.participant_ep_ids)
        if not participant_ep_ids:
            raise ValueError("MTP MoE replay produced no EP participants")

        effective_tokens = int(
            batch.get_effective_total_tokens_for_compute(cluster_type)
        )
        if effective_tokens <= 0:
            raise ValueError(
                "MTP MoE replay requires positive pre-routing effective tokens, "
                f"got {effective_tokens}"
            )

        phase_values: list[list[float]] = []
        for ep_id in participant_ep_ids:
            lane_workload = workload.lane(int(ep_id))
            lane_phases = predictor.predict_moe_lane_phase_times(
                batch=batch,
                lane_workload=lane_workload,
                pipeline_stage=pipeline_stage,
                cluster_type=cluster_type,
            )
            if len(lane_phases) != 5:
                raise ValueError(
                    "MTP MoE lane phase API must return five values, "
                    f"got ep_id={ep_id}, values={lane_phases!r}"
                )
            normalized_phases = [float(value) for value in lane_phases]
            if any(
                not math.isfinite(value) or value < 0
                for value in normalized_phases
            ):
                raise ValueError(
                    "MTP MoE lane phase times must be finite and non-negative, "
                    f"got ep_id={ep_id}, values={normalized_phases}"
                )
            phase_values.append(normalized_phases)

        phase_maxima = tuple(
            max(values[index] for values in phase_values) for index in range(5)
        )
        return attention_execution_time, phase_maxima

    def _predict_mtp_terminal_row_time_ms(
        self,
        *,
        batch: Batch,
        stage_id: int,
        cluster_type: ClusterType,
        num_layers: int,
        layer_id: int,
    ) -> float:
        """Predict a terminal MTP row with physical EP barriers when required."""

        model_config = getattr(self, "_model_config", None)
        if model_config is None or not bool(getattr(model_config, "is_moe", False)):
            return super()._predict_mtp_terminal_row_time_ms(
                batch=batch,
                stage_id=stage_id,
                cluster_type=cluster_type,
                num_layers=num_layers,
                layer_id=layer_id,
            )
        is_moe_layer = getattr(model_config, "is_moe_layer", None)
        if not callable(is_moe_layer):
            raise ValueError(
                "MTP terminal MoE prediction requires model_config.is_moe_layer"
            )
        if not bool(is_moe_layer(layer_id)):
            return super()._predict_mtp_terminal_row_time_ms(
                batch=batch,
                stage_id=stage_id,
                cluster_type=cluster_type,
                num_layers=num_layers,
                layer_id=layer_id,
            )
        if cluster_type not in (ClusterType.MONOLITHIC, ClusterType.DECODE):
            return super()._predict_mtp_terminal_row_time_ms(
                batch=batch,
                stage_id=stage_id,
                cluster_type=cluster_type,
                num_layers=num_layers,
                layer_id=layer_id,
            )
        if int(getattr(self, "_moe_ep_size", 1)) <= 1:
            return super()._predict_mtp_terminal_row_time_ms(
                batch=batch,
                stage_id=stage_id,
                cluster_type=cluster_type,
                num_layers=num_layers,
                layer_id=layer_id,
            )

        attention_execution_time, phase_maxima = (
            self._predict_mtp_moe_lane_phase_aggregate(
                predictor=self,
                batch=batch,
                pipeline_stage=stage_id,
                cluster_type=cluster_type,
                layer_id=layer_id,
                num_layers=num_layers,
            )
        )
        attention_time_ms = float(attention_execution_time.total_time * 1e3)
        if not math.isfinite(attention_time_ms) or attention_time_ms < 0:
            raise ValueError(
                "MTP terminal attention time must be finite and non-negative, "
                f"got {attention_time_ms}"
            )
        return attention_time_ms + sum(phase_maxima) * int(num_layers)

    def _predict_mtp_decoder_layer_time_ms(
        self,
        *,
        predictor,
        batch: Batch,
    ) -> float:
        layer_id = 0
        model_config = getattr(predictor, "_model_config", None)
        if model_config is None:
            raise ValueError(
                "MTP structural decoder prediction requires model_config"
            )
        if not bool(getattr(model_config, "is_moe", False)):
            return super()._predict_mtp_decoder_layer_time_ms(
                predictor=predictor,
                batch=batch,
            )

        is_moe_layer = getattr(model_config, "is_moe_layer", None)
        if not callable(is_moe_layer):
            raise ValueError(
                "MTP structural MoE decoder prediction requires "
                "model_config.is_moe_layer"
            )
        if not bool(is_moe_layer(layer_id)):
            return super()._predict_mtp_decoder_layer_time_ms(
                predictor=predictor,
                batch=batch,
            )

        cluster_type = getattr(predictor, "_cluster_type", None)
        if not isinstance(cluster_type, ClusterType):
            raise ValueError(
                "MTP structural MoE decoder prediction requires a valid cluster_type"
            )

        attention_execution_time, phase_maxima = (
            self._predict_mtp_moe_lane_phase_aggregate(
                predictor=predictor,
                batch=batch,
                pipeline_stage=0,
                cluster_type=cluster_type,
                layer_id=layer_id,
                num_layers=1,
            )
        )
        attention_time_ms = float(attention_execution_time.model_time_ms)
        return attention_time_ms + sum(phase_maxima)
