"""Forward-step identity and lifecycle management for cluster schedulers."""

from collections.abc import Callable, Mapping

from frontier.entities import Batch


def source_batches_by_lane(cohort_batches, batch):
    """Normalize a direct wave call to a non-empty lane-to-batch mapping."""
    if cohort_batches is None:
        lane_id = getattr(batch, "_stage_owner_replica_local_id", None)
        return {int(0 if lane_id is None else lane_id): batch}
    if not isinstance(cohort_batches, dict) or not cohort_batches:
        raise ValueError("cohort_batches must be a non-empty lane mapping")
    normalized = {}
    for lane_id, source_batch in cohort_batches.items():
        if type(lane_id) is not int or lane_id < 0:
            raise ValueError(f"cohort lane ID must be non-negative int, got {lane_id!r}")
        if not isinstance(source_batch, Batch):
            raise TypeError(
                "cohort_batches values must be Batch instances, "
                f"got {type(source_batch).__name__}"
            )
        normalized[lane_id] = source_batch
    return normalized


class ForwardSyncState:
    """Own forward-step identity bookkeeping shared by PREFILL and DECODE."""

    def __init__(self) -> None:
        self._open_steps_by_kind: dict[str, dict[tuple, int]] = {
            "prefill": {},
            "decode": {},
        }
        self._next_step_id_by_replica: dict[int, int] = {}

    @staticmethod
    def get_step_id(batch) -> int:
        step_id = getattr(batch, "_forward_cohort_id", None)
        if step_id is None:
            step_id = getattr(batch, "global_id", None)
        if type(step_id) is not int or step_id < 0:
            raise ValueError(
                "forward cohort ID must be an exact non-negative int, "
                f"got {step_id!r}"
            )
        return step_id

    @staticmethod
    def _validate_kind(sync_kind: str) -> None:
        if sync_kind not in ("prefill", "decode"):
            raise ValueError(f"unknown synchronization kind: {sync_kind!r}")

    def open_steps(self, sync_kind: str) -> dict[tuple, int]:
        self._validate_kind(sync_kind)
        return self._open_steps_by_kind[sync_kind]

    def resolve_step(
        self,
        *,
        sync_kind: str,
        replica_id: int,
        stage_id: int,
        batch,
        lane_id: int,
        layer_id: int,
        sync_stage: str,
        room_lookup: Callable[[int], Mapping | None],
    ) -> int | None:
        self._validate_kind(sync_kind)
        for value, field_name in (
            (replica_id, "replica_id"),
            (stage_id, "stage_id"),
            (lane_id, "lane_id"),
            (layer_id, "layer_id"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"sync {field_name} must be an exact non-negative int"
                )

        provisional_id = getattr(batch, "_forward_cohort_provisional_id", None)
        current_id = self.get_step_id(batch)
        if provisional_id is None:
            provisional_id = current_id
            batch._forward_cohort_provisional_id = provisional_id
        if type(provisional_id) is not int or provisional_id < 0:
            raise ValueError(
                "forward cohort provisional ID must be an exact non-negative int, "
                f"got {provisional_id!r}"
            )

        binding_key = (
            replica_id,
            stage_id,
            layer_id,
            sync_stage,
            provisional_id,
        )
        open_steps = self.open_steps(sync_kind)
        open_step_id = open_steps.get(binding_key)
        if open_step_id is not None:
            room = room_lookup(open_step_id)
            if room is None:
                raise RuntimeError(
                    "Forward-step state references a missing waiting room: "
                    f"kind={sync_kind}, replica={replica_id}, stage={stage_id}, "
                    f"layer={layer_id}, sync_stage={sync_stage}, step={open_step_id}"
                )
            existing_batch = room.get("batches", {}).get(lane_id)
            if (
                existing_batch is None
                or (existing_batch is batch and batch.is_idle)
                or (existing_batch.is_idle and not batch.is_idle)
                or (batch.is_idle and not existing_batch.is_idle)
            ):
                if current_id != open_step_id:
                    batch._forward_cohort_id = open_step_id
                return open_step_id
            raise ValueError(
                "one attention-DP lane cannot occupy two open sync cohorts: "
                f"replica={replica_id}, stage={stage_id}, lane={lane_id}, "
                f"layer={layer_id}, sync_stage={sync_stage}"
                )

        next_id = int(self._next_step_id_by_replica.get(replica_id, 0))
        if getattr(batch, "is_idle", False) and current_id < next_id:
            return None
        resolved_id = max(current_id, next_id)
        open_steps[binding_key] = resolved_id
        batch._forward_cohort_id = resolved_id
        self._next_step_id_by_replica[replica_id] = resolved_id + 1
        return resolved_id

    def close_step(
        self,
        *,
        sync_kind: str,
        replica_id: int,
        stage_id: int,
        layer_id: int,
        sync_stage: str,
        provisional_id: int,
    ) -> None:
        self._validate_kind(sync_kind)
        if type(provisional_id) is not int or provisional_id < 0:
            raise ValueError(
                "forward cohort provisional ID must be an exact non-negative int, "
                f"got {provisional_id!r}"
            )
        open_steps = self.open_steps(sync_kind)
        open_steps.pop(
            (replica_id, stage_id, layer_id, sync_stage, provisional_id),
            None,
        )
        self._next_step_id_by_replica[replica_id] = max(
            int(self._next_step_id_by_replica.get(replica_id, 0)),
            provisional_id + 1,
        )
