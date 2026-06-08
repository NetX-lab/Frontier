from typing import List, Optional

from frontier.entities.base_entity import BaseEntity
from frontier.entities.request import Request
from frontier.types import ClusterType
from frontier.logger import init_logger

logger = init_logger(__name__)


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Batch has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


class BatchStage(BaseEntity):
    def __init__(
        self,
        batch_id: int,
        replica_id: int,
        pipeline_stage: int,
        execution_time: float,
        model_execution_time: float,
        requests: List[Request],
        num_tokens: List[int],
        cluster_type: ClusterType,
        effective_total_tokens_compute: Optional[int] = None,
        effective_total_tokens_transfer: Optional[int] = None,
        effective_total_tokens_rounded: Optional[int] = None,
        tokens_are_post_routing: bool = False,
    ) -> None:
        self._id = BatchStage.generate_id()

        self._requests = requests
        self._request_runtime_epochs = [
            int(getattr(request, "runtime_epoch", 0)) for request in requests
        ]
        self._num_tokens = num_tokens
        total_tokens = sum(num_tokens)
        self._total_tokens = total_tokens
        self._batch_id = batch_id
        self._replica_id = replica_id
        self._pipeline_stage = pipeline_stage
        self._execution_time = execution_time
        self._model_execution_time = model_execution_time
        self._cluster_type = cluster_type
        self._effective_total_tokens_compute = (
            effective_total_tokens_compute
            if effective_total_tokens_compute is not None
            else total_tokens
        )
        self._effective_total_tokens_transfer = (
            effective_total_tokens_transfer
            if effective_total_tokens_transfer is not None
            else total_tokens
        )
        self._effective_total_tokens_rounded = (
            effective_total_tokens_rounded
            if effective_total_tokens_rounded is not None
            else (self._effective_total_tokens_compute + 7) // 8 * 8
        )
        self._tokens_are_post_routing = bool(tokens_are_post_routing)

        self._scheduled_at = None
        self._completed_at = None
        self._scheduled = False

    @property
    def num_tokens(self) -> List[int]:
        return self._num_tokens

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_scheduled
    def completed_at(self) -> float:
        return self._completed_at

    @property
    def execution_time(self) -> float:
        return self._execution_time

    @property
    def model_execution_time(self) -> float:
        return self._model_execution_time

    @property
    def pipeline_stage(self) -> int:
        return self._pipeline_stage

    @property
    def request_ids(self) -> List[int]:
        return [request.id for request in self._requests]

    @property
    def requests(self) -> List[Request]:
        return self._requests

    @property
    def size(self) -> int:
        return len(self._requests)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def effective_total_tokens_compute(self) -> int:
        return self._effective_total_tokens_compute

    @property
    def effective_total_tokens_transfer(self) -> int:
        return self._effective_total_tokens_transfer

    @property
    def effective_total_tokens_rounded(self) -> int:
        return self._effective_total_tokens_rounded

    @property
    def tokens_are_post_routing(self) -> bool:
        return self._tokens_are_post_routing

    def on_schedule(
        self,
        time: float,
    ) -> None:
        self._scheduled_at = time
        self._scheduled = True

        for request, runtime_epoch in zip(
            self._requests, self._request_runtime_epochs
        ):
            request_runtime_epoch = int(getattr(request, "runtime_epoch", 0))
            if request_runtime_epoch != int(runtime_epoch):
                logger.debug(
                    "[BATCH-STAGE-SCHEDULE][STALE] batch_id=%s req=%s batch_epoch=%s request_epoch=%s",
                    self._batch_id,
                    request.id,
                    runtime_epoch,
                    request_runtime_epoch,
                )
                continue
            request.on_batch_stage_schedule(time, self._cluster_type)

    def on_stage_end(
        self,
        time: float,
    ) -> None:
        assert (
            abs(time - (self._scheduled_at + self._execution_time)) < 1e-6
        ), f"{time} != {self._scheduled_at} + {self._execution_time}"

        self._completed_at = time

        for request, runtime_epoch in zip(
            self._requests, self._request_runtime_epochs
        ):
            request_runtime_epoch = int(getattr(request, "runtime_epoch", 0))
            if request_runtime_epoch != int(runtime_epoch):
                logger.debug(
                    "[BATCH-STAGE-END][STALE] batch_id=%s req=%s batch_epoch=%s request_epoch=%s",
                    self._batch_id,
                    request.id,
                    runtime_epoch,
                    request_runtime_epoch,
                )
                continue
            request.on_batch_stage_end(
                time, self._execution_time, self._model_execution_time, self._cluster_type
            )

    def override_execution_time(self, execution_time: float) -> None:
        self._execution_time = execution_time

    def override_model_execution_time(self, model_execution_time: float) -> None:
        self._model_execution_time = model_execution_time

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "size": self.size,
            "execution_time": self._execution_time,
            "model_execution_time": self._model_execution_time,
            "scheduled_at": self._scheduled_at,
            "completed_at": self._completed_at,
            "replica_id": self._replica_id,
            "batch_id": self._batch_id,
            "pipeline_stage": self._pipeline_stage,
            "scheduled": self._scheduled,
            "request_ids": self.request_ids,
            "num_tokens": self._num_tokens,
            "total_tokens": self._total_tokens,
            "effective_total_tokens_compute": self._effective_total_tokens_compute,
            "effective_total_tokens_transfer": self._effective_total_tokens_transfer,
            "effective_total_tokens_rounded": self._effective_total_tokens_rounded,
            "tokens_are_post_routing": self._tokens_are_post_routing,
        }

    def to_chrome_trace(self, time: int) -> dict:
        return {
            "name": f"{self.request_ids}",
            "ph": "X",
            "ts": (time - self._execution_time) * 1e6,
            "dur": self._execution_time * 1e6,
            "pid": self._replica_id,
            "tid": self._pipeline_stage,
            "args": {
                "batch_id": self._batch_id,
                "batch_size": self.size,
                "request_ids": self.request_ids,
                "num_tokens": self._num_tokens,
                # "requests": [request.to_dict() for request in self._requests],
            },
        }
