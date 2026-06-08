from typing import List

from frontier.config import SyntheticRequestGeneratorConfig
from frontier.entities import Request, RequestRoundPlan
from frontier.request_generator.base_request_generator import BaseRequestGenerator
from frontier.request_generator.request_interval_generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from frontier.request_generator.request_length_generator_registry import (
    RequestLengthGeneratorRegistry,
)
from frontier.types import RequestIntervalGeneratorType
from frontier.utils.random import set_seeds


class SyntheticRequestGenerator(BaseRequestGenerator):

    def __init__(self, config: SyntheticRequestGeneratorConfig):
        super().__init__(config)

        self.request_length_generator = RequestLengthGeneratorRegistry.get(
            self.config.length_generator_config.get_type(),
            self.config.length_generator_config,
        )
        self.request_interval_generator = RequestIntervalGeneratorRegistry.get(
            self.config.interval_generator_config.get_type(),
            self.config.interval_generator_config,
        )

    def _generate_next_request(self, last_arrived_at: float) -> Request:
        inter_request_time = (
            self.request_interval_generator.get_next_inter_request_time()
        )
        if inter_request_time is None:
            return None
        arrived_at = last_arrived_at + inter_request_time

        (
            prefill_tokens,
            decode_tokens,
            block_hash_ids,
            session_id,
        ) = self._normalize_request_length_output(
            self.request_length_generator.get_next_num_tokens()
        )

        if prefill_tokens is None or decode_tokens is None:
            return None

        return self._build_request(
            arrived_at=arrived_at,
            num_prefill_tokens=int(prefill_tokens),
            num_decode_tokens=int(decode_tokens),
            priority=self.config.default_priority,
            block_hash_ids=block_hash_ids,
            session_id=session_id,
        )

    @staticmethod
    def _normalize_request_length_output(output):
        if hasattr(output, "num_prefill_tokens") and hasattr(output, "num_decode_tokens"):
            return (
                output.num_prefill_tokens,
                output.num_decode_tokens,
                getattr(output, "block_hash_ids", None),
                getattr(output, "session_id", None),
            )

        if isinstance(output, tuple):
            if len(output) == 2:
                return output[0], output[1], None, None
            if len(output) == 4:
                return output[0], output[1], output[2], output[3]

        raise ValueError(
            "Request length generator output must be a 2-tuple, 4-tuple, or object "
            "with num_prefill_tokens/num_decode_tokens attributes."
        )

    def _generate_implicit_hidden_round_plans(
        self,
        final_prefill_tokens: int,
        final_decode_tokens: int,
    ) -> List[RequestRoundPlan]:
        hidden_round_plans = []
        for _ in range(max(self._thinking_depth - 1, 0)):
            output = self.request_length_generator.get_next_num_tokens()
            hidden_prefill_tokens, hidden_decode_tokens, _, _ = (
                self._normalize_request_length_output(output)
            )
            if hidden_prefill_tokens is None or hidden_decode_tokens is None:
                raise ValueError(
                    "SyntheticRequestGenerator could not generate hidden thinking "
                    "round lengths."
                )
            hidden_round_plans.append(
                RequestRoundPlan(
                    int(hidden_prefill_tokens),
                    int(hidden_decode_tokens),
                )
            )
        return hidden_round_plans

    def _generate_requests(self) -> List[Request]:
        requests = []

        current_time = 0

        # first priority is duration
        if self.config.duration is not None:
            while current_time < self.config.duration:
                request = self._generate_next_request(current_time)
                current_time = request.arrived_at
                requests.append(request)
        elif self.config.num_requests is not None:
            for _ in range(self.config.num_requests):
                request = self._generate_next_request(current_time)
                current_time = request.arrived_at
                requests.append(request)
        else:
            assert (
                self.config.interval_generator_config.get_type()
                == RequestIntervalGeneratorType.TRACE
            )

            while True:
                request = self._generate_next_request(current_time)
                if request is None:
                    break
                current_time = request.arrived_at
                requests.append(request)

        return requests

    def generate_requests(self) -> List[Request]:
        assert (
            self.config.duration
            or self.config.num_requests
            or self.config.interval_generator_config.get_type()
            == RequestIntervalGeneratorType.TRACE
        )

        set_seeds(self.config.seed)

        requests = self._generate_requests()

        # sort requests by arrival time
        requests.sort(key=lambda x: x.arrived_at)
        # remove any requests that arrived after the time limit
        if self.config.duration is not None:
            requests = [
                request
                for request in requests
                if request.arrived_at < self.config.duration
            ]

        return requests
