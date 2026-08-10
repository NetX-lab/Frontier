from typing import List

from frontier.config import ClosedLoopRequestGeneratorConfig
from frontier.entities import Request
from frontier.request_generator.base_request_generator import BaseRequestGenerator
from frontier.request_generator.request_length_generator_registry import (
    RequestLengthGeneratorRegistry,
)
from frontier.request_generator.synthetic_request_generator import (
    SyntheticRequestGenerator,
)
from frontier.utils.random import set_seeds


class ClosedLoopRequestGenerator(BaseRequestGenerator):
    """Builds a fixed-size request population for closed-loop (max_concurrency-capped)
    admission. Every request is stamped with arrived_at=0.0 as a placeholder; real
    arrival timestamps are assigned at release time by the simulator/scheduler
    (see BaseGlobalScheduler.configure_closed_loop_backlog and the completion-triggered
    release logic in GlobalBatchEndEvent), not by this generator."""

    def __init__(self, config: ClosedLoopRequestGeneratorConfig):
        super().__init__(config)
        self.request_length_generator = RequestLengthGeneratorRegistry.get(
            self.config.closed_loop_length_generator_config.get_type(),
            self.config.closed_loop_length_generator_config,
        )

    def generate_requests(self) -> List[Request]:
        set_seeds(self.config.seed)
        requests: List[Request] = []
        for _ in range(self.config.num_requests):
            output = self.request_length_generator.get_next_num_tokens()
            prefill_tokens, decode_tokens, block_hash_ids, session_id = (
                SyntheticRequestGenerator._normalize_request_length_output(output)
            )
            if prefill_tokens is None or decode_tokens is None:
                break
            requests.append(
                self._build_request(
                    arrived_at=0.0,
                    num_prefill_tokens=int(prefill_tokens),
                    num_decode_tokens=int(decode_tokens),
                    priority=self.config.default_priority,
                    block_hash_ids=block_hash_ids,
                    session_id=session_id,
                )
            )
        return requests
