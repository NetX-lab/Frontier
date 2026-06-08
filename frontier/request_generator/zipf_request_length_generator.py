import math
from typing import Tuple

from frontier.config import ZipfRequestLengthGeneratorConfig
from frontier.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
)
from frontier.utils.zipf_generator import ZipfGenerator


class ZipfRequestLengthGenerator(BaseRequestLengthGenerator):

    def __init__(self, config: ZipfRequestLengthGeneratorConfig):
        super().__init__(config)

        self.zipf_generator = ZipfGenerator(
            config.min_tokens,
            config.max_tokens,
            config.theta,
            config.scramble,
            config.seed,
        )

    def get_next_num_tokens(self) -> Tuple[int, int]:
        """
        Generate token counts with guaranteed minimum of 1 for both prefill and decode.

        The calculation ensures:
        - decode_tokens >= 1 (via math.ceil)
        - prefill_tokens >= 1 (via max(1, ...))

        Returns:
            Tuple of (prefill_tokens, decode_tokens) as integers, both >= 1
        """
        total_tokens = self.zipf_generator.next()

        # Ensure decode_tokens >= 1 using ceil
        decode_tokens = math.ceil(
            total_tokens / (1 + self.config.prefill_to_decode_ratio)
        )

        # Calculate prefill tokens and ensure >= 1 to prevent truncation to 0
        prefill_tokens = max(1, int(total_tokens - decode_tokens))

        # Validate constraints: both must be positive integers
        assert prefill_tokens >= 1 and decode_tokens >= 1, (
            f"Token count constraint violated: prefill={prefill_tokens}, decode={decode_tokens}"
        )

        return prefill_tokens, decode_tokens
