import math
import random
from typing import Tuple

from frontier.request_generator.base_request_length_generator import (
    BaseRequestLengthGenerator,
)


class UniformRequestLengthGenerator(BaseRequestLengthGenerator):

    def get_next_num_tokens(self) -> Tuple[int, int]:
        """
        Generate token counts with guaranteed minimum of 1 for both prefill and decode.

        The calculation ensures:
        - total_tokens >= 2 (at least 1 prefill + 1 decode)
        - decode_tokens >= 1 (via math.ceil)
        - prefill_tokens >= 1 (via max(1, ...))

        Returns:
            Tuple of (prefill_tokens, decode_tokens) as integers, both >= 1
        """
        total_tokens = random.uniform(
            self.config.min_tokens,
            self.config.max_tokens,
        )

        # Ensure decode_tokens >= 1 using ceil
        decode_tokens = math.ceil(
            total_tokens / (1 + self.config.prefill_to_decode_ratio)
        )

        # Calculate prefill tokens and ensure >= 1 to prevent truncation to 0
        # when total_tokens is small (e.g., 1.78 -> prefill=0.78 -> int=0)
        prefill_tokens = max(1, int(total_tokens - decode_tokens))

        # Validate constraints: both must be positive integers
        assert prefill_tokens >= 1 and decode_tokens >= 1, (
            f"Token count constraint violated: prefill={prefill_tokens}, decode={decode_tokens}"
        )

        return prefill_tokens, decode_tokens
