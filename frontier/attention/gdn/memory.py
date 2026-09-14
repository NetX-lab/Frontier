"""Fixed per-request state accounting for GDN layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class GatedDeltaNetStateLayout:
    """Physical state shapes for one GDN layer and one request slot.

    GDN state is fixed with respect to sequence history.  The simulator uses
    this value for capacity accounting and never materializes tensor state.
    """

    conv_state_shape: tuple[int, int]
    recurrent_state_shape: tuple[int, int, int]
    conv_bytes_per_element: int = 2
    recurrent_bytes_per_element: int = 4

    def __post_init__(self) -> None:
        if len(self.conv_state_shape) != 2 or any(
            type(dimension) is not int or dimension <= 0
            for dimension in self.conv_state_shape
        ):
            raise ValueError(
                "conv_state_shape must contain two positive integer dimensions, "
                f"got {self.conv_state_shape!r}"
            )
        if len(self.recurrent_state_shape) != 3 or any(
            type(dimension) is not int or dimension <= 0
            for dimension in self.recurrent_state_shape
        ):
            raise ValueError(
                "recurrent_state_shape must contain three positive integer "
                f"dimensions, got {self.recurrent_state_shape!r}"
            )
        for field_name in ("conv_bytes_per_element", "recurrent_bytes_per_element"):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value!r}")

    @staticmethod
    def _num_elements(shape: Iterable[int]) -> int:
        result = 1
        for dimension in shape:
            result *= int(dimension)
        return result

    @property
    def conv_state_elements(self) -> int:
        return self._num_elements(self.conv_state_shape)

    @property
    def recurrent_state_elements(self) -> int:
        return self._num_elements(self.recurrent_state_shape)

    @property
    def total_elements(self) -> int:
        return self.conv_state_elements + self.recurrent_state_elements

    @property
    def total_bytes(self) -> int:
        return (
            self.conv_state_elements * self.conv_bytes_per_element
            + self.recurrent_state_elements * self.recurrent_bytes_per_element
        )
