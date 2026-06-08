from enum import Enum


class MeasurementType(str, Enum):
    CUDA_EVENT = "CUDA_EVENT"
    KERNEL_ONLY = "KERNEL_ONLY"

    @classmethod
    def from_string(cls, value: str) -> "MeasurementType":
        normalized = str(value).strip().upper()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported measurement_type={value!r}. "
                f"Expected one of {[member.value for member in cls]}."
            ) from exc
