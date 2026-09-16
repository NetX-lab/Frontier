from random import randint


_COLLECTIVE_DTYPE_NAMES = {
    "FP16": "float16", "BF16": "bfloat16", "FP32": "float32",
}
SUPPORTED_COLLECTIVE_PRECISIONS = tuple(_COLLECTIVE_DTYPE_NAMES)


def precision_to_dtype(precision: str):
    """Resolve the collective precision contract lazily for either runner."""
    import torch

    try:
        name = _COLLECTIVE_DTYPE_NAMES[precision.upper()]
    except KeyError as exc:
        raise ValueError(
            f"Collective profiling does not support precision {precision!r}; "
            f"choose one of {sorted(SUPPORTED_COLLECTIVE_PRECISIONS)}."
        ) from exc
    return getattr(torch, name)


class CollectivesInput:
    used_comm_ids = set()

    def __init__(
        self,
        num_workers: int,
        num_workers_per_node: int,
        collective_size: int,
        collective: str,
        precision: str = "FP16",
    ):
        self.num_workers = num_workers
        self.num_workers_per_node = num_workers_per_node
        self.collective_size = collective_size
        self.collective = collective
        normalized_precision = str(precision).upper()
        if normalized_precision not in SUPPORTED_COLLECTIVE_PRECISIONS:
            raise ValueError(
                "Collective profiling precision must be one of "
                f"{SUPPORTED_COLLECTIVE_PRECISIONS}; got {precision!r}."
            )
        self.precision = normalized_precision
        self.comm_id = self._get_comm_id()

    @classmethod
    def _get_comm_id(cls):
        comm_id = randint(0, 65535)
        while comm_id in cls.used_comm_ids:
            comm_id = randint(0, 65535)
        cls.used_comm_ids.add(comm_id)
        return comm_id

    def is_valid(self, total_gpus_available: int, num_nodes: int):
        if self.num_workers == 1:
            return False

        if self.collective == "send_recv" and self.num_workers != 2:
            return False

        if self.num_workers < self.num_workers_per_node:
            return False

        if self.num_workers % self.num_workers_per_node != 0:
            return False

        num_nodes_required = self.num_workers // self.num_workers_per_node

        if self.num_workers > total_gpus_available or num_nodes_required > num_nodes:
            return False

        return True
