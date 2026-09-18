import gc
from typing import Optional

import ray
import torch

from frontier.logger import init_logger
from frontier.profiling.collectives.collectives_input import (
    CollectivesInput,
    precision_to_dtype as _precision_to_dtype,
)
from frontier.profiling.collectives.collectives_wrapper import CollectiveWrapper
from frontier.profiling.collectives.main import _configure_collective_environment
from frontier.profiling.common.accelerator import set_process_visible_device

logger = init_logger(__name__)


@ray.remote(num_gpus=1)
class BenchmarkRunner:
    def __init__(self, gpu_id: int, max_gpus_per_node: int, head_ip: str) -> None:
        self._gpu_id = gpu_id
        self._max_devices_per_node = max_gpus_per_node
        self._accelerator_visibility = self._set_accelerator_visible_device()
        self._last_num_workers_per_node = None
        self._last_num_workers = None
        self._head_ip = head_ip

    def _set_accelerator_visible_device(self) -> str:
        visibility = set_process_visible_device(
            self._gpu_id % self._max_devices_per_node,
            torch_module=torch,
        )
        _configure_collective_environment()
        return visibility

    def run_collective(
        self,
        collectives_input: CollectivesInput,
    ) -> Optional[dict]:
        if (
            collectives_input.num_workers != self._last_num_workers
            or collectives_input.num_workers_per_node != self._last_num_workers_per_node
        ) and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        rank = self._get_rank(
            collectives_input.num_workers, collectives_input.num_workers_per_node
        )
        if rank is None:
            return None

        if (
            collectives_input.num_workers != self._last_num_workers
            or collectives_input.num_workers_per_node != self._last_num_workers_per_node
        ):
            self._init_communication(
                collectives_input.comm_id,
                rank,
                collectives_input.num_workers,
                collectives_input.num_workers_per_node,
            )
            self._last_num_workers = collectives_input.num_workers
            self._last_num_workers_per_node = collectives_input.num_workers_per_node

        wrapper = CollectiveWrapper(
            rank,
            collectives_input.num_workers,
            collectives_input.comm_id,
            collectives_input.collective_size,
            collectives_input.collective,
            collectives_input.num_workers_per_node,
            self._max_devices_per_node,
            dtype=_precision_to_dtype(collectives_input.precision),
        )
        stats = wrapper.profile()
        del wrapper
        gc.collect()
        return stats

    def _init_communication(
        self, comm_id: int, rank: int, num_workers: int, devices_per_node: int
    ):
        logger.info(
            f"Initializing gpu id: {self._gpu_id}, Rank: {rank}, num_workers: {num_workers}, comm_id: {comm_id}, "
            f"devices_per_node: {devices_per_node}, max_devices_per_node: {self._max_devices_per_node}, "
            f"ip_addr: {ray.util.get_node_ip_address()}, visibility: {self._accelerator_visibility}"
        )

        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=num_workers,
            init_method=f"tcp://{self._head_ip}:{comm_id}",
        )

    def _get_rank(self, num_workers: int, devices_per_node: int):
        assert self._max_devices_per_node >= devices_per_node
        assert self._max_devices_per_node % devices_per_node == 0
        assert num_workers % devices_per_node == 0 or num_workers < devices_per_node

        num_nodes = num_workers // devices_per_node
        current_node = self._gpu_id // self._max_devices_per_node

        if current_node >= num_nodes:
            return None

        local_gpu_id = self._gpu_id % self._max_devices_per_node

        # # scatter devices uniformly across the node
        # node_devices = list(range(self._max_devices_per_node))
        # device_offset = self._max_devices_per_node // devices_per_node

        # # selected devices for this worker
        # selected_devices = node_devices[::device_offset]

        # pack devices in order
        selected_devices = list(range(devices_per_node))

        if local_gpu_id not in selected_devices:
            return None

        # rank of this worker
        rank = current_node * devices_per_node + selected_devices.index(local_gpu_id)

        return rank
