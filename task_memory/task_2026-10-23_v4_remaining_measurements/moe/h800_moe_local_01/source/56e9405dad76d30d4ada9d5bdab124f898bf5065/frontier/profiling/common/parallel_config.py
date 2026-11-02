"""Simple parallel configuration for profiling."""

from dataclasses import dataclass, field


@dataclass
class ParallelConfig:
    """Simple parallel configuration for profiling.
    
    This is a lightweight version of Sarathi's ParallelConfig,
    containing only the fields needed for profiling operations.
    """
    
    pipeline_parallel_size: int = field(
        default=2, metadata={"help": "Number of pipeline parallel groups."}
    )
    tensor_parallel_size: int = field(
        default=1, metadata={"help": "Number of tensor parallel groups."}
    )

    def __post_init__(self):
        self.world_size = self.pipeline_parallel_size * self.tensor_parallel_size

