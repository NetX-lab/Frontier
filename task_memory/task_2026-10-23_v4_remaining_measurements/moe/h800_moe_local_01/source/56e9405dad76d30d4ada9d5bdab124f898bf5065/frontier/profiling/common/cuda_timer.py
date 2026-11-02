"""Compatibility name for the existing CUDA event timer."""

from frontier.profiling.common.device_timer import DeviceTimer


class CudaTimer(DeviceTimer):
    """Preserve the existing CUDA timer import while sharing the lazy timer."""

    def __init__(self, name, layer_id=0, aggregation_fn=sum, filter_str=None):
        super().__init__(
            name,
            layer_id=layer_id,
            aggregation_fn=aggregation_fn,
            filter_str=filter_str,
        )
