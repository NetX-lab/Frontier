from dataclasses import dataclass, field

from frontier.config.base_fixed_config import BaseFixedConfig
from frontier.logger import init_logger
from frontier.types import DeviceSKUType

logger = init_logger(__name__)


@dataclass
class BaseDeviceSKUConfig(BaseFixedConfig):
    fp16_tflops: int
    total_memory_gb: int


@dataclass
class A40DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 150
    total_memory_gb: int = 45

    @staticmethod
    def get_type():
        return DeviceSKUType.A40


@dataclass
class A100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 312
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.A100

@dataclass
class A800DeviceSKUConfig(BaseDeviceSKUConfig):
    # A800的性能与A100相近，但有一些差异
    fp16_tflops: int = 320  # 可以根据实际情况调整
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.A800
    
@dataclass
class H100DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1000
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.H100


@dataclass
class H800DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 989
    total_memory_gb: int = 80

    @staticmethod
    def get_type():
        return DeviceSKUType.H800


@dataclass
class H200DeviceSKUConfig(BaseDeviceSKUConfig):
    fp16_tflops: int = 1979
    total_memory_gb: int = 141

    @staticmethod
    def get_type():
        return DeviceSKUType.H200


@dataclass
class H20DeviceSKUConfig(BaseDeviceSKUConfig):
    # H20 values are grounded in local H20 worker inventory and the bundled
    # MegaScale-Infer hardware table used by this repository's evaluation notes.
    fp16_tflops: int = 148
    total_memory_gb: int = 96

    @staticmethod
    def get_type():
        return DeviceSKUType.H20


@dataclass
class RtxPro6000DeviceSKUConfig(BaseDeviceSKUConfig):
    # NVIDIA RTX PRO 6000 Blackwell Server Edition: 96 GB GDDR7 and
    # 1 PFLOP FP16/BF16 Tensor Core throughput per NVIDIA's official specs.
    fp16_tflops: int = 1000
    total_memory_gb: int = 96

    @staticmethod
    def get_type():
        return DeviceSKUType.RTX_PRO_6000
