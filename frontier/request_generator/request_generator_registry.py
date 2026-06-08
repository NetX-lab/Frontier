from frontier.request_generator.synthetic_request_generator import (
    SyntheticRequestGenerator,
)
from frontier.request_generator.trace_replay_request_generator import (
    TraceReplayRequestGenerator,
)
from frontier.types import RequestGeneratorType
from frontier.utils.base_registry import BaseRegistry


class RequestGeneratorRegistry(BaseRegistry):
    pass


RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC, SyntheticRequestGenerator
)
RequestGeneratorRegistry.register(
    RequestGeneratorType.TRACE_REPLAY, TraceReplayRequestGenerator
)
