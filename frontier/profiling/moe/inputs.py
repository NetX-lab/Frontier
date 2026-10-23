"""Explicit source-local positions and physical gathered MoE profiling cells."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from frontier.moe_gating_runtime import validate_moe_gating_runtime_context
from frontier.moe_routing_policy import TokenInterval

SCHEMA_VERSION = "moe.position_cyclic.inputs.v1"


@dataclass(frozen=True)
class MoESourceFrame:
    source_lane_id: int
    query_lens: tuple[int, ...]
    context_lens: tuple[int, ...]
    physical_rows: int
    gating_runtime_context: str

    def __post_init__(self):
        if type(self.source_lane_id) is not int or self.source_lane_id < 0:
            raise ValueError("source_lane_id must be a non-negative int")
        queries, contexts = tuple(self.query_lens), tuple(self.context_lens)
        if len(queries) != len(contexts):
            raise ValueError("source query/context arrays must be aligned")
        for context, query in zip(contexts, queries):
            TokenInterval(context, query)
        if type(self.physical_rows) is not int or self.physical_rows <= 0 or sum(queries) > self.physical_rows:
            raise ValueError("source physical_rows must be positive and contain all live queries")
        validate_moe_gating_runtime_context(self.gating_runtime_context)
        object.__setattr__(self, "query_lens", queries)
        object.__setattr__(self, "context_lens", contexts)

    @property
    def live_rows(self):
        return sum(self.query_lens)


@dataclass(frozen=True)
class MoERoutingProfileInput:
    moe_input_id: str
    source_frames: tuple[MoESourceFrame, ...]
    gathered_physical_rows: int

    def __post_init__(self):
        if not isinstance(self.moe_input_id, str) or not self.moe_input_id.strip():
            raise ValueError("moe_input_id must retain a nonblank census/forward identity")
        frames = tuple(self.source_frames)
        if not frames or any(not isinstance(frame, MoESourceFrame) for frame in frames):
            raise ValueError("source_frames must contain explicit MoESourceFrame values")
        if tuple(frame.source_lane_id for frame in frames) != tuple(range(len(frames))):
            raise ValueError("source frames must be in contiguous DP order")
        if type(self.gathered_physical_rows) is not int or self.gathered_physical_rows != sum(frame.physical_rows for frame in frames):
            raise ValueError("gathered_physical_rows must equal the supplied source widths")
        object.__setattr__(self, "source_frames", frames)


@dataclass(frozen=True)
class MoERoutingProfileCell:
    routing_input: MoERoutingProfileInput
    moe_frame_scope: str
    source_lane_id: int
    ep_id: int | None = None

    def __post_init__(self):
        if not isinstance(self.routing_input, MoERoutingProfileInput):
            raise ValueError("cell requires an explicit routing input")
        if type(self.source_lane_id) is not int or not 0 <= self.source_lane_id < len(self.routing_input.source_frames):
            raise ValueError("cell source lane is outside its input")
        if self.moe_frame_scope == "source_local":
            if self.ep_id is not None:
                raise ValueError("source_local cells cannot declare an EP owner")
        elif self.moe_frame_scope == "dp_gathered":
            if type(self.ep_id) is not int or self.ep_id < 0:
                raise ValueError("dp_gathered cells require an explicit EP owner")
        else:
            raise ValueError("cell scope must be source_local or dp_gathered")

    def to_json(self):
        return asdict(self)


def source_frame_tensors(frame: MoESourceFrame, *, device):
    """Return (signed int64 positions, bool is_padding), once before native timing.

    Padding positions are inert storage, never logical position-zero requests.
    Both the profiler adapter and native dispatch probe consume this interface.
    """
    import torch
    if not isinstance(frame, MoESourceFrame):
        raise ValueError("source_frame_tensors requires a validated MoESourceFrame")
    positions = torch.zeros(frame.physical_rows, dtype=torch.int64, device=device)
    is_padding = torch.ones(frame.physical_rows, dtype=torch.bool, device=device)
    offset = 0
    for context, query in zip(frame.context_lens, frame.query_lens):
        positions[offset:offset + query] = torch.arange(context, context + query, dtype=torch.int64, device=device)
        is_padding[offset:offset + query] = False
        offset += query
    return positions, is_padding


def routing_profile_cell(value):
    """Reconstruct the same immutable cell at executor serialization boundaries."""
    if isinstance(value, MoERoutingProfileCell):
        return value
    record = dict(value)
    source = dict(record.pop("routing_input"))
    source["source_frames"] = tuple(MoESourceFrame(**frame) for frame in source["source_frames"])
    return MoERoutingProfileCell(MoERoutingProfileInput(**source), **record)


def read_routing_profile_inputs(path):
    payload = json.loads(Path(path).read_text())
    if payload["schema_version"] != SCHEMA_VERSION or payload["routing_runtime_path"] != "position_cyclic":
        raise ValueError("routing input schema/policy mismatch")
    inputs = tuple(MoERoutingProfileInput(
        item["moe_input_id"], tuple(MoESourceFrame(**frame) for frame in item["source_frames"]),
        item["gathered_physical_rows"],
    ) for item in payload["inputs"])
    if not inputs or len({item.moe_input_id for item in inputs}) != len(inputs):
        raise ValueError("routing inputs must retain distinct census/forward identities")
    return inputs
