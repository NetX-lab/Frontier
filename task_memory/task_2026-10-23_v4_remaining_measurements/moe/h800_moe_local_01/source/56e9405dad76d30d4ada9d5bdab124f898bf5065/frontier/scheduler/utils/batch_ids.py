"""Stable identifiers for Replica-local scheduler batches."""


def attention_batch_id(replica_id: int, lane_id: int | None, counter: int, lane_count: int) -> int:
    """Encode a batch counter and attention-DP lane into one integer."""
    if type(replica_id) is not int or replica_id < 0:
        raise ValueError("replica_id must be an exact non-negative int")
    if type(counter) is not int or counter < 0:
        raise ValueError(f"lane_batch_counter must be an exact non-negative int, got {counter!r}")
    if type(lane_count) is not int or lane_count <= 0:
        raise ValueError(f"attention-DP lane count must be positive, got {lane_count}")
    if lane_id is None:
        lane_id = 0
    elif type(lane_id) is not int or not 0 <= lane_id < lane_count:
        raise ValueError(
            "replica_local_id must be None or an exact lane ID in the "
            f"attention-DP domain [0, {lane_count}), got {lane_id!r}"
        )
    return counter * lane_count + lane_id


def decode_sync_id(lane_id: int, counter: int, lane_count: int) -> int:
    """Encode a MONOLITHIC decode-sync counter with lane scope."""
    lane_id = int(lane_id or 0)
    if lane_id < 0 or lane_id >= lane_count:
        raise ValueError(
            "MONOLITHIC decode sync lane id must be within the attention-DP "
            f"domain, got replica_local_id={lane_id}, lane_count={lane_count}"
        )
    counter = int(counter or 0)
    if counter < 0:
        raise ValueError(
            "lane_decode_sync_counter must be non-negative, "
            f"got {counter!r}"
        )
    return counter * lane_count + lane_id
