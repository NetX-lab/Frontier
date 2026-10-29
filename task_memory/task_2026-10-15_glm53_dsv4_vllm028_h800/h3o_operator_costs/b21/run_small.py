"""P2-32 operator replay, cold cache, with independent counts on both runs."""
import argparse
import json
import sys
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "a10"))
import a10_driver
from generate_trace import generate


def run(root, observe):
    if (root / "cache").exists():
        raise ValueError(f"cold run requires a new cache: {root}")
    trace = generate(root)
    from frontier.cc_backend.backends.measured_cc_backend import MeasuredCCBackend
    from frontier.events.batch_stage_end_event import BatchStageEndEvent
    calls, events = Counter(), Counter()
    predict, end = MeasuredCCBackend.predict_operator_collective, BatchStageEndEvent.handle_event

    def count_predict(backend, *args, **kwargs):
        result = predict(backend, *args, **kwargs)
        calls[backend.cluster_type.name.lower()] += 1
        return result

    def count_end(event, *args, **kwargs):
        events[event._cluster_type.name.lower()] += 1
        return end(event, *args, **kwargs)

    with ExitStack() as stack:
        stack.enter_context(patch.object(MeasuredCCBackend, "predict_operator_collective", count_predict))
        stack.enter_context(patch.object(BatchStageEndEvent, "handle_event", count_end))
        try:
            a10_driver.main("p2_32", root, observe=observe,
                            fields={"trace_request_generator_config_trace_file": trace["path"]})
        finally:
            (root / "independent_counts.json").write_text(json.dumps(
                {"collective_calls": dict(calls), "stage_end_events": dict(events), "trace": trace}, indent=1) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--without-observer", action="store_true")
    args = parser.parse_args()
    run(args.root.resolve(), not args.without_observer)
