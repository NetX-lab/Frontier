"""Compare the DES load model with an actual pinned vLLM coordinator loop."""

import argparse
import ast
import copy
from fractions import Fraction
import json
import logging
from pathlib import Path
from types import SimpleNamespace as NS

from frontier.scheduler.request_load import RequestLoad
from frontier.scheduler.utils.vllm_dp_load_balancer import VllmDPLoadBalancer


class ReplayComplete(Exception):
    pass


def reference_publications(source, reports, duration_ms):
    """Execute the source loop with deterministic message delivery and time."""
    tree = ast.parse(source.read_text())
    owner = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "DPCoordinatorProc")
    methods = [node for node in owner.body if isinstance(node, ast.FunctionDef)
               and node.name in ("process_input_socket", "_get_engine_counts")]
    clock = NS(ms=0, pending=None, cursor=0)
    publications = []

    class Socket:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def recv(self):
            return b"\x01" if self.path == "back" else clock.pending

        def send(self, message):
            if self.path == "front":
                publications.append((clock.ms, copy.deepcopy(message[0])))

    sockets = {name: Socket(name) for name in ("front", "output", "back")}

    class Poller:
        def register(self, *args):
            pass

        def poll(self, timeout):
            deadline = clock.ms + timeout
            if clock.cursor < len(reports) and reports[clock.cursor][0] <= deadline:
                clock.ms, engine, step, waiting, running = reports[clock.cursor]
                clock.cursor += 1
                clock.pending = NS(
                    outputs=[], utility_output=None, engine_index=engine,
                    scheduler_stats=NS(step_counter=step, current_wave=0,
                                       num_waiting_reqs=waiting, num_running_reqs=running),
                    wave_complete=None, start_wave=None,
                )
                return [(sockets["output"], 1)]
            if deadline > duration_ms:
                raise ReplayComplete
            clock.ms = deadline
            return []

    namespace = {
        "MsgpackDecoder": lambda _: NS(decode=lambda message: message),
        "EngineCoreOutputs": object,
        "make_zmq_socket": lambda *, path, **kwargs: sockets[path],
        "zmq": NS(XPUB=1, PULL=2, POLLIN=1, Poller=Poller),
        "time": NS(time=lambda: Fraction(1_000_000_000 + clock.ms, 1000)),
        "msgspec": NS(msgpack=NS(encode=copy.deepcopy)),
        "copy": copy,
        "logger": logging.getLogger("vllm_coordinator_reference"),
    }
    module = ast.Module(body=methods, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    coordinator_type = type("SourceCoordinator", (), {
        name: namespace[name] for name in ("process_input_socket", "_get_engine_counts")
    })
    coordinator = coordinator_type()
    coordinator.ctx = None
    coordinator.engines = [NS(request_counts=[0, 0]) for _ in range(2)]
    coordinator.stats_update_interval_ms = 100
    try:
        coordinator.process_input_socket("front", "output", "back")
    except ReplayComplete:
        pass
    return publications


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Same-step collection, previous-step snapshots, timeout ties, then idle.
    reports = [
        (10, 0, 0, 0, 10), (20, 1, 0, 0, 3),
        (100, 0, 1, 1, 8), (120, 1, 1, 0, 4),
        (140, 0, 2, 0, 9), (170, 1, 2, 0, 5),
        (230, 0, 3, 0, 8), (280, 1, 3, 0, 4),
        (310, 0, 4, 0, 0), (330, 1, 4, 0, 0),
    ]
    duration_ms = 6000
    publications = reference_publications(args.source, reports, duration_ms)
    router = VllmDPLoadBalancer(2)
    report_index = publication_index = 0
    expected = [[0, 0], [0, 0]]
    for now_ms in range(duration_ms + 1):
        while report_index < len(reports) and reports[report_index][0] == now_ms:
            _, engine, step, waiting, running = reports[report_index]
            router.report(now_ms / 1000, engine, step, RequestLoad(waiting, running))
            report_index += 1
        router._advance(now_ms / 1000)
        while publication_index < len(publications) and publications[publication_index][0] <= now_ms:
            expected = publications[publication_index][1]
            publication_index += 1
        actual = [list(load) for load in router.frontend_counts]
        assert actual == expected, (now_ms, actual, expected, publications)
    result = {"status": "PASS", "source": str(args.source), "reports": reports,
              "snapshots": publications, "compared_millisecond_states": duration_ms + 1,
              "scope": "Actual coordinator count-publication loop; deterministic zero-latency transport."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
