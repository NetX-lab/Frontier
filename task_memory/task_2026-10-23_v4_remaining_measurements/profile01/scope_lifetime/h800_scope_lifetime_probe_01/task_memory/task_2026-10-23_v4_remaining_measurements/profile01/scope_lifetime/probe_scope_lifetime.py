"""Standalone live-scope proof; importing this module calls no CUDA API."""

import argparse
import ctypes
import hashlib
import importlib.util
import json
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parent / "graph_probe/probe_diagnostic.py"
assert hashlib.sha256(FROZEN.read_bytes()).hexdigest() == "a9d6a24e0aff13ef61088d75b5f693b599f42e8c50c2702342fbb84d3754d1a0"
spec = importlib.util.spec_from_file_location("scope_frozen_probe", FROZEN)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
NAMES = ("add_a", "mul_b", "add_c", "mul_d", "sub_out")


def frame():
    return dict(monotonic_ns=time.monotonic_ns(), native_id=threading.get_native_id(),
                python_ident=threading.get_ident())


def serving_profile():
    root = HERE.parents[3]
    bindings = {r["path"]: r["sha256"] for r in json.loads((HERE / "source_binding.json").read_text())["bindings"]}
    for name in ("native_full_structure", "native_serve_profile"):
        path = root / f"tests/comparison/calibration/{name}.py"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == bindings[str(path.relative_to(root))]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    sys.modules["vllm.compilation.breakable_cudagraph"] = SimpleNamespace(
        BreakableCUDAGraphCapture=SimpleNamespace(is_active=lambda: False))
    return module.ServingProfile(SimpleNamespace(local=threading.local()))


class ScopeGraphCallbacks(probe.CuptiGraphCallbacks):
    def __init__(self, ids, runtime, profile):
        self.runtime, self.profile = runtime, profile
        super().__init__(ids)

    def start(self):
        super().start()
        if self.owned and not self.receipt["errors"]:
            self._call("cuptiEnableCallback", 1, self.handle, probe.RESOURCE, 13)

    def _callback(self, userdata, domain, callback_id, pointer):
        row = dict(callback_id=callback_id, **frame(), scope=getattr(self.profile.local, "scope", None))
        try:
            if domain != probe.RESOURCE or callback_id not in (13, 20, 18) or not pointer:
                raise ValueError("Unexpected or empty graph resource callback")
            resource = ctypes.cast(pointer, ctypes.POINTER(probe.CuptiResourceData)).contents
            if not resource.resourceDescriptor:
                raise ValueError("Graph resource callback has no descriptor")
            native = ctypes.cast(resource.resourceDescriptor, ctypes.POINTER(probe.CuptiGraphData)).contents
            row.update(context=int(resource.context or 0), resource_handle=int(resource.resourceHandle.stream or 0),
                       payload={name: int(getattr(native, name) or 0) for name, _ in probe.CuptiGraphData._fields_})
            if callback_id == 13:
                row.update(node=row["payload"]["node"], graph=row["payload"]["graph"], node_type=native.nodeType)
                self.receipt["events"].append(row)
                if not row["node"] or not row["graph"]:
                    raise ValueError("Created callback has no node/graph handle")
                row.update(node_id=self.ids.node_id(row["node"]), graph_id=self.ids.graph_id(row["graph"]))
                node_type, = probe.checked(self.runtime.cudaGraphNodeGetType(row["node"]))
                row["queried_type"] = int(node_type)
            else:
                before = len(self.receipt["events"])
                super()._callback(userdata, domain, callback_id, pointer)
                if len(self.receipt["events"]) > before:
                    self.receipt["events"][-1].update(row)
                    row = self.receipt["events"][-1]
            probe._diagnostic_stage("callback", row)
        except BaseException as error:
            self.receipt["errors"].append(dict(callback_id=callback_id, error_type=type(error).__name__, message=str(error)))


def capture_operations(operations, profile, report):
    for slot, (name, operation) in enumerate(zip(NAMES, operations)):
        with profile.range("component", name):
            window = dict(scope=name, slot=slot, **frame())
            window["start"] = window["monotonic_ns"]
            report["windows"].append(window)
            probe._diagnostic_stage("op.start", window)
            try:
                function, a, b, out = operation
                function(a, b, out=out)
            finally:
                window["end"] = time.monotonic_ns()
                probe._diagnostic_stage("op.end", window)


def lifetime(graph, report):
    try:
        try:
            graph.raw_cuda_graph()  # Discard any unexpected return without inspecting it.
            report["raw_source_runtime_error"] = False
        except RuntimeError:
            report["raw_source_runtime_error"] = True
        report["raw_exec"] = graph.raw_cuda_graph_exec()
    except Exception:
        report["errors"].append(traceback.format_exc())


def analyze(report, events):
    predicates = report["predicates"] = dict.fromkeys(("P1", "P2", "P3", "P4", "P5", "P6"), "UNVERIFIED")
    try:
        launch, kernels = probe.target_kernels(events)
        report.update(target_launch=launch, target_kernels=kernels)
        keys = [int(k["args"]["graph node id"]) for k in kernels]
    except Exception:
        report["errors"].append(traceback.format_exc())
    capture = report["callback_capture"]
    rows = capture["events"]
    created = [r for r in rows if r["callback_id"] == 13]
    clones = [r for r in rows if r["callback_id"] == 20]
    execs = [r for r in rows if r["callback_id"] == 18]
    originals = {(r["original_node"], r["original_node_id"]) for r in clones
                 if "original_node_id" in r}
    report["unmatched_created_count"] = sum((r["node"], r.get("node_id")) not in originals for r in created)
    links = report["links"] = []
    try:
        if not capture["subscribed"] or not capture["unsubscribed"] or capture["subscription_active"] or capture["errors"] or report["profile_errors"]:
            raise ValueError("Subscription, callback, close or real range observation failed")
        if len(clones) != 5 or len(execs) != 1 or len(rows) != len(created) + len(clones) + len(execs):
            raise ValueError("Expected five clones and one directly witnessed exec")
        relation = execs[0]
        for clone in clones:
            selected = [r for r in created if (r["node"], r.get("node_id")) ==
                        (clone["original_node"], clone["original_node_id"])]
            if len(selected) != 1:
                raise ValueError("Original handle/full-ID tuple does not select exactly one created row")
            source = selected[0]
            links.append(dict(source=source, clone=clone))
            if (source["node_type"] != 0 or source["queried_type"] != 0 or source["node_id"] != clone["original_node_id"]
                    or source["graph"] != relation["graph"] or source["graph_id"] != relation["graph_id"]):
                raise ValueError("Source type, full original ID or direct source graph mismatch")
        for field in ("node", "node_id"):
            for kind in ("source", "clone"):
                values = [link[kind][field] for link in links]
                if len(set(values)) != 5 or any(v <= 0 for v in values):
                    raise ValueError("Duplicate or invalid source/clone identity")
        predicates.update(P1="PASS", P5="PASS", P2="PASS")
        report["thread_observations"] = [{key: link["source"][key] == report["capture_frame"][key]
                                        for key in ("native_id", "python_ident")} for link in links]
        windows = report["windows"]
        owners = []
        for link in links:
            source = link["source"]
            matches = [w for w in windows if w["start"] <= source["monotonic_ns"] < w["end"]]
            owners.append(matches[0]["scope"] if len(matches) == 1 and matches[0]["scope"] == source["scope"] else None)
        fresh = (len(windows) == 5 and {w["slot"] for w in windows} == set(range(5))
                 and set(owners) == set(NAMES) and {w["scope"] for w in windows} == set(NAMES))
        predicates["P3"] = "PASS" if fresh else "NEGATIVE_LIVE_TLS_ROUTE"
        predicates["P4"] = "PASS" if report["capture_complete"] and report["default_constructor"] and report.get("raw_source_runtime_error") is True and report["raw_exec"] == relation["graph_exec"] else "UNVERIFIED"
        if report.get("target_kernels") and len(keys) == 5 and set(keys) == {r["clone"]["node_id"] for r in links} and {int(k["args"]["graph id"]) for k in kernels} == {relation["exec_graph_id"]}:
            report["full_id_replay_join"] = "PASS"
            predicates["P6"] = "PASS" if fresh and report["replay_complete"] else "UNVERIFIED"
    except Exception:
        report["errors"].append(traceback.format_exc())
    report["verdict"] = ("PASS" if all(v == "PASS" for v in predicates.values()) and not report["errors"] else
                         "NEGATIVE_LIVE_TLS_ROUTE" if predicates["P3"] == "NEGATIVE_LIVE_TLS_ROUTE" and report.get("full_id_replay_join") == "PASS" and predicates["P4"] == "PASS" and report["replay_complete"] and not report["errors"] else "UNVERIFIED")


def run_phase(torch, runtime, output, elements, warmup, mode, report):
    profile = serving_profile()
    report.update(verdict="UNVERIFIED", errors=[], windows=[], profile_errors=profile.errors, capture_complete=False, replay_complete=False)
    report["predicates"] = dict.fromkeys(("P1", "P2", "P3", "P4", "P5", "P6"), "UNVERIFIED")
    stream, x = torch.cuda.Stream(), torch.ones(elements, device="cuda")
    a, b, c, d, out = [torch.empty_like(x) for _ in NAMES]
    operations = ((torch.add, x, x, a), (torch.mul, a, x, b), (torch.add, b, x, c),
                  (torch.mul, c, x, d), (torch.sub, d, x, out))
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            for function, left, right, target in operations:
                function(left, right, out=target)
    torch.cuda.synchronize()
    expected = [tensor.cpu().tolist() for tensor in (a, b, c, d, out)]
    capture = ScopeGraphCallbacks(probe.CuptiGraphIds(probe.cupti_library()), runtime, profile) if mode == "observe" else None
    report["callback_capture"] = None if capture is None else capture.receipt
    try:
        if capture is not None:
            capture.start()
        graph = torch.cuda.CUDAGraph()
        report.update(default_constructor=True, capture_frame=frame())
        probe._diagnostic_stage(mode + ".capture.start", report["capture_frame"])
        with torch.cuda.graph(graph, stream=stream):
            capture_operations(operations, profile, report)
        report.update(capture_complete=True, capture_end=frame())
        probe._diagnostic_stage(mode + ".capture.end", report["capture_end"])
    finally:
        released = True if capture is None else capture.close()
        probe._diagnostic_stage(mode + ".close", released)
    lifetime(graph, report)
    if not released:
        report["errors"].append("Owned subscriber release failed; Kineto and control are blocked")
        return
    probe._diagnostic_stage(mode + ".kineto.start")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as trace:
        with torch.profiler.record_function(probe.TARGET):
            graph.replay()
        torch.cuda.synchronize()
        report["replay_complete"] = True
    path = output / (mode + "_trace.json")
    trace.export_chrome_trace(str(path))
    report["trace_path"] = str(path)
    events = json.loads(path.read_text())["traceEvents"]
    if capture is not None:
        analyze(report, events)
    else:
        launch, kernels = probe.target_kernels(events)
        report.update(target_launch=launch, target_kernels=kernels, topology=len(kernels) == len({k["args"]["graph node id"] for k in kernels}) == 5 and len({k["args"]["graph id"] for k in kernels}) == 1)
    report["outputs"] = [tensor.cpu().tolist() for tensor in (a, b, c, d, out)]
    report["numeric_identity"] = report["outputs"] == expected


def main():
    probe._diagnostic_start()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--elements", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.elements <= 0 or args.warmup <= 0:
        parser.error("elements and warmup must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(image_digest=args.image_digest, elements=args.elements, warmup=args.warmup, errors=[])
    try:
        import torch
        import cuda.bindings.runtime as runtime
        with torch.inference_mode():
            for mode in ("observe", "control"):
                phase = report[mode] = {}
                run_phase(torch, runtime, args.output_dir, args.elements, args.warmup, mode, phase)
                if mode == "observe" and phase["callback_capture"]["subscription_active"]:
                    break
        report["control_identity"] = report["observe"]["outputs"] == report["control"]["outputs"] if "control" in report else "NOT_RUN"
    except Exception:
        report["errors"].append(traceback.format_exc())
    (args.output_dir / "probe_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
