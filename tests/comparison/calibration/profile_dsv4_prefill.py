"""Run balanced isolated prefill after the accepted C26 bootstrap."""

import argparse
import hashlib
import importlib
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if any(name.split(".")[0] in ("torch", "vllm") for name in sys.modules):
        raise RuntimeError("Native modules loaded before the accepted C26 bootstrap")
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1":
        raise RuntimeError("V4 prefill requires the observed V4 V2 runner")
    package = Path(config["native_package"])
    for name, expected in config["expected_source_hashes"].items():
        source = package / name.removeprefix("vllm/")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Native source mismatch: {name}")
    sys.path[:0] = [config["source_snapshot"], config["worker_wheel_site"]]
    os.chdir(config["source_snapshot"])
    if config["moe_gating_routing_runtime_path"] != "balanced_topk":
        raise ValueError("Prefill bootstrap requires explicit balanced_topk")
    strategy = Path(config["script"]).with_name("balanced_routing.py")
    spec = PathFinder.find_spec("balanced_routing", sys.path)
    if ("balanced_routing" in sys.modules or spec is None
            or spec.origin != str(strategy) or spec.submodule_search_locations is not None
            or hashlib.sha256(strategy.read_bytes()).hexdigest() != config["tool_hashes"][str(strategy)]):
        raise ValueError("Balanced routing origin or tool hash mismatch")
    os.environ["VLLM_MOE_ROUTING_SIMULATION_STRATEGY"] = "balanced_topk"
    module = importlib.import_module("balanced_routing")
    if module.__file__ != str(strategy):
        raise ValueError("Loaded balanced routing origin mismatch")
    module.register_balanced_topk()
    from frontier.profiling.attention import dsv4_attention_runner

    dsv4_attention_runner.routing_config = config
    from frontier.profiling.attention.dsv4_attention_main import main as profile

    profile(config["profile_args"])


if __name__ == "__main__":
    main()
