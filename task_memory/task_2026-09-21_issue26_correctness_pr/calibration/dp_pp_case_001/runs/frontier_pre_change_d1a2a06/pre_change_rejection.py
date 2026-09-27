"""Construct the G4 Frontier configuration on the pre-change revision (plan section 18.5 G5)."""
import importlib.util, json, sys
from pathlib import Path
tree, case, out = map(Path, sys.argv[1:4])
spec = importlib.util.spec_from_file_location("run_frontier_case", tree / "tests/comparison/dp_placement_pp/run_frontier_case.py")
harness = importlib.util.module_from_spec(spec); spec.loader.exec_module(harness)
import frontier.config
assert Path(frontier.config.__file__).resolve().is_relative_to(out), frontier.config.__file__
from frontier.config import VllmLoadBalancingClusterSchedulerConfig
from frontier.simulator import Simulator
config = harness.case_config(
    json.loads((case / "inputs/engine_g4.json").read_text()),
    json.loads((case / "inputs/frontier_g4.json").read_text()),
    case / "inputs/trace_g4/trace.csv", out / "run", VllmLoadBalancingClusterSchedulerConfig,
)
print("frontier.config", frontier.config.__file__, "num_pipeline_stages", config.cluster_config.replica_config.num_pipeline_stages)
try:
    Simulator(config)
except ValueError as error:
    print("REJECTED", type(error).__name__, error)
    raise SystemExit(0)
print("CONSTRUCTED")
raise SystemExit(1)
