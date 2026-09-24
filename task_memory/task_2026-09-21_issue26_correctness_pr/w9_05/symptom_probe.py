"""Run the regression-test configuration on one tree and report completion."""
import importlib.util, sys, tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location("case", sys.argv[1])
case = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case)
from frontier.simulator import Simulator

with tempfile.TemporaryDirectory(dir="/data/ycfeng/tmp") as root:
    simulator = Simulator(case._config(Path(root)))
    simulator.run()
    for r in simulator._all_requests:
        print(r.id, "completed", r.completed, "decode", r.num_processed_decode_tokens, "/", r.num_decode_tokens)
