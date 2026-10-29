"""Profile the first 180 s wall of the H4a-4 run (scratch diagnostic, not an H4a input)."""
import cProfile, pstats, signal, sys
from pathlib import Path
sys.path.insert(0, "task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4a_p0_diagnostic")
import h4a4_driver
from frontier.utils.random import set_seeds
root = Path(sys.argv[1])
config, _ = h4a4_driver.build_config(root)
set_seeds(config.seed)
from frontier.simulator import Simulator
simulator = Simulator(config)
class Stop(Exception):
    pass
def stop(*_):
    raise Stop
signal.signal(signal.SIGALRM, stop)
profile = cProfile.Profile()
signal.alarm(180)
profile.enable()
try:
    simulator.run()
except Stop:
    pass
profile.disable()
print("simulated time s", getattr(simulator, "_time", None))
stats = pstats.Stats(profile).sort_stats("cumulative")
stats.print_stats(45)
stats.sort_stats("tottime").print_stats(30)
