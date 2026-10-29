import json, shlex, subprocess, sys
from pathlib import Path
from dsv4_graph_bridge_check import graph_mode_flags
scratch = Path(__file__).parent
config = json.loads(Path("/data/ycfeng/tmp/glm53_dsv4/h200/p1h03_verify/bridge_config.json").read_text())
command = [sys.executable, "-m", "frontier.profiling.attention.dsv4_attention_main", "aggregate",
           *config["plan_args"], *graph_mode_flags("FULL_AND_PIECEWISE", config["cudagraph_capture_sizes"]),
           "--raw_dir", str(scratch / "raw"), "--output_root", str(scratch / "profiling"),
           "--device", "h200", "--source_revision", "230239c8", "--model_config_revision", "package_03"]
(scratch / "aggregate-command.txt").write_text(shlex.join(command) + "\n")
print(shlex.join(command), flush=True)
raise SystemExit(subprocess.run(command, timeout=240).returncode)
