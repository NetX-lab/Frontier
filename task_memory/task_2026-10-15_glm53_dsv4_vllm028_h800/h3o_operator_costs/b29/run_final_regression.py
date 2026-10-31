import json, subprocess, sys
from pathlib import Path
s=Path(__file__).resolve().parent
root=Path(sys.argv[1]);label=sys.argv[2]
b=Path("/data/ycfeng/Frontier/.worktrees/b29-ingress-delays/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs/b29")
targets=json.loads((b/"regression_targets.json").read_text())
missing=[p for p in targets if not (root/p.split("::")[0]).exists()]
assert missing in ([],["tests/unit/test_serving_path_delays.py"]),missing
targets=[p for p in targets if p not in missing]
print(label,"targets",len(targets),"new-module absent on base",missing,flush=True)
result=subprocess.run(["timeout","3700",sys.executable,str(s/"run_pytest.py"),str(root),label,*targets],timeout=3760)
sys.exit(result.returncode)
