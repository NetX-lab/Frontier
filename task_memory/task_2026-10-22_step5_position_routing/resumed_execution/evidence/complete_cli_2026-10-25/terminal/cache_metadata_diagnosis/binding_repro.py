"""Reproduce only the original post-run fixture metadata assertion without pricing."""
from pathlib import Path
import json
import sys
preparation=Path('/data/ycfeng/Frontier/.worktrees/step5-preview/task_memory/task_2026-10-22_step5_position_routing/resumed_execution/evidence/execution_preparation_2026-10-25')
sys.path.insert(0,str(preparation))
from joint_timing import check_bindings
print(json.dumps(check_bindings(json.loads((preparation/'joint_bindings.json').read_bytes()),preparation)))
