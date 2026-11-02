"""Run the existing frozen balanced Frontier binding check against staged inputs."""
import importlib.util
import json
from pathlib import Path

root = Path(__file__).resolve().parent
repo = Path('/data/ycfeng/Frontier/.worktrees/v4-native-profile04-run')
package = repo / 'task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/v4_p0_native_20261021/candidate_h800_p0_profile_04'
config = json.loads((package / 'config.json').read_text())
lock = json.loads((package / 'source_lock.json').read_text())
spec = importlib.util.spec_from_file_location('frozen_checker', repo / 'task_memory/task_2026-10-22_v4_native_dispatch/check_groundtruth_run_successor_r3.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
result = checker.verify_balanced_frontier(config | {'harness_commit': lock['harness_commit']}, config['engine'])
assert len(result['sources']) == 84
assert all(role['world_size'] == 16 for role in result['roles'].values())
(root / 'balanced_frontier.receipt.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(status='PASS_EXISTING_BALANCED_FRONTIER', sources=len(result['sources']), roles=result['roles'])))
