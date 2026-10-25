import hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
T = Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028/task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h4b_lifecycle_measurement/package_01')
config = json.loads((T / 'config.json').read_text())
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
overlay = json.loads(Path(config['overlay_manifest']).read_text())
staged = Path(overlay['staged_dir'])
paths = [T / 'config.json', Path(config['script']), Path(config['overlay_manifest'])]
out = dict(
    staged_is_package=staged == T and all(p.parent == staged for p in paths),
    tool_hashes_ok=all(h(n) == v for n, v in config['tool_hashes'].items()) and config['script'] in config['tool_hashes'],
    overlay_source_ok=all(h(r['source']) == r['sha256'] and Path(r['source']).parent == T for r in overlay['files'].values()),
    v4_pin_ok=config['expected_source_hashes']['vllm/models/deepseek_v4/nvidia/model.py'] == overlay['files']['models/deepseek_v4/nvidia/model.py']['sha256'],
    observer_in_harness_dir=(Path(config['harness_dir']) / 'native_scheduler_lifecycle.py').is_file(),
    workspace=config['workspace'] == str(T), output_dir=config['output_dir'])
sys.path[:0] = [str(T), config['harness_dir'], str(Path(config['harness_dir']).parent / 'dp_placement_pp')]
from native_pd_worker import node_plan
from vllm_replay import server_command
peers = [dict(host=f'h{i}', ip=f'10.0.0.{i}') for i in range(4)]
for host in ('h0', 'h1', 'h2', 'h3'):
    plan = node_plan(config, peers, host, [str(i) for i in range(8)])
    try:
        command = server_command(plan['engine'], Path(config['model_dir']), config['http_ports'][plan['role']])
    except Exception as error:
        command = [repr(error)]
    i = command.index('--scheduler-cls') if '--scheduler-cls' in command else None
    out[host] = dict(role=plan['role'], lane=plan['lane'], headless=plan['engine']['headless'],
                     scheduler_cls=command[i + 1] if i is not None else None,
                     async_flag='--async-scheduling' in command)
print(json.dumps(out, indent=1))
