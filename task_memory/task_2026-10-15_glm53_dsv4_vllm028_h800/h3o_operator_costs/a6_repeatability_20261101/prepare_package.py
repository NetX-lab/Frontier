from pathlib import Path
import importlib.util
repo=Path('/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028')
task=Path('task_memory/task_2026-10-15_glm53_dsv4_vllm028_h800/h3o_operator_costs')
source=repo/task/'p1h_bridge/prepare_package.py'
spec=importlib.util.spec_from_file_location('a6_existing_package_builder',source)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.ROOT=Path('/data/ycfeng/Frontier/.worktrees/g1h-supplement-build')
module.HERE=module.ROOT/task/'a6_repeatability_20261101'
module.main()
