"""Verify the CPU simulator imports and binary dependencies in a mounted CPython prefix."""
from datetime import datetime, timezone
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import resource
import sys

PREFIX = Path(sys.argv[1]).resolve()
REPO = Path("/data/ycfeng/Frontier/.worktrees/step5-preview")
EXPECTED_VERSIONS = {
    "numpy": "2.2.6", "pandas": "2.3.3", "scipy": "1.15.3",
    "scikit-learn": "1.7.2", "plotly": "7.1.0", "fasteners": "0.20",
    "ddsketch": "3.0.1", "pytest": "9.1.1",
}
assert sys.version_info[:3] == (3, 10, 6), sys.version
assert Path(sys.prefix).resolve() == PREFIX, sys.prefix
modules = {}
for name in ("numpy", "pandas", "scipy", "sklearn", "plotly", "fasteners", "ddsketch", "pytest",
             "_ssl", "_sqlite3", "_bz2", "_lzma", "_ctypes", "_hashlib"):
    module = importlib.import_module(name)
    origin = Path(module.__file__).resolve()
    if not name.startswith("_"):
        assert origin.is_relative_to(PREFIX), (name, str(origin))
    modules[name] = {"origin": str(origin)}
versions = {name: metadata.version(name) for name in EXPECTED_VERSIONS}
assert versions == EXPECTED_VERSIONS, versions
from frontier.simulator import Simulator
from frontier.config import SimulationConfig
import frontier.simulator
import frontier.config
for name, module in (("frontier.simulator", frontier.simulator), ("frontier.config", frontier.config)):
    origin = Path(module.__file__).resolve()
    assert origin.is_relative_to(REPO), (name, str(origin))
    modules[name] = {"origin": str(origin)}
import numpy as np
import pandas as pd
from scipy.linalg import solve
from sklearn.linear_model import LinearRegression
matrix = np.array([[2.0, 0.0], [0.0, 4.0]])
vector = np.array([6.0, 8.0])
assert np.array_equal(np.linalg.solve(matrix, vector), np.array([3.0, 2.0]))
assert np.array_equal(solve(matrix, vector), np.array([3.0, 2.0]))
assert pd.DataFrame({"value": [1, 2, 3]}).value.sum() == 6
model = LinearRegression().fit(np.array([[0.0], [1.0], [2.0]]), np.array([1.0, 3.0, 5.0]))
assert np.allclose(model.predict(np.array([[3.0]])), np.array([7.0]))
result = {
    "task_id": "D29-MOUNTED-CPYTHON-RUNTIME", "status": "PASS_CPU_IMPORTS_ONLY",
    "collected_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version, "executable": sys.executable,
    "prefix": sys.prefix, "base_prefix": sys.base_prefix,
    "platform": platform.platform(), "package_versions": versions,
    "modules": modules, "binary_numerical_checks": "PASS",
    "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    "simulator_constructed": False, "pricing_wave_calls": 0, "gpu_calls": 0,
    "torch_required_for_cpu_fixture": False,
    "limitations": ["CPU imports and tiny binary operations only; no full workload memory or runtime claim.",
                    "No Simulator initialization, training, native vLLM or GPU qualification."]
}
print(json.dumps(result, indent=2))
