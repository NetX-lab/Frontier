"""Audit the actual simulator imports before allocating predictor work."""

import importlib
import importlib.metadata
import json
import sys
import traceback
from pathlib import Path

report = {"executable": sys.executable, "python": sys.version, "packages": {}, "errors": {}}
packages = {
    "ddsketch": "ddsketch", "fasteners": "fasteners", "numpy": "numpy",
    "pandas": "pandas", "plotly.express": "plotly", "yaml": "PyYAML",
    "sklearn": "scikit-learn", "scipy": "scipy", "tqdm": "tqdm",
}
for module, distribution in packages.items():
    try:
        imported = importlib.import_module(module)
        report["packages"][distribution] = {
            "version": importlib.metadata.version(distribution), "path": imported.__file__}
    except Exception:
        report["errors"][module] = traceback.format_exc()
try:
    from frontier.simulator import Simulator
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=2, random_state=0, n_jobs=1)
    model.fit([[0], [1], [2]], [0, 1, 2])
    assert len(model.predict([[1]])) == 1
    report["simulator_import"] = Simulator.__module__
except Exception:
    report["errors"]["simulator_and_predictor"] = traceback.format_exc()
report["status"] = "FAIL" if report["errors"] else "PASS"
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
raise SystemExit(bool(report["errors"]))
