"""Rebind the retained wave to committed source without candidate installation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
ORIGINAL = REPO / "task_memory/task_2026-10-22_step5_position_routing/runtime_model_view/evidence"
ROOT = Path(sys.argv[1]).resolve()


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


frozen = json.loads((ORIGINAL / "freeze.json").read_text())
original_hashes = frozen["sources"]["baseline"]["hashes"]
registry = "frontier/execution_time_predictor/prediction_model_registry.py"
owned = {
    registry, "frontier/execution_time_predictor/moe_routing_workload.py",
    "tests/unit/test_prediction_measurement_bindings.py", "tests/unit/test_moe_routing_runtime.py",
}
baseline = Path(frozen["sources"]["baseline"]["root"])
sources = {}
for label, source in (("baseline", baseline), ("candidate", REPO)):
    head = git(source, "rev-parse", "HEAD")
    assert not git(source, "diff", "HEAD", "--name-only", "--", "frontier", "tests", "data")
    hashes = {}
    for relative, expected in original_hashes.items():
        value = sha(source / relative)
        if label == "baseline" or relative not in owned:
            assert value == expected, relative
        committed = subprocess.check_output(["git", "show", f"{head}:{relative}"], cwd=source)
        assert value == hashlib.sha256(committed).hexdigest(), relative
        hashes[relative] = value
    hashes[registry] = sha(source / registry)
    sources[label] = {"root": str(source), "head": head, "hashes": hashes,
        "submodule_pin": "3dac53c544e546574cb6e8078cd14053d0916113"}
frozen.update(task_id="D29-NAMED-MODEL-QUERY", session_id="ses_e93af1ff7ffe0YosowRNrc24Xa", sources=sources)
assert sha(Path(frozen["protected_body"])) == frozen["protected_body_sha256"]
raw = Path(frozen["collective_original"]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == frozen["collective_sha256"]
for label in sources:
    target = ROOT / label / "collective-fixture/h800/collective_local.json"
    target.parent.mkdir(parents=True, exist_ok=False)
    target.write_bytes(raw)
(ROOT / "fixture_wiring.py").write_bytes((ORIGINAL / "fixture_wiring.py").read_bytes())
driver = (ORIGINAL / "run.py").read_text()
installation = "from narrow_view import install\nview_control = install(setup_namespace, LABEL, ROOT)"
assert driver.count(installation) == 1
production = '''from frontier.execution_time_predictor.prediction_model_registry import PredictionModelRegistry
from frontier.execution_time_predictor.moe_routing_workload import MoeRoutingWorkload
for function, relative in (
    (PredictionModelRegistry._models_view_for_family, 'frontier/execution_time_predictor/prediction_model_registry.py'),
    (MoeRoutingWorkload._step5_position_queries, 'frontier/execution_time_predictor/moe_routing_workload.py'),
):
    assert Path(function.__code__.co_filename).resolve() == SOURCE / relative
manager = setup_namespace['manager']
predictor = setup_namespace['predictor']
family = predictor._measurement_family_name(predictor._active_measurement_type)
full = manager._models_view_for_family(family, predictor._cluster_type)
expected = {key: model for key, model in full.items()
            if model._frontier_cache_model_name == 'moe_grouped_gemm'}
if LABEL == 'candidate':
    selected = manager._models_view_for_family(family, predictor._cluster_type, model_name='moe_grouped_gemm')
    again = manager._models_view_for_family(family, predictor._cluster_type, model_name=None)
    assert list(again) == list(full) and all(again[key] is model for key, model in full.items())
    assert manager._models_view_for_family(family, model_name='D29_ABSENT_MODEL') == {}
else:
    selected = expected
assert len(selected) == 2 and list(selected) == list(expected)
assert all(selected[key] is model for key, model in expected.items())
view_control = {'applied': False, 'production_method_files_verified': True,
    'full_order': list(full), 'selected_order': list(selected), 'selected_same_objects': True}
'''
driver = driver.replace(installation, production.rstrip())
driver = driver.replace("'task_id':'D29-RUNTIME-MODEL-VIEW'", "'task_id':'D29-NAMED-MODEL-QUERY'")
assert "from narrow_view" not in driver and "install(" not in driver
(ROOT / "run.py").write_text(driver)
(ROOT / "freeze.json").write_text(json.dumps(frozen, indent=2) + "\n")
print(json.dumps({"sources": sources, "protected_body_sha256": frozen["protected_body_sha256"],
    "original_driver_sha256": sha(ORIGINAL / "run.py"), "production_driver_sha256": sha(ROOT / "run.py"),
    "candidate_install_removed": True}, indent=2))
