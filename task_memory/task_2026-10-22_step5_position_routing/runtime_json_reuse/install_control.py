"""Install explicitly identified scratch functions in one diagnostic process."""
import ast
import hashlib
import importlib
import json
from pathlib import Path


def install(source, root, label):
    control = json.loads((root / 'control.json').read_text())
    if label == 'baseline':
        return {'applied': False, 'variant': 'baseline'}
    specifications = []
    if control['variant'] in ('registry', 'combined'):
        specifications.append(('prediction_model_registry', 'PredictionModelRegistry', ['_get_family_model']))
    if control['variant'] in ('point', 'combined'):
        specifications.append(('moe_routing_workload', 'MoeRoutingWorkload', ['_step5_plan_candidates', '_step5_position_queries']))
    installed = []
    for module_name, class_name, methods in specifications:
        relative = Path('frontier/execution_time_predictor') / (module_name + '.py')
        path = Path(control['patch_root']) / relative
        raw = path.read_bytes()
        tree = ast.parse(raw)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        definitions = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
        assert len(definitions) == len(methods)
        module = importlib.import_module('frontier.execution_time_predictor.' + module_name)
        namespace = dict(vars(module))
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
        for name in methods:
            assert Path(getattr(getattr(module, class_name), name).__code__.co_filename).resolve() == source / relative
            setattr(getattr(module, class_name), name, namespace[name])
        installed.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'methods': methods})
    return {'applied': True, 'variant': control['variant'], 'source_files_unchanged': True, 'installed': installed}
