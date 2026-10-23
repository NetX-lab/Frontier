"""External source/path/owner wiring; never replace prediction behavior."""
import ast
import dataclasses
import hashlib
import os
from pathlib import Path
import sys

def setup(frozen, label, root):
    source = Path(frozen['sources'][label]['root'])
    original = Path(frozen['original_setup']).read_bytes()
    assert hashlib.sha256(original).hexdigest() == frozen['setup_sha256']
    original_tree = ast.parse(original)
    original_wave = next(node for node in original_tree.body if isinstance(node, ast.Assign) and
        any(isinstance(target, ast.Name) and target.id == 'phases' for target in node.targets))
    derived = original.replace(b'/data/ycfeng/Frontier/.worktrees/step5-d29-cost-fields', str(source).encode())
    derived = derived.replace(b'48d34106c026b824097f9910a0bbf89b253a8669', frozen['sources'][label]['head'].encode())
    tree = ast.parse(derived)
    wave = next(node for node in tree.body if isinstance(node, ast.Assign) and
        any(isinstance(target, ast.Name) and target.id == 'phases' for target in node.targets))
    assert ast.dump(wave, include_attributes=False) == ast.dump(original_wave, include_attributes=False)
    path = root / label / 'derived_setup.py'
    path.write_bytes(derived)
    cwd = root / label / 'cwd'
    relative = Path('data/config/models/stepfun-ai__Step-5-Preview.json')
    (cwd / relative).parent.mkdir(parents=True)
    (cwd / relative).write_bytes((source / relative).read_bytes())
    os.chdir(cwd)  # Before any CLI/configuration parsing.
    sys.path.insert(0, str(source))
    namespace = {'__name__': '__main__', '__file__': str(path)}
    sys.argv = [str(path), '--output', str(root / label / 'setup-output')]
    prefix = [node for node in tree.body if node.lineno < wave.lineno]
    exec(compile(ast.Module(body=prefix, type_ignores=[]), str(path), 'exec'), namespace)
    backend = namespace['predictor']._cc_backend
    config = dataclasses.replace(backend._config, profiling_data_dir=str(root / label / 'collective-fixture'))
    backend._config = config
    return namespace, compile(ast.Module(body=[wave], type_ignores=[]), str(path), 'exec'), tree, wave, path

def bind_production_owner(namespace, root, label):
    from frontier.cc_backend.collective_profile import CollectiveProfileInputs
    from frontier.types import ClusterType
    manager = namespace['manager']
    # The accepted helper reconstructs a cached manager via __new__, bypassing
    # __init__. Initialize precisely the new real constructor-owned field.
    manager._collective_profile_inputs = CollectiveProfileInputs()
    inputs = manager.collective_profile_inputs
    p_backend = namespace['Cluster'](namespace['clusters'][ClusterType.PREFILL],
        namespace['config'].metrics_config, namespace['config'].request_generator_config).cc_backend
    p_backend._config = dataclasses.replace(p_backend._config,
        profiling_data_dir=str(root / label / 'collective-fixture'))
    p_backend.bind_collective_profile_inputs(inputs)
    original = p_backend.collective_profile_bytes()
    path = root / label / 'collective-fixture' / p_backend._device_type / 'collective_local.json'
    path.write_bytes(b'{')  # Scratch copy only: first success stays fixed mid-run.
    d_backend = namespace['predictor']._cc_backend
    d_backend.bind_collective_profile_inputs(inputs)
    assert p_backend.collective_profile_bytes() == d_backend.collective_profile_bytes() == original
    assert p_backend._collective_profile_inputs is d_backend._collective_profile_inputs is inputs
    return p_backend, d_backend, original, path
