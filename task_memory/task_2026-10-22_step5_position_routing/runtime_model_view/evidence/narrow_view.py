"""Change only the two query-view method bodies in the candidate process."""
import hashlib
import inspect
import textwrap


def install(namespace, arm, root):
    import frontier.execution_time_predictor.prediction_model_registry as registry
    import frontier.execution_time_predictor.moe_routing_workload as workload
    owner = registry.PredictionModelRegistry
    original = owner._models_view_for_family
    view = textwrap.dedent(inspect.getsource(original))
    changes = {
        '    cluster_type: Optional[ClusterType] = None,\n': '    cluster_type: Optional[ClusterType] = None,\n    *, model_name: Optional[str] = None,\n',
        '    names = set(self._legacy_model_registry(family_name))': '    names = {name for name in self._legacy_model_registry(family_name)\n             if model_name is None or name == model_name}',
        '        model_name\n        for model_name, identity in self._contract_model_registry(family_name)\n        if not isinstance(identity, tuple)': '        name\n        for name, identity in self._contract_model_registry(family_name)\n        if not isinstance(identity, tuple) and (model_name is None or name == model_name)',
        '        if isinstance(identity, tuple) and not self._is_ffn_typed_model_for_cluster(cluster_type, name):': '        if model_name is not None and name != model_name:\n            continue\n        if isinstance(identity, tuple) and not self._is_ffn_typed_model_for_cluster(cluster_type, name):',
    }
    for old, new in changes.items():
        assert view.count(old) == 1, old
        view = view.replace(old, new)
    view_path = root / arm / 'candidate_view.py'
    view_path.write_text(view)
    globals_view = dict(vars(registry))
    exec(compile(view, str(view_path), 'exec'), globals_view)
    candidate = globals_view['_models_view_for_family']
    predictor = namespace['predictor']
    manager = predictor._model_manager
    family = predictor._measurement_family_name(predictor._active_measurement_type)
    role = predictor._cluster_type
    full = original(manager, family, role)
    assert candidate(manager, family, role) == full
    selected = lambda models: {key: model for key, model in models.items()
        if getattr(model, '_frontier_cache_model_name', None) == 'moe_grouped_gemm'}
    expected = selected(full)
    actual = selected(candidate(manager, family, role, model_name='moe_grouped_gemm'))
    assert actual.keys() == expected.keys() and all(actual[key] is expected[key] for key in expected)
    point = textwrap.dedent(inspect.getsource(workload.MoeRoutingWorkload._step5_position_queries))
    old = 'self._model_manager._models_view_for_family(family, self._cluster_type)'
    assert point.count(old) == 1
    point = point.replace(old, "self._model_manager._models_view_for_family(family, self._cluster_type, model_name='moe_grouped_gemm')")
    point_path = root / arm / 'candidate_point.py'
    point_path.write_text(point)
    globals_point = dict(vars(workload))
    exec(compile(point, str(point_path), 'exec'), globals_point)
    if arm == 'candidate':
        owner._models_view_for_family = candidate
        workload.MoeRoutingWorkload._step5_position_queries = globals_point['_step5_position_queries']
    return {'applied': arm == 'candidate', 'full_view_equal': True, 'selected_objects_identical': True,
        'full_models': len(full), 'selected_models': len(expected),
        'view_sha256': hashlib.sha256(view.encode()).hexdigest(),
        'point_sha256': hashlib.sha256(point.encode()).hexdigest()}
