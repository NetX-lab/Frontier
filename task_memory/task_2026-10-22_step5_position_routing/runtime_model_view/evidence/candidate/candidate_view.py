def _models_view_for_family(
    self,
    family_name: str,
    cluster_type: Optional[ClusterType] = None,
    *, model_name: Optional[str] = None,
) -> Dict[str, BaseEstimator]:
    """Project legacy names and enumerate every native binding by its cache hash."""

    names = {name for name in self._legacy_model_registry(family_name)
             if model_name is None or name == model_name}
    names.update(
        name
        for name, identity in self._contract_model_registry(family_name)
        if not isinstance(identity, tuple) and (model_name is None or name == model_name)
    )
    models: Dict[str, BaseEstimator] = {}
    for (name, identity), model in self._contract_model_registry(family_name).items():
        if model_name is not None and name != model_name:
            continue
        if isinstance(identity, tuple) and not self._is_ffn_typed_model_for_cluster(cluster_type, name):
            label = f"{name}:{model._frontier_model_hash}"
            if (label in models
                    and models[label]._frontier_measurement_identity != model._frontier_measurement_identity):
                raise ValueError(f"Conflicting native model artifacts for {label!r}")
            models[label] = model
    for model_name in sorted(names):
        if self._is_ffn_typed_model_for_cluster(cluster_type, model_name):
            continue
        contract = self._resolve_cluster_model_contract(cluster_type, model_name)
        identity = _serialize_selected_layer_cache_identity(contract)
        model = self._get_family_model(
            family_name,
            model_name,
            requested_identity=identity,
            include_native=False,
        )
        if identity is not None and model is None:
            raise ValueError(
                f"No trained model for {model_name!r} matches the typed contract "
                f"requested by cluster {cluster_type!r}: {identity}"
            )
        if model is not None:
            models[model_name] = model
    return models
