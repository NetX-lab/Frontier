def _models_view_for_family(
    self,
    family_name: str,
    cluster_type: Optional[ClusterType] = None,
) -> Dict[str, BaseEstimator]:
    """Project legacy names and enumerate every native binding by its cache hash."""

    names = set(self._legacy_model_registry(family_name))
    names.update(
        model_name
        for model_name, identity in self._contract_model_registry(family_name)
        if not isinstance(identity, tuple)
    )
    models: Dict[str, BaseEstimator] = {}
    for (name, identity), model in self._contract_model_registry(family_name).items():
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
