def _resolve_moe_layer_ids(cls, config: Any) -> tuple[int, ...]:
    """Resolve one canonical MoE layer set and verify custom adapters."""
    num_layers = getattr(config, 'num_layers', None)
    if type(num_layers) is not int or num_layers <= 0:
        raise ValueError('moe layer validation requires a positive num_layers')
    getter = getattr(config, 'get_moe_layer_ids', None)
    predicate = getattr(config, 'is_moe_layer', None)
    if callable(getter):
        try:
            typed_getter = cast(Callable[[], Iterable[int]], getter)
            raw_ids = tuple(typed_getter())
        except TypeError as exc:
            raise ValueError('get_moe_layer_ids() must return an iterable') from exc
        if any((type(layer_id) is not int for layer_id in raw_ids)):
            raise ValueError('get_moe_layer_ids() must return integer layer IDs')
        if len(set(raw_ids)) != len(raw_ids):
            raise ValueError('get_moe_layer_ids() returned duplicate layer IDs')
        if any((layer_id < 0 or layer_id >= num_layers for layer_id in raw_ids)):
            raise ValueError(f'get_moe_layer_ids() returned an out-of-range layer ID for [0, {num_layers})')
        canonical = tuple(sorted(raw_ids))
        if callable(predicate) and (not (getattr(getter, '__func__', None) is D29_DEFAULT_GETTER and getattr(predicate, '__func__', None) is D29_DEFAULT_PREDICATE)):
            predicate_ids = tuple((layer_id for layer_id in range(num_layers) if bool(predicate(layer_id))))
            if predicate_ids != canonical:
                raise ValueError(f'custom MoE layer getter and predicate disagree: get_moe_layer_ids={list(canonical)}, is_moe_layer={list(predicate_ids)}')
        return canonical
    if callable(predicate):
        return tuple((layer_id for layer_id in range(num_layers) if bool(predicate(layer_id))))
    return parse_moe_layer_ids(getattr(config, 'moe_layers_enum', None), num_layers)
