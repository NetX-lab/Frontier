from __future__ import annotations

from dataclasses import dataclass, replace

from frontier.operators.spec import (
    OperatorFamilySpec,
    OperatorSpec,
    TensorParallelMode,
)


@dataclass(frozen=True)
class FamilyBinding:
    """Concrete model-to-operator-family binding."""

    family_id: str
    variant_id: str
    family: OperatorFamilySpec
    frozen: bool = False
    reason: str = ""

    def require_enabled_for_execution(self) -> None:
        if self.frozen:
            raise NotImplementedError(
                f"Operator family binding {self.family_id}/{self.variant_id} "
                f"is frozen: {self.reason}"
            )
        self.family.require_enabled_for_execution()


@dataclass(frozen=True)
class OperatorQueryBinding:
    """Registry facts bound to one runtime operator query."""

    family_id: str
    family: OperatorFamilySpec
    operator: OperatorSpec
    physical_name: str
    profiling_name: str
    tp_mode: TensorParallelMode | None


def _registered_operator_matches(
    operator_name: str,
):
    from frontier.operators.families import iter_operator_families

    families = tuple(iter_operator_families())
    physical_matches = tuple(
        (family, operator)
        for family in families
        for operator in family.operators
        if operator.name == operator_name
    )
    profiling_matches = tuple(
        (family, operator)
        for family in families
        for operator in family.operators
        if operator.profiling_name() == operator_name
    )
    return families, physical_matches, profiling_matches


def bind_operator_query(
    operator_name: str,
    *,
    family_id: str | None = None,
) -> OperatorQueryBinding:
    """Bind a physical or unambiguous profiling operator name to the registry."""

    if not isinstance(operator_name, str) or not operator_name:
        raise ValueError("Operator query name must be a non-empty string")

    families, physical_matches, profiling_matches = _registered_operator_matches(
        operator_name
    )
    if family_id is not None:
        family_matches = tuple(family for family in families if family.family_id == family_id)
        if not family_matches:
            raise ValueError(f"Unknown operator family: {family_id}")

    if len(physical_matches) > 1:
        raise ValueError(
            f"Duplicate physical operator registration for {operator_name!r}"
        )
    if len(physical_matches) == 1:
        family, operator = physical_matches[0]
        conflicting_aliases = tuple(
            (candidate_family, candidate_operator)
            for candidate_family, candidate_operator in profiling_matches
            if candidate_operator is not operator
        )
        if conflicting_aliases:
            names = [
                f"{candidate_family.family_id}/{candidate_operator.name}"
                for candidate_family, candidate_operator in conflicting_aliases
            ]
            raise ValueError(
                f"Operator query {operator_name!r} collides with profiling aliases: "
                f"{names}"
            )
    elif not profiling_matches:
        raise ValueError(f"Unknown operator query: {operator_name!r}")
    elif len(profiling_matches) > 1:
        scoped_matches = profiling_matches
        if family_id is not None:
            scoped_matches = tuple(
                (candidate_family, candidate_operator)
                for candidate_family, candidate_operator in profiling_matches
                if candidate_family.family_id == family_id
            )
        scoped_families = {candidate_family.family_id for candidate_family, _ in scoped_matches}
        scoped_tp_modes = {
            candidate_operator.tp_mode for _, candidate_operator in scoped_matches
        }
        if (
            family_id is not None
            and scoped_matches
            and len(scoped_families) == 1
            and len(scoped_tp_modes) == 1
        ):
            family, operator = scoped_matches[0]
        else:
            names = [
                f"{candidate_family.family_id}/{candidate_operator.name}"
                for candidate_family, candidate_operator in profiling_matches
            ]
            raise ValueError(
                f"Operator query {operator_name!r} is an ambiguous profiling alias: {names}"
            )
    else:
        family, operator = profiling_matches[0]

    if family_id is not None and family.family_id != family_id:
        raise ValueError(
            f"Operator {operator_name!r} does not belong to operator family "
            f"{family_id!r}; registered family is {family.family_id!r}"
        )

    family.require_enabled_for_execution()
    return OperatorQueryBinding(
        family_id=family.family_id,
        family=family,
        operator=operator,
        physical_name=operator.name,
        profiling_name=operator.profiling_name(),
        tp_mode=operator.tp_mode,
    )


def _resolve_architecture_linear_tp_mode(
    operator_name: str,
    architecture_profile,
) -> TensorParallelMode:
    if architecture_profile is None:
        raise ValueError(f"Unsupported operator query: {operator_name!r}")

    attention_linear_ops = getattr(architecture_profile, "attention_linear_ops", None)
    if attention_linear_ops is None:
        raise TypeError(
            "architecture_profile must expose a attention_linear_ops declaration"
        )

    replicated_ops = tuple(attention_linear_ops.replicated_ops)
    sharded_ops = tuple(attention_linear_ops.sharded_ops)
    if operator_name in replicated_ops and operator_name in sharded_ops:
        raise ValueError(
            f"Architecture profile {architecture_profile.profile_id!r} declares "
            f"operator {operator_name!r} as both replicated and sharded"
        )
    if operator_name in replicated_ops:
        return TensorParallelMode.REPLICATED
    if operator_name in sharded_ops:
        return TensorParallelMode.ATTENTION_TP
    raise ValueError(f"Unsupported operator query: {operator_name!r}")


def resolve_operator_query_tp_mode(
    operator_name: str,
    *,
    family_id: str | None = None,
    architecture_profile=None,
) -> TensorParallelMode:
    """Resolve the declared TP mode for one exact operator query.

    Generic operators come from the unified operator registry. Architecture-
    specific linear attention operators come from the selected, already
    registered model architecture profile. A profiling alias can be resolved
    when all matching physical operators share one family and TP mode; this
    preserves explicit collision checks while supporting timing aliases such
    as the two memory residual operators that publish ``add``.
    """

    try:
        binding = bind_operator_query(operator_name, family_id=family_id)
    except ValueError as exc:
        _, physical_matches, profiling_matches = _registered_operator_matches(
            operator_name
        )
        if physical_matches:
            raise
        if profiling_matches:
            if family_id is not None:
                raise
            profiling_families = {
                candidate_family.family_id
                for candidate_family, _ in profiling_matches
            }
            profiling_tp_modes = {
                candidate_operator.tp_mode
                for _, candidate_operator in profiling_matches
            }
            if len(profiling_families) != 1 or len(profiling_tp_modes) != 1:
                raise
            binding = bind_operator_query(
                operator_name,
                family_id=next(iter(profiling_families)),
            )
        else:
            if family_id is not None:
                raise
            return _resolve_architecture_linear_tp_mode(
                operator_name,
                architecture_profile,
            )

    if binding.tp_mode is None:
        raise ValueError(
            f"Unsupported operator query TP mode for {operator_name!r}: None"
        )
    return binding.tp_mode


@dataclass(frozen=True)
class OperatorManifest:
    """Ordered operator families selected for one model config."""

    family_bindings: tuple[FamilyBinding, ...]

    def families(self) -> tuple[OperatorFamilySpec, ...]:
        return tuple(binding.family for binding in self.family_bindings)

    def operators(self) -> tuple[OperatorSpec, ...]:
        return tuple(dict.fromkeys(
            operator
            for binding in self.family_bindings
            for operator in binding.family.operators
        ))


def _get_model_architecture_profile(config):
    get_profile = getattr(config, "get_model_architecture_profile", None)
    if callable(get_profile):
        return get_profile()

    from frontier.model_architectures import get_model_architecture_profile

    return get_model_architecture_profile(config)


def _deepseek_v4_layer_manifest(config, layer) -> OperatorManifest:
    """Select reachable source scopes, without admitting runtime execution."""

    from frontier.attention.families import (
        DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT,
        get_attention_family,
    )
    from frontier.operators.families import (
        DEEPSEEK_V4_MHC_FAMILY,
        MOE_FAMILY,
        SHARE_EXPERT_FAMILY,
    )

    attention_family = get_attention_family(layer.family_id)
    attention_operators = DEEPSEEK_V4_ATTENTION_OPERATORS_BY_VARIANT[layer.variant_id]
    first = layer.global_layer_id == 0
    last = layer.global_layer_id == config.num_layers - 1
    mhc_names = {"dsv4_mhc_attn_post_ffn_pre"}
    if first:
        mhc_names.update(("dsv4_embedding", "dsv4_mhc_first_pre"))
    else:
        mhc_names.add("dsv4_mhc_ffn_post_attn_pre")
    if last:
        mhc_names.update((
            "dsv4_mhc_final_post", "dsv4_mhc_head", "dsv4_final_norm", "dsv4_lm_head",
        ))
    mhc_variant = "first_last" if first and last else (
        "first" if first else "last" if last else "middle"
    )
    reason = "DeepSeek-V4 layer scopes are source inventory; runtime execution is disabled"
    bindings = [
        FamilyBinding(
            family_id=attention_family.family_id,
            variant_id=layer.variant_id,
            family=replace(attention_family, operators=attention_operators),
            frozen=True,
            reason=reason,
        ),
        FamilyBinding(
            family_id=DEEPSEEK_V4_MHC_FAMILY.family_id,
            variant_id=mhc_variant,
            family=replace(
                DEEPSEEK_V4_MHC_FAMILY,
                operators=tuple(
                    operator for operator in DEEPSEEK_V4_MHC_FAMILY.operators
                    if operator.name in mhc_names
                ),
            ),
            frozen=True,
            reason=reason,
        ),
    ]
    # Preserve the shared family identities, but do not expose their generic
    # producer/predictor targets through this inventory-only model binding.
    for family, variant in (
        (MOE_FAMILY, "hash" if layer.global_layer_id < config.num_hash_layers else "score"),
        (SHARE_EXPERT_FAMILY, "shared_dense"),
    ):
        bindings.append(FamilyBinding(
            family_id=family.family_id,
            variant_id=variant,
            family=replace(
                family,
                supported_variants=(variant,),
                operators=tuple(
                    replace(operator, predictor_target=False, profiling_target=False,
                            e2e_trace_target=False, execution_time_attr=None)
                    for operator in family.operators
                ),
                profiling_order=(),
            ),
            frozen=True,
            reason=reason,
        ))
    return OperatorManifest(family_bindings=tuple(bindings))


def build_operator_manifest(config, *, global_layer_id: int | None = None) -> OperatorManifest:
    """Build the complete model inventory or one global layer's reachable scopes."""
    from frontier.attention.families import get_attention_family
    from frontier.attention.model_binding import bind_attention_family, bind_layer_attention
    from frontier.operators.families import (
        FFN_FAMILY, MEMORY_FAMILY, MOE_FAMILY, SHARE_EXPERT_FAMILY, get_operator_family,
    )

    architecture_profile = _get_model_architecture_profile(config)
    architecture_profile.validate_structural_requirements(config)
    get_specs = getattr(config, "get_layer_attention_specs", None)
    bindings = []
    contracts = ()
    if global_layer_id is not None:
        layer = bind_layer_attention(config, global_layer_id)
        if layer.family_id == "deepseek_v4_attention":
            return _deepseek_v4_layer_manifest(config, layer)
        specs = (layer,)
        layer_kind = architecture_profile.resolve_layer_contract(
            config, layer_id=global_layer_id,
        ).layer_kind
        contracts = tuple(
            contract for contract in architecture_profile.iter_active_layer_contracts(config)
            if layer_kind in contract.base_layer_kinds
        )
    elif callable(get_specs):
        specs = tuple(get_specs())
        if tuple(spec.global_layer_id for spec in specs) != tuple(range(config.num_layers)):
            raise ValueError("Operator manifest requires every backbone layer in order")
        # V4 source scopes require a layer to select routing and mHC endpoints.
        if any(spec.family_id == "deepseek_v4_attention" for spec in specs):
            raise ValueError("DeepSeek-V4 operator scopes require an explicit global layer id")
        contracts = tuple(architecture_profile.iter_active_layer_contracts(config))
    else:
        specs = ()
        contracts = tuple(architecture_profile.iter_active_layer_contracts(config))
        attention = bind_attention_family(config)
        bindings.append(FamilyBinding(
            family_id=attention.family_id, variant_id=attention.variant_id,
            family=attention.family, frozen=attention.frozen, reason=attention.reason,
        ))

    seen = set()
    for spec in specs:
        if bind_layer_attention(config, spec.global_layer_id) != spec:
            raise ValueError("Operator manifest disagrees with the model-owned layer binding")
        identity = (spec.family_id, spec.variant_id)
        if identity in seen:
            continue
        family = get_attention_family(spec.family_id)
        if spec.variant_id not in family.supported_variants:
            raise ValueError(f"Unknown attention variant {spec.variant_id!r} for {spec.family_id!r}")
        bindings.append(FamilyBinding(
            family_id=spec.family_id, variant_id=spec.variant_id, family=family,
            frozen=not family.execution_enabled,
            reason="Resolved from the model-owned layer attention bindings",
        ))
        seen.add(identity)
    bindings.append(FamilyBinding(
        family_id=MEMORY_FAMILY.family_id, variant_id="replicated", family=MEMORY_FAMILY,
    ))

    layer_is_moe = bool(getattr(config, "is_moe", False))
    if global_layer_id is not None and layer_is_moe:
        from frontier.model_architectures import parse_moe_layer_ids
        layer_is_moe = global_layer_id in parse_moe_layer_ids(
            config.moe_layers_enum, config.num_layers,
        )
    elif architecture_profile._is_mixed_model(config):
        bindings.append(FamilyBinding(
            family_id=FFN_FAMILY.family_id, variant_id="dense", family=FFN_FAMILY,
        ))
    if layer_is_moe:
        bindings.append(FamilyBinding(
            family_id=MOE_FAMILY.family_id, variant_id="routed", family=MOE_FAMILY,
        ))
        if not hasattr(config, "supports_share_expert"):
            raise TypeError("MoE operator manifests require config.supports_share_expert()")
        if config.supports_share_expert():
            bindings.append(FamilyBinding(
                family_id=SHARE_EXPERT_FAMILY.family_id,
                variant_id="shared_dense", family=SHARE_EXPERT_FAMILY,
            ))
    else:
        bindings.append(FamilyBinding(
            family_id=FFN_FAMILY.family_id, variant_id="dense", family=FFN_FAMILY,
        ))

    present = {binding.family_id for binding in bindings}
    for contract in contracts:
        for family_id in contract.operator_family_ids:
            if family_id in present:
                continue
            family = get_operator_family(family_id)
            if len(family.supported_variants) != 1:
                raise ValueError(f"Layer family {family_id!r} requires an explicit variant binding")
            bindings.append(FamilyBinding(
                family_id=family_id, variant_id=family.supported_variants[0], family=family,
                frozen=not family.execution_enabled,
                reason="Declared by the model-owned layer contract",
            ))
            present.add(family_id)
    return OperatorManifest(family_bindings=tuple(bindings))
