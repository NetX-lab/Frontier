"""Bounded resident identity and logical native-state witnesses, never native repair."""

import hashlib
import time

SCALAR_SCALE_FIELDS = ('_k_scale_float', '_v_scale_float', '_q_scale_float', '_prob_scale_float')


def resident_identity(observer, worker, request_id, phase, tp_rank):
    """Hash every resident view using the frozen chunk iterator, not state_dict/copies."""
    import torch

    capacity, state = observer.frozen_helpers()
    settings = observer.config['resident_identity']
    started, scanned, peak = time.monotonic(), 0, 0
    storages, rows, seen_views, scalars = {}, [], {}, {}
    cache_storage = set()
    cache_owners = {id(value) for value in worker.vllm_config.compilation_config.static_forward_context.values()}
    for ref in observer.connectors:
        connector = ref()
        if connector is None:
            continue
        for tensor in connector.device_kv_caches.values():
            cache_storage.add(capacity.register_storage(tensor, {})[0])
        context = connector.vllm_config.compilation_config.static_forward_context
        for registration in observer.connector_states[connector]['registrations']:
            if registration['component'] != 'summary_mean':
                continue
            cache = context[registration['registration_name']]._cache
            for name in ('numerator', 'denominator', 'maximum', 'last_slot',
                         'request_journal_values', 'request_journal_last', 'request_journal_tag'):
                value = getattr(cache, name, None)
                if isinstance(value, torch.Tensor):
                    cache_storage.add(capacity.register_storage(value, {})[0])
    excluded = []
    for module_name, module in worker.model_runner.model.named_modules(remove_duplicate=False):
        tensors = list(capacity.module_tensors(module_name, module, torch.Tensor))
        registered = set(module._parameters) | set(module._buffers)
        for name, value in vars(module).items():
            if name not in registered and isinstance(value, torch.Tensor):
                tensors.append((module_name+'.'+name, 'tensor_attribute', name, value, None))
            elif name in SCALAR_SCALE_FIELDS and type(value) in (bool, int, float):
                scalars[module_name+'.'+name] = repr(value)
        quant = getattr(module, 'quant_method', None)
        if quant is not None:
            for name, value in vars(quant).items():
                if isinstance(value, torch.Tensor):
                    tensors.append((module_name+'.quant_method.'+name, 'quant_tensor', name, value, None))
                elif type(value) in (bool, int, float, str):
                    scalars[module_name+'.quant_method.'+name] = repr(value)
        for name, kind, _, value, persistent in tensors:
            storage, _ = capacity.register_storage(value, {})
            if storage in cache_storage or (id(module) in cache_owners and name == module_name+'.kv_cache'):
                excluded.append(name)
                continue
            record = capacity.storage_record(name, kind, value, storages, persistent=persistent)
            key = state.parameter_view_key(record)
            if key not in seen_views:
                digest, size, chunks = hashlib.sha256(), 0, 0
                layout = state.parameter_chunk_layout(record, settings['chunk_bytes']//value.element_size())
                for chunk in state.parameter_chunks(value, layout):
                    host = chunk.detach().contiguous().cpu().reshape(-1).view(torch.uint8)
                    raw = host.numpy().tobytes()
                    digest.update(raw)
                    size += len(raw)
                    scanned += len(raw)
                    chunks += 1
                    peak = max(peak, len(raw))
                    if scanned > settings['max_bytes'] or time.monotonic()-started > settings['max_seconds']:
                        raise RuntimeError('Resident identity diagnostic scan budget exceeded')
                if size != value.numel()*value.element_size():
                    raise ValueError('Resident identity did not cover the complete tensor view')
                seen_views[key] = dict(sha256=digest.hexdigest(), bytes=size, chunks=chunks)
            rows.append(dict(record, **seen_views[key]))
    if not rows:
        raise ValueError('Resident model identity is empty')
    observer.emit('resident_identity', worker, request_id, phase=phase, tp_rank=tp_rank, tensors=rows,
                  scalar_attributes=scalars, excluded_native_cache_views=excluded,
                  bytes_scanned=scanned, peak_chunk_bytes=peak, elapsed_s=time.monotonic()-started,
                  complete=True, scope='all_registered_parameters_buffers_and_direct_tensor_quant_attributes',
                  observer_fence=True)


def checkpoint(observer, connector, req, request, boundary):
    """Observe valid logical cache bytes, excluding unused padding by native geometry."""
    capacity, state = observer.frozen_helpers()
    observer.synchronize()
    computed, block, region = request['computed'], observer.config['block_size'], observer.config['region_size']
    last_page = (computed-1)//block
    config = connector.vllm_config.compilation_config.static_forward_context
    metadata = observer.connector_states[connector]
    groups = [request['blocks'][g] for g in connector.kv_cache_config.transfer_group_ids]
    components = []
    for registration in metadata['registrations']:
        group = registration['group_id']
        window = metadata['group_specs'][group]['sliding_window']
        start = max(0, computed-window) if window is not None else 0
        cache = None
        if registration['component'] == 'summary_mean':
            cache = config[registration['registration_name']]._cache
            tensor, units = cache.mean, block//region
        else:
            # Locked Step backend/G1 geometry is [page, head, token, channel].
            # Only the logical witness changes axes; wire rows remain untouched.
            tensor = connector.device_kv_caches[registration['registration_name']].transpose(1, 2)
            units = block
        if tensor.shape[1] != units:
            raise ValueError('Unsupported native logical token/region axis')
        for logical in range(start//block, (computed+block-1)//block):
            physical = groups[group][logical]
            if physical == 0:
                raise ValueError('Required logical native state is null/retired')
            lo, hi = max(0, start-logical*block), min(block, computed-logical*block)
            if cache is not None:
                lo, hi = lo//region, (hi+region-1)//region
            sample, _ = observer.tensor_sample(req, registration['component'], tensor[physical, lo:hi], capacity, state)
            if boundary != 'first_local_tail_complete' or not last_page-1 <= logical <= last_page:
                sample.pop('raw_hex')
            components.append(dict(group_id=group, layer=registration['layer'],
                component=registration['component'], logical_page=logical, physical_page=physical,
                valid_units=[lo, hi], native_units_per_page=units,
                unit_shape=list(tensor.shape[2:]), sample=sample))
    observer.emit('state_checkpoint', connector, req, boundary=boundary, num_computed_tokens=computed,
                  logical_components=components, observer_fence=True)


def reset_content(observer, zeroer, arguments):
    """Observe actual mean/marker/tag reset; native accumulators are not zeroed here."""
    import torch

    requests = list(getattr(observer.local, 'scheduled_request_ids', ()))
    if not any(observer.content_selected(req) for req in requests):
        return
    observer.synchronize()
    groups = arguments.get('block_ids_by_group')
    if groups is None:
        raise ValueError('Native reset lacks owner-group IDs')
    for ref in observer.connectors:
        connector = ref()
        if connector is None:
            continue
        records = []
        context = connector.vllm_config.compilation_config.static_forward_context
        for registration in observer.connector_states[connector]['registrations']:
            pages = groups[registration['scheduler_group_id']]
            if not pages:
                continue
            row = dict(layer=registration['layer'], component=registration['component'],
                       group_id=registration['group_id'], pages=list(pages),
                       alias_storage=registration['alias_storage'])
            if registration['component'] == 'raw_kv':
                enabled = zeroer._meta is not None
                row.update(native_zero_enabled=enabled,
                           reset_policy='native_base_zero' if enabled else 'native_overwrite_before_use')
                if enabled:
                    tensor = connector.device_kv_caches[registration['registration_name']]
                    row['reset_valid'] = all(not bool(tensor[p].view(torch.uint8).any()) for p in pages)
                else:
                    row['reset_valid'] = None
            else:
                cache = context[registration['registration_name']]._cache
                row.update(reset_policy='mean_zero_last_slot_invalid_journal_page_tags_invalid',
                           accumulators_policy='unchanged_until_native_fresh_region_update',
                           reset_valid=all(not bool(cache.mean[p].view(torch.uint8).any())
                                           and int(cache.last_slot[p]) == -1 for p in pages))
                tags = cache.request_journal_tag if cache.request_journal_width else None
                row['journal_tags_invalid'] = (tags is None or not bool(torch.isin(tags//cache.block_size,
                                                         torch.tensor(pages, device=tags.device)).any()))
                row['speculative_tails_invalid'] = not any(p in cache._speculative_tails for p in pages)
            records.append(row)
        observer.emit('reset_content', connector, scheduled_request_ids=requests,
                      components=records, observer_fence=True,
                      boundary='native_zeroer_return_then_diagnostic_cuda_synchronize')


def canonical_identity(row):
    """Compare complete logical content and alias geometry, never device addresses."""
    aliases, tensors = {}, []
    for value in sorted(row['tensors'], key=lambda r: r['name']):
        aliases.setdefault(value['storage'], value['name'])
        tensors.append({k: value[k] for k in ('name', 'kind', 'shape', 'dtype', 'stride',
                                             'storage_offset', 'persistent', 'bytes', 'sha256')} |
                       {'alias': aliases[value['storage']]})
    return {'tensors': tensors, 'scalar_attributes': row['scalar_attributes']}


def components(observer, connector):
    capacity, _ = observer.frozen_helpers()
    storages, tensors, journals = {}, [], {}
    context = connector.vllm_config.compilation_config.static_forward_context
    for row in observer.connector_states[connector]['registrations']:
        value = connector.device_kv_caches[row['registration_name']]
        tensors.append(dict(layer=row['layer'], group_id=row['group_id'], component=row['component'],
                            tensor=capacity.storage_record(row['registration_name'], row['component'], value, storages)))
        if row['component'] != 'summary_mean':
            continue
        cache = context[row['registration_name']]._cache
        journals[row['layer']] = cache.request_journal_width
        names = ['numerator', 'denominator', 'maximum', 'last_slot']
        if cache.request_journal_width:
            names += ['request_journal_values', 'request_journal_last', 'request_journal_tag']
        for name in names:
            tensors.append(dict(layer=row['layer'], group_id=row['group_id'], component=name,
                tensor=capacity.storage_record(row['registration_name']+'.'+name, name, getattr(cache, name), storages)))
    observer.emit('component_inventory', connector, tensors=tensors,
                  request_journal_widths=journals, unique_resident_storage_bytes=sum(storages.values()),
                  graph_padding_is_not_allocatable_capacity=True, boundary='native_cache_registration_return')
