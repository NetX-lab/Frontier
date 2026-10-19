"""Owned cache-range and resident-scale witnesses within the native device observer.

This child supplies layouts and ordered operations to DeviceObservation. It has
no writer, transport, synchronization, reservation, or native execution entry.
CPU assignments choose candidate ranges; the existing drain checks those ranges
against actual device inputs before any content witness can be qualified.
"""
import copy
import hashlib


STATE_FIELDS = ('raw', 'closed_summary', 'partial_summary', 'live_tail')
SCALE_PARAMETERS = ('w13_weight_scale_inv', 'w2_weight_scale_inv')
SCALE_COUNTS = ('finite', 'zero', 'negative')
SUMMARY_TENSORS = ('_mean', 'mean', 'numerator', 'denominator', 'maximum', 'last_slot')


def finite_count(torch, source, absolute, finite, counter):
    """Reduce every source element using the caller's already owned scratch."""
    shape = tuple(source.shape)
    elements = 1
    for extent in shape:
        elements *= extent
    values = absolute.view(-1)[:elements].view(shape)
    mask = finite.view(-1)[:elements].view(shape)
    if source.dtype == values.dtype:
        torch.abs(source, out=values)
    else:
        values.copy_(source, non_blocking=True)
        torch.abs(values, out=values)
    torch.lt(values, float('inf'), out=mask)
    torch.sum(mask, dim=tuple(range(len(shape))), dtype=torch.int64, out=counter)


def dtype_bytes(torch, dtype):
    if dtype.is_floating_point:
        return torch.finfo(dtype).bits//8
    return {torch.uint8: 1, torch.int32: 4, torch.int64: 8, torch.bool: 1}[dtype]


def tensor_bytes(torch, tensor):
    elements = 1
    for extent in tensor.shape:
        elements *= extent
    return elements*dtype_bytes(torch, tensor.dtype)


class StateObservation:
    """Inspect the explicitly bound packed KV and inverse-scale owners only."""
    def __init__(self, runner, sites, limits, metadata_limits, describe, scratch_elements):
        self.runner, self.sites, self.torch = runner, sites, sites.torch
        self.limits, self.describe = limits, describe
        self.scratch_elements = scratch_elements
        self.owners, self.scales, self.gaps, self.assignments = [], [], [], {}
        self.step_layout, self.once_layout, self.device_layout = {}, {}, {}
        self.scale_submitted, self.summary_count = False, 0
        context = getattr(getattr(runner, 'compilation_config', None), 'static_forward_context', None)
        if context is None:
            self.gaps.append(dict(reason='UNVERIFIED_NATIVE_CACHE_CONTEXT_UNAVAILABLE'))
            return
        summary_layers = {}
        for name, layer in context.items():
            main = getattr(layer, 'main_layer_name', None)
            if main is not None:
                if main in summary_layers:
                    raise ValueError('More than one native summary owner targets a KV layer')
                summary_layers[main] = (name, layer)
        raw_bytes, side_bytes, summary_index = 0, 0, 0
        for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
            grouped_specs = getattr(group.kv_cache_spec, 'kv_cache_specs', None)
            for name in group.layer_names:
                if len(self.owners)+len(self.gaps) >= metadata_limits.tracked_entries:
                    raise ValueError('Cache content owners exceed the existing entry cap')
                layer, summary = context.get(name), summary_layers.get(name)
                spec = grouped_specs[name] if grouped_specs is not None else group.kv_cache_spec
                raw = getattr(layer, 'kv_cache', None)
                if (not isinstance(raw, sites.tensor_type) or len(raw.shape) != 4 or
                        raw.device != runner.device or raw.shape[2] != runner.block_tables.block_sizes[group_id] or
                        raw.dtype not in (self.torch.bfloat16, self.torch.float16, self.torch.float32,
                                          self.torch.float8_e4m3fn, self.torch.uint8) or
                        any(extent <= 0 for extent in raw.shape) or raw.stride()[-1] != 1):
                    self.gaps.append(dict(layer=name, reason='UNVERIFIED_NATIVE_PACKED_KV_LAYOUT'))
                    continue
                cache_dtype = getattr(layer, 'kv_cache_dtype', None)
                if raw.dtype in (self.torch.uint8, self.torch.float8_e4m3fn) and cache_dtype not in ('fp8', 'fp8_e4m3'):
                    self.gaps.append(dict(layer=name, reason='UNVERIFIED_NATIVE_FP8_KV_INTERPRETATION'))
                    continue
                window = getattr(spec, 'sliding_window', None)
                if summary is None and (type(window) is not int or window <= 0):
                    self.gaps.append(dict(layer=name, reason='UNVERIFIED_CACHE_RANGE_FAMILY'))
                    continue
                if raw.shape[1]*raw.shape[3] > scratch_elements:
                    raise ValueError('One native raw KV token exceeds prepared reduction scratch')
                owner = dict(layer=name, layer_owner=layer, group_id=group_id, raw=raw,
                    family='DSA' if summary is not None else 'SWA', window=window,
                    raw_offset=raw_bytes, raw_bytes=raw.shape[1]*raw.shape[3]*dtype_bytes(self.torch, raw.dtype),
                    metadata=dict(layer=name, group_id=group_id, cache=self.describe(raw),
                        spec_type=type(spec).__name__, sliding_window=window,
                        extra_retained_tokens=getattr(spec, 'extra_retained_tokens', 0), kv_cache_dtype=cache_dtype))
                raw_bytes += owner['raw_bytes']
                if summary is not None:
                    summary_name, summary_layer = summary
                    cache = summary_layer._cache
                    if (cache is None or cache.request_journal_width or
                            summary_layer._target_impl.summary_cache is not cache):
                        raise ValueError('The native summary cache must be initialized with speculation off')
                    mean, numerator, denominator, maximum = (getattr(cache, key) for key in
                                                              ('mean', 'numerator', 'denominator', 'maximum'))
                    if (mean.device != runner.device or mean.dtype != self.torch.uint8 or
                            len(mean.shape) != 3 or mean.shape[0] != raw.shape[0] or mean.shape[1] <= 0 or
                            raw.shape[2] % mean.shape[1] or
                            any(tensor.dtype != self.torch.float32 or tensor.device != runner.device or
                                tensor.shape[0] < raw.shape[0] for tensor in (numerator, denominator, maximum)) or
                            cache.last_slot.dtype != self.torch.int64 or cache.last_slot.device != runner.device):
                        raise ValueError('Actual native summary/accumulator layouts do not match packed KV')
                    if mean.shape[2] > scratch_elements:
                        raise ValueError('One native summary row exceeds prepared reduction scratch')
                    region_size = raw.shape[2]//mean.shape[1]
                    if (cache.block_size != raw.shape[2] or cache.pages != raw.shape[0] or
                            tuple(numerator.shape[1:]) != (mean.shape[2],) or
                            tuple(denominator.shape) != tuple(numerator.shape) or
                            tuple(maximum.shape) != (numerator.shape[0], 1) or
                            tuple(cache.last_slot.shape) != (numerator.shape[0],) or
                            tuple(cache._mean.shape) != (numerator.shape[0], mean.shape[1], mean.shape[2])):
                        raise ValueError('Native summary/accumulator dimensions disagree')
                    owner.update(summary=cache, summary_layer=summary_layer, summary_name=summary_name, summary_index=summary_index,
                        region_size=region_size, side_offset=side_bytes,
                        summary_bytes=mean.shape[2], tail_bytes=sum(tensor_bytes(self.torch, tensor[0])
                            for tensor in (numerator, denominator, maximum)))
                    side_bytes += 2*owner['summary_bytes']+owner['tail_bytes']
                    summary_index += 1
                    owner['metadata'].update(summary_owner=summary_name, region_size=region_size,
                        summary=self.describe(mean), numerator=self.describe(numerator),
                        denominator=self.describe(denominator), maximum=self.describe(maximum),
                        last_slot=self.describe(cache.last_slot), scratch_pages=cache._mean.shape[0]-raw.shape[0])
                    owner['summary_layouts'] = {key: self.describe(getattr(cache, key)) for key in SUMMARY_TENSORS}
                    owner['summary_transfer_layout'] = self.describe(summary_layer.kv_cache)
                self.owners.append(owner)
        self.summary_count = summary_index
        if self.owners:
            requests, owners = runner.max_num_reqs, len(self.owners)
            self.step_layout = dict(state_counts=((requests, owners, len(STATE_FIELDS)), self.torch.int64),
                state_raw_samples=((requests, raw_bytes), self.torch.uint8))
            if summary_index:
                self.step_layout.update(state_side_samples=((requests, side_bytes), self.torch.uint8),
                    state_markers=((requests, summary_index), self.torch.int64))
            self.device_layout['state_counts'] = ((limits.steps, requests, owners, len(STATE_FIELDS)), self.torch.int64)
            self.device_layout['state_count_scratch'] = ((), self.torch.int64)
        # The original census has already bound these resident module objects.
        for reference, binding in getattr(sites, 'census_owners', {}).values():
            owner = reference()
            if owner is None or 'routed_modular' not in binding['sites']:
                continue
            for name in SCALE_PARAMETERS:
                scale = getattr(owner, name, None)
                if scale is None:
                    continue
                if len(self.scales) >= metadata_limits.tracked_entries:
                    raise ValueError('Resident inverse scales exceed the existing entry cap')
                if (not isinstance(scale, sites.tensor_type) or scale.device != runner.device or
                        scale.dtype != self.torch.float32 or len(scale.shape) < 2 or
                        any(extent <= 0 for extent in scale.shape)):
                    raise ValueError('Native resident inverse-scale layout is unsupported')
                width = 1
                for extent in scale.shape[1:]:
                    width *= extent
                if width > scratch_elements:
                    raise ValueError('One resident expert scale exceeds the prepared reduction scratch')
                rows = min(limits.scan_rows, max(1, scratch_elements//width))
                self.scales.append(dict(owner=owner, name=name, tensor=scale, width=width, rows=rows,
                    chunks=(scale.shape[0]+rows-1)//rows,
                    metadata=dict(module_path=binding['module_path'], layer_id=binding['layer_id'],
                                  parameter=name, tensor=self.describe(scale))))
        if self.scales:
            chunks = sum(row['chunks'] for row in self.scales)
            samples = sum(row['width']*dtype_bytes(self.torch, row['tensor'].dtype) for row in self.scales)
            self.once_layout = dict(scale_counts=((chunks, len(SCALE_COUNTS)), self.torch.int64),
                scale_extrema=((chunks, 2), self.torch.float32), scale_samples=((samples,), self.torch.uint8))
            self.device_layout.update(scale_counts=self.once_layout['scale_counts'],
                                      scale_extrema=self.once_layout['scale_extrema'])
        elif self.owners:
            self.gaps.append(dict(reason='UNVERIFIED_RESIDENT_INVERSE_SCALE_BINDINGS_UNAVAILABLE'))
        if self.owners or self.scales:
            self.device_layout['state_absolute'] = ((scratch_elements,), self.torch.float32)

    def byte_count(self, layout):
        total = 0
        for shape, dtype in layout.values():
            elements = 1
            for extent in shape:
                elements *= extent
            total += elements*dtype_bytes(self.torch, dtype)
        return total

    def allocate(self, absolute, finite):
        """Allocate only after DeviceObservation has reserved the complete layout."""
        self.once = {name: self.torch.empty(shape, dtype=dtype, device='cpu', pin_memory=True)
                     for name, (shape, dtype) in self.once_layout.items()}
        self.device = {}
        if self.device_layout:
            self.device['state_finite'] = finite.view(-1)
        for name, (shape, dtype) in self.device_layout.items():
            if name == 'state_absolute' and absolute.dtype == dtype:
                self.device[name] = absolute.view(-1)
            else:
                self.device[name] = self.torch.empty(shape, dtype=dtype, device=self.runner.device)

    def inventory(self):
        return dict(status='CANDIDATE_RANGES_AND_RESIDENT_SCALES_NOT_NATIVE_QUALIFICATION',
            owners=[dict(owner['metadata'], family=owner['family']) for owner in self.owners],
            resident_scales=[scale['metadata'] for scale in self.scales], gaps=copy.deepcopy(self.gaps),
            transient_activation_scales='UNMEASURED_FULL_REPLAY_BYPASSES_PYTHON',
            raw_writer_and_physical_kernel_completion='UNMEASURED',
            unclassified_allocator_history='UNMEASURED')

    def assignment(self, row):
        if row['outcome'] != 'RETURNED_CPU_CALL_NOT_GPU_COMPLETION':
            return
        name, arguments = row['site'], row['arguments']
        owner = self.runner.req_states if name == 'request_state_remove' else self.runner.block_tables
        if row['owner_binding'] != 'RESIDENT_BOUND_OWNER' or row['owner_id'] != id(owner):
            raise ValueError('Lifecycle observation does not belong to the current native state owner')
        if name == 'request_state_remove':
            self.assignments.pop(arguments['req_id'], None)
        elif name == 'block_append':
            index = arguments['req_index']
            matches = [item['req_id'] for item in row['lifecycle_after']['request_indices']
                       if item['persistent_index'] == index]
            if len(matches) != 1:
                raise ValueError('A returned block assignment has no unique original request')
            req = matches[0]
            blocks = arguments['new_block_ids']
            if len(blocks) != len(self.runner.kv_cache_config.kv_cache_groups):
                raise ValueError('Observed block assignment omits a native KV group')
            if arguments['overwrite']:
                current = [[] for _ in blocks]
            else:
                current = self.assignments.get(req)
                if current is None:
                    raise ValueError('Append has no observed original block assignment')
                current = copy.deepcopy(current)
            for group, ids in enumerate(blocks):
                if type(ids) is not list or any(type(page) is not int or page < 0 for page in ids):
                    raise ValueError('Observed native page assignment must contain nonnegative integer IDs')
                current[group].extend(ids)
                if len(current[group]) != row['lifecycle_after']['block_counts_cpu'][group][index]:
                    raise ValueError('Returned native assignment and CPU active block count disagree')
            if len(current) != len(self.runner.kv_cache_config.kv_cache_groups) or any(
                    len(ids) > self.runner.block_tables.input_block_tables[group].shape[1]
                    for group, ids in enumerate(current)):
                raise ValueError('Observed native assignment exceeds prepared table dimensions')
            self.assignments[req] = current

    def select(self, entry):
        """Detach a candidate CPU plan without reading any prepared device value."""
        selected = []
        for row in entry['request_frontiers']:
            req = row['req_id']
            blocks = self.assignments.get(req)
            if blocks is None:
                selected.append(dict(req_id=req, status='UNVERIFIED_NO_OBSERVED_BLOCK_ASSIGNMENT'))
                continue
            start, count = row['num_computed_tokens_np'], row['num_scheduled_tokens']
            if type(start) is not int or type(count) is not int or start < 0 or count <= 0 or count != row['query_end']-row['query_start']:
                raise ValueError('Candidate range has an incomplete native CPU frontier')
            selected.append(dict(req_id=req, persistent_index=row['idx_mapping_np'],
                start=start, end=start+count, query_start=row['query_start'], query_end=row['query_end'],
                blocks=copy.deepcopy(blocks), status='CPU_CANDIDATE_REQUIRES_DEVICE_INPUT_JOIN'))
        entry['state_selection'] = selected

    def count(self, source, counter):
        finite_count(self.torch, source, self.device['state_absolute'], self.device['state_finite'],
                     self.device['state_count_scratch'])
        self.torch.add(counter, self.device['state_count_scratch'], out=counter)

    def raw_ranges(self, owner, blocks, start, end):
        raw, block_size = owner['raw'], owner['raw'].shape[2]
        per_row = raw.shape[1]*raw.shape[3]
        rows = min(self.limits.scan_rows, self.scratch_elements//per_row)
        if rows < 1:
            raise ValueError('One actual raw KV token exceeds prepared reduction scratch')
        while start < end:
            logical_block, offset = divmod(start, block_size)
            if logical_block >= len(blocks):
                raise ValueError('Candidate range exceeds the observed active table prefix')
            page = blocks[logical_block]
            if not 0 <= page < raw.shape[0]:
                raise ValueError('Candidate physical page is outside actual native KV capacity')
            count = min(rows, block_size-offset, end-start)
            source = raw[page, :, offset:offset+count, :]
            if raw.dtype == self.torch.uint8:
                source = source.view(self.torch.float8_e4m3fn)
            yield source
            start += count

    def copy_bytes(self, destination, offset, tensor):
        raw = tensor.view(self.torch.uint8)
        count = tensor_bytes(self.torch, tensor)
        destination[offset:offset+count].view(tuple(raw.shape)).copy_(raw, non_blocking=True)
        return offset+count

    def check_owner(self, owner):
        context = self.runner.compilation_config.static_forward_context
        layer = context.get(owner['layer'])
        if layer is not owner['layer_owner'] or self.describe(layer.kv_cache) != owner['metadata']['cache']:
            raise ValueError('Native cache owner changed after content preparation')
        if owner['family'] == 'DSA':
            summary_layer = context.get(owner['summary_name'])
            if (summary_layer is not owner['summary_layer'] or summary_layer._cache is not owner['summary'] or
                    summary_layer.main_layer_name != owner['layer'] or
                    summary_layer._target_impl.summary_cache is not owner['summary'] or
                    self.describe(summary_layer.kv_cache) != owner['summary_transfer_layout'] or
                    any(self.describe(getattr(summary_layer._cache, key)) != layout
                        for key, layout in owner['summary_layouts'].items())):
                raise ValueError('Native summary owner changed after content preparation')

    def scan(self, entry, destinations):
        index = entry['index']
        if not self.owners:
            entry['state_content_status'] = 'UNVERIFIED_NO_SUPPORTED_CACHE_CONTENT_OWNERS'
            return
        counts = self.device['state_counts'][index]
        counts.zero_()
        coverage = []
        for request_index, selection in enumerate(entry['state_selection']):
            if selection['status'].startswith('UNVERIFIED'):
                continue
            length = selection['end']
            for owner_index, owner in enumerate(self.owners):
                self.check_owner(owner)
                blocks = selection['blocks'][owner['group_id']]
                start = 0 if owner['family'] == 'DSA' else max(0, length-owner['window'])
                raw = owner['raw']
                expected = dict.fromkeys(STATE_FIELDS, 0)
                for source in self.raw_ranges(owner, blocks, start, length):
                    self.count(source, counts[request_index, owner_index, 0].view(()))
                    expected['raw'] += tensor_bytes(self.torch, source)//dtype_bytes(self.torch, source.dtype)
                page, offset = blocks[(length-1)//raw.shape[2]], (length-1)%raw.shape[2]
                sample = raw[page, :, offset, :]
                self.copy_bytes(destinations['state_raw_samples'][request_index], owner['raw_offset'], sample)
                row = dict(owner_index=owner_index, request_index=request_index, req_id=selection['req_id'],
                    layer=owner['layer'], group_id=owner['group_id'], family=owner['family'],
                    logical_raw_range=[start, length], raw_sample_position=length-1,
                    raw_sample_page=page, raw_sample_offset=offset, expected=expected)
                if owner['family'] == 'DSA':
                    cache, region = owner['summary'], owner['region_size']
                    closed, opened = divmod(length, region)
                    block_size, regions_per_page = raw.shape[2], cache.mean.shape[1]
                    first = 0
                    while first < closed:
                        logical_block, local = divmod(first, regions_per_page)
                        current_page = blocks[logical_block]
                        if not 0 <= current_page < raw.shape[0]:
                            raise ValueError('Closed summary candidate is outside native page capacity')
                        count = min(closed-first, regions_per_page-local, self.limits.scan_rows,
                                    self.scratch_elements//cache.mean.shape[2])
                        source = cache.mean[current_page, local:local+count].view(self.torch.float8_e4m3fn)
                        self.count(source, counts[request_index, owner_index, 1].view(()))
                        expected['closed_summary'] += count*cache.mean.shape[2]
                        first += count
                    side = destinations['state_side_samples'][request_index]
                    sample_offset = owner['side_offset']
                    if closed:
                        closed_page, local = divmod(closed-1, regions_per_page)
                        self.copy_bytes(side, sample_offset, cache.mean[blocks[closed_page], local])
                    sample_offset += owner['summary_bytes']
                    if opened:
                        local = offset//region
                        source = cache.mean[page, local:local+1].view(self.torch.float8_e4m3fn)
                        self.count(source, counts[request_index, owner_index, 2].view(()))
                        expected['partial_summary'] = cache.mean.shape[2]
                        self.copy_bytes(side, sample_offset, cache.mean[page, local])
                        sample_offset += owner['summary_bytes']
                        for tensor in (cache.numerator, cache.denominator, cache.maximum):
                            source = tensor[page:page+1]
                            self.count(source, counts[request_index, owner_index, 3].view(()))
                            expected['live_tail'] += tensor_bytes(self.torch, source)//dtype_bytes(self.torch, source.dtype)
                            sample_offset = self.copy_bytes(side, sample_offset, tensor[page])
                    destinations['state_markers'][request_index, owner['summary_index']:owner['summary_index']+1].copy_(
                        cache.last_slot[page:page+1], non_blocking=True)
                    row.update(closed_summary_rows=closed, open_tail_tokens=opened,
                        expected_last_slot=page*block_size+offset if opened else -1,
                        summary_fp8_interpretation='UINT8_BIT_VIEW_AS_FLOAT8_E4M3FN',
                        inactive_and_released_accumulators='EXCLUDED_NOT_CURRENT_STATE')
                coverage.append(row)
        destinations['state_counts'].copy_(counts, non_blocking=True)
        entry['state_coverage'] = coverage
        entry['state_content_status'] = 'SUBMITTED_CANDIDATE_RANGES_NOT_COMPLETED_OR_WRITE_PROOF'

    def scan_scales(self, entry):
        if self.scale_submitted or not self.scales:
            return
        offset, sample_offset, ranges = 0, 0, []
        for scale in self.scales:
            tensor = getattr(scale['owner'], scale['name'])
            if self.describe(tensor) != scale['metadata']['tensor']:
                raise ValueError('Resident scale owner changed after content preparation')
            first = offset
            for start in range(0, tensor.shape[0], scale['rows']):
                source = tensor[start:start+scale['rows']]
                values = self.device['state_absolute'][:tensor_bytes(self.torch, source)//4].view(tuple(source.shape))
                mask = self.device['state_finite'][:tensor_bytes(self.torch, source)//4].view(tuple(source.shape))
                values.copy_(source, non_blocking=True)
                for column, value in ((1, 0), (2, 0)):
                    (self.torch.eq if column == 1 else self.torch.lt)(values, value, out=mask)
                    self.torch.sum(mask, dim=tuple(range(len(source.shape))), dtype=self.torch.int64,
                                   out=self.device['scale_counts'][offset, column].view(()))
                self.torch.amin(values, out=self.device['scale_extrema'][offset, 0].view(()))
                self.torch.amax(values, out=self.device['scale_extrema'][offset, 1].view(()))
                finite_count(self.torch, source, self.device['state_absolute'], self.device['state_finite'],
                             self.device['scale_counts'][offset, 0].view(()))
                offset += 1
            # Contiguous owned FP32 scratch also handles noncontiguous expert views.
            sample = self.device['state_absolute'][:scale['width']].view(tuple(tensor.shape[1:]))
            sample.copy_(tensor[0], non_blocking=True)
            sample_offset = self.copy_bytes(self.once['scale_samples'], sample_offset, sample)
            ranges.append(dict(scale['metadata'], first_chunk=first, chunks=offset-first,
                elements=tensor_bytes(self.torch, tensor)//4, sample_offset=sample_offset-scale['width']*4,
                sample_bytes=scale['width']*4, sampled_expert_index=0,
                sample_policy='FIRST_RESIDENT_EXPERT_NOT_ALL_EXECUTED_EXPERTS'))
        self.once['scale_counts'].copy_(self.device['scale_counts'], non_blocking=True)
        self.once['scale_extrema'].copy_(self.device['scale_extrema'], non_blocking=True)
        entry['resident_scale_scan'] = ranges
        self.scale_submitted = True

    def selection_matches(self, selection, values):
        if selection['status'].startswith('UNVERIFIED'):
            return False
        indices = values['idx_mapping']
        if indices.count(selection['persistent_index']) != 1:
            return False
        request_index = indices.index(selection['persistent_index'])
        start, end = selection['query_start'], selection['query_end']
        if (values['positions'][start:end] != list(range(selection['start'], selection['end'])) or
                values['final_positions'][start:end] != values['positions'][start:end] or
                any(values['is_padding'][start:end]) or values['seq_lens'][request_index] != selection['end'] or
                values['request_state'][request_index] != selection['persistent_index'] or
                values['query_start_loc'][request_index:request_index+2] != [start, end]):
            return False
        for group, blocks in enumerate(selection['blocks']):
            count = values['block_counts'][group][selection['persistent_index']]
            table = values['block_table_'+str(group)][request_index]
            size = self.runner.block_tables.block_sizes[group]
            slots = values['slot_mapping_'+str(group)][start:end]
            expected = [blocks[position//size]*size+position%size for position in
                        range(selection['start'], selection['end'])]
            if count != len(blocks) or table[:count] != blocks or slots != expected:
                return False
        return True

    @staticmethod
    def raw_sample(tensor, offset, count, shape, dtype):
        raw = tensor[offset:offset+count].numpy().tobytes()
        return dict(shape=list(shape), dtype=dtype, raw_hex=raw.hex(), sha256=hashlib.sha256(raw).hexdigest())

    def drain(self, entry, destinations, values):
        rows, samples = [], []
        counts = destinations['state_counts'].numpy().tolist() if self.owners else []
        markers = destinations['state_markers'].numpy().tolist() if self.summary_count else []
        matched = [self.selection_matches(selection, values) for selection in entry.get('state_selection', [])]
        for coverage in entry.get('state_coverage', []):
            row = copy.deepcopy(coverage)
            request, owner_index = row['request_index'], row['owner_index']
            owner = self.owners[owner_index]
            observed = dict(zip(STATE_FIELDS, counts[request][owner_index]))
            row.update(observed=observed, device_inputs_match_candidate_range=matched[request],
                status='OBSERVED_FINITE_CANDIDATE_CONTENT_NOT_NATIVE_WRITE_PROOF' if
                    matched[request] and observed == row['expected'] else 'UNVERIFIED_CONTENT_OR_RANGE_JOIN')
            sample = self.raw_sample(destinations['state_raw_samples'][request], owner['raw_offset'],
                owner['raw_bytes'], (owner['raw'].shape[1], owner['raw'].shape[3]), str(owner['raw'].dtype))
            samples.append(dict(layer=owner['layer'], req_id=row['req_id'], kind='raw_current_tail', **sample))
            if owner['family'] == 'DSA':
                row['observed_last_slot'] = markers[request][owner['summary_index']]
                if row['observed_last_slot'] != row['expected_last_slot']:
                    row['status'] = 'UNVERIFIED_LIVE_TAIL_OR_CLOSURE_MARKER'
                offset = owner['side_offset']
                if row['closed_summary_rows']:
                    sample = self.raw_sample(destinations['state_side_samples'][request], offset,
                        owner['summary_bytes'], (owner['summary_bytes'],), 'torch.uint8; bit view torch.float8_e4m3fn')
                    samples.append(dict(layer=owner['layer'], req_id=row['req_id'], kind='last_closed_summary', **sample))
                offset += owner['summary_bytes']
                if row['open_tail_tokens']:
                    for kind, count in (('partial_summary', owner['summary_bytes']), ('live_tail', owner['tail_bytes'])):
                        sample = self.raw_sample(destinations['state_side_samples'][request], offset,
                            count, (count,), 'native storage bytes')
                        samples.append(dict(layer=owner['layer'], req_id=row['req_id'], kind=kind, **sample))
                        offset += count
            rows.append(row)
        scale_rows = []
        if 'resident_scale_scan' in entry:
            scale_counts, extrema = self.once['scale_counts'].numpy(), self.once['scale_extrema'].numpy()
            for selected in entry['resident_scale_scan']:
                first, end = selected['first_chunk'], selected['first_chunk']+selected['chunks']
                observed = dict(zip(SCALE_COUNTS, scale_counts[first:end].sum(axis=0).tolist()))
                minimum, maximum = extrema[first:end, 0].min(), extrema[first:end, 1].max()
                # Nonfinite statistics retain bytes and an explicit verdict, not JSON NaN.
                bounds = dict(minimum=float(minimum), maximum=float(maximum)) if observed['finite'] == selected['elements'] else dict(
                    minimum=None, maximum=None, extrema_raw_hex=extrema[first:end].tobytes().hex())
                sample = self.raw_sample(self.once['scale_samples'], selected['sample_offset'],
                    selected['sample_bytes'], selected['tensor']['shape'][1:], selected['tensor']['dtype'])
                scale_rows.append(dict(selected, observed=observed, **bounds, sample=sample,
                    status='OBSERVED_FINITE_RESIDENT_SCALE_NOT_TRANSIENT_OR_CONSUMPTION_PROOF' if
                        observed['finite'] == selected['elements'] else 'UNVERIFIED_NONFINITE_RESIDENT_SCALE'))
        return dict(status='UNVERIFIED' if self.gaps or not rows or not all(matched) or
                    any(row['status'].startswith('UNVERIFIED') for row in rows+scale_rows) else
                    'OBSERVED_APPLICABLE_CONTENT_ONLY_NOT_NATIVE_WRITE_OR_PHYSICAL_QUALIFICATION',
            rows=rows, samples=samples, resident_scales=scale_rows, gaps=copy.deepcopy(self.gaps),
            completion='OWNED_COPY_EVENT_QUERY_READY_AFTER_NATIVE_GENERATE',
            full_resident_parameter_finite_census='UNMEASURED', transient_activation_scales='UNMEASURED',
            raw_writer_and_capture_to_replay_physical_join='UNMEASURED')

    @staticmethod
    def compact(record):
        result = copy.deepcopy(record)
        for row in result['samples']:
            row.pop('raw_hex', None)
        for row in result['resident_scales']:
            row['sample'].pop('raw_hex', None)
        return result
