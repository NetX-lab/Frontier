"""Consume actual canonical native reference, alias, reset and pool ownership records."""

from collections import defaultdict
import json

try:
    from .r2_reference import canonical_identity
except ImportError:
    from r2_reference import canonical_identity


def resident(records, observation, check):
    startup = {}
    for row in records:
        if row['event'] != 'resident_identity':
            continue
        key = row['role'], row['dp_rank'], row['tp_rank']
        check.require(row.get('complete') is True and row.get('tensors'), f'{key}: incomplete resident identity')
        check.expect(row['peak_chunk_bytes'] <= observation['resident_identity']['chunk_bytes'],
                     f'{key}: resident copy exceeded the frozen chunk budget')
        value = canonical_identity(row)
        if row['phase'] == 'startup':
            check.expect(key not in startup, f'{key}: duplicate startup identity')
            startup[key] = value
        else:
            check.require(key in startup, f'{key}: no same-worker startup identity')
            check.require(startup.get(key) == value,
                          f'{key}: complete resident identity changed before independent execution')
    required = {(role, dp, tp) for role in ('prefill', 'decode') for dp in range(2) for tp in range(8)}
    check.require(set(startup) == required, 'Incomplete32-worker resident parameter/buffer/scales identity')
    for dp in range(2):
        for tp in range(8):
            check.require(startup.get(('prefill', dp, tp)) == startup.get(('decode', dp, tp)),
                          f'DP{dp}/TP{tp}: full P/D resident contents or alias geometry differ')
    check.details.update(startup_workers=len(startup), comparison='complete contents and alias geometry; no sampled/equal-seed proof')


def token_ids(row):
    return [value for chunk in row.get('token_id_deltas', ()) for value in chunk.get('token_ids', ())]


def logical_state(row):
    return {(item['group_id'], item['layer'], item['component'], item['logical_page']):
            (item['valid_units'], item['sample']['dtype'], item['sample']['shape'], item['sample']['sha256'])
            for item in row['logical_components']}


def checkpoint_census(value, registration, block, region, check):
    specs = {row['group_id']: row for row in registration['group_specs']}
    expected = set()
    for layer in registration['expected_layers']:
        group = layer['group_id']
        window = specs[group]['sliding_window']
        end = value['num_computed_tokens']
        start = max(0, end-window) if window is not None else 0
        for component in layer['components']:
            for page in range(start//block, (end+block-1)//block):
                expected.add((group, layer['layer'], component, page))
    check.require(set(logical_state(value)) == expected, 'Independent native checkpoint lacks complete logical component/page census')
    check.expect(len(logical_state(value)) == len(value['logical_components']), 'Duplicated logical component/page state')
    for item in value['logical_components']:
        units = block if item['component'] == 'raw_kv' else block//region
        check.expect(item['native_units_per_page'] == units and item['unit_shape']
                     and all(type(v) is int and v > 0 for v in item['unit_shape']), 'Unsupported native logical component geometry')
        check.expect(item['sample']['shape'] == [item['valid_units'][1]-item['valid_units'][0]]+item['unit_shape'],
                     'Logical native snapshot contains padding or omitted valid elements')


def first_tail_errors(a, b):
    import torch
    try:
        from compare_attention_state import errors
    except ModuleNotFoundError:
        from tests.integration.step5_environment.compare_attention_state import errors
    result = {}
    for name in ('numerator', 'denominator', 'maximum'):
        actual = torch.frombuffer(bytearray.fromhex(a['exact_tensors'][name]['raw_hex']), dtype=torch.float32)
        expected = torch.frombuffer(bytearray.fromhex(b['exact_tensors'][name]['raw_hex']), dtype=torch.float32)
        result[name] = errors(actual, expected)
    return result


def active_tail(row, block):
    """G1/native reset contract: last_slot=-1 makes old accumulators inactive."""
    result = []
    for page in row['pages']:
        slot = page['last_slot']
        if slot < 0:
            continue
        if slot//block != page['block_id']:
            raise ValueError('Active last_slot does not belong to its observed physical page')
        result.append((slot % block, {name: sample['sha256']
            for name, sample in page['exact_tensors'].items() if name != 'last_slot'}))
    return result


def journal(row, block):
    """Normalize valid native physical journal slots through the checkpoint map."""
    import struct

    value = row['journal']
    if value['width'] == 0:
        return []
    if value['status'] != 'OBSERVED_EXACT_BYTES':
        raise ValueError('Independent reference journal bytes unavailable')
    tensors = value['tensors']
    tags_raw = bytes.fromhex(tensors['request_journal_tag']['raw_hex'])
    last_raw = bytes.fromhex(tensors['request_journal_last']['raw_hex'])
    payload = bytes.fromhex(tensors['request_journal_values']['raw_hex'])
    width = value['width']
    if len(tags_raw) != width*8 or len(last_raw) != width*8 or len(payload) % width:
        raise ValueError('Native request journal dtype/width mismatch')
    tags, last = struct.unpack('<'+'q'*width, tags_raw), struct.unpack('<'+'q'*width, last_raw)
    stride = len(payload)//width
    # The logical page map is attached by reference() from independent native tables.
    mapping = row['_physical_to_logical']
    output = []
    for index, tag in enumerate(tags):
        if tag < 0:
            continue
        page, offset = divmod(tag, block)
        if page not in mapping:
            raise ValueError('Live journal tag points outside observed request ownership')
        previous = last[index]
        previous_logical = -1 if previous < 0 else mapping[previous//block]*block+previous % block
        output.append((mapping[page]*block+offset, previous_logical,
                       payload[index*stride:(index+1)*stride].hex()))
    return sorted(output)


def reference(stage, records, config, expected, clients, check):
    observation = config['pd_state_observation']
    refs = set(observation['local_reference_client_request_ids'])
    by_client = defaultdict(list)
    for row in records:
        if row.get('client_request_id'):
            by_client[row['client_request_id']].append(row)
    summary = json.loads((stage/'evidence/replay/r2_replay_summary.json').read_text())
    reference_by_case = {}
    for client in refs:
        trace = expected[client]
        rows = by_client[client]
        report = summary['waves'][trace['wave']]
        check.require(report['before']['verified'] and report['after']['verified'], f'{client}: independent execution lacks idle boundaries')
        check.expect(report['dispatch'] == 'direct_native_D_no_proxy_or_transfer_metadata', f'{client}: reference used PD/external route')
        delta = report.get('transfer_delta', {})
        sources = delta.get('decode', {}).get('prompt_tokens_by_source', {})
        for source, value in (('local_compute', trace['num_prefill_tokens']), ('local_cache_hit', 0), ('external_kv_transfer', 0)):
            check.require(sources.get(source, {}).get('total') == value, f'{client}: observed D prompt source {source} differs from full-local reference')
        p_sources = delta.get('prefill', {}).get('prompt_tokens_by_source', {})
        check.require(p_sources and all(value.get('total') == 0 for value in p_sources.values()), f'{client}: independent reference unexpectedly used P')
        client_rows = clients.get((trace['wave'], client), [])
        check.require(len(client_rows) == 1, f'{client}: missing unique reference client record')
        if len(client_rows) != 1:
            continue
        client_row = client_rows[0]
        check.require(len(token_ids(client_row)) == trace['num_decode_tokens'], f'{client}: incomplete real reference output IDs')
        starts = [row for row in rows if row['event'] == 'reference_start']
        check.require({row['tp_rank'] for row in starts} == set(range(8)), f'{client}: incomplete native reference prompt witness')
        check.expect(not any(row['event'].startswith('read_') or row['event'] == 'target_reserve' for row in rows),
                     f'{client}: independent reference touched external KV')
        for row in starts:
            check.expect(row['initial_computed_tokens'] == 0 and row['actual_prompt_tokens'] == trace['num_prefill_tokens']
                         and row['actual_prompt_sha256'] == client_row['prompt_sha256'], f'{client}: native reference reused cache or changed prompt IDs')
            phase = 'reference:'+client
            check.require(any(value['event'] == 'resident_identity' and value['phase'] == phase
                              and value['tp_rank'] == row['tp_rank'] for value in rows),
                          f'{client}: missing complete reference resident identity')
        allocations = [row for row in rows if row['event'] == 'block_allocate']
        check.require(allocations, f'{client}: no actual local allocation evidence')
        for row in allocations:
            check.expect(row.get('kv_transfer_params_present') is False and row.get('num_external_tokens') == 0
                         and row.get('num_local_cache_tokens') == 0, f'{client}: external/local-cache allocation in independent execution')
        for rank in range(8):
            compute = sorted((row for row in rows if row['event'] == 'target_compute' and row['tp_rank'] == rank),
                             key=lambda row: row['monotonic_ns'])
            end = 0
            for row in compute:
                check.expect(row['num_computed_tokens'] == end, f'{client}/TP{rank}: local compute is not independent/contiguous')
                end = row['num_computed_tokens']+row['num_scheduled_tokens']
            check.require(end == trace['num_prefill_tokens']+trace['num_decode_tokens']-1,
                          f'{client}/TP{rank}: missing full local prefill and native decode')
        reference_by_case[trace['num_prefill_tokens']] = (client_row, rows)
    compared = 0
    for client in observation['content_client_request_ids']:
        if client in refs:
            continue
        trace = expected[client]
        match = reference_by_case.get(trace['num_prefill_tokens'])
        check.require(match is not None, f'{client}: no case-aligned independent full-prefill execution')
        actual_clients = clients.get((trace['wave'], client), [])
        if match is None or len(actual_clients) != 1:
            check.require(False, f'{client}: missing client/reference pair')
            continue
        baseline, reference_rows = match
        actual = actual_clients[0]
        check.expect(actual['prompt_sha256'] == baseline['prompt_sha256'], f'{client}: reference prompt IDs changed')
        output = token_ids(actual)
        check.require(output and output == token_ids(baseline)[:len(output)],
                      f'{client}: native output differs; no numerical/batch-drift waiver exists')
        checkpoints = [row for row in by_client[client] if row['event'] == 'state_checkpoint']
        boundaries = {'first_local_tail_complete'} if trace.get('cancel_after_first_content') else {
            'first_local_tail_complete', 'first_decode_region_crossing', 'final_decode_input_complete'}
        for boundary in boundaries:
            for rank in range(8):
                pair = [[row for row in values if row['event'] == 'state_checkpoint' and row['boundary'] == boundary
                         and row['tp_rank'] == rank] for values in (by_client[client], reference_rows)]
                check.require(all(len(values) == 1 for values in pair), f'{client}/{boundary}/TP{rank}: missing unambiguous native state pair')
                if not all(len(values) == 1 for values in pair):
                    continue
                a, b = pair[0][0], pair[1][0]
                for value in (a, b):
                    registrations = [row for row in records if row['event'] == 'state_registration'
                        and (row['host'], row['tp_rank'], row['dp_rank']) == (value['host'], value['tp_rank'], value['dp_rank'])]
                    check.require(len(registrations) == 1, f'{client}: missing unique native checkpoint registration')
                    if len(registrations) == 1:
                        checkpoint_census(value, registrations[0], config['engine']['block_size'], observation['region_size'], check)
                check.expect(a['num_computed_tokens'] == b['num_computed_tokens'], f'{client}: compared different token positions')
                check.require(logical_state(a) == logical_state(b), f'{client}/{boundary}/TP{rank}: native logical state differs; no cross-batch error bound')
                for value, rows in ((a, by_client[client]), (b, reference_rows)):
                    tails = [row for row in rows if row['event'] == 'tail_state' and row['boundary'] == boundary and row['tp_rank'] == rank]
                    check.require(tails, f'{client}/{boundary}/TP{rank}: missing tail/journal state')
                    value['_tails'] = {row['layer']: row for row in tails}
                    for tail in tails:
                        tail['_physical_to_logical'] = {item['physical_page']: item['logical_page']
                            for item in value['logical_components'] if item['layer'] == tail['layer'] and item['component'] == 'summary_mean'}
                check.require(set(a['_tails']) == set(b['_tails']), f'{client}: tail layer census differs')
                for layer in a['_tails'].keys() & b['_tails'].keys():
                    x, y = a['_tails'][layer], b['_tails'][layer]
                    check.require(active_tail(x, config['engine']['block_size']) == active_tail(y, config['engine']['block_size']),
                                  f'{client}/{layer}: active native tail differs; no new tolerance')
                    if (active_tail(x, config['engine']['block_size']) != active_tail(y, config['engine']['block_size'])
                            and 'g1_first_tail_difference' not in check.details):
                        xp = next((p for p in x['pages'] if p['last_slot'] >= 0), None)
                        yp = next((p for p in y['pages'] if p['last_slot'] >= 0), None)
                        if xp is not None and yp is not None:
                            check.details['g1_first_tail_difference'] = first_tail_errors(xp, yp)
                    check.require(journal(x, config['engine']['block_size']) == journal(y, config['engine']['block_size']),
                                  f'{client}/{layer}: live native request journal differs')
                compared += 1
    check.details.update(reference_requests=len(reference_by_case), compared_state_pairs=compared,
                         numerical_policy='Exact logical bytes/output; mismatches UNVERIFIED, never presumed batch drift')


def ownership(records, config, check):
    """Global physical ownership is deduplicated across groups/components/aliases."""
    owners, current, released, reuse = {}, {}, {}, []
    pools = {}
    for row in records:
        if row['event'] == 'pool_snapshot':
            pools[row['host'], row['dp_rank'], row['role']] = row
    check.require(len(pools) == 4, 'Missing actual startup pool capacity for every role/lane')
    relevant = [row for row in records if row['event'] in {
        'block_allocate', 'target_reserve', 'block_remove_skipped', 'source_free', 'target_free', 'pressure'}]
    for row in sorted(relevant, key=lambda value: (value['host'], value['monotonic_ns'])):
        pool = row['host'], row['dp_rank'], row['role']
        req = pool+(row['request_id'],)
        event = row['event']
        if event == 'pressure':
            check.expect(row['cause'] == 'native_allocate_slots_returned_none', 'Pressure was not a native failed allocation')
            prior = {v for group in row['before']['blocks'] for v in group if v}
            final = current.get(req, set())
            check.expect(final <= prior, 'Failed native allocation created new physical ownership')
            check.expect(row['pool_free_blocks_after']-row['before']['pool_free_blocks'] == len(prior-final),
                         'Failed native allocation changed capacity without observed native retirement')
            demands = row['before'].get('native_demands', [])
            check.require(demands, 'Native pressure lacks the actual allocator block-demand result')
            rejected = False
            for demand in demands:
                watermark = (demand['native_watermark_blocks'] if demand['has_scheduled_reqs']
                             and demand['request_status'] in {'WAITING', 'PREEMPTED'} else 0)
                available = demand['pool_free_blocks']-(0 if demand['apply_admission_cap'] else demand['reserved_blocks'])
                rejected |= demand['native_blocks_to_allocate']+watermark > available
            check.require(rejected, 'Native None result does not prove requested blocks exceeded actual available capacity')
            continue
        group_ids = row.get('block_ids', row.get('blocks'))
        if not isinstance(group_ids, list):
            check.require(False, f'{req}: missing native ownership table')
            continue
        observed = {block for group in group_ids for block in group if block != 0}
        before = row.get('before', {}).get('blocks', [])
        prior = {block for group in before for block in group if block != 0}
        old = current.get(req, set())
        freeing = event in {'source_free', 'target_free'}
        final = set() if freeing else observed
        if freeing:
            check.expect(observed == old, f'{req}: final free differs from actual outstanding ownership')
            prior = observed
        # Nested native retirement can occur inside allocation; its actual event
        # already updated ownership, while the outer before snapshot remains old.
        expected_delta = len(prior)-len(final)
        a, b = row.get('pool_free_before'), row.get('pool_free_after')
        check.require(type(a) is int and type(b) is int, f'{req}: missing native per-call pool counts')
        if type(a) is int and type(b) is int:
            check.expect(b-a == expected_delta, f'{req}: native pool/component-alias conservation mismatch')
            if pool in pools:
                check.expect(0 <= a <= pools[pool]['total_blocks']-1 and 0 <= b <= pools[pool]['total_blocks']-1,
                             f'{req}: free count exceeds native immutable capacity')
        for block in old-final:
            key = pool+(block,)
            check.expect(owners.get(key) == req, f'{req}: released another request component alias')
            owners.pop(key, None)
            released[key] = (req, row)
        for block in final-old:
            key = pool+(block,)
            check.expect(key not in owners, f'{req}: physical page aliased across live requests')
            if key in released and released[key][0] != req:
                reuse.append((row, block, released[key][1]))
            owners[key] = req
        current[req] = final
    check.expect(not owners, 'Native run ended with unreleased physical page/component owners')
    controlled = [(row, block, old) for row, block, old in reuse if row.get('wave') == 'reuse6144']
    check.require(controlled, 'No actual released-page reuse in the declared post-pressure reuse control')
    registrations = [row for row in records if row['event'] == 'state_registration']
    reset_rows = [row for row in records if row['event'] == 'reset_content']
    inventories = [row for row in records if row['event'] == 'component_inventory']
    accounting = []
    for registration in registrations:
        views = registration['registrations']
        storages = {item['tensor']['storage']: item['tensor']['storage_bytes'] for item in views}
        regions = {item['region_index']: item for item in views}
        check.expect(sum(storages.values()) == registration['unique_resident_storage_bytes'], 'Resident aliases counted more than once')
        complete = [row for row in inventories if (row['host'], row['tp_rank'], row['dp_rank']) ==
                    (registration['host'], registration['tp_rank'], registration['dp_rank'])]
        check.require(len(complete) == 1, 'Missing complete raw/mean/accumulator/journal component inventory')
        full_bytes = None
        if len(complete) == 1:
            inventory = complete[0]
            native_storages = {row['tensor']['storage']: row['tensor']['storage_bytes'] for row in inventory['tensors']}
            full_bytes = sum(native_storages.values())
            check.expect(full_bytes == inventory['unique_resident_storage_bytes'], 'Full component aliases were charged more than once')
            actual_keys = {(row['group_id'], row['layer'], row['component']) for row in inventory['tensors']}
            expected_keys = {(row['group_id'], row['layer'], row['component']) for row in views}
            for row in views:
                if row['component'] == 'summary_mean':
                    names = ['numerator', 'denominator', 'maximum', 'last_slot']
                    if inventory['request_journal_widths'][row['layer']]:
                        names += ['request_journal_values', 'request_journal_last', 'request_journal_tag']
                    expected_keys |= {(row['group_id'], row['layer'], name) for name in names}
            check.expect(actual_keys == expected_keys, 'Complete component ownership inventory omits live state')
        accounting.append(dict(host=registration['host'], dp_rank=registration['dp_rank'], tp_rank=registration['tp_rank'],
            unique_resident_bytes=sum(storages.values()),
            all_component_resident_bytes=full_bytes,
            unique_wire_bytes=sum(item['wire_block_bytes']*item['tensor']['shape'][0] for item in regions.values()),
            resident_plus_wire_is_not_a_capacity_sum=True))
        for allocation, block, old in controlled:
            if (allocation['host'], allocation['dp_rank'], allocation['role']) != (registration['host'], registration['dp_rank'], registration['role']):
                continue
            matches = [row for row in reset_rows if row['host'] == allocation['host'] and row['tp_rank'] == registration['tp_rank']
                       and allocation['request_id'] in row['scheduled_request_ids']
                       and allocation['monotonic_ns'] <= row['monotonic_ns']]
            groups = {index for index, group in enumerate(allocation['block_ids']) if block in group}
            for view in views:
                if view['scheduler_group_id'] not in groups:
                    continue
                components = [item for row in matches for item in row['components'] if item['layer'] == view['layer']
                              and item['component'] == view['component'] and block in item['pages']]
                check.require(components, f'{view["layer"]}: reused component has no actual GPU-visible reset witness')
                for item in components:
                    if item.get('reset_valid') is not None:
                        check.expect(item['reset_valid'] is True, f'{view["layer"]}: invalid native component reset')
                    if item['component'] == 'summary_mean':
                        check.expect(item['journal_tags_invalid'] and item['speculative_tails_invalid'], 'Reused page retained native request state')
                    check.expect(item['alias_storage'] == view['alias_storage'], 'Reset referenced a different component alias')
    check.require(registrations and accounting, 'Missing actual resident/wire component alias registration')
    pressure = [row for row in records if row['event'] == 'pressure' and row['role'] == 'decode'
                and row.get('wave') == 'capacity_pressure']
    check.require(pressure, 'Explicit frozen-capacity128-request trigger did not observe target allocation pressure')
    check.details.update(actual_target_pressure_events=len(pressure), physical_reuses=len(reuse),
                         controlled_reuses=len(controlled), outstanding_physical_owners=len(owners),
                         resident_wire_accounting=accounting)
