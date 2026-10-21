#!/usr/bin/env python3
"""Compare retained native module content with independent CPU mathematical stages."""
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time

import torch

CLOUD = Path('/mnt/codesign-exp/ycfeng/step5-preview/g1_precision_ffn_20261017_01')


def tensor(record, cloud):
    relative = Path(record['path']).relative_to(CLOUD)
    path = (cloud/relative).resolve()
    assert path.is_relative_to(cloud.resolve())
    assert path.stat().st_size == record['size_bytes']
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(piece)
    assert digest.hexdigest() == record['sha256'], str(path)
    value = torch.load(path, map_location='cpu', weights_only=True)
    assert list(value.shape) == record['shape'] and str(value.dtype) == record['dtype']
    return value


def raw_tensor_sha256(value, chunk_bytes):
    """Hash logical C-order dtype bytes with CPU copies limited to one row chunk."""
    if chunk_bytes <= 0:
        raise ValueError('A positive tensor-copy byte budget is required')
    if sys.byteorder != 'little':
        raise ValueError('Native tensor attestation requires the recorded little-endian host')
    value = value.detach()
    if value.numel() == 0:
        return hashlib.sha256().hexdigest()
    row_elements = math.prod(value.shape[1:]) if value.ndim else 1
    row_bytes = row_elements * value.element_size()
    if row_bytes > chunk_bytes:
        raise ValueError('The tensor-copy budget cannot hold one logical row')
    rows_per_chunk = chunk_bytes // row_bytes
    digest = hashlib.sha256()
    rows = value if value.ndim else value.reshape(1)
    for start in range(0, rows.shape[0], rows_per_chunk):
        # A one-element view counts as contiguous with its original stride; force standard strides.
        cpu = rows[start:start+rows_per_chunk].to('cpu', memory_format=torch.contiguous_format, copy=True)
        digest.update(memoryview(cpu.reshape(-1).view(torch.uint8).numpy()))
        del cpu
    return digest.hexdigest()


def weight_attestation(module, chunk_bytes, expert_count):
    """Read current parameters and buffers; no weights or storage padding are saved."""
    records = []
    for kind, values in (('parameter', module.named_parameters()), ('buffer', module.named_buffers())):
        for name, value in values:
            pieces = value.unbind(0) if value.ndim == 3 and value.shape[0] == expert_count else (value,)
            for index, piece in enumerate(pieces):
                records.append(dict(name=name, kind=kind, expert=index if len(pieces) == expert_count else None,
                    shape=list(piece.shape), stride=list(piece.stride()), dtype=str(piece.dtype),
                    full_shape=list(value.shape), full_stride=list(value.stride()),
                    storage_offset=piece.storage_offset(), logical_bytes=piece.numel()*piece.element_size(),
                    raw_sha256=raw_tensor_sha256(piece, chunk_bytes)))
    return dict(scope='current_weights_after_last_measured_root_not_continuous_content_monitoring',
                producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                hash_order='logical_C_order_dtype_bytes', byte_order=sys.byteorder,
                chunk_bytes=chunk_bytes, tensors=records)


def compare_weight_attestation(snapshot, records, cloud):
    """Require complete logical tensor identity, separately from serialized layout."""
    if snapshot['hash_order'] != 'logical_C_order_dtype_bytes' or snapshot['byte_order'] != sys.byteorder:
        raise ValueError('Incompatible tensor hash interpretation')
    def keyed(values):
        result = {}
        for row in values:
            key = (row['name'], row['expert'])
            if key in result:
                raise ValueError('Duplicate tensor identity')
            result[key] = row
        return result
    current, historical = keyed(snapshot['tensors']), keyed(records)
    if current.keys() != historical.keys():
        raise ValueError('Current and historical tensor inventories differ')
    for key, observed in current.items():
        record = historical[key]
        value = tensor(record, cloud)
        assert observed['shape'] == record['shape'] and observed['dtype'] == record['dtype'], key
        assert observed['logical_bytes'] == value.numel()*value.element_size(), key
        assert observed['raw_sha256'] == raw_tensor_sha256(value, snapshot['chunk_bytes']), key
    return dict(status='PASS_CURRENT_WEIGHT_BYTES_NOT_MEASUREMENT_ADMISSION', tensors=len(current),
                layout_scope='Native and serialized strides remain separate observations')


def errors(actual, expected):
    actual, expected = actual.double(), expected.double()
    delta = (actual-expected).abs()
    maximum = actual.abs().max().item()
    return dict(elements=actual.numel(), finite=bool(torch.isfinite(actual).all()),
        max_absolute_error=delta.max().item(), mean_absolute_error=delta.mean().item(),
        rms_error=delta.square().mean().sqrt().item(), actual_max_absolute=maximum,
        relative_max_error=None if maximum == 0 else delta.max().item()/maximum,
        unequal_elements=int((actual != expected).sum()),
        expected_bf16_unequal=int((actual != expected.bfloat16().double()).sum()))


def activation(value, variant='step'):
    gate, up = value.double().chunk(2, dim=-1)
    if variant == 'input_clamp':
        gate = gate.clamp(max=7)
    activated = gate*torch.sigmoid(gate)
    if variant != 'unclamped':
        if variant != 'input_clamp':
            activated = activated.clamp(max=7)
        up = up.clamp(-7, 7)
    return (activated*up).bfloat16()


def linear_stages(frame, parameters, prefix, report):
    packed = parameters[prefix+'gate_up_proj.weight'].double()
    down = parameters[prefix+'down_proj.weight'].double()
    x = frame['input'].double()
    projected = x@packed.T
    observed = frame[prefix+'gate_up_proj']
    report[prefix+'projection'] = errors(observed, projected)
    expected = activation(observed)
    report[prefix+'activation_given_native_projection'] = errors(frame[prefix+'act_fn'], expected)
    report[prefix+'clamp_alternatives'] = {variant: int((expected != activation(observed, variant)).sum())
        for variant in ('unclamped', 'input_clamp')}
    report[prefix+'down_given_native_activation'] = errors(frame[prefix+'down_proj'], frame[prefix+'act_fn'].double()@down.T)
    report[prefix+'end_to_end'] = errors(frame[prefix+'down_proj'], activation(projected.bfloat16()).double()@down.T)


def fp8_input(value, power_of_two):
    blocks = value.float().reshape(value.shape[0], -1, 128)
    scales = blocks.abs().amax(dim=-1).clamp(min=1e-10)*(1.0/448)
    if power_of_two:
        scales = torch.exp2(torch.ceil(torch.log2(scales)))
    scaled = (blocks/scales.unsqueeze(-1)).clamp(-448, 448)
    encodings = torch.arange(127, dtype=torch.int64)
    exponents, mantissas = encodings//8, encodings%8
    values = torch.where(exponents == 0, mantissas.double()*2**-9,
        (1+mantissas.double()/8)*torch.exp2(exponents.double()-7))
    magnitudes = scaled.abs().double()
    upper = torch.searchsorted(values, magnitudes).clamp(max=126)
    lower = (upper-1).clamp(min=0)
    below = magnitudes-values[lower]
    above = values[upper]-magnitudes
    choose_upper = (above < below) | ((above == below) & (upper%2 == 0))
    rounded = values[torch.where(choose_upper, upper, lower)]*scaled.sign().double()
    return (rounded*scales.double().unsqueeze(-1)).reshape(value.shape)


def expert_weights(records, expert, cloud, fp8):
    weights = {r['name']: tensor(r, cloud).double() for r in records if r['expert'] == expert}
    result = []
    for projection in ('w13', 'w2'):
        weight = weights[f'experts.routed_experts.{projection}_weight']
        if fp8:
            scale = weights[f'experts.routed_experts.{projection}_weight_scale_inv']
            weight = weight*scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        result.append(weight)
    return result


def routed(frame, records, cloud, fp8=False, power_of_two=False):
    ids, probabilities = frame['routing_ids'], frame['routing_weights']
    contributions = torch.zeros((4, 8, 4096), dtype=torch.bfloat16)
    witnesses = dict(gate_silu_over_limit=0, up_over_limit=0)
    for expert in ids.unique().tolist():
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        w13, w2 = expert_weights(records, expert, cloud, fp8)
        x = frame['input'][rows]
        x = fp8_input(x, power_of_two) if fp8 else x.double()
        projected = (x@w13.T).bfloat16()
        gate, up = projected.double().chunk(2, dim=-1)
        witnesses['gate_silu_over_limit'] += int((gate*torch.sigmoid(gate) > 7).sum())
        witnesses['up_over_limit'] += int((up.abs() > 7).sum())
        h = activation(projected)
        h = fp8_input(h, power_of_two) if fp8 else h.double()
        value = h@w2.T
        contributions[rows, slots] = (value*probabilities[rows, slots].double().unsqueeze(1)).bfloat16()
    return errors(frame['output_1'], contributions.double().sum(dim=1).bfloat16()), witnesses


def compare(case, cloud, operation):
    started = time.perf_counter()
    torch.set_num_threads(4)
    report = dict(case=case['case'], operation=operation, python=platform.python_version(), torch=torch.__version__,
        numerical_guarantee='UNVERIFIED: mathematical reference is not a compiled-kernel error bound')
    if operation in ('pre', 'post'):
        count, size, layouts = 0, 0, []
        for record in case[operation]:
            value = tensor(record, cloud)
            layouts.append(dict(name=record['name'], expert=record['expert'],
                native_stride=record['stride'], serialized_stride=list(value.stride())))
            assert torch.isfinite(value.float()).all(), record['name']
            count += 1
            size += record['size_bytes']
        report.update(records=count, bytes=size, integrity='PASS', layouts=layouts)
    else:
        parameters = {r['name']: tensor(r, cloud) for r in case['post'] if r['expert'] is None}
        report['frames'] = []
        for native in case['frames']:
            frame = {r['name']: tensor(r, cloud) for r in native['tensors']}
            x = frame['input']
            columns = torch.arange(4096, dtype=torch.float32)
            expected = torch.stack([((columns*(2*r+1)+7*r)%257-128)/128 for r in range(4)]).bfloat16()
            if native['input'] == 'zero':
                expected.zero_()
            elif native['input'] == 'large':
                expected *= 16384
            if case['case'] == 'output_projection':
                expected = expected.repeat(1, 3)
            assert torch.equal(x, expected)
            row = dict(input=native['input'], tensor_integrity='PASS', layouts=[dict(name=r['name'],
                native_stride=r['stride'], serialized_stride=list(frame[r['name']].stride())) for r in native['tensors']])
            if case['case'] == 'dense':
                linear_stages(frame, parameters, '', row)
                assert torch.equal(frame['output_0'], frame['down_proj'])
            elif case['case'].startswith('moe_'):
                linear_stages(frame, parameters, 'share_expert.', row)
                assert torch.equal(frame['output_0'], frame['share_expert.down_proj'])
                assert torch.equal(frame['gate'], frame['routing_logits'])
                logits = x.double()@parameters['gate.weight'].double().T
                row['gate'] = errors(frame['gate'], logits)
                scores = torch.sigmoid(frame['gate'].double())
                corrected = scores+parameters['router_bias'].double()
                ordered = corrected.argsort(dim=-1, descending=True)
                ids = frame['routing_ids'].long()
                row['selection_sets_match'] = bool(torch.equal(ids.sort(dim=-1).values, ordered[:, :8].sort(dim=-1).values))
                sorted_scores = corrected.gather(1, ordered)
                row['rank8_rank9_margins'] = (sorted_scores[:, 7]-sorted_scores[:, 8]).tolist()
                chosen = scores.gather(1, ids)
                weights = 3*chosen/chosen.sum(dim=1, keepdim=True)
                row['weights_given_native_logits_and_ids'] = errors(frame['routing_weights'], weights)
                biased = corrected.gather(1, ids)
                row['biased_weight_alternative'] = errors(frame['routing_weights'], 3*biased/biased.sum(dim=1, keepdim=True))
                independent_scores = torch.sigmoid(logits)
                independent_order = (independent_scores+parameters['router_bias'].double()).argsort(dim=-1, descending=True)
                row['selection_sets_from_independent_logits_match'] = bool(torch.equal(ids.sort(dim=-1).values,
                    independent_order[:, :8].sort(dim=-1).values))
                independent_chosen = independent_scores.gather(1, ids)
                independent_probabilities = 3*independent_chosen/independent_chosen.sum(dim=1, keepdim=True)
                row['weights_given_independent_logits_and_native_ids'] = errors(frame['routing_weights'], independent_probabilities)
                if case['case'] == 'moe_excluded':
                    row['routed_given_exposed_native_routing'], row['routed_clamp_witnesses'] = routed(frame, case['post'], cloud)
                else:
                    row['fp8_conditional_quantization_candidates'] = {}
                    for mode in (False, True):
                        error, witnesses = routed(frame, case['post'], cloud, fp8=True, power_of_two=mode)
                        row['fp8_conditional_quantization_candidates']['ue8m0' if mode else 'floating_scale'] = dict(
                            error=error, clamp_witnesses=witnesses,
                            qualification='UNVERIFIED: mode is a mathematical candidate, not observed native binding')
            else:
                projected = x.double()@parameters['weight'].double().T
                if case['case'] == 'dsa_projection':
                    row['projection'] = errors(torch.cat((frame['output_0'], frame['output_1']), dim=-1), projected)
                    assert torch.equal(frame['output_1'], torch.cat([frame[f'indexer_{i}'] for i in range(4)], dim=-1))
                else:
                    row['projection'] = errors(frame['output_0'], projected)
            if native['input'] == 'zero':
                assert all(not torch.count_nonzero(v) for n, v in frame.items() if n.startswith('output_'))
            report['frames'].append(row)
    report.update(wall_seconds=time.perf_counter()-started,
        host_peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        checked_native_strides='recorded separately; CPU materialization may be contiguous')
    return report


if __name__ == '__main__':
    case, cloud, operation = json.loads(sys.stdin.read())
    print(json.dumps(compare(case, Path(cloud), operation)), flush=True)
