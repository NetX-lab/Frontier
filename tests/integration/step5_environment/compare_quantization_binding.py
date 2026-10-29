#!/usr/bin/env python3
"""Check retained native block-FP8 inputs, bytes and finite scale diagnostics."""
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

import torch


def check(value, quantized, scales, use_ue8m0, block_size):
    assert value.dtype == torch.bfloat16
    assert quantized.dtype == torch.float8_e4m3fn and scales.dtype == torch.float32
    assert value.shape == quantized.shape
    assert list(scales.shape) == [value.shape[0], value.shape[1] // block_size]
    assert torch.isfinite(value).all() and torch.isfinite(quantized.float()).all()
    assert torch.isfinite(scales).all() and (scales > 0).all()
    blocks = value.double().reshape(value.shape[0], -1, block_size)
    maximum = blocks.abs().amax(-1).clamp(min=float(torch.tensor(1e-10)))
    raw = maximum / 448
    expected_scales = torch.exp2(torch.ceil(torch.log2(raw))) if use_ue8m0 else raw
    zero_input = not bool(torch.count_nonzero(value))
    if use_ue8m0 and not zero_input:
        assert torch.equal(scales.double(), expected_scales)
    elif not use_ue8m0:
        nearest = expected_scales.float()
        lower = torch.where(nearest.double() > expected_scales,
                            torch.nextafter(nearest, torch.full_like(nearest, -float('inf'))), nearest)
        upper = torch.where(nearest.double() < expected_scales,
                            torch.nextafter(nearest, torch.full_like(nearest, float('inf'))), nearest)
        assert ((scales >= lower) & (scales <= upper)).all()
    else:
        # Exact zero bytes do not establish agreement with the Triton scale floor.
        assert (torch.frexp(scales.double())[0] == 0.5).all()
    positive_codes = torch.arange(127, dtype=torch.int64)
    exponent, fraction = positive_codes // 8, positive_codes % 8
    representable = torch.where(exponent == 0, fraction.double() * 2**-9,
                                (1 + fraction.double() / 8) * torch.exp2(exponent.double() - 7))
    normalized = (blocks / scales.double().unsqueeze(-1)).clamp(-448, 448)
    magnitude = normalized.abs()
    above = torch.searchsorted(representable, magnitude).clamp(max=126)
    below = (above - 1).clamp(min=0)
    upper_distance = representable[above] - magnitude
    lower_distance = magnitude - representable[below]
    chosen = torch.where((upper_distance < lower_distance) |
                         ((upper_distance == lower_distance) & (above % 2 == 0)), above, below)
    expected_bytes = (chosen + torch.signbit(normalized).long() * 128).byte().reshape(value.shape)
    assert torch.equal(quantized.view(torch.uint8), expected_bytes), 'Quantized E4M3 bytes differ'
    alternate = raw if use_ue8m0 else torch.exp2(torch.ceil(torch.log2(raw)))
    return dict(quantized_elements=value.numel(), scale_elements=scales.numel(), byte_mismatches=0,
                maximum_scale_absolute_difference_from_triton_equation=(scales.double() - expected_scales).abs().max().item(),
                triton_scale_equation_mismatches=int((scales.double() != expected_scales).sum()),
                zero_scale_floor_equivalence='UNVERIFIED' if zero_input and use_ue8m0 else 'not_applicable',
                mode_discriminating_scale_elements=int((scales.double() != alternate).sum()))


def read_tensor(record, cloud, original):
    path = (cloud / Path(record['path']).relative_to(original)).resolve()
    assert path.is_relative_to(cloud.resolve()) and path.stat().st_size == record['size_bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
    value = torch.load(path, map_location='cpu', weights_only=True)
    assert list(value.shape) == record['shape'] and str(value.dtype) == record['dtype']
    # These three small contiguous quantizer tensors retain their native strides on CPU.
    assert list(value.stride()) == record['stride'] and value.is_contiguous()
    return value


def check_input(value, expected):
    assert value.dtype == expected.dtype and value.shape == expected.shape
    assert torch.equal(value.view(torch.int16), expected.view(torch.int16)), 'Input fixture bits differ'


def analyze(binding, cloud, original):
    columns = torch.arange(4096, dtype=torch.float32)
    rows = torch.stack([((columns * (2*r + 1) + 7*r) % 257 - 128) / 128 for r in range(4)]).bfloat16()
    inputs = {'zero': torch.zeros_like(rows), 'nonconstant': rows, 'large': rows * 16384}
    assert sorted(frame['input'] for frame in binding['frames']) == sorted(inputs)
    reports = []
    for frame in binding['frames']:
        assert sorted(record['name'] for record in frame['tensors']) == ['input', 'quantized', 'scales']
        tensors = {record['name']: read_tensor(record, cloud, original) for record in frame['tensors']}
        check_input(tensors['input'], inputs[frame['input']])
        result = check(tensors['input'], tensors['quantized'], tensors['scales'],
                       binding['use_ue8m0'], binding['block_shape'][1])
        reports.append(dict(input=frame['input'], **result))
    return reports


if __name__ == '__main__':
    torch.set_num_threads(4)
    started = time.perf_counter()
    binding, cloud, original = json.load(sys.stdin)
    reports = analyze(binding, Path(cloud), Path(original))
    print(json.dumps(dict(scope='finite_quantizer_diagnostics_not_historical_04e_or_compiled_bound',
                          verdict='PARTIAL: scale-floor equivalence remains UNVERIFIED; byte checks are independent',
                          frames=reports, python=platform.python_version(), torch=torch.__version__,
                          wall_seconds=time.perf_counter() - started,
                          host_peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
