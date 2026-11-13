"""Regressions for retained block-FP8 finite-content diagnostics."""
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).with_name('compare_quantization_binding.py')
spec = importlib.util.spec_from_file_location('quantization_comparison', SCRIPT)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_exact_power_two_not_rounded_logarithm():
    value = torch.zeros(4, 128, dtype=torch.bfloat16)
    quantized = value.to(torch.float8_e4m3fn)
    scales = torch.full((4, 1), 2**-33, dtype=torch.float32)
    result = comparison.check(value, quantized, scales, True, 128)
    assert result['triton_scale_equation_mismatches'] == 4
    assert result['zero_scale_floor_equivalence'] == 'UNVERIFIED'
    bad = torch.nextafter(scales, torch.full_like(scales, float('inf')))
    assert torch.equal(torch.log2(bad), torch.log2(bad).round())
    with pytest.raises(AssertionError):
        comparison.check(value, quantized, bad, True, 128)


def test_raw_scale_enclosing_values_only():
    value = torch.ones(1, 128, dtype=torch.bfloat16)
    exact = value.double().abs().max() / 448
    nearest = exact.float().reshape(1, 1)
    direction = -float('inf') if nearest.double() > exact else float('inf')
    other = torch.nextafter(nearest, torch.full_like(nearest, direction))
    lower, upper = (other, nearest) if direction < 0 else (nearest, other)
    assert lower.double() < exact < upper.double()
    for scales in (lower, upper):
        quantized = (value.float() / scales).to(torch.float8_e4m3fn)
        comparison.check(value, quantized, scales, False, 128)
    for scales in (torch.nextafter(lower, torch.zeros_like(lower)),
                   torch.nextafter(upper, torch.full_like(upper, float('inf')))):
        quantized = (value.float() / scales).to(torch.float8_e4m3fn)
        with pytest.raises(AssertionError):
            comparison.check(value, quantized, scales, False, 128)


def test_signed_zero_fixture_and_quantized_bytes():
    expected = torch.zeros(1, 128, dtype=torch.bfloat16)
    value = expected.clone()
    value[0, 0] = -0.0
    assert torch.equal(value, expected)
    with pytest.raises(AssertionError, match='Input fixture bits differ'):
        comparison.check_input(value, expected)
    scales = torch.full((1, 1), 2**-33, dtype=torch.float32)
    quantized = value.to(torch.float8_e4m3fn)
    comparison.check(value, quantized, scales, True, 128)
    quantized.view(torch.uint8)[0, 0] = 0
    with pytest.raises(AssertionError, match='Quantized E4M3 bytes differ'):
        comparison.check(value, quantized, scales, True, 128)


def test_retained_layout_metadata_rejected(tmp_path):
    path = tmp_path / 'scales.pt'
    value = torch.ones(4, 32, dtype=torch.float32)
    torch.save(value, path)
    record = dict(path='/original/scales.pt', size_bytes=path.stat().st_size,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest(), shape=[4, 32],
                  dtype='torch.float32', stride=[32, 1])
    assert torch.equal(comparison.read_tensor(record, tmp_path, Path('/original')), value)
    record['stride'] = [1, 4]
    with pytest.raises(AssertionError):
        comparison.read_tensor(record, tmp_path, Path('/original'))


def test_missing_or_duplicate_input_frames_rejected():
    binding = {'frames': []}
    with pytest.raises(AssertionError):
        comparison.analyze(binding, Path('/unused'), Path('/original'))
    binding['frames'] = [{'input': name} for name in ('zero', 'zero', 'large')]
    with pytest.raises(AssertionError):
        comparison.analyze(binding, Path('/unused'), Path('/original'))
