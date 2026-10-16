"""Regression checks for retained tensor materialization and independent math."""
import hashlib
import importlib.util
from pathlib import Path

import pytest


torch = pytest.importorskip('torch')
SCRIPT = Path(__file__).with_name('compare_precision_ffn.py')


def module():
    spec = importlib.util.spec_from_file_location('precision_comparison', SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_native_slice_materialization_preserves_content_not_stride(tmp_path):
    worker = module()
    original = torch.arange(24, dtype=torch.float32).reshape(3, 8)[:, :4]
    path = tmp_path/'slice.pt'
    materialized = original.clone()
    torch.save(materialized, path)
    record = dict(path=str(worker.CLOUD/'slice.pt'), size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), shape=list(original.shape),
        dtype=str(original.dtype), stride=list(original.stride()))
    actual = worker.tensor(record, tmp_path)
    assert torch.equal(actual, original)
    assert actual.stride() != original.stride()
    record['sha256'] = '0'*64
    with pytest.raises(AssertionError):
        worker.tensor(record, tmp_path)


def test_clamp_order_and_zero_relative_error():
    worker = module()
    value = torch.tensor([[10., -6.625]], dtype=torch.bfloat16)
    expected = worker.activation(value)
    assert expected.item() == -46.5
    assert not torch.equal(expected, worker.activation(value, 'input_clamp'))
    assert not torch.equal(expected, worker.activation(value, 'unclamped'))
    assert worker.errors(torch.zeros(2), torch.zeros(2))['relative_max_error'] is None


def test_e4m3_encoding_matches_representable_rounding():
    worker = module()
    value = torch.linspace(-448, 448, 4096).reshape(1, 4096)
    blocks = value.reshape(1, -1, 128)
    scales = blocks.abs().amax(-1).clamp(min=1e-10)*(1./448)
    encoded = (blocks/scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn).double()
    expected = (encoded*scales.double().unsqueeze(-1)).reshape(1, 4096)
    assert torch.equal(worker.fp8_input(value, False), expected)
    assert not torch.count_nonzero(worker.fp8_input(torch.zeros(1, 128), False))
