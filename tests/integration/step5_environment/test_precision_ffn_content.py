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


@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float8_e4m3fn])
def test_raw_hash_preserves_dtype_bytes_and_chunk_boundaries(dtype):
    worker = module()
    value = torch.tensor([0., -0., 1., -1., 2., -2., 3., -3.]).to(dtype).reshape(2, 4)
    expected = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert worker.raw_tensor_sha256(value, value[0].numel()*value.element_size()) == expected
    assert worker.raw_tensor_sha256(value, 1024) == expected
    sliced = value[:, ::2]
    assert worker.raw_tensor_sha256(sliced, 1024) == hashlib.sha256(
        sliced.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    with pytest.raises(ValueError):
        worker.raw_tensor_sha256(value, 1)


def test_current_weight_comparison_requires_complete_raw_identity(tmp_path):
    worker = module()
    target = torch.nn.Module()
    target.register_parameter('weight', torch.nn.Parameter(torch.tensor([0., -0., 1.], dtype=torch.bfloat16)))
    target.register_buffer('scales', torch.tensor([1., -1.]))
    snapshot = worker.weight_attestation(target, 16, 352)
    records = []
    for name, value in [('weight', target.weight), ('scales', target.scales)]:
        path = tmp_path/(name+'.pt')
        torch.save(value.detach(), path)
        records.append(dict(name=name, expert=None, path=str(worker.CLOUD/path.name),
            size_bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            shape=list(value.shape), dtype=str(value.dtype)))
    assert worker.compare_weight_attestation(snapshot, records, tmp_path)['tensors'] == 2
    with pytest.raises(ValueError):
        worker.compare_weight_attestation(snapshot, records[:1], tmp_path)
    with pytest.raises(ValueError):
        worker.compare_weight_attestation(snapshot, records+records[:1], tmp_path)
    snapshot['tensors'][0]['raw_sha256'] = '0'*64
    with pytest.raises(AssertionError):
        worker.compare_weight_attestation(snapshot, records, tmp_path)


def test_expert_hash_split_and_scalar_buffer():
    worker = module()
    target = torch.nn.Module()
    target.register_parameter('weight', torch.nn.Parameter(torch.ones(2, 2, 2)))
    target.register_buffer('scale', torch.tensor(-0.))
    records = worker.weight_attestation(target, 16, 2)['tensors']
    assert [(r['name'], r['expert']) for r in records] == [('weight', 0), ('weight', 1), ('scale', None)]
    assert records[-1]['raw_sha256'] == hashlib.sha256(b'\x00\x00\x00\x80').hexdigest()


def test_weight_snapshot_is_after_all_timed_roots_and_diagnostic():
    import ast
    source = SCRIPT.with_name('local_ffn_measurement.py').read_text()
    measure = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'measure')
    sample_loop = next(n for n in measure.body if isinstance(n, ast.For))
    attest = next(n for n in measure.body if isinstance(n, ast.FunctionDef) and n.name == 'attest')
    assert attest.lineno > sample_loop.end_lineno
    assert 'diagnostic(attest)' in source
    assert 'weight_attestation(' not in ast.unparse(sample_loop)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.int64])
@pytest.mark.parametrize('budget_elements', [1, 2, 3, 5])
def test_strided_single_element_tail_uses_standard_layout(dtype, budget_elements):
    worker = module()
    value = torch.arange(10).to(dtype)[::2]
    expected = hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert worker.raw_tensor_sha256(value, budget_elements*value.element_size()) == expected


@pytest.mark.parametrize('shape', [(2, 0), (0,), (0, 4096), (2, 0, 3)])
def test_empty_payload_and_invalid_copy_budget(shape):
    worker = module()
    assert worker.raw_tensor_sha256(torch.empty(shape), 8) == hashlib.sha256(b'').hexdigest()
    with pytest.raises(ValueError):
        worker.raw_tensor_sha256(torch.ones(1), 0)
