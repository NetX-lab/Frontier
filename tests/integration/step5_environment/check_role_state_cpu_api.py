"""Check the current resident observer with locked Torch CPU tensor operations."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import weakref

import torch

import role_capacity_observation as capacity
import role_model_input_observation as model
import role_state_observation as state


class TorchCpuApi:
    """Keep real Torch tensors and explicitly substitute CUDA ownership primitives."""
    def __init__(self):
        self.stream = SimpleNamespace(cuda_stream=1, device=torch.device('cpu'))
        self.allocations, self.events = [], []
        self.cuda = SimpleNamespace(is_current_stream_capturing=lambda: False,
                                    current_stream=lambda device: self.stream, Event=self.event)

    def __getattr__(self, name):
        return getattr(torch, name)

    def empty(self, shape, *, dtype, device, pin_memory=False):
        # A CPU-only probe cannot establish native pinned allocation or CUDA copies.
        tensor = torch.empty(shape, dtype=dtype, device=device)
        self.allocations.append(dict(tensor=tensor, requested_pin_memory=pin_memory))
        return tensor

    def event(self, *, enable_timing):
        assert enable_timing is False
        events, stream = self.events, self.stream
        class Event:
            def record(self, selected):
                assert selected is stream
                events.append('record_cpu_substitute')
            def query(self):
                events.append('query_cpu_substitute')
                return True
        return Event()


def unique_bytes(tensors):
    return sum({(str(tensor.device), tensor.untyped_storage().data_ptr()):
                tensor.untyped_storage().nbytes() for tensor in tensors}.values())


def resident_model():
    actual = torch.nn.Module()
    first = torch.tensor([56, 127, 255, 184], dtype=torch.uint8).view(torch.float8_e4m3fn)
    actual.register_buffer('fp8', first)
    actual.register_buffer('bf16', torch.tensor([1.5, float('nan'), -2.25], dtype=torch.bfloat16))
    actual.register_buffer('fp16', torch.tensor([2.5, float('inf'), -3.5], dtype=torch.float16))
    actual.register_buffer('fp32', torch.tensor([3.25, -1.75], dtype=torch.float32))
    actual.register_buffer('scalar', torch.tensor(7.25, dtype=torch.float32))
    actual.register_buffer('empty', torch.empty((2, 0, 3), dtype=torch.float32), persistent=False)
    matrix = torch.arange(105, dtype=torch.float32).reshape(3, 5, 7).transpose(0, 1)
    actual.register_buffer('strided', matrix)
    actual.register_buffer('alias', matrix[:])
    actual.register_buffer('overlap', matrix[:, :, ::2])
    actual.register_buffer('opaque', torch.tensor([127, 255], dtype=torch.uint8))
    scales = torch.nn.Module()
    scales.register_parameter('w13_weight_scale_inv', torch.nn.Parameter(
        torch.tensor([[0, -1], [2, 3], [4, 5]], dtype=torch.float32)))
    scales.register_parameter('w2_weight_scale_inv', torch.nn.Parameter(
        torch.arange(12, dtype=torch.float32).reshape(3, 4)[:, ::2]))
    actual.add_module('experts', scales)
    return actual, scales


def runner_inputs(api, actual, hidden_dtype):
    dtype = torch.float32 if hidden_dtype is torch.float32 else torch.bfloat16
    raw = torch.arange(40, dtype=torch.float32).reshape(5, 1, 4, 2)
    context = {'layer': SimpleNamespace(kv_cache=raw)}
    table = torch.tensor([[2, 3, 0], [0, 0, 0]], dtype=torch.int32)
    runner = SimpleNamespace(model=actual, device=torch.device('cpu'), main_stream=api.stream,
        use_pp=False, pcp_manager=None, num_speculative_steps=0, is_first_pp_rank=True, is_last_pp_rank=True,
        max_num_tokens=8, max_num_reqs=2,
        model_config=SimpleNamespace(dtype=dtype, hf_text_config=SimpleNamespace(
            hidden_size=3, fp32_residual_connection=hidden_dtype is torch.float32)),
        compilation_config=SimpleNamespace(static_forward_context=context),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=['layer'],
            kv_cache_spec=SimpleNamespace(sliding_window=3, extra_retained_tokens=0))]),
        block_tables=SimpleNamespace(cp_size=1, cp_rank=0, block_sizes=[4], input_block_tables=[table],
            num_blocks=SimpleNamespace(gpu=torch.tensor([[1, 0]], dtype=torch.int32))))
    return runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    report = dict(scope='CURRENT_RESIDENT_OBSERVER_REAL_TORCH_CPU_API_ONLY_NOT_CUDA_OR_G2',
        python=sys.version, torch_version=torch.__version__, torch_git_revision=torch.version.git_version,
        cuda=torch.version.cuda, checks=[], cases=[], sources={}, G2='UNVERIFIED',
        substitutions=['CPU destinations are unpinned', 'CUDA streams and events are CPU substitutes'],
        limits=['No CUDA allocation/copy/reduction qualification', 'No native model execution',
                'No scan timing/headroom qualification', 'No transient scale or consumption qualification'])
    for path in (Path(__file__), Path(capacity.__file__), Path(model.__file__), Path(state.__file__)):
        report['sources'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()

    def check(name, condition, **details):
        report['checks'].append(dict(name=name, passed=bool(condition), **details))
        if not condition:
            raise AssertionError(name)

    try:
        for hidden_dtype in (torch.float32, torch.bfloat16):
            api = TorchCpuApi()
            actual, scales = resident_model()
            runner = runner_inputs(api, actual, hidden_dtype)
            sites = SimpleNamespace(torch=api, tensor_type=torch.Tensor, intermediate_type=None,
                padding_mask_used=True, census_owners={id(scales): (weakref.ref(scales),
                    dict(module_path='experts', layer_id=0, sites=['routed_modular']))})
            limits = model.DeviceObservationLimits(steps=2, scan_rows=4, witness_rows=1,
                owned_bytes=65536, rpc_bytes=65536)
            reservations = []
            def reserve(owned):
                assert not api.allocations
                reservations.append(owned)
            deadline = time.monotonic()+20
            observed = model.DeviceObservation(runner, sites, limits,
                [dict(name='A', prompt_token_ids=[31, 32])], dict(n=1, seed=None),
                65536, reserve=reserve, deadline=deadline)
            prefix = str(hidden_dtype)+': '
            host = [tensor for row in observed.destinations for tensor in row.values()]+list(observed.state.once.values())
            device = [observed.absolute, observed.finite, observed.counts, *observed.state.device.values()]
            check(prefix+'one complete reservation precedes allocations', reservations == [observed.owned_bytes])
            check(prefix+'exact host storage bytes', unique_bytes(host) == observed.host_bytes,
                  bytes=observed.host_bytes)
            check(prefix+'exact device scratch storage bytes', unique_bytes(device) == observed.device_bytes,
                  bytes=observed.device_bytes)
            check(prefix+'exact total owned storage bytes', unique_bytes(host+device) == observed.owned_bytes,
                  bytes=observed.owned_bytes)
            check(prefix+'every new allocation is counted', unique_bytes([row['tensor'] for row in api.allocations]) ==
                  observed.owned_bytes)
            entry = {}
            observed.state.attach_resident(entry)
            result = observed.state.drain_parameters(entry)
            rows = {row['name']: row for row in result['rows']}
            for prepared in observed.state.parameters:
                selected, tensor = rows[prepared['metadata']['name']], prepared['tensor']
                expected = int(torch.isfinite(tensor.to(torch.float32)).sum())
                check(prefix+selected['name']+' complete real numeric finite count',
                      selected['finite_elements'] == expected, actual=selected['finite_elements'], expected=expected)
                if selected['elements']:
                    with torch.no_grad():
                        sample = torch.empty((1,), dtype=tensor.dtype)
                        sample.copy_(tensor[tuple(slice(0, 1) for _ in tensor.shape)].reshape(1))
                    check(prefix+selected['name']+' original raw sample bytes',
                          selected['sample']['raw_hex'] == sample.view(torch.uint8).numpy().tobytes().hex())
                    check(prefix+selected['name']+' aligned sample destination',
                          selected['sample_offset'] % tensor.element_size() == 0)
                else:
                    check(prefix+selected['name']+' empty view has no scan or raw sample',
                          selected['chunks'] == 0 and selected['sample'] is None)
                chunks = list(state.parameter_chunks(tensor, prepared['layout']))
                check(prefix+selected['name']+' bounded basic-index chunk coverage',
                    len(chunks) == prepared['layout']['chunks'] and
                    all(chunk.numel() <= observed.state.scratch_elements for chunk in chunks) and
                    sum(chunk.numel() for chunk in chunks) == tensor.numel() and
                    all(chunk.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr() for chunk in chunks))
            check(prefix+'both FP8 NaN bit patterns are rejected', rows['.fp8']['finite_elements'] == 2)
            check(prefix+'BF16 numeric conversion retains NaN', rows['.bf16']['finite_elements'] == 2)
            check(prefix+'FP16 numeric conversion retains infinity', rows['.fp16']['finite_elements'] == 2)
            check(prefix+'opaque integer bytes remain explicit gap', result['status'] == 'UNVERIFIED' and
                  any(row['name'] == '.opaque' for row in result['gaps']))
            check(prefix+'identical strided aliases share one view', len(rows['.strided']['aliases']) == 2 and
                  '.alias' not in rows)
            check(prefix+'overlapping distinct views have independent coverage', '.overlap' in rows)
            check(prefix+'resident inverse-scale reductions reused', all(rows['experts.'+name]['finite_source'] ==
                'EXISTING_RESIDENT_SCALE_REDUCTION' for name in state.SCALE_PARAMETERS))
            frozen = copy.deepcopy(result)
            with torch.no_grad():
                actual.fp32.fill_(float('nan'))
            check(prefix+'owned copies survive source content mutation', observed.state.drain_parameters(entry) == frozen)
            registered = observed.state.parameter_registrations
            check(prefix+'capacity and content share all registrations', len(registered) ==
                  len(capacity.inventory(runner, torch)['tensors']))
            with torch.no_grad():
                actual.alias.transpose_(0, 1)
            try:
                observed.state.check_parameters()
            except ValueError as failure:
                check(prefix+'real alias layout mutation is rejected', 'view changed' in str(failure))
            else:
                check(prefix+'real alias layout mutation is rejected', False)
            before = len(api.allocations)
            try:
                observed.state.submit_resident(deadline)
            except ValueError as failure:
                check(prefix+'second submission rejected without allocation', 'only once' in str(failure) and
                      len(api.allocations) == before)
            else:
                check(prefix+'second submission rejected without allocation', False)
            report['cases'].append(dict(hidden_dtype=str(hidden_dtype), host_bytes=observed.host_bytes,
                device_bytes=observed.device_bytes, owned_bytes=observed.owned_bytes,
                registrations=len(registered), unique_views=len(rows), parameters=result,
                native_consumption='UNMEASURED', cuda_events='CPU_SUBSTITUTES_ONLY'))
        check('CUDA remains uninitialized', not torch.cuda.is_initialized())
        report['status'] = 'PASS'
    except BaseException as failure:
        report.update(status='FAIL', error_type=type(failure).__name__, error=str(failure))
        raise
    finally:
        report.update(cuda_initialized=torch.cuda.is_initialized(),
                      passed=sum(row['passed'] for row in report['checks']),
                      failed=sum(not row['passed'] for row in report['checks']))
        with (output/'api_report.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps({key: report.get(key) for key in ('status', 'scope', 'passed', 'failed', 'cuda_initialized')},
                         allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
