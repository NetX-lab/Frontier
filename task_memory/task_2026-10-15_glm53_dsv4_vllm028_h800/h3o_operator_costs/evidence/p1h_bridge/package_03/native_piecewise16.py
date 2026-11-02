"""Print the eager-break attention kernel grids of native PIECEWISE forwards at 16 padded rows."""
import sys
from pathlib import Path

from dsv4_kernel_identity_check import native_candidates

NAMES = ('_compute_global_topk_indices_and_lens_kernel', 'void smxx::decode::flash_fwd_mla_combine_kernel', 'void vllm::cooperative::cooperative_topk')
for candidate in native_candidates([Path(sys.argv[1])]):
    if candidate.mode != 'PIECEWISE':
        continue
    rows = {grid for (name, grid, block) in candidate.kernels if name == '_fused_q_kv_rmsnorm_kernel'}
    print(candidate.step_class, candidate.mode, 'padded', rows, 'forwards', candidate.forward_count)
    for (name, grid, block), count in sorted(candidate.kernels.items()):
        if name.startswith(NAMES):
            print('   ', count, grid, block, name[:100])
