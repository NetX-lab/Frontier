## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Implemented six bounded compute scopes and physical DP token metadata after YC's D019 approval; source checks pass, H200 numerical validation pending. |

# D019 lane A: bounded vLLM compute measurement

Status: implementation/source validation PASS; GPU import, operation-count and numerical validation PENDING. No GPU job was launched by this lane. Clean vLLM checkout `/data/ycfeng/tmp/vLLM-BS` was not modified. The approved first-forward scope overrides the skill's later three-logical-batch gate for this substep; it does not establish full-case operator closure.

## Checkout and commits

Diagnostic checkout: `/data/ycfeng/tmp/issue26-vllm-diagnostics-20260908`, branch `feature/frontier-comparison-instrumentation`.

- Starting commit: `8453dd342c6aa2721aaf4b410998aab2f38bc2ec`.
- `a259cee5c`: expose pure row-parallel GEMM, embedding computation, activation, expert-result sum and final normalization.
- `452262322`: record physical DP token counts using the already computed CPU DP metadata.
- Final pin: `4bc1bc026c91dff78bd7cf5ba6411f14d15e043d`; expose attention output initialization.
- Final diagnostic tracked/untracked status: clean (`git status --short` empty).

All six operation additions wrap the existing work with the existing `record_function_or_nullcontext`. They do not alter tensor operations, introduce CUDA synchronization, change default operation allowlists, or enable probes without an active explicitly selected logger. New names must be passed through `VLLM_FRONTIER_CUDA_EVENT_OP_SCOPES`. This external diagnostic checkout is not a critical module under Frontier's `frontier/` 2,000-line cleanup gate; modifying shared upstream giant modules only at existing measurement seams avoids an unrelated refactor.

## Scope contract

| Scope | Source | Included work | Excluded work | Expected count on first real DP worker |
| --- | --- | --- | --- | ---: |
| `row_parallel_gemm` | `vllm/model_executor/layers/linear.py:1328` | `quant_method.apply` for row-parallel linear | Following TP allreduce | 48; this Qwen MoE case reaches attention output projection |
| `embedding_compute` | `vllm/model_executor/layers/vocab_parallel_embedding.py:435` | Input index/mask, gather, output masked fill | Embedding TP allreduce | 1 |
| `final_layernorm` | `vllm/model_executor/models/qwen3_moe.py:623` | Final model residual/RMSNorm call | Decoder-layer input/post-attention norms | 1 |
| `moe_activation` | `vllm/model_executor/layers/fused_moe/fused_moe.py:1745` | Actual activation dispatch, gated SiLU for this case | W1/W2 GEMM, following quantization | 48 |
| `moe_sum` | `vllm/model_executor/layers/fused_moe/fused_moe.py:1798` | Expert-output reduction into hidden-state output | W1/activation/W2, EP combine | 48 |
| `attn_output_init` | `vllm/attention/layer.py:250` | Output `torch.zeros` allocation and initialization | Attention backend, KV write | 48 |

Existing `moe_grouped_gemm` includes W1, activation, intermediate quantization and W2. It ends **before** `moe_sum`. A repaired Frontier profile that contains sum must compare against same-run `moe_grouped_gemm + moe_sum`. The prior 18.180160 ms grouped-GEMM parent does not include that reduction. Child W1/activation/W2 measurements must not be added to their inclusive parent.

K09 zeros/fill source is now identified: `Attention.forward` uses `self.use_output = self.attn_backend.accept_output_buffer` (layer.py:195), and FlashInfer sets `accept_output_buffer=True` (flashinfer.py:138). Each Qwen attention call initializes the output with `torch.zeros` before entering the FlashInfer scope. A fresh `attn_output_init` count/timing will verify the source-to-trace assignment directly.

`row_parallel_gemm` is a generic operation label, not a model-name dispatch or case-specific timing. Other row-parallel layers would share it. The current first-forward inventory has only the 48 attention output projections at this seam.

## Minimal physical DP metadata

`gpu_model_runner.py` takes a reference to `get_forward_context().dp_metadata` inside the existing model context, only when batch logging is enabled. After the forward-end event and existing synchronization, the existing batch JSON record writes `batch_dp_token_counts`, obtained by differencing `cu_tokens_across_dp_cpu`.

No extra collective, tensor transfer or dummy model call is added. Existing real request IDs, local batch ID, rank tuple and timestamp bind this field to the measured real forward. Expected physical counts are `[4096, 1]` for the first real DP0 prefill. This proves peer physical input size, but a value of 1 alone does not distinguish a real decode token from a dummy token. Use the corresponding peer scheduler/real-request evidence to identify the idle lane. It does not create an invented global batch ID or claim that dummy-lane compute timing has been measured.

## Proposed low-density GPU groups

Use the existing worker's explicit scope allowlist and first-formal `VLLM_FRONTIER_PROFILE_BATCH_LIMIT=1` mechanism; root owns the exact run invocation and new disjoint output directories. Prefix remains `cmpl-pf4096_dc1024:`. Keep `VLLM_FRONTIER_OP_TIMING_MODE=cuda_event`, default scope mode (no per-op synchronize), `VLLM_FRONTIER_OP_AGG_MODE=per_scope`, metadata flag off. Run batch-only references before/after probes. Expected counts below apply to the selected first **real** batch per TP worker; later DP1's first selected real batch is not automatically the same global forward.

1. Attention: `attn_pre_proj,attn_rope,attn_kv_cache_save,attn_prefill,row_parallel_gemm` (240 scopes).
2. MoE parents: `moe_gating,moe_shuffling,moe_grouped_gemm,moe_sum` (240 scopes, because `moe_gating` has 96 invocations: alternating router linear and topk).
3. Missing terms and decomposition: `input_layernorm,post_attention_layernorm,embedding_compute,final_layernorm,attn_output_init,moe_grouped_gemm_w1,moe_activation,moe_grouped_gemm_w2` (290 scopes). If perturbation remains material, split normalization/missing terms (146 scopes) from MoE children (144 scopes), while retaining the same case and controls.

No nested parent/child duplicate is selected within these groups. Distinct runs cannot supply an exact additive critical-path decomposition. Compare scope values with matching Frontier profile boundaries and evaluate probe spread against batch-only references. Existing K device activity remains inventory evidence, not a substitute for CUDA-event measurements.

## Source verification execution and evidence

Environment: CPU master; conda `dev-vidur-v03-hopper-e2e`, `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`, Python 3.13.13. Parsing explicitly uses Python 3.10 syntax. Working directory is the diagnostic checkout. No GPU, Docker, network, package import or environment mutation was needed.

The following reproducible command combines the direct checks run for the three committed substeps. It detects changed computational statements hidden inside new measurement wrappers, malformed Python 3.10 syntax, missing/new duplicate scopes, and incorrect conversion of the existing cumulative CPU DP counts.

```bash
cd /data/ycfeng/tmp/issue26-vllm-diagnostics-20260908
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PY'
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace

baseline = '8453dd342c6aa2721aaf4b410998aab2f38bc2ec'
scopes = {'row_parallel_gemm', 'embedding_compute', 'final_layernorm',
          'moe_activation', 'moe_sum', 'attn_output_init'}
files = ['vllm/model_executor/layers/linear.py',
         'vllm/model_executor/layers/vocab_parallel_embedding.py',
         'vllm/model_executor/models/qwen3_moe.py',
         'vllm/model_executor/layers/fused_moe/fused_moe.py',
         'vllm/attention/layer.py']
seen = []
class StripScopes(ast.NodeTransformer):
    def visit_With(self, node):
        self.generic_visit(node)
        if len(node.items) == 1:
            ctx = node.items[0].context_expr
            if (isinstance(ctx, ast.Call) and isinstance(ctx.func, ast.Name)
                    and ctx.func.id == 'record_function_or_nullcontext'
                    and len(ctx.args) == 1 and isinstance(ctx.args[0], ast.Constant)
                    and ctx.args[0].value in scopes):
                seen.append(ctx.args[0].value)
                return node.body
        return node
    def visit_ImportFrom(self, node):
        if (node.module == 'vllm.v1.utils' and len(node.names) == 1
                and node.names[0].name == 'record_function_or_nullcontext'):
            return None
        return node
for name in files:
    current = StripScopes().visit(ast.parse(Path(name).read_text(), feature_version=(3, 10)))
    original = StripScopes().visit(ast.parse(subprocess.check_output(
        ['git', 'show', f'{baseline}:{name}'], text=True), feature_version=(3, 10)))
    assert ast.dump(current) == ast.dump(original), name
assert set(seen) == scopes and len(seen) == 6
print('PASS six new scopes, computational AST unchanged')
path = Path('vllm/v1/worker/gpu_model_runner.py')
tree = ast.parse(path.read_text(), feature_version=(3, 10))
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name == '_frontier_dp_token_counts')
module = ast.Module(body=[ast.ImportFrom(module='__future__',
    names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
namespace = {}
exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
read = namespace['_frontier_dp_token_counts']
for cumulative, expected in [([4096, 4097], [4096, 1]),
                             ([1, 4097], [1, 4096]),
                             ([4097, 8194], [4097, 4097]), ([1, 2], [1, 1])]:
    metadata = SimpleNamespace(cu_tokens_across_dp_cpu=SimpleNamespace(
        tolist=lambda: list(cumulative)))
    assert read(metadata, expected[0]) == expected
assert read(None, 4096) == [4096]
print('PASS five physical DP metadata controls')
PY
git diff --check 8453dd342c6aa2721aaf4b410998aab2f38bc2ec HEAD
```

Observed individual substep results: all five initial scope checks PASS, attention-init scope check PASS, all five DP metadata controls PASS, and each `git diff --check` PASS. One preparatory Python edit command failed with `SyntaxError: unexpected character after line continuation character` because its command string contained a literal backslash before a newline; it wrote no files. The command was corrected and all checks above completed. Two discovery `rg` calls referenced absent legacy paths; actual current paths were then used. These are tooling failures, not runtime findings.

## Pending tasks and limits

1. Root runs fresh H200 import/low-density measurements at the final pin. Validate finite positive timings, expected per-layer counts, physical DP counts and batch-only perturbation.
2. Update the full P/S/V mapping with exact profile producer coverage. Compare repaired complete MoE path with grouped-GEMM parent plus separate sum, preserving global 4097 input and deterministic expert layout.
3. Keep first-forward CUDA numerical RCA/correction and clean TTFT acceptance open until the fresh measurements close them. CPU/workflow integration remains after the approved CUDA gate.

New unresolved design issues: none introduced by this substep. Known remaining evidence gaps: dummy-participant timings are not measured by these real-batch scopes; a peer physical count of 1 needs scheduler evidence to establish dummy status; low-density perturbation and GPU runtime import must be checked in the approved environment.

## Analyzer handoff

`tests/e2e/issue26_first_batch_op_rca_compute.py` was added and committed as `9a1585ac`. It validates any completed attention/moe/detail group, selects request0 on its actual DP lane, checks all four TP ranks and pairs grouped-GEMM with its sibling reduction by layer. Synthetic valid/invalid controls passed; actual GPU numerical validation remains pending. Exact commands and evidence are in `test_report_2026-09-08_d019_vllm_scopes.md`.
