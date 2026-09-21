## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Recorded source-preserving scope checks, physical-DP metadata controls and bounded compute parser fault checks. |

# D019 lane A verification

PASS for source preservation and parser controls. GPU runtime, operator timings, P/S/V comparison and CUDA-span correction remain PENDING. No performance number in the synthetic parser fixture is a measurement.

## Execution

CPU master, conda `dev-vidur-v03-hopper-e2e`, Python 3.13.13 at `/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python`. Python source is parsed against the GPU runtime's Python 3.10 syntax. All scratch fixtures and the temporary one-use fault driver are under `/data/ycfeng/tmp/issue26-d019-compute-parser-check`. Production diagnostic checkout remains frozen at `4bc1bc026c91dff78bd7cf5ba6411f14d15e043d`.

Full scope-check command is retained inline in [analysis/d019-vllm-scopes.md](analysis/d019-vllm-scopes.md), under Source verification execution and evidence. It was executed again directly from that Markdown block and passed. It strips only six named diagnostic wrappers/local imports and compares each affected model module's AST with the pre-change commit. It executes the actual CPU DP conversion function in isolation with four physical layouts and a DP1 fallback.

Parser source: `tests/e2e/issue26_first_batch_op_rca_compute.py`, committed as `9a1585ac246bf0b6b0c051ce4389904f75fd6e3c`. Reproduction commands, from the active Frontier worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python /data/ycfeng/tmp/issue26-d019-compute-parser-check/check.py
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_first_batch_op_rca_compute.py --run moe=/data/ycfeng/tmp/issue26-d019-compute-parser-check/valid --output /data/ycfeng/tmp/issue26-d019-compute-parser-check/output
git diff --check
```

The temporary fixture is explicitly tagged `fixture_only=true`. It places the real request on DP1, exercises all four TP workers, uses 48 decoder layers, 96 alternating gating scopes, 48 GG scopes and 48 sibling reduction scopes. Each synthetic duration is 1 ms solely to make expected pairing arithmetic independently inspectable. No fixture is merged into measurement data.

## Criteria and actual evidence

| Check | Expected behavior | Observed |
| --- | --- | --- |
| Six new model scopes | Removing only wrappers/imports produces original computational AST | PASS, six scopes and five model modules |
| Physical DP cumulative counts | `[4096,4097] -> [4096,1]`; reverse lane and mixed shapes retained | PASS, five controls |
| First-request rank join | Real request on DP1 joins its four TP peers | PASS |
| Gating split | 96 ordered scopes become 48 router-linear and 48 topk scopes | PASS |
| GG plus reduction | 48 same-run/same-rank sibling pairs; no rank summation | PASS, each rank's synthetic sum 96 ms |
| Missing sum | Reject incomplete op coverage | PASS rejection |
| Duplicate scope identity | Reject double counting | PASS rejection |
| `record_function` timing substituted | Reject non-event timing family | PASS rejection |
| NaN duration | Reject nonfinite timing | PASS rejection |
| Wrong peer physical-token counts | Reject incompatible global first-forward shape | PASS rejection |
| Wrong real request token count | Reject incompatible prefill shape | PASS rejection |
| Missing first-request TP rank | Reject incomplete participant set | PASS rejection |
| CSV/JSON CLI path | Produce four rank records with fixture provenance | PASS, `PASS_COLLECTION_DIAGNOSTIC_ONLY` |
| Whitespace/error check | No malformed diff | PASS |

Observed stdout ends with:

```text
PASS six new scopes, computational AST unchanged
PASS five physical DP metadata controls
PASS real request on DP1 joins all TP ranks; 48 same-layer GG+sum pairs; alternating gating scopes
PASS rejected missing_sum
PASS rejected duplicate_scope
PASS rejected wrong_timing_family
PASS rejected nan_timing
PASS rejected wrong_physical_tokens
PASS rejected wrong_request_tokens
PASS rejected missing_rank
PASS_COLLECTION_DIAGNOSTIC_ONLY groups ['moe'] first-request rank groups 4
```

A preparatory edit command had a Python heredoc string `SyntaxError` before writing any files; it was corrected. Legacy path-discovery `rg` misses were corrected to current locations. There was no runtime check failure in this substep. No H200 runtime import or numerical assertion has yet been executed by this lane.

## Fresh measurement handoff

Root runs the approved H200 diagnostic modes. For each completed group, use:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python tests/e2e/issue26_first_batch_op_rca_compute.py --run attention=/ABSOLUTE/attention/operators --run moe=/ABSOLUTE/moe/operators --run detail=/ABSOLUTE/detail/operators --output /ABSOLUTE/fresh-analysis
```

The uppercase paths above are explicit placeholders pending actual run directories; they are not reported as an executed command. Any subset of completed groups may be supplied. The parser validates real request0, its actual DP lane and all TP peers, physical peer input, matching batch metadata, timing family, unique contiguous scope sequence, positive durations and per-layer coverage. It emits separate groups and ranks. GG+sum is an additional same-run diagnostic field, not a second term added to the component table. P/S/V input selection and the material-perturbation decision remain root's integration checks.
