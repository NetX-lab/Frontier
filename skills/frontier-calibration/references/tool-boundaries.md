# Tool Boundaries

The skill directory holds no runtime helper. Entries run the repository tools
below when their verified input formats and engine support match the case,
pinned by the Frontier commit that the command receipt records. Run them
from the repository root with `PYTHONPATH=$PWD`. Before a tool's first use in a
task, read its `--help` and module docstring; they own the flags, inputs, and
outputs, and this table owns only the role and the owning entry.

The table describes available implementations, including vLLM-specific ones.
It is not a compatibility promise for another engine. `engine-integration.md`
owns the capability checks and metric-conversion boundary.

## Repository Tools and Owners

| Tool | Owner entries | Use |
| --- | --- | --- |
| `tests/comparison/e2e_gate/run_gate.py` | `e2e-metrics-gap`, `simulator-run`, `cpu-overhead-supplement`, `error-correction` | single entry point: `cells`, `normalize`, `gate`, `frontier` (pinned replay), `cpu-table` (byte-exact rebuild), `admit` (CPU-probe admission) |
| `tests/comparison/e2e_gate/gate.py`, `normalize.py`, `dp_pp_modes.py`, `admission.py` | `e2e-metrics-gap`, `cpu-overhead-supplement` | gate kinds, row normalization, DP x PP mode labels, admission rules; called through `run_gate.py` |
| `tests/comparison/e2e_gate/cases/scenario_matrix.json`, `artifacts.json`, `cases/published_readings.json` | `parity-run`, `error-correction` | claimed cells and their recorded commands, pinned artifacts, published readings |
| `tests/comparison/calibration/e2e_metrics_gap.py` | `e2e-metrics-gap` | five-metric gap of one vLLM run against Frontier runs |
| `tests/comparison/dp_placement_pp/run_vllm_worker.sh`, `vllm_replay.py` | `groundtruth-run`, `groundtruth-run-check` | co-location and DP x PP ground truth; `vllm_replay.py` owns the mode switches |
| `tests/comparison/calibration/run_pd_worker.sh`, `pd_replay.py`, `pd_proxy.py`, `pd_metrics.py` | `groundtruth-run` | existing vLLM 1P1D ground truth: proxy, two instances, request join |
| `tests/comparison/dp_placement_pp/make_trace.py` | `parity-run`, `dispatch-align-trace` | trace and request ids from a case `workload.json` (warmup groups, bursts, Poisson segments) |
| `tests/comparison/dp_placement_pp/extract_vllm_placement.py`, `compare_placement.py` | `workflow-gap-analysis` | DP placement, engine iterations, dummy passes, workflow gap table |
| `tests/comparison/dp_placement_pp/run_frontier_case.py` | `simulator-run` | Frontier runs of a DP placement study (not the E2E gate) |
| `tests/comparison/calibration/op_gap_analysis.py` | `op-gap-analysis` | native operator gap from one clean, one `op_timing`, and one `kernel_timing` run |
| `tests/comparison/calibration/forward_device_timeline.py` | `op-gap-analysis`, `residual-lane-classification` | per-forward device busy, idle, and all-reduce time from torch profiler traces |
| `tests/comparison/calibration/flashinfer_decode_plan_bench.py` | `op-gap-analysis` | graph-plan against eager-plan decode attention |
| `tests/comparison/calibration/nccl_p2p_bench.py`, `vllm_tp_allreduce_bench.py` | `residual-lane-classification`, `residual-repair` | KV P2P and TP all-reduce bandwidth and latency fits |
| `tests/comparison/calibration/run_profiling_plan.py` | `op-supplement` | runs a profiling plan on a GPU worker and writes a manifest and `COMPLETE` |
| `tests/comparison/calibration/merge_profiling_csvs.py` | `op-supplement` | lossless merge of a passing profiling run into a new table set |
| `frontier/profiling/kernel_gap/main.py` | `op-supplement` | `kernel_gap.csv` per execution mode and attention TP size |
| `frontier/profiling/cpu_overhead/vllm_cpu_probe.py` | `cpu-overhead-analysis`, `cpu-overhead-supplement` | CPU-overhead tables from CPU-probe logs (`python -m frontier.profiling.cpu_overhead.vllm_cpu_probe`) |
| `tests/e2e/refactor_fidelity/run_matrix.py` | every repair entry, `error-correction` | baseline-against-candidate fidelity matrix over the public examples |

Operator profiling producers (`python -m frontier.profiling.<family>.main`)
follow `docs/profiling/README.md` and `frontier/profiling/README.md`. Verify that
the profiled implementation and backend match the selected engine; a matching
operator name or shape alone does not establish the same kernel behavior.

## Existing vLLM Collection Profile

Use this profile only after checking the selected vLLM build or fork against
its collectors and probes. No particular remote, branch, or base version is
required. `tests/comparison/dp_placement_pp/vllm_replay.py` (`MODES`,
`server_env`) owns the switches; its mode names are not cross-engine flags.

| Mode | Records and scope |
| --- | --- |
| `clean` | Request metrics; diagnostic probes off. |
| `schedule_timing` | Scheduler step membership, queueing, and periods. |
| `op_timing` | CUDA-event scopes plus batch/PP records; scope times can include launch gaps. |
| `kernel_timing` | Synchronized scope times; device synchronization distorts step timing. |
| `instrumented` | MoE routing records, separate from operator timing. |
| `cpu` | Engine and per-stage host observations. |
| `device_timeline` | CPU probe plus torch profiler device activity. |
| `kv_save_timing`, `kv_save_device_ids` | CPU and KV-save observations for the existing connector. |

These modes are mutually exclusive in the existing replay. Its clean mode
writes request metrics, and its diagnostic modes disable those metrics. Other
collectors may emit both, but diagnostic rows never become clean gate evidence.

The published vLLM recipes use at least three warmups per prompt-length class,
then drain them; eager MoE also gets an unmeasured compile-cache pass. Pair
clean/probe runs in one job on the same node group and take the KV block count
from that cell's clean run. Inspect the collector's actual warmup and lifecycle
behavior before reusing the recipe for a different build.

Some existing cells use `groundtruth_weight_mode=dummy`,
`real_weight_download=false`, and `moe_uniform_routing` with Frontier's
`balanced` split. These are explicit synthetic-case settings. They never
replace the user's native weights or routing automatically; their limits follow
`contract.md`, Weights and routing. The same restriction applies to the
`pd_s33` endpoint correction and `dp_pp_pairs` mode-conditioned comparison.

For the existing CPU producer, `run_gate.py admit` defines the 0.97-1.03
probe/reference ratio and the `client_e2e`, `send_span`, and `isolated_prefill`
rules. Their specific observations and cohorts are part of this profile;
applying them to another engine requires equivalent timing boundaries and
validation, not just a matching field name.

## Case-Local Tools

Some steps have no repository tool yet: the ground-truth run check, the case
manifest writer, an engine-specific collector or metric converter, the case
runner that turns the native engine configuration and a Frontier
settings file into a `frontier.main` command (and, for a DP x PP ensemble
member, draws its route delays), the GPU job launcher, the coverage
check, the routing audit, and the error budget. The task keeps them in its
artifacts bundle. A case-local tool:

- follows this contract (for example, a runner stops on a null setting and
  takes the settings file by explicit argument);
- has its SHA-256 recorded in every receipt that uses it;
- is proposed for a move under `tests/comparison/` when a second task needs it.

`site.local.md`, when present beside `SKILL.md`, names this host's copies.

## Command Receipt

`exec capture: UNKNOWN`. Each entry records, for each command it launches, the
exact command, working directory, declared environment, UTC start and end, exit
code, Frontier commit and dirty state, ground-truth build identity, and expected output
paths. The receipt covers that entry's own calls only.

## Tool Failure

A missing tool, a non-zero exit, malformed output, a stale or pre-existing
output path, a hash mismatch, or a completion-marker mismatch is an explicit
failure. Record it in the task's progress log and stop at the owning entry.
Fix the cause under the existing change rules; never silently substitute a
parser or hand-edit a published table. A planned engine collector/converter is
case-local work with recorded inputs, validation, and provenance, not a fallback
that turns a failed or unsupported comparison into a passing one.
