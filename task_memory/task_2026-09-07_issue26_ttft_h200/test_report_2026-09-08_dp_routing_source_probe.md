## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Verified exact routing methods with controlled load state. |

# DP routing source-method verification

## Execution

Working directory: /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
Frontier revision: 91f7147a5ff8b7ea2fccaf48b0227d10df747b4f.
vLLM revision: 46f7b179fd3bf42b9616dc4670cba419afdb2085.
Conda environment: dev-vidur-v03-hopper-e2e; Python 3.13.13.
The vLLM method is AST-extracted unchanged to avoid importing GPU dependencies. Frontier methods are imported from the actual active worktree.

Exact command:

```bash
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python - <<'PY'
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from frontier.scheduler.cluster_scheduler.round_robin_cluster_scheduler import RoundRobinClusterScheduler as RR
from frontier.scheduler.cluster_scheduler.lor_cluster_scheduler import LORClusterScheduler as LOR

source = Path('/data/ycfeng/tmp/vLLM-BS/vllm/v1/engine/core_client.py')
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DPLBAsyncMPClient')
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'get_core_engine_for_request')
module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
namespace = {'sys': sys}
exec(compile(ast.fix_missing_locations(module), str(source), 'exec'), namespace)
choose = namespace[method.name]

def client(counts):
    return NS(lb_engines=[list(x) for x in counts], eng_start_index=0, client_count=1,
              core_engines=[0, 1], reqs_in_flight={})

def request(index):
    return NS(request_id=str(index), data_parallel_rank=None)

def frontier(kind, waiting=(0, 0)):
    obj = kind.__new__(kind)
    obj._num_replicas = 1
    obj._replica_dp_size = 2
    obj._cluster = NS(replicas={7: object()})
    obj._request_counter = 0
    obj._request_queue = []
    obj._replica_schedulers = {(7, d): NS(num_pending_requests=waiting[d]) for d in range(2)}
    return obj

rr = frontier(RR)
singleton = []
for i in range(6):
    rr._request_queue = [request(i)]
    singleton.append(rr._schedule_batch_mode()[0][1])
rr = frontier(RR)
rr._request_queue = [request(i) for i in range(6)]
burst = [row[1] for row in rr._schedule_batch_mode()]
c = client([(0, 0), (0, 0)])
vllm_burst = [choose(c, request(i)) for i in range(6)]
vllm_idle_refresh = [choose(client([(0, 0), (0, 0)]), request(i)) for i in range(6)]
cases = []
for counts in [[(0, 10), (0, 0)], [(1, 0), (0, 3)], [(0, 3), (0, 4)]]:
    lor = frontier(LOR, [x[0] for x in counts])
    lor._request_queue = [request(0)]
    cases.append({'waiting_running': counts, 'vllm': choose(client(counts), request(0)),
                  'frontier_lor': lor._schedule_lor()[0][1]})
assert singleton == [0]*6 and burst == [0, 1]*3
assert vllm_burst == [0, 1]*3 and vllm_idle_refresh == [0]*6
assert [(x['vllm'], x['frontier_lor']) for x in cases] == [(1, 0), (1, 1), (0, 0)]
result = {'status': 'PASS', 'python': sys.version, 'frontier_rr_singleton': singleton,
          'frontier_rr_burst': burst, 'vllm_no_snapshot_refresh': vllm_burst,
          'vllm_fresh_empty_snapshot_each_arrival': vllm_idle_refresh, 'load_cases': cases,
          'scope': 'Actual routing methods with controlled state; no GPU or E2E timing validation.'}
Path('task_memory/task_2026-09-07_issue26_ttft_h200/analysis/dp_routing_source_probe.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
PY
```

## Criteria and evidence

PASS: All explicit assertions succeeded. Machine-readable results: analysis/dp_routing_source_probe.json.

- Current Frontier RR: six singleton calls choose [0,0,0,0,0,0]; one six-request call chooses [0,1,0,1,0,1]. This reproduces a policy-state defect.
- vLLM, initially empty counts and no snapshot refresh: [0,1,0,1,0,1].
- vLLM, a new empty snapshot before each request: [0,0,0,0,0,0]. This models all previous work having drained and the empty state reaching the frontend before the next arrival.
- Counts DP0=(waiting0,running10), DP1=(waiting0,running0): actual vLLM chooses DP1; Frontier LOR chooses DP0 because its current vllm_v1 pending property excludes running requests.
- Counts (1,0)/(0,3): both methods choose DP1; equal outputs here do not establish equal algorithms.
- Counts (0,3)/(0,4): both methods choose DP0.

## Limits

Additional source verification: company-proxied upstream download succeeded:

```bash
source /data/ycfeng/tmp/issue26-h200-network/company-proxy.sh
curl --connect-timeout 8 --max-time 15 --fail --silent --show-error https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/v1/engine/core_client.py -o /data/ycfeng/tmp/issue26-vllm-v0102-core-client-research.py
```

AST comparison of DPLBAsyncMPClient.get_core_engine_for_request observed official_v0102_routing_method_equal=True (upstream lines1132–1157). The same comparison between the clean and diagnostic local checkouts observed clean_vs_diagnostic_routing_method_equal=True. These checks verify policy source identity, not runtime trace identity.

These are deterministic source-method checks with supplied state, not fresh GPU measurements or validation of load-report timing. The probe does not test coordinator transport, snapshot publication, scheduler state transitions, multi-client behavior, or E2E latency. No simulated/actual TTFT or latency error is claimed by this check. No production code was changed.
