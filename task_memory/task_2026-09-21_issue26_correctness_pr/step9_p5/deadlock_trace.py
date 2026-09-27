"""Trace layer-sync entries and stage-context ownership up to the stall."""
import sys
from pathlib import Path
tree = sys.argv[1]; case = sys.argv[2]; root = Path(sys.argv[3])
sys.meta_path[:] = [f for f in sys.meta_path if "editable" not in getattr(type(f), "__module__", "").lower()]
sys.path.insert(0, tree)
sys.path.insert(1, str(Path(__file__).parent))
import c2_pp1_policy_matrix as m
import frontier.config as fc
fc.VllmLoadBalancingClusterSchedulerConfig = fc.RoundRobinClusterSchedulerConfig
from frontier.simulator import Simulator
import frontier.scheduler.utils.sync_entry as se
from frontier.scheduler.replica_stage_scheduler.stage_execution_context import StageExecutionContext
from frontier.scheduler.replica_stage_scheduler.replica_stage_schduler import ReplicaStageScheduler

log = []
def ctx_state(c):
    return f"g={c._forward_group_id} sealed={c._forward_group_sealed} full={sorted(str(t.operation_id[:2]) for t in c._active_full_stage_tickets)} ep={c._active_ep_ticket is not None} fifo={[str(t.operation_id[:2]) for t in c._ready_fifo]}"
orig_enter = se.enter_layer_sync
def enter(scheduler, time, replica_id, stage_id, batch, replica_local_id, sync_stage, layer_id, t, *, mode, metrics_store=None):
    pre = getattr(batch, "_forward_cohort_id", None)
    out = orig_enter(scheduler, time, replica_id, stage_id, batch, replica_local_id, sync_stage, layer_id, t, mode=mode, metrics_store=metrics_store)
    ctx = scheduler.get_stage_execution_context(replica_id, stage_id)
    log.append(f"{time:.6f} ENTER lane={replica_local_id} b={batch.id} idle={batch.is_idle} reqs={[r.id for r in batch.requests]} L{layer_id} {mode} prov={getattr(batch,'_forward_cohort_provisional_id',None)} step {pre}->{getattr(batch,'_forward_cohort_id',None)} out={[type(e).__name__ + ':' + str(getattr(e,'_replica_local_id',None)) for e in (out or [])]} | {ctx_state(ctx)}")
    return out
se.enter_layer_sync = enter
orig_try = StageExecutionContext.try_acquire
def try_acquire(self, ticket):
    ok = orig_try(self, ticket)
    log.append(f"      TRY {ticket.operation_id[:3]} {ticket.scope} ok={ok} | {ctx_state(self)}")
    return ok
StageExecutionContext.try_acquire = try_acquire
orig_rel = StageExecutionContext.release
def release(self, ticket):
    orig_rel(self, ticket)
    log.append(f"      REL {ticket.operation_id[:3]} | {ctx_state(self)}")
StageExecutionContext.release = release
orig_bind = StageExecutionContext.bind_forward_group
def bind(self, ticket):
    g = orig_bind(self, ticket)
    log.append(f"      BIND {ticket.operation_id[:3]} -> {g}")
    return g
StageExecutionContext.bind_forward_group = bind
for name in ("replace_full_stage_owners_with_ep_wave", "replace_ep_wave_with_full_stage_owners", "transition_active_scope"):
    orig = getattr(StageExecutionContext, name)
    def w(orig=orig, name=name):
        def f(self, *a, **k):
            r = orig(self, *a, **k)
            log.append(f"      {name} | {ctx_state(self)}")
            return r
        return f
    setattr(StageExecutionContext, name, w())
orig_pop = ReplicaStageScheduler.pop_batch_if_not_busy
def pop(self):
    b = orig_pop(self)
    log.append(f"      POP lane={self._replica_local_id} -> {None if b is None else (b.id, [r.id for r in b.requests])}")
    return b
ReplicaStageScheduler.pop_batch_if_not_busy = pop

root.mkdir(parents=True, exist_ok=True)
sim = Simulator(m._config(root, case))
try:
    sim.run(); print("DRAINED")
except RuntimeError as e:
    print("STUCK")
for line in log[-int(sys.argv[4]):]:
    print(line)
