## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-08 | Verified local mixed-batch reachability and clarified request-level preservation in the proposed repair. |
| 2026-09-08 | Reproduced the predictor-timed mixed-phase stall and prepared a shared-protocol repair proposal. |

# Mixed-phase forward synchronization RCA and proposed repair

## Observed result

The fresh100request run at a4a0c496 exits1: event_queue_length0 while DP0 owns running requests0/1 and46queued, and DP1 owns running request2 and51queued. The reported final47.156283407937735s is the last scheduled arrival, not the instant execution stopped. Event tracing locates the last progress at0.29563780122719624s.

The exact same command, with this same-commit run's newly trained caches, fresh diagnostic output, and event tracing, reproduces the failure. Reducing the input to the first3requests still fails at the same timestamp. Reducing output length to8tokens also preserves the failure. Removing the third request from that minimized fixture completes2/2requests, each4096/8. These minimized fixtures are control-flow diagnostics only; the calibration case remains4096/1024.

## Claim ledger

| Claim | Label | Direct evidence | Strength / limit |
| --- | --- | --- | --- |
| Same admitted group is split into different phase wait rooms | Evidence | full_MONOLITHIC_sync_state.json: provisional5, replica0/stage0/layer0/pre_moe; decode step240 has lane0 batch2166 requests0/1, prefill step241 has lane1 batch1742 request2 | Direct |
| Both participants are real and occupied, so idle synthesis cannot join them | Evidence | Both batches idle=false; runtime lane stages busy; sync_entry._can_supply_idle_lane returnsFalse for busy siblings | Direct |
| Phase separation prevents the shared EP wave from being scheduled | Inference | Actual entry-handler control: prefill+prefill emits1wave, decode+decode emits1wave, decode+prefill emits0waves and leaves two open groups | Strong controlled support; production repair not applied |
| Snapshot selection itself caused the stall | Inference, contradicted as direct mechanism | First3routes0,0,1 match fresh vLLM diagnostic; isolated sync-entry reproduction uses no load-balancer selection | Selection exposes the pre-existing path but does not explain missing collective readiness |
| The earlier dummy-timing check proves predictor-timed execution | Inference, rejected | Dummy3requests completed but actual predictor3requests stall; their phase overlaps differ | Earlier check's scope remains functional only |
| Entire TTFT residual is CPU overhead | Unknown | No full repaired Frontier output; no matched CPU critical-path evidence | Do not assign the residual to CPU |

## Source mechanism

`frontier/events/replica_stage_schedule_event.py` selects PrefillSyncEvent or DecodeSyncEvent from each lane's local batch phase. `frontier/scheduler/utils/sync_entry.py` enters different waiting rooms. `frontier/scheduler/utils/forward_sync_state.py` also partitions open bindings by phase, even when the admitted forward-group identity is the same. Each room waits for attention-DP2 participants, while the peer is busy in the other room. No next collective event is emitted.

Shared EP work already lives in `ep_wave_schedule.schedule_layer_wave`, so the repair should extend this mechanism. Merely aliasing the two wait rooms is insufficient: prefill/decode completion paths currently require different stage-start metadata, layer progress, and metric ledgers. Both completion helpers also select a sample batch for some attention predictions; a mixed cohort must retain each lane's own request shape and completion accounting.

## First formal batch finding

Fresh vLLM diagnostic DP0/TP0/PP0 batch4769 contains onlyrequest0/4096tokens. Fresh Frontier trace has onlyrequest0 arriving before batch0 completes onDP0 at65.79007690374297ms; next arrival is81.2080018222332ms. Thus the first formal prefill request composition matches. DP1's first vLLM formal prefill is request2/4096tokens. First3Frontier routes are0,0,1.

The vLLM diagnostic batch forward duration77.1250228881836ms and clean request0 officialTTFT121.48427963256836ms use different boundaries and separate executions. The Frontier first-stage endpoint65.79007690374297ms is partial execution evidence, not a completed100request result. Their differences must not be labeled measured CPU overhead.

## Proposed next sub-step, pending YC decision

Update: YC subsequently authorized the design-only step. The resulting design and revised implementation estimate are maintained in ../design_shared_forward_sync.md; the preliminary proposal below is historical. Runtime repair and calibration validation remain outstanding.

Local mixed batches are already reachable with chunked prefill disabled. Fresh vLLM DP0/TP0 batch4770 contains request0/decode1 and request1/prefill4096. The validated diagnostic has49mixed batches among50formal-prefill batches on DP0 and48among49on DP1 (TP0 only, avoiding TP duplication). This is local-batch evidence, not a cross-DP round join. Frontier V1 schedules RUNNING then WAITING requests into one batch; disabling chunked prefill only skips an entire waiting prefill when its tokens exceed the remaining budget. Batch retains request-level token counts and aggregate prefill/decode counts. Its stage event chooses the prefill synchronization path whenever any prefill tokens exist. Consequently, mixed+prefill enters the same phase path, whereas mixed+decode can expose the existing split. The proposed repair must retain the complete per-request phase/token composition of each source lane, not assume every lane is purely prefill or decode. Merely selecting the prefill event does not prove decode requests are reclassified; numerical handling of mixed attention and per-source completion requires validation.

1. Share the synchronization identity and waiting state for participants of one admitted MONOLITHIC forward group, independent of their local prefill/decode phase.
2. Reuse the existing EP wave builder to aggregate real participants once, enter one shared wave, and restore stage ownership once.
3. Preserve phase and shape per source lane for attention prediction, layer progression, final completion and TTFT/model-time accounting. Adapt the existing prefill/decode helpers instead of sending decode batches through a prefill-only ledger.
4. Add a deterministic same-group mixed-phase regression and same-phase/idle controls; rerun the3request predictor-timed reproduction, then the full100request4096/1024case with fresh caches and artifacts.
5. Complete request-level metric comparison and first/all-prefill batch membership analysis.

Expected production touch points: sync_entry.py, forward_sync_state.py, ep_wave_schedule.py, prefill_collective.py, decode_collective.py, with a small shared helper only if needed to keep one ownership-restoration mechanism. Rough scope:5–7production modules plus focused tests. This changes a shared forward-synchronization contract beyond the D016 RR/load-snapshot patch; implementation awaits the workspace Approval Gate decision.

Rejected alternatives: forcing all requests ontoDP0 violates the frozenDP2reference; serializing prefill and decode merely avoids a vLLM-reachable mixed cohort; merging rooms without per-lane completion breaks existing metadata/metric invariants. No such substitute is applied.

## Reproduction

CPU conda dev-vidur-v03-hopper-e2e, Python3.13.13. The observer only wraps Simulator._build_sequential_scheduler_state_report to retain state at the existing failure boundary; it does not alter scheduling.

```bash
cd /data/ycfeng/stepfun-performance-optimization/Frontier/.worktrees/issue26-ttft-h200-20260907
export FRONTIER_LOG_LEVEL=ERROR PYTHONDONTWRITEBYTECODE=1 TMPDIR=/data/ycfeng/tmp PYTHONPATH="$PWD"
/home/i-fengyicheng/miniconda3/envs/dev-vidur-v03-hopper-e2e/bin/python /data/ycfeng/tmp/issue26-h200-network/observe_deadlock.py /data/ycfeng/tmp/issue26-dp-deadlock-repro-03req
```

The prepared command file declares the exact trace, profiling files, current-generation caches, and output. For another reproduction, prepare a fresh output directory and replace only its output argument; preserve failure evidence. Persistent copies of commands, traces, sync states, observer source, and the two-request negative control are in analysis/mixed-phase-stall-evidence/. Handler control results are in analysis/mixed_phase_sync_control.json.

Additional control completed: the first2requests with the original4096/1024shape also finish2/2with exit0. Thus the three-request reproduction does not depend on shortening decode length; the mixed-phase third request is required for this observed stall. Evidence: analysis/mixed-phase-stall-evidence/two_request_decode1024_metrics.jsonl.
