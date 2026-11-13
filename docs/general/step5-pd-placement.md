# Step structural P/D placement

The opt-in `vllm_load_balancing` cluster scheduler supports sequential Step
`pd-disaggregation` with dummy timing and the structural NIXL connector. Each
role has one endpoint represented by one Replica; its attention-DP lanes are
the request owners. At P32/D32, each role uses attention TP8 x DP4, MoE TP1 x
EP32 and PP1. There are four schedulers per role, not 32.

Select `--cluster_scheduler_config_type vllm_load_balancing` and
`--vllm_load_balancing_cluster_scheduler_config_policy_version v0.28.0`.
Legacy defaults and round-robin placement remain unchanged. Multiple endpoint
Replicas, other PD attention families, PD-AF, pipeline parallel PD, measured
timing and non-NIXL PD are not admitted by this extension.

P placement occurs at request arrival. The P frontend's inflight ownership
ends at its response, independently of the retained source KV pages. The
proxy's D request is placed after that response and before C16 reservation.
Reservation retries, READ, readiness, admission and decode retain the selected
endpoint and lane. D never reenters the cluster placement queue at readiness.
Remote-KV receiving and capacity-blocked requests count as waiting on D.

Both endpoints reuse `VllmDPLoadBalancer`. The v0.28 rule is:

```
max(client_count * inflight, waiting + running)
    + waiting * 6.0 * max(0, kv_cache_usage - 0.5)
```

Ties follow a rotating scan. Local waiting increases by `client_count` after
selection, and unfinished-request counts survive snapshot replacement. A
repeated unfinished request ID retains affinity without another increment.
The simulator models one frontend per endpoint; the balancer also accepts
client count/index for source-rule tests, not concurrent frontend execution.

Publication retains the existing lazy coordinator model: changed snapshots
have a 100 ms interval, unchanged snapshots a 5 s interval, and synchronized
steps a 50 ms collection minimum unless a previous-step snapshot is latched.
That previous-step snapshot publishes first without clearing `stats_changed`;
publishing current counts clears it. KV-only changes do not emit reports.
Publication replaces local waiting estimates, not the inflight floor.

Run `tests/integration/run_step5_pd_placement.py --mode online --output-root
<new-directory>` with the CPU simulator environment for expected/actual
placement and lifecycle traces. The driver also supports offline mode.

This is structural evidence only. Actual native endpoint selection, DP4
report arrival delays, ties, update ordering, readiness/release ordering,
wire bytes and the D prompt-tail/first-output-token differences remain
separate native qualification requirements. No predictive parity is claimed.
