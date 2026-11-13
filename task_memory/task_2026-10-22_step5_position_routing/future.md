# Deferred work outside D29 acceptance

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Defer the independent role-scoped placement override gap; D29 uses the working global placement option. |
| 2026-10-23 | Index deferred work identified by the checkpoint without expanding D29. |

## Independent public consumer

Retain the latent None-manager/name-projection issue for a separately requested
monolithic measured Step capability. The approved P/D entrypoint supplies a
shared manager; this is not a D29 blocker. Evidence and the withdrawn proposal
remain in repairs/independent_consumer_scope.md. Do not implement a binder,
registry mixin or fallback under the current task.

## Module splits and scheduler duplication

Five existing size exceptions and their proposed split boundaries are retained in
docs/general/module-size-exceptions.md (D29 update3b78f625). The scheduler's
duplicate width assignment was also retained. Revisit only under a separate
cleanup/split request with its own scope and validation; no D29 implementation
allocation or acceptance dependency is added here.

## Role-scoped communication placement override

The actual configuration probe finds role-scoped placement flags declared but not
propagated through cluster_role_config.py456–494; the existing global
collective_sim placement option reaches both P/D backends correctly. This does not
block the approved D29 shared-placement configuration. Revisit under a separate
role-specific backend configuration task. Evidence: CLI placement repro's
probe_cluster_scoped.py/run_probe2.log in the scratch path recorded in
repairs/cli_placement_repair.md. No production default or role-config change is
allocated to the current two-file runner repair.
