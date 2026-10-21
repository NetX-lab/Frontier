# Pinned native grouped NIXL decision

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded actual source-body controls and the implemented structural rule. |

Revision: `2cf0a6915ce544dc493a0990f2ea38d81601128a`, read with `git show` from
`/data/ycfeng/Frontier/.real-engine/vLLM-BS`; its working HEAD is not the source pin.

## Callsite evidence

- `vllm/models/deepseek_v4/attention.py:716-738,762-774`: main compressed KV and
  indexer use `MLAAttentionSpec`, with logical block256 and compression metadata.
- `vllm/models/deepseek_v4/compressor.py:155-203`: float32 state, CSA window8 /
  block4, HCA window128 / block8, `SlidingWindowMLASpec`. These are not Mamba
  states. `vllm/v1/kv_cache_interface.py:595` derives that spec from sliding-window
  spec; `single_type_kv_cache_manager.py:1903-1913` selects SlidingWindowManager.
- `vllm/v1/core/kv_cache_manager.py:498-511`: ordinary allocation prunes on the
  processed frontier, excluding in-flight tokens. Thus a single initial6144
  prefill at frontier0 really allocates [24,96,96,1536,768], not the window tail.
- `vllm/v1/core/sched/scheduler.py:2677-2700`: at finish, call remove_skipped_blocks
  with processed computed tokens before computed-token clipping and connector
  `request_finished_all_groups`. `kv_cache_manager.py:714-732` clips to computed
  logical coverage, not a guessed page count.
- `single_type_kv_cache_manager.py:597-661,1087-1098`: released whole pages become
  nulls; skipped count is floor(max(0,computed-window+1-extra)/group_block_size).
- `nixl/base_scheduler.py:128-150,240-284`: SW groups exchange their last
  ceil(window/block)+1 positions; Mamba speculative/single-state trimming does
  not apply to V4 states. SWA uses the accepted native window128/block64 layout.
- `nixl/pull_scheduler.py:153-167,261-279`: D exchanges unhashed/non-null IDs with
  the same clipper; P advertises clipped group IDs while retaining ownership.
- `nixl/base_worker.py:2412-2446`: without Mamba, take the P suffix matching each
  D group's count after kernel-block expansion. Here both sides share layout,
  so page selection is exactly the corresponding logical group suffix.
- `sched/scheduler.py:918-921,1033-1043`: remote reservation schedules zero local
  tokens and uses external computed tokens. `single_type_kv_cache_manager.py:
  230-279,288-325` creates skipped null prefixes then allocates only required
  external live pages. `sched/scheduler.py:2755-2762` replays prompt-1 after load.

## CPU observations at6144

| Group | Initial P allocation | P held at handoff | P connector list (incl. null) | D reserved | Selected transfer | Component bytes/page | Payload B |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compressed | 24 | 24 | 24 | 24 | 24 | 1002240 | 24053760 |
| swa_0 | 96 | 2 | 3 | 2 | 2 | 823680 | 1647360 |
| swa_1 | 96 | 2 | 3 | 2 | 2 | 786240 | 1572480 |
| csa_state_0 | 1536 | 2 | 3 | 2 | 2 | 870912 | 1741824 |
| hca_state_0 | 768 | 16 | 17 | 16 | 16 | 656640 | 10506240 |
| **Total** | **2520** | **46** | **50** | **46** | **46** | n/a | **39521664** |

`native_rules.json` binds executed native bodies, source hashes and results at
prompts191/6144/6145. Fixture objects replace platform/runtime imports; these
are actual extracted methods, not a native GPU execution or wire-byte claim.

The implementation separately snapshots full logical P and D tables (including
nulls) and selected P transfer IDs. Bytes sum existing component-page sizes,
never maximum backing-slab size or ranks. Source and destination block ID spaces
remain separate; numeric ID coincidence across the two pools is harmless.

Structural completion still uses the accepted deterministic D-ready then P-release
approximation at one timestamp. Native readiness/notification ordering and time,
actual wire payloads, measured execution and formal acceptance remain open.
