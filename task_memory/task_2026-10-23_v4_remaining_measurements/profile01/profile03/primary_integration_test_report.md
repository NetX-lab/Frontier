# Integrated collection repair verification

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-24 | Accepted merged diagnostic DP and collection repairs after twelve independent-process suites and source/evidence checks. |

## Scope and source

Runtime source1437ace6 includes DP sourcebb50186e (merge19f9ce60) and collection source20660f4c. Both components received independent Standards and Spec reviews with No findings. The collection reviewers were `ses_e93b9299effeDSxga0hQqLu7Yr` and `ses_e93b92a39ffefgem0OQuvAwWAN`; DP reviews are recorded in `../dp_sequence/review.md`.

The collection review inspected both full changed files, source diff, actual stop_server call path, requirements/design, original RED controls and the existing77-case comparison. It confirmed role override copying, shared known-peer paths, peer-budget post-stop-marker waits, independent monotonic timings and preserved native/soft-error semantics. Reviewers did not execute tests or rehash inputs; root verification below supplies that evidence.

## Actual verification

Working directory `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`, HEAD1437ace6; `/data/ycfeng/envs/asyncopd-py312/bin/python`, Python3.12.13, no conda activation. Each suite used a separate process, at most two concurrent, through `sudo -n systemd-run --scope --quiet -p MemoryMax=30G timeout 60s runuser -u brainpp -- env ... python -B -m pytest`. CUDA_VISIBLE_DEVICES was empty; exact commands, environments, temporary paths and timestamps are in `receipt.json`. C26/legacy fixtures match the earlier root checks byte-for-byte.

| Suite | Expected/actual |
| --- | --- |
| native_pd_worker |36PASS|
| native_pd_manifest |14PASS|
| native_serve_profile |36PASS|
| native_streaming_replay |39PASS|
| native_cold_start |18PASS|
| native_pd_canonical_http |1PASS|
| calibration_pd_harness |68PASS|
| dp_placement_harness |75PASS|
| native_v2_serving_inventory |46PASS|
| native_serving_inventory |28PASS|
| native_groundtruth_check |98PASS/1known baseline FAIL|
| native_full_structure |6PASS|

Total465PASS/1FAIL/0SKIP, all expected counts and failure identity matched. The sole failed32-rank positive fixture still omits dp_utils, exactly as baseline44f2ddb9; it is not counted as passed. Both versions'32 counter/hook dictionaries were already independently compared (`../dp_sequence/baseline_diagnosis.json`). No source, test or checker was modified to suppress that failure.

Root also passed46 checks of results, all six merged code/test hashes against Git, collection record hashes, unchanged native stop source and the complete historical77-case/domain binding. The retained logs/XML and `primary_acceptance.json` carry file hashes. No new matrix capture was needed because simulator/input domains remain identical.

## Decision and limits

Accept source/CPU components for a fresh diagnostic package. Per-role512/64 windows and600000record cap remain explicit next-package inputs. CPU checks do not establish actual >=3 useful steps, raw cross-DP pairing, no drops, stream dependencies, GPU shutdown order or captured operator ownership. Those are measured and audited separately; no native/C26/reference/operator/E2E admission changes.
