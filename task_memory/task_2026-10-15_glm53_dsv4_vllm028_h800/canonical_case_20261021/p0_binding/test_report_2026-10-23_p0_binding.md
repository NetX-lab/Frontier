# P0 workload and metric binding verification

## Modification History
| Date | Change |
| --- | --- |
| 2026-10-23 | Recorded actual CPU results, source-derived prompt limits and P0-only acceptance. |
| 2026-10-23 | Root independently replayed the frozen command and verified every non-clock field; retained the initial wrapper-metadata comparison failure. |

Task W128-V4-P0-CASE-BINDING. **PASS_P0_WORKLOAD_METRIC_BINDING_ONLY**, not numerical calibration or all-case completion. Original/R4 case paths and120 exact input hashes are in `receipt.json`; runtime admission is reused from `../../native06_acceptance_20261023/primary_acceptance_r4.json` (R4 all8PASS), not rerun or newly granted here. No GPU/platform/network/Git/source/package operation.

## Commands and environment
- `command.md`: exact inline existing-reader command, executed via `/bin/bash -c`; `/usr/bin/python3`3.10.6, pandas2.0.1, no conda, `PYTHONDONTWRITEBYTECODE=1`, timeout60s. Exit0,3.150102s, empty stderr (`validation.stderr.log`). Parent-supplied root4425bf8b is recorded, not independently Git-queried.
- `test_command.md`: existing `test_canonical_fixed_manifest.py -k 'not builder'`; frontier-py310 Python3.10.6, no conda, timeout60s. Exit0, **3PASS/1deselected in1.03s** (`fixed_tests.log`, `fixed_tests.xml`, `test_receipt.json`). Existing accepted rebuild is not repeated; no duplicate manifest is created.
- Initial environment inspection attempted `python` and exited127 (`command not found`); site binding specifies python3. This did not execute validation or change dependencies.

## Acceptance results
| Condition | Actual | Result |
| --- | --- | --- |
| Shared files and unchanged original evidence |5/5 shared/original/R4/package copies exact;120 accessed input SHA256/bytes/mtime unchanged | PASS |
| Formal count/arrivals/lengths/totals |128 each round,0..127s,6144/1024;786432 input/131072 output/917504 total | PASS |
| All populations and independent warmups |432/432 complete;384 formal/48 warmup;16 warmups×THREE;zero missing/failed/unfinished | PASS |
| Stable template and valid prompt IDs |144 templates,884736 reconstructed IDs;100..129179 inside vocab129280, identical across3 repeats | PASS — frozen-source reconstruction |
| Existing native client/idle/metric functions |Full reader PASS,7 ordered boundaries,128 formal rows per round;stored aggregates exact, normalized formulas within1e-9 | PASS |
| Existing Frontier loader |All128 preserved, no clipping | PASS |
| Admitted P0 runtime/source/layout |R4 hash/result chain exact;native32rank scalar/group records match original/R4 bytes;TP8/DP2/EP16 | PASS — existing R4 scope |
| Full memory/Frontier/promotion/final calibration |No qualified comparison or numerical error receipt produced | OPEN |

Metrics are arithmetic means over all128 formal requests; TTFT=`1000*(first-issue)`, E2E=`1000*(last-issue)`, request TPOT=`(E2E-TTFT)/1023`, THPT=`917504/(last formal last-token-first formal submission)`. Warmups excluded; no round pooling/filtering. These are native reference values, with **no Frontier prediction and no numerical error**.

| Repeat | TTFT ms | TPOT ms | E2E ms | Total THPT tokens/s | Window s |
| --- | ---: | ---: | ---: | ---: | ---: |
|1 |403.30500297932304 |19.03109878276599 |19872.119057748932 |6273.332979026521 |146.25463100196794 |
|2 |393.21618992471485 |19.016502763090106 |19847.098516565893 |6273.129500109898 |146.2593750031665 |
|3 |399.9459093065525 |18.98073687169182 |19817.239729047287 |6269.771959563796 |146.33769871015102 |

## Decisive hashes and source limits
- Shared manifest:`5c3acb69017a9088ce20179c4a1bbfa99eadb8fbc15f6ced510496edc78c55d3`;SHA256SUMS:`594c3cb671f13dff4804dc269f957820c51fd6fe5af5f841194091c623e013b9`.
- Actual client JSONL:`5cd4858ea2110cbf4c974aa318e343e47a3d913dee593c11111df60467f894a5`;model config:`b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818`.
- Reconstructed prompt aggregate:`a585b96b162ef2e7746241ab093a638b67000f0e5a1242687ae51a3355051859` (encoding/order in receipt). This is a newly computed source reconstruction digest, **not a captured HTTP/device-input hash**. Existing pinned native client uses stable template IDs, not repeat-qualified transport IDs (`vllm_replay.py:513–525`);native stream usage counts derive from returned prompt IDs (`vllm/entrypoints/openai/completion/serving.py:312–326`).
- Source inspection `vllm/v1/engine/input_processor.py:484–510` shows a tokenizer-conditional vocabulary check. Actual native command uses `--skip-tokenizer-init`; guard execution is not claimed. Direct reconstructed bounds settle prompt validity, not retained raw arrays/output-ID vocab coverage.
- R4 checker:`67f3192302b6d12c0b8296581ee8b2788a772bce80b158ad45bc349eacb32a7b`;result:`71bc110b8162af3672460d59a4ff95e9e8b224ad45605eb43753b983b83c5faa`. OriginalR3 FAIL/native05 UNVERIFIED untouched.

## Native memory and remaining gates
All32 rank records retain54740 grouped cache blocks;native available KV54863013684..54879790900B,requested72274785076B,total_consumed15437135872..15453913088B,peak_activation1957858304B,graph estimate0. These are native scalars, not a complete independent stage-peak/non-Torch/graph ledger or Frontier capacity equality. No naive block×256 token capacity is claimed for grouped caches. R4's physical P NONE/D PIECEWISE+FULL, MXFP4/MARLIN and fp8_ds_mla remain reference execution evidence, not measured costs. Other hardware32/64, qualified physicalTP8/operator/graph costs, full non-dummy prediction, independent10% and30% gates, READ/device completion and observer-overhead claims remain unverified.

## Root independent replay

At2026-10-23T13:02:49Z, root asserted owner receipt SHA256 `56583585cf83fa0b82676e637342b2c932309fb9f81f491e26f000ec4f7698fb` and its command hash, extracted the exact Python body from `command.md`, and executed `/usr/bin/python3 -B -` with that body on stdin, `PYTHONDONTWRITEBYTECODE=1`, the recorded root working directory and timeout60s. Result: exit0, empty stderr,3.190564s. Evidence is `/data/ycfeng/tmp/glm53_dsv4/p0_binding_root_bs0volv_/{stdout.json,stderr.log}`; hashes and acceptance are retained in `root_verification.json`.

The initial receipt-equality harness failed because the owner appended `command_execution` outside the command's stdout. Root compared all24 common non-clock fields exactly (including environment,120 input pins, metrics and all scope limitations) and separately asserted the owner runner's exit0/empty stderr. Only three clock fields differ. The successful analysis was not rerun and no evidence was rewritten. **PASS / ACCEPTED_P0_WORKLOAD_METRIC_BINDING**, within the scope above.
