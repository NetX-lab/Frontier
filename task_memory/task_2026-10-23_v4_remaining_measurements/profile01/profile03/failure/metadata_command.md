# W128-PROFILE03-METADATA commands and checks
## Modification History
| Date | Change |
| --- | --- |
| 2026-10-24 | Recorded bounded CPU replay, hashes, limits and phase checkpoint. |

## Original commands and environment
Adapter: `/data/ycfeng/tmp/glm53_dsv4/profile03_metadata_cpu/audit.py` (133 lines; SHA-256 `41aa293f5e4e7010d23468306a39697d5854eb7a835eabc1eb689465e230e1ab`). Parent/ranks: `/usr/bin/python3`, Python 3.10.6, GCC 11.3.0; standard library only, no runtime imports or changes.
```bash
python3 /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_cpu/audit.py run /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_cpu/run1.json
python3 /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_cpu/audit.py run /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_cpu/run2.json
cmp /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_cpu/run1.json /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_cpu/run2.json
```
Each run uses at most two concurrent rank processes; each rank's published snapshot, baseline and tmp are parsed inside:
```bash
sudo -n systemd-run --scope -p MemoryMax=30G timeout 60s /usr/bin/python3 /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_cpu/audit.py rank nodes/HOST/serving_profile/rankRANK.json
```
Replace HOST/RANK with an inventory-listed rank; parent does so deterministically. Raw bytes use 8-MiB hashing chunks, never decompression. Original outputs remain immutable; current `metadata_audit.json` matches corrected scratch `run1.json` below.

## Original results and pass conditions
- PASS: receipt inventory hash, exact 190 paths, all sizes/SHA-256, 8,208,668,144 bytes; 32 role/rank/generation joins, 16 ordered two-member DP groups and staged source bindings.
- PASS: two fresh successful CPU runs have identical full JSON and stdout digest `d4a8382cee45643ff1143cb04fdc996cc6a6c97989c0ecae331b926b37f781c7`; no execution time enters the digest.
- PASS: both original output files and preserved original `metadata_audit.json` SHA-256 `02e1d2df3a779a4902800094e098def776fcefc1dff35e7a60ea3b907ffc77b0` (3,761,389 bytes). This checks immutable artifacts, not GPU failure reproduction.
- PASS: additional CPU assertions confirm all 32 source/collector mappings agree; collector/baseline/inventory losses/errors are zero; each map has six non-null streams and equals its baseline.
- PASS: prior timeline.json/timeline.md/command.md hashes respectively remain `82406421d9c21ccda4b0b1191aef7c68d4bc058230a5519b2c23918ee14f1b5c`, `58b1901d38b258a96df876b9f6d4807983122dea7a569fcb0cb083e6bebe01e6`, `fdf63d41e188aefb22a0b3476743b2e5eba4da8bddcd85e25e8e8f4a67f9a4c0`.
- Artifact shortfall: 0/32 `.profile.json`, 16 stale snapshots, one empty tmp and one parseable unpublished tmp. Formal published two-sided pairing is unavailable; unmatched values are explicit.
- Initial adapter attempt failed locally on absent `window`; None-aware classification and separate tmp handling were corrected before both fresh successful runs. No worker/runtime behavior changed.

## Phase checkpoint
- Plan: full hash census, 32 isolated rank audits, raw-sequence partitions, source identity and two-run equality completed; raw traces/source feedback remain root-owned and unexecuted here.
- Drift: one None-window parser correction; no tooling expansion, trace parsing, platform work, serving-success inference or data admission. Final checks PASS: combined MD 86 lines; adapter 133 lines; exact candidate/pair partitions and prior three files immutable.
- Spec: requested `.profile.json` files are absent from the authoritative full inventory; stages and unpublished tmp evidence are explicit, not silently substituted.
- Route: root may correlate retained traces but must preserve missing P0/D1 formal publication and unmatched partitions; post-autostop completion timing remains unknown. Run remains FAILED.
## W128-PROFILE03-METADATA-CENSUS-FIX
- Original copies: `/data/ycfeng/tmp/glm53_dsv4/profile03_metadata_census_fix_20261024/original/profile03_metadata_cpu/{audit.py,run1.json,run2.json}` and `/data/ycfeng/tmp/glm53_dsv4/profile03_metadata_census_fix_20261024/original/failure/{metadata_audit.json,metadata_audit.md,metadata_command.md}`; exact preservation hashes: `/data/ycfeng/tmp/glm53_dsv4/profile03_metadata_census_fix_20261024/preservation.json`.
- Pinned source: `/data/ycfeng/tmp/glm53_dsv4/profile03_metadata_census_fix_20261024/candidate/audit.py`, SHA-256 `ff614c992e291f9282fb1fd4450c7d27de8c7505ca6a6de8292fec7f122fd77d`, 134 lines; source delta 3gross, maintained MD delta 20gross. Pin: `/data/ycfeng/tmp/glm53_dsv4/profile03_metadata_census_fix_20261024/pin.json`; rank limits remain max2, MemoryMax30G/timeout60s.
- PASS twice: 190 hashes/32 ranks, census32 raw traces/32 tables; corrected JSON `ab476ca22619818e50b7b142f7d2f7dedd6141740ffa979b9081cea52c1c4376` (3,761,419 bytes), nonclock `a6827373a842d0480037f89f5b1acdbfe3ef7d7c6fce529ea769b97a23bfad51`. All other fields equal after removing only adapter hash/raw count/new table count; original receipts and14 required immutable guards PASS. `/data/ycfeng/tmp/glm53_dsv4/profile03_metadata_census_fix_20261024/verification.json` records proof and concurrent non-owned test-report drift. Final command is root's fresh-output replay, not executed here; FAILED/GPU-cause limits unchanged.
```bash
python3 /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_census_fix_20261024/candidate/audit.py run /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_census_fix_20261024/run1.json
python3 /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_census_fix_20261024/candidate/audit.py run /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_census_fix_20261024/run2.json
python3 /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_census_fix_20261024/candidate/audit.py run /data/ycfeng/tmp/glm53_dsv4/profile03_metadata_census_fix_20261024/independent_root.json
```
