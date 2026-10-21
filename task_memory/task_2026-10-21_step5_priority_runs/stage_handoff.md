# D21-RUNS immutable package handoff

## Modification History

| Date | Change |
| --- | --- |
| 2026-10-21 | Recorded first runnable P0 package and formal-release interface. |

## P0 first runnable package

`/data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification01/stage`

- Source commit `2033376877ac713670cdd5574b5e95fc00b0cdb4`; original config SHA256 `df52565282a2ab800577776f581c128406f644f5e27260c3a090f8aa7780cdd8`.
- Full executable worker, existing HTTP proxy/client, metadata/tokenizers, native source pins, probe helpers, role preflight and launcher callback are sealed.28 staged tools verified; no parse-only stub or R2 instrumentation.
-4 nodes x8 H800; codesign; ResourceConfig `preemptible=false`; P16/D16 with TP8/DP2/EP16/PP1 each. Locked immutable image and actual installed Step version remain unchanged.
- Qualification16 warmups -> verified idle ->8 concurrent same-shape completions. All native flags remain7168/8192/128, native graph mode, FP8 exclusions and no extra NCCL override.
- `timeout_seconds=3600`, `startup_timeout_s=1800`, `replay_timeout_s=1200`. Reservation-budget projection60 minutes /32 GPU-hours, **not** a measured duration prediction. Formal run retains90 minutes /48 GPU-hours at32 GPUs;64-GPU budgets are64/96 GPU-hours respectively.
- Native readiness and reference admission remain UNVERIFIED. Parent owns live capacity/creator/source mounts/device observations, all native runtime qualification and launch decisions. No platform call was performed here.

Parent submit command (only after live P0 quota/capacity reconciliation):

```bash
bash /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification01/stage/submit.sh
```

The wrapper first verifies the sealed package, then sources the existing StepMind environment and uses the existing personal Python RJob launcher. It introduces no new launch path. Offline dry-run:

```bash
python3 -B /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification01/stage/tests/integration/step5_pd/submit_native_pd.py \
 --config /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification01/stage/config.json \
 --launcher-scope /data/ycfeng/tmp/step5_d21_priority_20261021/h800_32_qualification01/stage/launcher_scope.py --dry-run
```

This first package is preserved byte-for-byte. Later source commits and other stages do not edit it. Do not rebuild an existing stage or replace already submitted files.

## Four case definitions

`tests/integration/step5_pd/priority_cases.json` has four entries; `canonical_stage.py build --case {h800_32,h800_64,h200_32,h200_64}` applies those entries to the existing canonical definition. Use `--kind qualification` for basic validation. This never consults30% accuracy. Native/Frontier identity declarations are in `tests/comparison/e2e_gate/cases/step5_priority.json`; legacy `step5_canonical.json` is unchanged.

H80032/H80064 are codesign non-preemptible; H20032 is step_main non-preemptible; H20064 is step_main preemptible. Parent must count **all** personal nonterminal H200 non-preemptible reservations across sessions: total<=32. There is no80/130-GPU exception. Existing historical H20064 non-preemptible evidence is preserved; do not duplicate its basic launch before parent judges it.

## Formal input interface

`--kind canonical --formal-inputs /absolute/path/inputs.json` is mandatory for a formal build. The builder freezes explicit pins, and standalone stage verification plus the submission seam rerun the same checks before credentials or SDK construction. Example schema (placeholders, **not** fabricated evidence):

```json
{
 "readiness": {
  "run_directory": "/local/case/runs/groundtruth_clean/qualification",
  "run_check": {"path": "/local/case/analysis/run_check_status.groundtruth_clean.json", "sha256": "ACTUAL_SHA256"}
 },
 "accuracy": {
  "matrix": {"path": "/local/same-hardware32/frozen_matrix.json", "sha256": "ACTUAL_SHA256"},
  "artifacts_root": "/local/same-hardware32",
  "run_directory": "/local/same-hardware32/runs/groundtruth_clean/clean_run",
  "frontier_run_directory": "/local/same-hardware32/frontier/measured"
 }
}
```

32-GPU formal runs require `readiness`, not `accuracy`.64-GPU formal runs require both. The readiness check consumes the existing `groundtruth-run-check` schema and actual pinned platform/source/native-qualification receipts, not a manually asserted boolean. Its case_id is the registry entry (for example `h800_64`). `runtime_identity` includes the case's `native_runtime`, original harness_commit/source_hashes, and `topology` exactly as declared in that case's `native_topology`. Original config/model/image/source identity and actual GPU inventory/role lane population must agree. Actual content/lifetime/runtime disposition remains owned by the existing native checker.

The accuracy matrix uses the existing matrix schema, matching same-hardware32 cell and immutable workload pins, case-bound run_check, and admitted routing. No supplied scalar error or published JSON reading releases a run: the implementation reads actual raw three-repeat client records, validates full populations, re-normalizes with the existing normalizer, and calls the existing comparator. Every one of all12 independent errors must be <=.30; exact equality passes. Pooled/diagnostic metrics cannot release. The four final10% gates and metric formulas are unchanged: a valid30% prerequisite may still have a final verdict of FAIL. Original Frontier config/truth files are pinned by the builder and must match resolved Step hardware/model/P/D TP/DP/EP/PP; no cross-case timing substitution.

No formal package is sealed here because genuine case-bound native readiness/accuracy evidence has not yet been admitted. Missing evidence blocks only formal builds/submissions, never basic qualification. No cloud reads or writes were performed while preparing these local packages; cloud paths are future worker outputs only.
