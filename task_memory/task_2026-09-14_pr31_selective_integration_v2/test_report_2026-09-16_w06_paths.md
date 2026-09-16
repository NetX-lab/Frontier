## Modification History

| Date | Summary of Changes |
| --- | --- |
| 2026-09-16 | Recorded canonical path precedence, public API preservation and strict configured-device selection. |

# W06 paths and device-family boundary

W06 / R07 / N03. Full public dictionary resolution now projects the existing MeasurementInputPaths resolver; manager, predictor initialization and family tuple APIs share that resolver. Explicit per-field values, including empty paths, win over configured/derived values. Historical DEVICE_EVENT key aliases are read centrally. Kernel-only CPU None/absent inherits eager, explicit empty stays empty. PP-specific dictionary keys and distinct tuple ordering remain unchanged. Standalone training already consumes explicit dataset_path and needs no duplicate resolver.

Both event selectors now delegate to one device-metadata resolver. A valid supplied device_config or configured SKU determines CUDA_EVENT/DEVICE_EVENT; unknown SKU or invalid platform raises rather than silently selecting CUDA. Tests use explicit valid device metadata instead of missing required predictor state.

## Verification

Environment: /usr/bin/python 3.12.3; no conda active. Command:

```bash
python -m pytest tests/unit/test_measurement_path_precedence.py tests/unit/test_measurement_family_selector.py tests/unit/test_device_timer_contract.py tests/unit/test_gdn_hybrid_e2e_increment14ab.py -q -p no:cacheprovider --tb=short > /data/ycfeng/tmp/pr33-w06-second.log 2>&1
```

**51 PASS in 9.25s.** Expected outputs include preserved partial DEVICE_EVENT overrides, explicit empties, legacy aliases, configured/derived paths, PP paths, family-specific tuple/dictionary agreement, standalone training handoff and invalid-device rejection. Real hybrid production-constructor E2E included. First run failures from new test factory misuse and old missing device fixtures were corrected in tests; observed None CPU fallback gap was corrected at the canonical boundary. No failure hidden as a production fallback. `git diff --check` PASS.

Scope limits: source includes uncommitted W03 runtime migration; that lane still has D01 pending and is not claimed green by this result. W06 uses source-confirmed old N03 plus actual post-fix regression checks; no pre-fix RED result is claimed.

## Ownership and readability

The manager's duplicated 80-line path algorithm and predictor's override/default branches were removed. Existing resolver remains the owner; a dictionary projection preserves the existing public interface instead of adding a second path policy. Strict device-family mapping is shared by two existing selectors. No standalone-training changes were needed. Giant modules retain unrelated training/prediction responsibilities; the coherent path algorithm is consolidated in the existing dependency-light owner. No further unrelated split is included.

| File | Before LOC | Current worktree LOC |
| --- | ---: | ---: |
| `frontier/execution_time_predictor/measurement_input_paths.py` | 116 | 163 |
| `frontier/execution_time_predictor/shared_prediction_model_manager.py` | 4716 | 4646 |
| `frontier/execution_time_predictor/sklearn_execution_time_predictor.py` | 8371 | 8277 |

Counts include the pending W03 changes in sklearn_execution_time_predictor; the selective W06 index patch contains only imports/path initialization/device selection. Final hunk reconciliation remains W11.
