# F2 local interpolation diagnostic

## Modification History

| Date | Change |
| --- | --- |
| 2026-11-01 | Added a measured-bracket prediction without changing the original single-capacity verdict. |

## Evidence and interpretation

Decision46 requests the one-GPU launch-depth comparison. The original 84-trial output and October31 analysis are retained. All six repetitions agree at every graph size. A single launch or kernel capacity has poor fit because the curve has a plateau followed by an inverse-size segment. As a descriptive check only, `min(1020, 39888.505221861 / kernels_per_graph)` has log RMSE0.017160503; these two quantities are the plateau median and geometric mean of the observed kernel totals for graph sizes64–4096. They do not establish CUDA's internal mechanism or native queue equivalence.

The current native count2240 is inside the measured2048/4096 bracket. Piecewise log-log interpolation uses only the bracket's measured depths20/10: `20 * (10 / 20) ** (log(2240 / 2048) / log(4096 / 2048)) = 18.285714285714285`. Its continuous difference from19 is -0.714285714285715 (-3.759398496%). The diagnostic replay input is18 complete forwards, one below the formal19 (-5.263157895%). No native metric is used in this interpolation. No extrapolation is allowed.

The existing global status remains **AMBIGUOUS**, `predicted_K=null`; local interpolation is separately labeled. Leave-one-size-out error at2048 is0%, at1024 is1.892915648%. The transition at32 fails a10% comparison (22.667004734%); this method is not a globally qualified launch-capacity model. Synthetic single-stream add graphs omit native concurrent streams, memory nodes and collectives. K18 is a diagnostic scenario, not a replacement of formalK19.

## Verification

`PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 timeout 60s /data/ycfeng/envs/frontier-py310/bin/python -m pytest tests/unit/test_b30b_ex_situ_probes.py -q -p no:cacheprovider --basetemp=/data/ycfeng/tmp/f2_interpolation_final_20261101 --junitxml=/data/ycfeng/tmp/f2_interpolation_final_20261101.xml`: **32 PASS in2.99s**. Checks cover the retained ambiguity, exact bracket, complete-forward count, omitted-point check, exact-endpoint integer result and refusal to extrapolate.

The complete raw84-trial CLI run returned0 using Python3.10.6 from `/data/ycfeng/envs/frontier-py310/bin/python`; exact argv, source hash, environment and timestamps are in `h3o_operator_costs/b30b/f2_analysis_20261101/command.json`. `analysis.json` retains absolute measured/predicted depth and relative error for every interior held-out size. Raw input SHA256 remains `5dfd37990d9f48679f4b28dff55a725f3f8316a9b79872042c4713aa16ac2b0f`.

Next: diagnostic P2-32/P2-64 replay with the existing max-forwards field set to18, unchanged formal CC and reference bindings; run the existing per-repeat four-metric gate on completion.
