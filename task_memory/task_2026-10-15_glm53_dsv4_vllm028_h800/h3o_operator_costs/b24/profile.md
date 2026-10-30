# Candidate 300-second cProfile window

| Function | Calls | Self s | Cumulative s | Cause |
| --- | ---: | ---: | ---: | --- |
| `simulator.py:423:run` | 1 | 0.000 | 300.000 | Total sequential event-loop execution in the measured window. |
| `simulator.py:595:_run_sequential` | 1 | 0.495 | 300.000 | Sequential event queue processing. |
| `sync_entry.py:34:enter_layer_sync` | 33788 | 0.413 | 191.694 | Collects DP lanes in a layer waiting room and launches a complete EP wave. |
| `ep_wave_schedule.py:52:schedule_layer_wave` | 8447 | 0.373 | 188.031 | Builds source-cohort inputs, predicts the wave, and schedules its lane/collective events. |
| `base_cluster_scheduler.py:1035:_prepare_moe_ep_wave_plan` | 8447 | 0.077 | 185.163 | Delegates workload, phase prediction, logging and barrier preparation. |
| `ep_wave.py:80:prepare_moe_wave_from_inputs` | 8447 | 0.414 | 185.086 | Materializes wave workload, predictions and collective barriers. |
| `kernel_gap.py:209:_predict_layer_with_kernel_gap` | 79753 | 0.749 | 131.579 | Calls selected layer predictor and adds segment-specific gaps. |
| `expert_parallel.py:114:predict_ep_wave_phase_times` | 8447 | 5.404 | 130.058 | Remaining distinct V4 predictions plus required per-participant batch construction and logging. |
| `decode_sync_event.py:36:handle_event` | 22780 | 0.388 | 130.023 | Event-handler work, including scheduling and phase prediction. |
| `base_cluster_scheduler.py:1282:on_decode_sync` | 22780 | 0.061 | 129.435 | Delegates each decode-lane arrival to the layer synchronization path. |
| `sync_entry.py:197:enter_decode_sync` | 22780 | 0.033 | 129.374 | Runs the shared layer rendezvous for a decode source lane. |
| `base_cluster_scheduler.py:1109:_on_decode_ep_wave_ready` | 5695 | 0.138 | 126.981 | Schedules the completed decode cohort through the shared wave helper. |
| `sklearn_moe_execution_time_predictor.py:95:_predict_dsv4_structural_execution` | 79753 | 0.680 | 126.823 | Dispatches V4 operator prediction for attention sources or distinct FFN groups. |
| `deepseek_v4.py:342:predict_structural_execution` | 79753 | 4.560 | 126.142 | Builds operators, collective contexts, workload metadata and timing objects. |
| `sklearn_disaggregation_execution_time_predictor.py:1391:predict_stage_execution_time` | 34075 | 0.271 | 102.155 | Source attention stage prediction and reporting assembly. |

Wall time: 300.000140 s; simulated time reached: 3.612744670137 s; peak RSS: 1352464 KiB.
Cumulative values overlap because parent functions include their callees.
