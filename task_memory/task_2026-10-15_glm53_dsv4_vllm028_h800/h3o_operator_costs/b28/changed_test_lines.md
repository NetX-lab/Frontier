# Changed test lines

Every added/removed line from the four affected test files is listed below. Base refers to the read-only b28-base tree; candidate refers to the final working file. No skip, xfail or assertion relaxation was added. Blank lines are included for completeness.

## tests/unit/test_glm_sp_moe_comm.py

| Side | Line | Text | Reason |
| --- | ---: | --- | --- |
| base removed | 100 | def test_equal_sp_rows_with_unequal_raw_tokens_are_covered(role): | Check uniform routing for dispatch and combine; rename the exact-path fail hook. |
| candidate added | 100 | @pytest.mark.parametrize("name", ["expert_parallel_alltoall_dispatch", "expert_parallel_alltoall_combine"]) | Check uniform routing for dispatch and combine; rename the exact-path fail hook. |
| candidate added | 101 | def test_equal_sp_rows_with_unequal_raw_tokens_are_covered(role, name): | Check uniform routing for dispatch and combine; rename the exact-path fail hook. |
| base removed | 104 | predictor._cc_backend._predict_unequal = lambda *_, **__: pytest.fail("uniform SP must not use exact-pattern lookup") | Check uniform routing for dispatch and combine; rename the exact-path fail hook. |
| base removed | 105 | operator = get_comm_operator("expert_parallel_alltoall_combine") | Check uniform routing for dispatch and combine; rename the exact-path fail hook. |
| candidate added | 105 | predictor._cc_backend._predict_exact_sp = lambda *_, **__: pytest.fail("uniform SP must not use exact-pattern lookup") | Check uniform routing for dispatch and combine; rename the exact-path fail hook. |
| candidate added | 106 | operator = get_comm_operator(name) | Check uniform routing for dispatch and combine; rename the exact-path fail hook. |
| base removed | 134 | def test_unequal_dispatch_keeps_largest_rank_uniform_curve(): | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| candidate added | 135 | def test_unequal_dispatch_uses_its_exact_measured_row(): | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| candidate added | 136 | import csv | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| candidate added | 137 | from statistics import median | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| candidate added | 138 | (blank) | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| base removed | 138 | predictor._cc_backend._predict_unequal = lambda *_, **__: pytest.fail("dispatch must not use exact-pattern lookup") | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| base removed | 140 | assert predictor._predict_comm_operator_with_context(operator, ctx) == ( | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| base removed | 141 | predictor._cc_backend.predict_allgather(395264, 32, comm_domain="EP") | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| base removed | 142 | ) | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| candidate added | 143 | size = operator.build_payload_bytes(ctx) | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| candidate added | 144 | assert size == 395264 | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| candidate added | 145 | measured = [float(row["median_us"]) for row in csv.DictReader( | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| candidate added | 146 | (ROOT / "data/profiling/network/h800_dgx/summary.csv").open()) | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| candidate added | 147 | if row["op"] == "all_gatherv" and row["tokens_per_dp_rank"] == "256/1/1/1"] | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| candidate added | 148 | assert len(measured) == 6 | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| candidate added | 149 | exact = predictor._predict_comm_operator_with_context(operator, ctx) | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| candidate added | 150 | assert exact == median(measured) / 1000 | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| candidate added | 151 | assert exact != predictor._cc_backend.predict_allgather(size, 32, comm_domain="EP") | Replace the superseded uniform rule with exact H800 median evidence and require a different scalar-curve price. |
| base removed | 145 | def test_unequal_combine_other_backends_keep_scalar_prediction(): | Extend the unchanged scalar-backend contract to dispatch; preserve the combine assertion. |
| candidate added | 154 | @pytest.mark.parametrize("name,alias,size", [("expert_parallel_alltoall_dispatch", "allgather", 395264), | Extend the unchanged scalar-backend contract to dispatch using the fixture actual alias-specific divisors. |
| candidate added | 155 | ("expert_parallel_alltoall_combine", "reduce_scatter", 3440640)]) | Extend the unchanged scalar-backend contract to dispatch using the fixture actual alias-specific divisors. |
| candidate added | 156 | def test_unequal_ep_other_backends_keep_scalar_prediction(name, alias, size): | Extend the unchanged scalar-backend contract to dispatch using the fixture actual alias-specific divisors. |
| base removed | 148 | operator = get_comm_operator("expert_parallel_alltoall_combine") | Extend the unchanged scalar-backend contract to dispatch; preserve the combine assertion. |
| base removed | 149 | assert predictor._predict_comm_operator_with_context(operator, ctx) == 3440640 / 5000 + 32 | Extend the unchanged scalar-backend contract to dispatch; preserve the combine assertion. |
| base removed | 150 | assert predictor._cc_backend.calls[-1]["collective_alias"] == "reduce_scatter" | Extend the unchanged scalar-backend contract to dispatch; preserve the combine assertion. |
| candidate added | 159 | operator = get_comm_operator(name) | Extend the unchanged scalar-backend contract to dispatch using the fixture actual alias-specific divisors. |
| candidate added | 160 | divisor = 3000 if alias == "allgather" else 5000 | Extend the unchanged scalar-backend contract to dispatch using the fixture actual alias-specific divisors. |
| candidate added | 161 | assert predictor._predict_comm_operator_with_context(operator, ctx) == size / divisor + 32 | Extend the unchanged scalar-backend contract to dispatch using the fixture actual alias-specific divisors. |
| candidate added | 162 | assert predictor._cc_backend.calls[-1]["collective_alias"] == alias | Extend the unchanged scalar-backend contract to dispatch using the fixture actual alias-specific divisors. |

## tests/unit/test_measured_cc_backend.py

| Side | Line | Text | Reason |
| --- | ---: | --- | --- |
| base removed | 61 | def _unequal_combine_row(time_us=339.5): | Rename the synthetic row helper and add dispatch evidence with its reversed input/output geometry. |
| candidate added | 61 | def _unequal_ep_row(time_us=339.5, *, dispatch=False): | Rename the synthetic row helper and add dispatch evidence with its reversed input/output geometry. |
| candidate added | 69 | if dispatch: | Rename the synthetic row helper and add dispatch evidence with its reversed input/output geometry. |
| candidate added | 70 | row.update(call="moe_dispatch_all_gatherv", op="all_gatherv", | Rename the synthetic row helper and add dispatch evidence with its reversed input/output geometry. |
| candidate added | 71 | max_input_bytes=395264, max_output_bytes=3458560, | Rename the synthetic row helper and add dispatch evidence with its reversed input/output geometry. |
| candidate added | 72 | kernels="ncclDevKernel_AllGatherV_RING_LL(args)") | Rename the synthetic row helper and add dispatch evidence with its reversed input/output geometry. |
| base removed | 156 | def test_uneven_dispatch_uses_caller_supplied_largest_rank_bytes(tmp_path): | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| base removed | 157 | rows = [_row(), _row(400, 40)] | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| base removed | 158 | for row in rows: | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| base removed | 159 | row.update(op="all_gatherv", group="ep", group_size=32) | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| base removed | 160 | backend = _backend(tmp_path, rows) | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| base removed | 161 | rank_bytes = [100, 100, 100, 400] | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| base removed | 162 | assert backend.predict_allgather(max(rank_bytes), 32, comm_domain="EP") == .04 | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| base removed | 163 | # Unmeasured size vectors still fail instead of using the uniform curve. | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| base removed | 164 | error = MeasuredCCOutOfRangeError( | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| base removed | 165 | ("reduce_scatterv", "ep", 32), sum(rank_bytes), (100, 400), | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| base removed | 166 | reason="unequal SP rows select an unmeasured combine kernel", | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| base removed | 167 | ) | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| base removed | 168 | assert "unmeasured combine kernel" in str(error) | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| candidate added | 160 | def test_uneven_dispatch_uses_exact_ordered_rows_and_caller_supplied_bytes(tmp_path): | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| candidate added | 161 | from frontier.operators.spec import CollectiveImplementation | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| candidate added | 162 | (blank) | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| candidate added | 163 | uniform = _row(395264, 40) | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| candidate added | 164 | uniform.update(op="all_gatherv", group="ep", group_size=32) | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| candidate added | 165 | backend = _backend(tmp_path, [uniform, _unequal_ep_row(60, dispatch=True)]) | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| candidate added | 166 | args = ("allgather", CollectiveImplementation.GROUP_COORDINATOR_DIM0) | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| candidate added | 167 | assert backend.predict_operator_collective(*args, 395264, 32, "EP", (32, 1, 1, 1)) == .06 | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| candidate added | 168 | assert backend.predict_allgather(395264, 32, comm_domain="EP") == .04 | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| candidate added | 169 | for size, rows in [(395265, (32, 1, 1, 1)), (395264, (1, 32, 1, 1))]: | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| candidate added | 170 | with pytest.raises(MeasuredCCOutOfRangeError, match="no exact collective measurement"): | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| candidate added | 171 | backend.predict_operator_collective(*args, size, 32, "EP", rows) | Replace the scalar-only check with actual exact-row lookup and typed rejection of missing bytes/order. |
| base removed | 175 | rows = [uniform, *[_unequal_combine_row(t) for t in (339.0, 339.5, 340.0)]] | Update the shared synthetic row-helper name; preserve the existing combine assertion and behavior. |
| candidate added | 178 | rows = [uniform, *[_unequal_ep_row(t) for t in (339.0, 339.5, 340.0)]] | Update the shared synthetic row-helper name; preserve the existing combine assertion and behavior. |
| base removed | 191 | backend = _backend(tmp_path, [_row(), _unequal_combine_row()]) | Update the shared synthetic row-helper name; preserve the existing combine assertion and behavior. |
| candidate added | 194 | backend = _backend(tmp_path, [_row(), _unequal_ep_row()]) | Update the shared synthetic row-helper name; preserve the existing combine assertion and behavior. |
| base removed | 199 | backend = _backend(tmp_path, [_row(), _unequal_combine_row()]) | Update the shared synthetic row-helper name; preserve the existing combine assertion and behavior. |
| candidate added | 202 | backend = _backend(tmp_path, [_row(), _unequal_ep_row()]) | Update the shared synthetic row-helper name; preserve the existing combine assertion and behavior. |
| base removed | 216 | row = _unequal_combine_row() | Use the shared row helper and call-neutral validation error; preserve the full combine rejection matrix. |
| candidate added | 219 | row = _unequal_ep_row() | Use the shared row helper and call-neutral validation error; preserve the full combine rejection matrix. |
| base removed | 218 | with pytest.raises(ValueError, match="retained SP combine"): | Use the shared row helper and call-neutral validation error; preserve the full combine rejection matrix. |
| candidate added | 221 | with pytest.raises(ValueError, match="retained SP collective"): | Use the shared row helper and call-neutral validation error; preserve the full combine rejection matrix. |
| base removed | 222 | def test_equal_sp_holdout_stays_out_of_exact_and_uniform_samples(tmp_path): | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| base removed | 223 | row = _unequal_combine_row(1000) | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 225 | @pytest.mark.parametrize("change", [ | Reject an ordinary AllGather kernel, a noninteger input-row byte width and inconsistent output bytes. |
| candidate added | 226 | {"kernels": "ncclDevKernel_AllGather_RING_LL(args)"}, | Reject an ordinary AllGather kernel, a noninteger input-row byte width and inconsistent output bytes. |
| candidate added | 227 | {"max_input_bytes": 395265}, | Reject an ordinary AllGather kernel, a noninteger input-row byte width and inconsistent output bytes. |
| candidate added | 228 | {"max_output_bytes": 3458561}, | Reject an ordinary AllGather kernel, a noninteger input-row byte width and inconsistent output bytes. |
| candidate added | 229 | ]) | Reject an ordinary AllGather kernel, a noninteger input-row byte width and inconsistent output bytes. |
| candidate added | 230 | def test_invalid_retained_unequal_dispatch_kernel_or_shape_is_rejected(tmp_path, change): | Reject an ordinary AllGather kernel, a noninteger input-row byte width and inconsistent output bytes. |
| candidate added | 231 | row = _unequal_ep_row(dispatch=True) | Reject an ordinary AllGather kernel, a noninteger input-row byte width and inconsistent output bytes. |
| candidate added | 232 | row.update(change) | Reject an ordinary AllGather kernel, a noninteger input-row byte width and inconsistent output bytes. |
| candidate added | 233 | with pytest.raises(ValueError, match="retained SP collective"): | Reject an ordinary AllGather kernel, a noninteger input-row byte width and inconsistent output bytes. |
| candidate added | 234 | _backend(tmp_path, [_row(), row]) | Reject an ordinary AllGather kernel, a noninteger input-row byte width and inconsistent output bytes. |
| candidate added | 235 | (blank) | Spacing supporting the adjacent test change. |
| candidate added | 236 | (blank) | Spacing supporting the adjacent test change. |
| candidate added | 237 | @pytest.mark.parametrize("dispatch", [False, True]) | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 238 | def test_equal_sp_holdout_stays_out_of_exact_and_uniform_samples(tmp_path, dispatch): | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 239 | row = _unequal_ep_row(1000, dispatch=dispatch) | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 243 | if dispatch: | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 244 | row.update(max_input_bytes=12352, max_output_bytes=395264, | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 245 | kernels="ncclDevKernel_AllGather_RING_LL(args)") | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| base removed | 229 | assert not backend._unequal_combine_times | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 248 | assert len(backend.describe_holdouts()) == 1 | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 249 | assert not backend._exact_sp_times | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| base removed | 231 | _combine(backend, 393216, 32, (1, 1, 1, 1), comm_domain="EP") | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 251 | if dispatch: | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 252 | from frontier.operators.spec import CollectiveImplementation | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 253 | (blank) | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 254 | backend.predict_operator_collective("allgather", CollectiveImplementation.GROUP_COORDINATOR_DIM0, | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 255 | 12352, 32, "EP", (1, 1, 1, 1)) | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 256 | else: | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| candidate added | 257 | _combine(backend, 393216, 32, (1, 1, 1, 1), comm_domain="EP") | Check D4 for both EP calls, including an empty exact table, one holdout and missing uniform coverage. |
| base removed | 235 | small = _unequal_combine_row(525.3) | Update the shared synthetic row-helper name; preserve the existing combine assertion and behavior. |
| candidate added | 261 | small = _unequal_ep_row(525.3) | Update the shared synthetic row-helper name; preserve the existing combine assertion and behavior. |
| base removed | 238 | backend = _backend(tmp_path, [_row(), small, _unequal_combine_row()]) | Update the shared synthetic row-helper name; preserve the existing combine assertion and behavior. |
| candidate added | 264 | backend = _backend(tmp_path, [_row(), small, _unequal_ep_row()]) | Update the shared synthetic row-helper name; preserve the existing combine assertion and behavior. |
| base removed | 426 | def test_checked_in_unequal_combine_samples_match_producer_medians(): | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| candidate added | 452 | def test_checked_in_unequal_ep_samples_match_producer_medians(): | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 455 | from frontier.config.model_config import BaseModelConfig | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 456 | from frontier.config.parallel_semantics import sequence_parallel_moe_token_widths | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 457 | from frontier.operators.sequence_parallel_moe import ep_workload | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 458 | from frontier.operators.spec import CollectiveImplementation | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 463 | model = BaseModelConfig.create_from_name("zai-org/GLM-5.3") | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 464 | calls = {"all_gatherv": ("expert_parallel_alltoall_dispatch", "allgather"), | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 465 | "reduce_scatterv": ("expert_parallel_alltoall_combine", "reduce_scatter")} | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| base removed | 436 | if "single_active" not in row["scenario"] or row["op"] != "reduce_scatterv": | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| candidate added | 469 | if "single_active" not in row["scenario"] or row["op"] not in calls: | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| base removed | 439 | sp_rows = tuple(math.ceil(tokens / 8) for tokens in raw_tokens) | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| base removed | 440 | if len(set(sp_rows)) == 1: | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| candidate added | 472 | widths = sequence_parallel_moe_token_widths(raw_tokens, 8) | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 473 | if len(set(widths.local_tokens)) == 1: | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| base removed | 442 | assert "ncclDevKernel_Reduce_Sum_bf16_" in row["kernels"] | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| candidate added | 475 | name, _ = calls[row["op"]] | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 476 | prefix = ("ncclDevKernel_AllGatherV_" if row["op"] == "all_gatherv" | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 477 | else "ncclDevKernel_Reduce_Sum_bf16_") | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 478 | assert prefix in row["kernels"] | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| base removed | 444 | assert size == 8 * sum(sp_rows) * 6144 * 2 | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| base removed | 445 | samples[(sp_rows, size)].append(float(row["median_us"])) | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| base removed | 446 | assert sum(len(times) for times in samples.values()) == 30 | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| base removed | 447 | assert len(samples) == len(backend._unequal_combine_times) == 9 | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| base removed | 448 | assert {rows for rows, size in samples} == { | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| base removed | 449 | (x, 1, 1, 1) for x in (4, 8, 16, 32, 64, 128, 256, 512, 1024) | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| base removed | 450 | } | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| base removed | 451 | for (sp_rows, size), times in samples.items(): | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| base removed | 452 | assert _combine(backend, size, 32, sp_rows, "EP") == median(times) / 1000 | Generalize the sample census to both EP calls and preserve combine kernel/size/median checks. |
| candidate added | 480 | assert size == ep_workload(name, model, model.num_experts_per_tok, widths)["data_size_bytes"] | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 481 | expected_output = (widths.gathered_tokens * (size // max(widths.local_tokens)) | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 482 | if row["op"] == "all_gatherv" | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 483 | else max(widths.local_tokens) * (size // widths.gathered_tokens)) | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 484 | assert int(row["max_output_bytes"]) == expected_output | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 485 | samples[((row["op"], "ep", 32), widths.local_tokens, size)].append(float(row["median_us"])) | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 486 | assert sum(len(times) for times in samples.values()) == 60 | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 487 | assert len(samples) == len(backend._exact_sp_times) == 18 | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 488 | for op, (_, alias) in calls.items(): | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 489 | assert {rows for (key, rows, size) in samples if key[0] == op} == { | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 490 | (x, 1, 1, 1) for x in (4, 8, 16, 32, 64, 128, 256, 512, 1024) | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 491 | } | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 492 | for (key, sp_rows, size), times in samples.items(): | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 493 | if key[0] == op: | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 494 | assert backend.predict_operator_collective( | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 495 | alias, CollectiveImplementation.GROUP_COORDINATOR_DIM0, | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 496 | size, 32, "EP", sp_rows, | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |
| candidate added | 497 | ) == median(times) / 1000 | Generalize the sample census to 60 raw rows/18 pooled entries, with both kernel proofs and producer geometry. |

## tests/unit/test_v4_measured_collective_consumer.py

| Side | Line | Text | Reason |
| --- | ---: | --- | --- |
| base removed | 130 | if len(set(widths.local_tokens)) > 1: | Assert supplied ordered rows and measured prices for both calls, including uniform cases. |
| base removed | 131 | key = (("reduce_scatterv", "ep", len(lanes) * 8), widths.local_tokens, size) | Assert supplied ordered rows and measured prices for both calls, including uniform cases. |
| base removed | 132 | assert measured_collective_time(combine, ctx, h200, size) == h200._unequal_combine_times[key] / 1000 | Assert supplied ordered rows and measured prices for both calls, including uniform cases. |
| candidate added | 130 | for operator, curve in [(dispatch, "all_gatherv"), (combine, "reduce_scatterv")]: | Assert supplied ordered rows and measured prices for both calls, including uniform cases. |
| candidate added | 131 | size = operator.build_payload_bytes(ctx) | Assert supplied ordered rows and measured prices for both calls, including uniform cases. |
| candidate added | 132 | key = (curve, "ep", len(lanes) * 8) | Assert supplied ordered rows and measured prices for both calls, including uniform cases. |
| candidate added | 133 | assert operator.sp_rows_builder(ctx) == widths.local_tokens | Assert supplied ordered rows and measured prices for both calls, including uniform cases. |
| candidate added | 134 | if len(set(widths.local_tokens)) > 1: | Assert supplied ordered rows and measured prices for both calls, including uniform cases. |
| candidate added | 135 | assert h200.uses_exact_sp_rows(key, widths.local_tokens) | Assert supplied ordered rows and measured prices for both calls, including uniform cases. |
| candidate added | 136 | expected = h200._exact_sp_times[(key, widths.local_tokens, size)] | Assert supplied ordered rows and measured prices for both calls, including uniform cases. |
| candidate added | 137 | else: | Assert supplied ordered rows and measured prices for both calls, including uniform cases. |
| candidate added | 138 | expected = h200._curves[key].predict_us(size) | Assert supplied ordered rows and measured prices for both calls, including uniform cases. |
| candidate added | 139 | assert measured_collective_time(operator, ctx, h200, size) == expected / 1000 | Assert supplied ordered rows and measured prices for both calls, including uniform cases. |
| candidate added | 150 | ("deepseek-ai/DeepSeek-V4-Flash", (1, 6144), "dsv4_ep_dispatch_allgather"), | Generalize renamed-operator coverage to both V4/GLM calls, proving selection ignores operator names. |
| candidate added | 152 | ("zai-org/GLM-5.3", (64, 1, 1, 1), "expert_parallel_alltoall_dispatch"), | Generalize renamed-operator coverage to both V4/GLM calls, proving selection ignores operator names. |
| base removed | 145 | def test_exact_combine_is_spec_driven_with_renamed_operator(profile, lanes, name, h200): | Generalize renamed-operator coverage to both V4/GLM calls, proving selection ignores operator names. |
| candidate added | 154 | def test_exact_ep_collective_is_spec_driven_with_renamed_operator(profile, lanes, name, h200): | Generalize renamed-operator coverage to both V4/GLM calls, proving selection ignores operator names. |
| base removed | 147 | operator = replace(get_comm_operator(name), name="another_combine_name") | Generalize renamed-operator coverage to both V4/GLM calls, proving selection ignores operator names. |
| candidate added | 156 | operator = replace(get_comm_operator(name), name="another_collective_name") | Generalize renamed-operator coverage to both V4/GLM calls, proving selection ignores operator names. |
| base removed | 149 | if name == "expert_parallel_alltoall_combine": | Generalize renamed-operator coverage to both V4/GLM calls, proving selection ignores operator names. |
| candidate added | 158 | if profile == "zai-org/GLM-5.3": | Generalize renamed-operator coverage to both V4/GLM calls, proving selection ignores operator names. |
| base removed | 154 | key = (("reduce_scatterv", "ep", len(lanes) * 8), operator.sp_rows_builder(ctx), size) | Generalize renamed-operator coverage to both V4/GLM calls, proving selection ignores operator names. |
| base removed | 155 | assert measured_collective_time(operator, ctx, backend, size) == backend._unequal_combine_times[key] / 1000 | Generalize renamed-operator coverage to both V4/GLM calls, proving selection ignores operator names. |
| candidate added | 163 | curve = "all_gatherv" if operator.resolve_collective_alias(ctx) == "allgather" else "reduce_scatterv" | Generalize renamed-operator coverage to both V4/GLM calls, proving selection ignores operator names. |
| candidate added | 164 | key = ((curve, "ep", len(lanes) * 8), operator.sp_rows_builder(ctx), size) | Generalize renamed-operator coverage to both V4/GLM calls, proving selection ignores operator names. |
| candidate added | 165 | assert measured_collective_time(operator, ctx, backend, size) == backend._exact_sp_times[key] / 1000 | Generalize renamed-operator coverage to both V4/GLM calls, proving selection ignores operator names. |
| base removed | 164 | assert "exact per-root Reduce" in caught.value.reason | Use the call-neutral exact-measurement error text; keep the combine typed-key assertion. |
| candidate added | 174 | assert "exact collective measurement" in caught.value.reason | Use the call-neutral exact-measurement error text; keep the combine typed-key assertion. |
| candidate added | 175 | (blank) | Spacing supporting the adjacent test change. |
| candidate added | 176 | (blank) | Spacing supporting the adjacent test change. |
| candidate added | 177 | @pytest.mark.parametrize("size_delta,lanes", [(1, (6144, 1)), (0, (6144, 16))]) | Exercise V4 operator/context rejection for both an absent byte size and an absent SP vector. |
| candidate added | 178 | def test_unmeasured_dispatch_size_or_ordered_vector_is_typed(h200, size_delta, lanes): | Exercise V4 operator/context rejection for both an absent byte size and an absent SP vector. |
| candidate added | 179 | ctx, _ = context(lanes) | Exercise V4 operator/context rejection for both an absent byte size and an absent SP vector. |
| candidate added | 180 | operator = get_comm_operator("dsv4_ep_dispatch_allgather") | Exercise V4 operator/context rejection for both an absent byte size and an absent SP vector. |
| candidate added | 181 | with pytest.raises(MeasuredCCOutOfRangeError) as caught: | Exercise V4 operator/context rejection for both an absent byte size and an absent SP vector. |
| candidate added | 182 | measured_collective_time(operator, ctx, h200, operator.build_payload_bytes(ctx) + size_delta) | Exercise V4 operator/context rejection for both an absent byte size and an absent SP vector. |
| candidate added | 183 | assert caught.value.curve_key == ("all_gatherv", "ep", 16) | Exercise V4 operator/context rejection for both an absent byte size and an absent SP vector. |
| candidate added | 184 | assert "exact collective measurement" in caught.value.reason | Exercise V4 operator/context rejection for both an absent byte size and an absent SP vector. |
| candidate added | 185 | (blank) | Spacing supporting the adjacent test change. |
| candidate added | 186 | (blank) | Spacing supporting the adjacent test change. |
| candidate added | 187 | @pytest.mark.parametrize("lanes,dispatch_us,combine_us", [ | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 188 | ((6144, 1), 560.2, 734.6), ((6144, 1, 1, 1), 1043.0, 2172.0), | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 189 | ]) | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 190 | def test_h200_eager_ep_prices_equal_isolated_decision(h200, lanes, dispatch_us, combine_us): | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 191 | import csv | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 192 | from statistics import median | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 193 | (blank) | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 194 | ctx, _ = context(lanes) | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 195 | ep = len(lanes) * 8 | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 196 | source = ROOT / "data/profiling/network/h200_dgx/summary.csv" | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 197 | # build_table verifies these copied lines against the hashed isolated sources. | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 198 | scenario = "eager_t" + "_".join(map(str, lanes)) | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 199 | with source.open(newline="") as stream: | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 200 | rows = [row for row in csv.DictReader(stream) if row["scenario"] == scenario and int(row["group_size"]) == ep] | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 201 | for name, call, expected_us in [ | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 202 | ("dsv4_ep_dispatch_allgather", "moe_dispatch_all_gatherv", dispatch_us), | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 203 | ("dsv4_ep_output_reduce_scatter", "moe_combine_reduce_scatterv", combine_us), | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 204 | ]: | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 205 | operator = get_comm_operator(name) | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 206 | samples = [float(row["median_us"]) for row in rows if row["call"] == call] | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 207 | assert len(samples) == 3 | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 208 | time_ms = measured_collective_time(operator, ctx, h200, operator.build_payload_bytes(ctx)) | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 209 | assert time_ms == median(samples) / 1000 | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 210 | assert round(time_ms * 1000, 1) == expected_us | Assert the four decision prices and unrounded medians of the three checked-in isolated rows. |
| candidate added | 211 | (blank) | Spacing supporting the adjacent test change. |
| candidate added | 212 | (blank) | Spacing supporting the adjacent test change. |
| candidate added | 213 | @pytest.mark.parametrize("network", ["h200_dgx", "h800_dgx"]) | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 214 | def test_exact_dispatch_and_combine_vectors_have_identical_coverage(network): | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 215 | backend = MeasuredCCBackend( | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 216 | MeasuredCCBackendConfig(summary_csv_path=str(ROOT / "data/profiling/network" / network / "summary.csv")), | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 217 | ClusterType.MONOLITHIC, network.split("_")[0], network, 32, | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 218 | ) | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 219 | dispatch = {(key[2], rows) for key, rows, size in backend._exact_sp_times if key[0] == "all_gatherv"} | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 220 | combine = {(key[2], rows) for key, rows, size in backend._exact_sp_times if key[0] == "reduce_scatterv"} | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 221 | assert dispatch and dispatch == combine | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 222 | import csv | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 223 | (blank) | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 224 | model = BaseModelConfig.create_from_name( | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 225 | "deepseek-ai/DeepSeek-V4-Flash" if network == "h200_dgx" else "zai-org/GLM-5.3") | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 226 | measured_rows = 0 | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 227 | with Path(backend._source_path).open(newline="") as stream: | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 228 | for row in csv.DictReader(stream): | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 229 | if row["group"] != "ep": | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 230 | continue | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 231 | tokens = tuple(int(t) for t in row["tokens_per_dp_rank"].split("/")) | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 232 | ep = int(row["group_size"]) | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 233 | widths = sequence_parallel_moe_token_widths(tokens, ep // len(tokens)) | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 234 | size = int(row["max_input_bytes"]) | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 235 | key = ((row["op"], "ep", ep), widths.local_tokens, size) | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 236 | if key not in backend._exact_sp_times: | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 237 | continue | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 238 | name = ("dsv4_ep_dispatch_allgather" if row["op"] == "all_gatherv" | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 239 | else "dsv4_ep_output_reduce_scatter") | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 240 | producer = ep_workload(name, model, model.num_experts_per_tok, widths) | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 241 | assert size == producer["data_size_bytes"] | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 242 | input_rows, output_rows = ((max(widths.local_tokens), widths.gathered_tokens) | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 243 | if row["op"] == "all_gatherv" | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 244 | else (widths.gathered_tokens, max(widths.local_tokens))) | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 245 | assert size % input_rows == 0 | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 246 | assert int(row["max_output_bytes"]) == output_rows * (size // input_rows) | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 247 | measured_rows += 1 | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| candidate added | 248 | assert measured_rows == (96 if network == "h200_dgx" else 60) | Check vector parity and producer input/output geometry for all exact raw rows in both checked-in tables. |
| base removed | 296 | assert len(h200._unequal_combine_times) == 16 | Require 32 entries (16 per call) and zero H200 holdouts after dispatch retention. |
| base removed | 297 | assert len(h200.describe_holdouts()) == 48 | Require 32 entries (16 per call) and zero H200 holdouts after dispatch retention. |
| base removed | 298 | assert {row["call"] for row in h200.describe_holdouts()} == {"moe_dispatch_all_gatherv"} | Require 32 entries (16 per call) and zero H200 holdouts after dispatch retention. |
| base removed | 299 | assert all(row["curve_us"] is not None for row in h200.describe_holdouts()) | Require 32 entries (16 per call) and zero H200 holdouts after dispatch retention. |
| candidate added | 380 | assert len(h200._exact_sp_times) == 32 | Require 32 entries (16 per call) and zero H200 holdouts after dispatch retention. |
| candidate added | 381 | assert sum(key[0] == "all_gatherv" for key, _, _ in h200._exact_sp_times) == 16 | Require 32 entries (16 per call) and zero H200 holdouts after dispatch retention. |
| candidate added | 382 | assert sum(key[0] == "reduce_scatterv" for key, _, _ in h200._exact_sp_times) == 16 | Require 32 entries (16 per call) and zero H200 holdouts after dispatch retention. |
| candidate added | 383 | assert h200.describe_holdouts() == [] | Require 32 entries (16 per call) and zero H200 holdouts after dispatch retention. |

## tests/unit/test_v4_replay_census.py

| Side | Line | Text | Reason |
| --- | ---: | --- | --- |
| base removed | 90 | from frontier.cc_backend.backends.measured_cc_backend import _Curve | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |
| candidate added | 90 | from frontier.cc_backend.backends.measured_cc_backend import MeasuredCCBackend, _Curve | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |
| candidate added | 93 | uses_exact_sp_rows=MeasuredCCBackend.uses_exact_sp_rows, | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |
| base removed | 104 | backend._curve_key = lambda *args: ("reduce_scatterv", "ep", 16) | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |
| base removed | 105 | backend._unequal_combine_times = {(("reduce_scatterv", "ep", 16), (1, 2), 16): 4.} | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |
| base removed | 106 | row = observer.collective_request(backend, None, None, 16, 16, None, (1, 2), .004) | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |
| base removed | 107 | assert row["unequal_rows"] == (1, 2) and row["exact"] | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |
| candidate added | 105 | for call in ("reduce_scatterv", "all_gatherv"): | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |
| candidate added | 106 | key = (call, "ep", 16) | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |
| candidate added | 107 | backend._curve_key = lambda *args: key | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |
| candidate added | 108 | backend._exact_sp_times = {(key, (1, 2), 16): 4.} | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |
| candidate added | 109 | row = observer.collective_request(backend, None, None, 16, 16, None, (1, 2), .004) | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |
| candidate added | 110 | assert row["unequal_rows"] == (1, 2) and row["exact"] | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |
| candidate added | 111 | assert row["lower_time_us"] == row["upper_time_us"] == 4. | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |
| candidate added | 112 | assert row["lower_bytes"] == row["upper_bytes"] == 16 | Use the backend rule in the observer fixture and test exact attribution of both EP calls. |

