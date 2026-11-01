"""Render the Q1 input ledger from fixed dispositions and checked retained evidence."""
import json
from pathlib import Path
from audit_inputs import load, digest, HERE, TASK, RUNS


def entry(name, origin, consumer, criterion, evidence, status, missing='', limitation=''):
    return dict(input=name, origin=origin, consumer=consumer, criterion=criterion,
                evidence=evidence, status=status, missing_evidence=missing, limitation=limitation)


def main():
    a = load(HERE / 'evidence/audit.json')
    checks = load(HERE / 'evidence/retained_checks.json')
    assert a['status'] == 'PASS_EVIDENCE_CHECKS_NOT_FORMAL_ACCEPTANCE'
    assert checks['status'] == 'PASS_RETAINED_CPU_CHECKS_NOT_FORMAL_ACCEPTANCE'
    ex = '../../test_report_2026-10-31_ex_situ_input_audit.md'
    prior = '../qualification_codex_20261031/qualification_report.md'
    requirements = '../../requirements.md'
    inputs = []
    definitions = {
        'attention_graph_replay_kernel_only.csv': ('G1h-02 and supplement; one H200, native TP1 engine with per-rank TP8 shapes',
            'D-A6-FORWARD-ATTENTION; KFold5 seed1234; full-model measured windows; median5%/p90 10%',
            'NEEDS_DECISION', '48 heldout omissions, including seven directly queried geometries; independent-pass supplement requires YC attention choice.', 'All measured windows exist; full table overall NOT_EVALUABLE.'),
        'kernel_gap.csv': ('G1h-02 one-H200 operator replay', 'Decision46/ex-situ compute criterion; exact measured segment/TP constants and actual kernel counts',
            'QUALIFIED', '', 'EP unequal-lane eager dispatch undercounts two gaps/layer: <=0.0468ms/step, explicitly deferred by D-WARM-EP-DEFER.'),
        'linear_op_kernel_only.csv': ('G2h-01 one-H200 profiler at per-rank linear/mHC/shared-expert shapes', 'Tickets05/09/19 typed same-hardware operator coverage; ex-situ audit compute PASS; exact-only lookup', 'QUALIFIED', '', 'Measured-coordinate qualification, no interpolation/generalization claim.'),
        'model_runner_block_kernel_only.csv': ('G2h-02/03/04 one-H200 per-operator block preparation/zero/write profiling', 'Tickets05/09/19; exact-only or source-owned legal zero; no whole-step timing', 'QUALIFIED', '', 'Per-operator preparation only, not a bound whole-runner/step table.'),
        'model_runner_replay_kernel_only.csv': ('G2h-02/03/04 one-H200 input preparation/sampling scope profiling', 'Tickets05/09/19; exact-only or source-owned legal zero; source-selected P/D sampling', 'QUALIFIED', '', 'Native-default fixed min/max sampling contracts only.'),
        'moe_kernel_only.csv': ('G3h one-H200 profiler, balanced rank-plan token classes', 'Tickets05/09/19; typed balanced_topk per-rank exact gating/shuffle/GEMM points', 'QUALIFIED', '', 'Balanced routing only, no skew/other routing claim.'),
        'summary.csv': ('G5h H200 TP8/DP2/EP16 and TP8/DP4/EP32 topologies plus isolated reruns', 'Decision46 permits measured formal CC; D-COLLECTIVE-EXACT-REPEATABILITY exact-only zero violations plus <=10% time-weighted maximum pass deviation', 'QUALIFIED', '', 'A8 census calls 6,310,733/9,003,672; billed pricing calls 6,310,644/9,002,176 are a different counter. Interpolating consumers retain the old failed LOO gate.'),
    }
    for record in a['inputs']:
        name = Path(record['path']).name
        origin, criterion, status, missing, limitation = definitions[name]
        evidence = ['evidence/audit.json#/inputs', ex,
                    '../a6_codex_20261101/result.json#/other_families']
        if name == 'attention_graph_replay_kernel_only.csv':
            evidence += ['evidence/audit.json#/formal_attention', 'evidence/audit.json#/independent_attention',
                         '../a6_forward_attention_20261101/current_phase_heldout.json', requirements + '#D-A6-FORWARD-ATTENTION']
        if name == 'summary.csv': evidence += ['evidence/retained_checks.json#/collectives', requirements + '#D-COLLECTIVE-EXACT-REPEATABILITY']
        item = entry(name, origin, ', '.join(record['consumers']), criterion, evidence, status, missing, limitation)
        item.update(path=record['path'], sha256=record['sha256'], rows=record['rows'])
        inputs.append(item)
    extra = [
        ('Canonical trace and original request population', '68f891b fixed trace, native client shared artifact', 'arrival generator; P/D scheduler; native/Frontier comparison', 'Tickets17/18/19:128 requests,6144 input/1024 output,QPS1,three complete repeats; exact same artifact', ['evidence/audit.json#/runs', 'evidence/retained_checks.json#/warm_readings', prior+'#Q2'], 'QUALIFIED', '', '128 Frontier requests;384 native request-repeat pairs per scale before event exclusions; no slow-request dropping.'),
        ('Native reference hardware, source/image, roles and lifecycle', 'Original clean H200 native downloads/receipts', 'model/parallel/hardware binding and comparison reference', 'Tickets11/18/19 R4; own H20032/64 roles; D-CALIBRATION-PRIORITIES current overrides', [prior+'#Q1-Q2-Q12', '../qualification_codex_20261031/evidence/qualification_evidence.json#/sources', 'evidence/audit.json#/native_identity_and_capacity', 'evidence/retained_checks.json#/original_native_reference_receipts'], 'QUALIFIED', '', '32/64 H200 ranks,4/8 nodes,TP8 DP2/4 EP16/32 PP1; original R4 host-return graph scope, not device-timing proof or retrospective promotion.'),
        ('Model architecture, precision and effective per-role hardware', 'Pinned model config in two-fix config.json and native effective configuration', '43-layer V4 inventory; operator shapes, memory/layout, predictor admission', 'Tickets05/09/18/19; H200, FLASHMLA_SPARSE_DSV4,FP8 KV/indexer,balanced_topk,dummy weights,matching typed operators', ['evidence/audit.json#/runs', ex, prior], 'QUALIFIED', '', 'Use prefill_replica_config/decode_replica_config. Generic monolithic a100/TP1 defaults in config.json are inactive; no hardware substitution follows.'),
        ('Scheduler, topology, graph and feature flags', 'Existing vLLM0.28 source/launch rules and formal config', 'PDD sequential VLLMv1EngineReplicaScheduler, DP-lane placement/sync, batch/graph selection', 'AGENTS parallel mapping; Tickets17/18/19 deployment equivalence; unchunked/non-prefix/non-thinking/non-MTP path', ['evidence/audit.json#/runs', ex, prior], 'QUALIFIED', '', 'TP8 DP2/4 each role, graph full_and_piecewise; capture1,2,4,8,16..256 step8; decode queue2; sync32steps; max_num_seqs128. Effective configs are authoritative over CLI added-field defaults.'),
        ('Run-ahead depth K19', 'Native P2-64 F2 finding, retained by decisions43/46 and current Q1 decision', 'Host forward queue depth', 'D-P2-INPUT-QUALIFICATION explicitly retains K19; decision46 keeps F2 as finding', [requirements, ex, '../../test_report_2026-11-01_f2_interpolation.md'], 'QUALIFIED', '', 'Allowed formal finding, not ex-situ autonomous prediction; one-GPU interpolation18.29 and K18 comparisons are diagnostic, not substituted.'),
        ('Ingress10.3ms and request handoff11.4ms', 'Approved ingress optionA and existing handoff input', 'arrival handling and P-to-D handoff', 'Decision46 and D-A6-PASS-REPLAYS explicitly hold ingress/handoff fixed; no new residual offsets', [requirements, ex, 'evidence/audit.json#/runs'], 'QUALIFIED', '', 'No new fitted delay, host-zero change or handshake timing.'),
        ('Host cost zero and disabled alternative timing tables', 'D2(c) modeling simplification; a1 receipt inactive paths', 'host scheduling; predictor family selection', 'D-NO-STEP-COST-SHORTCUT/decision46: no whole-step cost binding, dummy mode off', ['evidence/audit.json#/runs', ex], 'QUALIFIED', '', 'No dsv4_step_cost file/binding; CPU overhead/device-event/legacy attention files and PP inputs are configured inactive paths, not consumed cost samples. Host modeling is a recorded fidelity limitation.'),
        ('Trainer configuration, seeds and derived cache', 'Fresh-cache two-fix runs and pinned fit configuration/code', 'production attention estimators, exact lookup and gap models', 'Tickets09/19 input identity; same frozen table/config; A6 KFold5 seed1234; simulator seed1234/routing42', ['evidence/audit.json#/runs', '../a6_codex_20261101/result.json#/trainer', '../two_fix_20261101/test_report_2026-11-01_two_fix.md'], 'QUALIFIED', '', 'Caches are generated artifacts of these pinned inputs, not independently measured costs. No new full fit/replay was executed in Q1.'),
        ('Packed KV registration/layout and group configuration', '96 native registrations plus retained base_worker source and logical group contract', 'grouped cache allocator and packed payload price', 'Spec5.5 actual registration/stride; homogeneous TP8 and explicit grouped representation', [prior+'#Q3', '../b30b/kv_layout_evidence.json', '../two_fix_20261101/nixl_boundaries.json', 'evidence/audit.json#/native_identity_and_capacity'], 'QUALIFIED', '', 'One region,1,002,240B stride; selected-page model uses groups256/full,64/window128 twice,4/window8,8/window128. Registration qualification does not witness request selection.'),
        ('Actual native request-selected descriptors/rank wire bytes', 'Source-derived selection plus observed registration; native request final arrays absent', 'NIXL request payload and rank receive ranges', 'Spec5.5 actual send/receive descriptors,trim/null/alias/stride/rank mapping', [prior+'#Q4', '../../source_spec_v2/Frontier_GLM53_DeepSeekV4Flash_vLLM028_H80064_Spec_v2.md', 'payload_observation_contract.md', '../two_fix_20261101/test_report_2026-11-01_two_fix.md'], 'OPEN', 'Native per-request/per-TP-rank final selected IDs and lengths after trimming, paired local/remote ranges and aliases; existing synthetic producer cannot emit this serving witness.', '46pages x1,002,240 =46,103,040B/handle is source-derived; structural39,521,664B preserved. Payload decisionA/B required to waive or obtain witness.'),
        ('H800 analytical NIXL bandwidth/latency formal pair', 'H800 two-node READ affine fit126.2Gbps/0.0647ms', 'analytical KV service-time price', 'Ticket19 own-hardware compute/collective/NIXL; no cross-hardware substitution', ['evidence/audit.json#/runs', '../../tickets/19-comparison-and-acceptance.md', '../../test_report_2026-11-01_f3_h200_nixl.md'], 'OPEN', 'A qualified H200 replacement/approved case-specific timing disposition, exact T6144 concurrency/tail support and model/criterion binding.', 'Current price unchanged; warm numerical PASS and observed old/new invariance do not waive ticket19.'),
        ('H200 NIXL02 measurements/candidate timing model', 'Completed two-node16H200 C_nixl phase;240records', 'prospective H200 replacement only; not consumed by current P2', 'Integrity criteria satisfied; existing affine fit physical intercept check failed; new model/bound needs YC', ['evidence/retained_checks.json#/nixl_integrity', '../b30b/f3_nixl02_20261101/fit_diagnosis.json', 'decisions.md', 'packages/nixl6144/cpu_check.json'], 'NEEDS_DECISION', 'YC timing choice; no accepted physical fit or tail/concurrency bound. Targeted package supplies exact prompt evidence after lead submission.', 'Negative intercept retained; do not clamp/select passes/divide aggregate tails or fit native E2E.'),
        ('NIXL completion/ownership mechanism', 'vLLM READ source and existing engine-step Frontier workflow', 'READ posts, polling boundaries and KV release', 'Existing source-based engine_step workflow; D-WARM-TWO-FIX no lifecycle changes', ['evidence/retained_checks.json#/nixl_completion_windows', '../two_fix_20261101/nixl_window_check.py', ex], 'QUALIFIED', '', 'Zero window hits for2.570035277->2.987238193ms; minima18.724965326/24.380075963ms. Only this two-price invariance is established; no arbitrary timing qualification.'),
        ('Case-bound measured KV pool capacity', 'Native startup logs109934/112914 blocks', 'explicit grouped decode KV allocator/admission', 'Spec10.4 measured capacity isolation, expressibility and topology/configuration binding', [prior+'#Q9', 'evidence/audit.json#/native_identity_and_capacity', 'evidence/retained_checks.json#/measured_capacity_f4', '../../source_spec_v2/Frontier_GLM53_DeepSeekV4Flash_vLLM028_H80064_Spec_v2.md'], 'QUALIFIED', '', 'capacity_source=measured for these cases only. No autonomous planner accuracy. Max-model admission2940 vs native2944 remains a recorded limitation; no cross-layout capacity transfer.'),
        ('Native startup non-KV/cache budget measurements', 'All96 rank records: requested,available,total consumed,transient headroom,workspaces,graph logs', 'memory provenance ledger and capacity binding', 'Spec10.4 report native allocations; do not assume disjoint counters', ['evidence/audit.json#/native_identity_and_capacity', prior+'#Q6'], 'QUALIFIED', '', 'All96 exact identities hold; requested127603313869B minus availableKV = total_consumed +1957858304B transient headroom. Workspace1.5/3GiB; graph2.35..2.38GiB rounded, not disjoint sums.'),
        ('Independent phase/allocation memory peaks', 'Startup counters overlap; no independent phase peak witness', 'possible future autonomous/pressure capacity validation, not this isolation input', 'Spec10.4 excludes new exact quantized capacity framework; priorQ7 limitation', [prior+'#Q7'], 'OPEN', 'Independent per-phase peak/category evidence is unverified.', 'Not a prerequisite to the qualified measured-capacity isolation; do not create a new GPU requirement for Q1.'),
        ('Autonomous planner/admission agreement', 'F4 planner116034/120623 vs native109934/112914; receipt max-model allocation2940 vs2944', 'possible future autonomous/pressure capacity behavior', 'Spec10.4 separation of measured isolation and autonomous capacity; priorQ8', ['evidence/retained_checks.json#/measured_capacity_f4', 'evidence/audit.json#/runs', prior+'#Q8'], 'OPEN', 'Autonomous budget/category accounting and admission geometry agreement not demonstrated; current planner mismatches remain.', 'F4 five metric files per scale byte-identical B28, not proof of planner accuracy. Deferred under measured isolation; no new source repair.'),
        ('Warm reference definitions, exclusions and gate', 'D-WARM-TARGET source/event-openers; retained original all-request and paired warm readings', 'TTFT/TPOT/E2E/THPT compare per repeat and aggregate', 'D-WARM-TARGET one event rule both sides; four errors<=10% every repeat; preserve all-request FAIL', ['evidence/retained_checks.json#/warm_readings', '../two_fix_20261101/comparison/comparison.json', requirements], 'QUALIFIED', '', '32/32 checks PASS; max5.675932671%; excluded0/5 request-repeat pairs; P2-64 repeat01 all-request TTFT12.254940665% FAIL preserved. Numerical part only.'),
        ('Formal hardware/operator coverage as a whole', 'Native R4 plus current typed tables/omission ledger', 'Tickets05/09/11/19 full qualification', 'Ticket19 own-hardware complete coverage without omitted workloads/fallbacks; current forward rule', ['ledger.json#/attention_omissions', 'evidence/audit.json', '../../tickets/19-comparison-and-acceptance.md'], 'OPEN', 'Attention48 disposition, H200 NIXL timing and native payload qualification remain missing.', 'Hardware identity itself closes; full input coverage does not.'),
        ('Future32-to64 promotion receipt', 'Original clean64/basic receipt exists under historical authorization; future binding absent', 'eligibility for a new formal64 campaign', 'D-CALIBRATION-PRIORITIES and Tickets11/18/19: admitted32 numerical<=30%,qualified coverage,successor/basic binding and prelaunch receipt', [prior+'#Q11-Q13', requirements], 'OPEN', 'Claude must bind qualified32 case, all repeats/generations/workload and intended64 successor/basic evidence in a receipt before any new formal64 launch.', 'Original clean native32/64 receipts are not retroactive promotion. All currently warm errors also satisfy30% numerical predicate, separate from qualification. No platform query.'),
    ]
    inputs.extend(entry(*args) for args in extra)
    for row in inputs:
        assert row['status'] in {'QUALIFIED','OPEN','NEEDS_DECISION'}
        assert row['evidence'] and row['criterion']
        if row['status'] != 'QUALIFIED': assert row['missing_evidence']
    omitted = a['omissions']
    all_queries = sum(1 for o in omitted if any(o['exact_workload_forward_queries'].values()))
    assert len(omitted) == 48 and all_queries == 7
    config_records=[]
    for run in a['runs']:
        receipt = load(Path(run['root']) / 'receipt.json')
        config = Path(receipt['metrics_dir']) / 'config.json'
        config_records.append(dict(case=receipt['case'], input_path=receipt['input'], input_sha256=receipt['input_sha256'],
            config_path=str(config), config_sha256=digest(config), config=load(config),
            note='All configured fields preserved. Generic monolithic defaults, inactive table paths and disabled modes are not consumed PDD costs.'))
    (HERE / 'evidence/configurations.json').write_text(json.dumps(config_records,indent=2)+'\n')
    result=dict(schema_version=1, assignment='D-P2-INPUT-QUALIFICATION Q1', date='2026-11-02 Asia/Hong_Kong',
        disposition='INPUT_QUALIFICATION_OPEN; qualification audit and available CPU package delivered',
        inputs=inputs, profiling_table_records=a['inputs'], formal_complete_attention=a['formal_attention'],
        attention_omissions=omitted, proposed_attention_check=a['independent_attention'],
        retained_cpu_checks='evidence/retained_checks.json',
        prior_audit_edge_map={
            'Q1':'QUALIFIED original hardware/configuration','Q2':'QUALIFIED original request/lifecycle/host graph admission',
            'Q3':'QUALIFIED registration','Q4':'OPEN actual selected payload','Q5':'OPEN H200 timing binding',
            'Q6':'QUALIFIED startup measurements','Q7':'OPEN independent peaks; deferred by Spec10.4 isolation',
            'Q8':'OPEN autonomous planner agreement; deferred by Spec10.4 isolation','Q9':'QUALIFIED measured capacity isolation',
            'Q10':'QUALIFIED current warm numerical predicate; old all-request FAIL preserved',
            'Q11':'QUALIFIED numerical30% predicate only','Q12':'QUALIFIED historical64 basic/reference scope',
            'Q13':'OPEN future prelaunch promotion receipt','Q14':'OPEN aggregate05/09/11/19 qualification'},
        gpu_package='packages/nixl6144/cpu_check.json',
        unavailable_prerequisite='Native payload observation producer/source-instrumentation authorization; see payload_observation_contract.md',
        decision_proposals='decisions.md')
    (HERE/'ledger.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Formal P2-H200 input qualification ledger (Q1)', '',
        'Audit date: **2026-11-02 Asia/Hong_Kong**. The requested artifact directory keeps its20261101 name. Working root: `/data/ycfeng/Frontier/.worktrees/glm53-dsv4-vllm028`. Assigned writable scope: this directory and focused analysis/unit tests; no Frontier or published-table edits, commits, platform/cloud/GPU actions or retained-data writes.', '',
        '**Formal input qualification remains OPEN.** Native hardware/configuration/lifecycle, case-bound measured capacity, the non-attention operator inputs and exact collective input have usable existing qualification evidence. Attention48 omissions need a new retained-pass criterion; H200 timing and actual native request payload remain open. Warm numerical PASS is separate. The targeted NIXL GPU package passed CPU checks; native payload collection lacks an authorized observation producer. Claude retains acceptance, recording, commits and submission.', '',
        'QUALIFIED means the stated existing criterion is discharged within its named scope; OPEN names exact missing evidence; NEEDS_DECISION means evidence is available but a new criterion/model/promotion must be selected by YC. Open autonomous memory/phase-peak limitations are deferred by the existing measured-capacity isolation rule and are not new formal P2 prerequisites.', '',
        'The full machine ledger is [ledger.json](ledger.json), including every one of the48 omissions, precise heldout exceptions/row indices/measured-window presence/query counts, full table hashes and proposed cross-pass comparisons. [Configuration inventory](evidence/configurations.json) preserves both complete pinned configurations and input file identities. The seven consumed timing tables are distinguished from inactive configured paths. Every table digest/count and trace/input digest was rechecked against B29 and two-fix; neither a full replay nor new fitting was needed.', '',
        '## Input ledger', '', '| Input and consumer | Origin / controlling criterion | Status, evidence and exact remaining issue |', '| --- | --- | --- |']
    for row in inputs:
        evidence='; '.join(f'[{Path(e.split("#")[0]).name}]({e.split("#")[0]})' for e in row['evidence'])
        identity=f" {row['rows']}rows; SHA256 `{row['sha256']}`." if 'sha256' in row else ''
        lines.append(f"| **{row['input']}**{identity}<br>{row['consumer']} | {row['origin']}<br>**Criterion:** {row['criterion']} | **{row['status']}**. {row['missing_evidence']} {row['limitation']}<br>{evidence} |")
    lines += ['', '## Attention qualification and omitted-workload consumers', '',
        'Frozen KFold5/seed1234 forward aggregation reproduces6,923 admitted layer rows. Of1,421 canonical workloads,1,373 are complete,24partial and24fully unevaluable. All182 missed layer rows are present in the original7105-row table with positive measured layer windows. The failure is model-key/exact-width/joint-geometry support after holding out a workload.', '',
        '| Forward class | Complete / total | Existing median / p90 error | Omission cause and status |', '| --- | ---: | --- | --- |']
    for group in a['formal_attention']['by_group']:
        key=group['key']; n=group['count']
        if n:
            errors=f"{100*group['median_absolute_relative_error']:.6f}% / {100*group['p90_absolute_relative_error']:.6f}%"
            cause=f"Complete subset QUALIFIED; {group['omitted_workloads']} joint-boundary/geometry heldout misses NEEDS_DECISION."
        else:
            errors='NOT_EVALUABLE'
            cause='NEEDS_DECISION. Exact padded-width group removed by holding out its workload.' if key[0]!='NONE' else 'NEEDS_DECISION. Its sole canonical workload is held out, removing the required phase/model keys.'
        lines.append(f"| {'/'.join(key)} | {n} / {group['total_workloads']} | {errors} | {cause} |")
    lines += ['', 'FULL/dummy and PIECEWISE/dummy each omit widths1,2,4,8,16,24. NONE/dummy has width1; NONE/prefill is the6144/C0 eager prefill that dominates TTFT. No new measurement of identical coordinates repairs this fold-support issue. [Evidence](evidence/audit.json), `omissions[].heldout_rejections`, records every exact operator/key/boundary error.', '',
        '**Seven of48 omitted geometries are directly queried**, in both B29 and two-fix, with these complete-run counts:', '',
        '| Unit | Mode/phase and geometry | P2-32 forwards | P2-64 forwards | Status |', '| ---: | --- | ---: | ---: | --- |']
    for o in omitted:
        if not any(o['exact_workload_forward_queries'].values()): continue
        w=o['workload']; geometry=f"{w['graph_runtime_mode']}/{w['phase']}, width{w['padded_rows']}, P={w['prefill_seq_lens'] or []}, PC={w['prefill_kv_cache_sizes'] or []}, DC={w['decode_kv_cache_sizes'] or []}"
        lines.append(f"| {o['unit_id']} | {geometry} | {o['exact_workload_forward_queries']['p2_32']} | {o['exact_workload_forward_queries']['p2_64']} | NEEDS_DECISION |")
    lines += ['', 'The other41 omitted exact geometries have zero direct queries, but this does **not** prove they cannot support fitting/interpolation for another queried geometry. No independence waiver is applied. The census preserves all128 requests and1024 decode outputs per request, reconstructs the first-tail C6143 then decode C6144..7166 sequence, checks the recorded first-tail classifier, and checks one prefill per request. Request-list order is normalized with query/context pairing preserved.', '',
        'The original production exact lookup, including its mean reducer over duplicate coordinates, predicts all48 omitted workloads without interpolation. Compare to each independent pass\'s measured full-forward window; these **proposed**, not formal, class statistics pool both passes for display. All individual errors are below3.663152%, so the same5%/10% per-class limits also hold when each pass is checked separately.', '',
        '| Class | Comparisons (two passes) | Median / p90 / maximum error | Status |', '| --- | ---: | --- | --- |']
    for g in a['independent_attention']['by_class']:
        lines.append(f"| {'/'.join(g['class'])} | {g['count']} | {100*g['median_absolute_relative_error']:.6f}% / {100*g['p90_absolute_relative_error']:.6f}% / {100*g['maximum_absolute_error']:.6f}% | NEEDS_DECISION; retained-pass criterion proposed |")
    lines += ['', 'Table promotion is independent: the existing median3 table also retains48 omissions. [YC choices](decisions.md) recommend retaining the original and approving an independent-pass supplement. Scope-level CSA FULL FAIL is preserved diagnostically; D-A6-FORWARD-ATTENTION already superseded its gating role.', '',
        '## Reused closures and remaining qualification edges', '',
        'The original qualification audit Q1-Q14 maps as follows. Sources and exact missing evidence are the input rows above and [prior audit](../qualification_codex_20261031/qualification_report.md). No historical FAIL is relabeled: the current *warm* numerical rule supersedes which reading decides numerical comparison, while all-request failures remain beside it.', '',
        '| Prior edge | Current disposition |', '| --- | --- |']
    for key,value in result['prior_audit_edge_map'].items(): lines.append(f'| {key} | {value} |')
    lines += ['',
        'Closure commands and actual results are in [verification.md](verification.md). Hardware/rank-memory rehash:108node/rank files,96 exact memory identities PASS. Collective check:44/38points; weighted maximum pass deviations **5.462299% / 0.783291%**, all repeated, exact-only and zero violations PASS. Measured-capacity binding:109934/112914blocks,5F4/B28metric artifacts per scale byte-identical PASS; planner mismatch remains. Warm reading identity and32metric predicates PASS, max**5.675932671%**; P2-64 all-request repeat01 TTFT**12.254940665% FAIL** preserved.', '',
        'NIXL price-change invariance check:128transfers per scale,510/507step starts checked, minimum subsequent gaps18.724965326/24.380075963ms, zero window hits PASS. This closes the effect of the existing two-fix price change only. It does not close H200 timing; exact T6144 and concurrent-serving bounds are absent, H200 affine fit is rejected, and observed full-batch tails reach38.814150ms.', '',
        '## Packages and decisions', '',
        '[Targeted NIXL package](packages/nixl6144/README.md):16H200,2nodes, planned wholly preemptible; the lead may choose non-preemptible only within the personal aggregate32 ceiling and must recheck the config/lock. Estimated2.059449933containerGPUh,8.0GPUh allocation timeout.30cases x3passes;90records,12150reader trials,80352notifications expected. Native packed-layout/source pins, geometry, placement, actual producer list-cases and existing planner checks PASS. Runtime/GPU checks and committed-source submission lock NOT_RUN; Claude owns review/commit/lock/resource checks and submission. No platform or cloud action occurred.', '',
        '[Native payload observation prerequisite](payload_observation_contract.md): no existing inspected producer observes final serving descriptor arrays/lengths. A bounded host observer requires separate authorization and implementation before a CPU-verified native package can exist. Conservative both-scale clean-campaign proxy cost35.633333GPUh (historical11.311111+24.322222);64GPU campaign must be wholly preemptible. This package is **NOT_BUILT**, not silently replaced by the synthetic timing probe.', '',
        '[Decision proposals](decisions.md) give independent choices for attention supplemental criterion, original/median3 table, H200 timing target/model, and native payload observation/source-derived exception. No new criterion, table, fit or source change was applied. Final qualification and Tickets05/09/11/19 stay OPEN.']
    (HERE/'ledger.md').write_text('\n'.join(lines)+'\n')
    print(f'PASS: {len(inputs)} input/disposition rows;48 omissions;7 directly queried;Q1-Q14 mapped')


if __name__ == '__main__':
    main()
