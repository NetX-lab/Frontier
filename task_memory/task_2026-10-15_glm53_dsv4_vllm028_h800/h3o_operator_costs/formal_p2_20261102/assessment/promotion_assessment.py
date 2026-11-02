"""Draft promotion contents; never issue or backdate a launch receipt."""
import datetime
import math
from pathlib import Path
from assess_inputs import HERE, FORMAL, H3O, TASK, load, record, save

command=load(FORMAL/'p2_32_command.json')
terminal=command['status']=='FINISHED'
reading_path=(FORMAL if terminal else H3O/'two_fix_20261101')/'readings/p2_32_warm_reading.json'
reading=load(reading_path)
binding=load(HERE/'binding.json')['p2_32']
repeats={}
for side,r in reading['warm']['repetitions'].items():
    values=[]
    for m in r['metrics']:
        native,frontier=m['vllm'],m['frontier']
        finite=all(math.isfinite(v) for v in [native,frontier,m['relative_error']]) and native!=0
        values.append(dict(metric=m['metric'],unit='tokens/s' if m['metric']=='token_throughput_tps' else 'ms',native=native,frontier=frontier,
                           relative_error=m['relative_error'],finite=finite,passes_30pct=finite and m['relative_error']<=.30))
    repeats[side]=dict(metrics=values,all_four_pass=all(x['passes_30pct'] for x in values),count=128)
metric_names={'ttft_ms','tpot_ms','token_throughput_tps','request_e2e_time_ms'}
assert all({m['metric'] for m in r['metrics']}==metric_names for r in repeats.values()), 'Required promotion metric set differs'
numerical=len(repeats)==3 and {name.rsplit('-',1)[-1] for name in repeats}=={'repeat01','repeat02','repeat03'} and all(len(r['metrics'])==4 and r['all_four_pass'] for r in repeats.values())
qualified=load(HERE/'input_status.json')['coverage_verdict']=='PASS' and binding['checks_pass']
root=Path(command['run_root']); rec=load(root/'receipt.json')
native=TASK/'h200_mainline/diagnostic/inputs/p2_32r'
successor=TASK/'h200_mainline/diagnostic/inputs/p2_64r'
manifest=load(successor/'run_manifest.json')
platform_path=next((TASK/'h200_mainline/p2_64_reference').glob('package_*/case/runs/groundtruth_clean/*/platform_terminal_receipt.json'))
platform=load(platform_path)
runtime_path=HERE/'runtime.json'
runtime=load(runtime_path) if runtime_path.exists() else {}
coverage_complete=(terminal and rec.get('status')=='FINISHED' and runtime.get('p2_32',{}).get('status')=='PASS'
                   and runtime['p2_32']['completion_count']==128 and runtime['p2_32']['native_r4_status']=='PASS')
result=dict(schema_version=1,generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  draft_only=True,issued_receipt=False,receipt_timestamp=None,
  numerical_predicate_30pct=numerical,qualified_input_predicate=qualified,
  formal32_terminal=terminal,formal32_runtime_coverage_verified=coverage_complete,runtime_audit=record(runtime_path) if runtime_path.exists() else None,
  promotion_predicate=numerical and qualified and coverage_complete,
  predicate_status='PASS' if numerical and qualified and coverage_complete else 'UNVERIFIED_PENDING_FORMAL32',
  numeric_evidence_generation='formal H200-pair run' if terminal else 'retained two-fix control only; update when formal finishes',
  draft=dict(model='deepseek-ai/DeepSeek-V4-Flash',hardware='H200',topology='P16+D16; TP8/DP2/EP16/PP1 per role',
             runtime='vLLM0.28.0 native; torch2.13.0+cu130; Frontier Python3.10.6, sequential PDD, measured CC',
             workload=dict(requests=128,prompt_tokens=6144,output_tokens=1024,qps=1,repetitions=3,trace=next(x for x in binding['inputs'] if x['kind']=='trace'),population=next(x for x in binding['inputs'] if x['kind']=='request_population')),
             source_commit=command['source_commit'],source_and_table_bindings=str(HERE/'binding.json'),
             formal_command=record(FORMAL/'p2_32_command.json'),native_manifest=record(native/'run_manifest.json'),native_r4=record(native/'run_check_status.groundtruth_clean.json'),
             readings=record(reading_path),repetitions=repeats,
             intended_successor=dict(model='deepseek-ai/DeepSeek-V4-Flash',hardware='H200',topology='P32+D32; TP8/DP4/EP32/PP1 per role',
                                     basic_validation=record(successor/'run_check_status.groundtruth_clean.json'),basic_native_manifest=record(successor/'run_manifest.json'),future_resource_class='64 wholly preemptible H200, step_main'),
             issue_condition='Claude issues an actual timestamped receipt after formal32 input/runtime/reference checks pass, before a NEW formal64 native submission. Draft is not authorization.'),
  chronology=dict(existing_native64_job=platform['job'],submission_utc='2026-10-25T15:54:41Z',submission_evidence=str(TASK/'test_report_2026-10-25_h200_package_preparation.md')+':74',
                  manifest_before_launch_utc=manifest['written_before_launch_utc'],manifest_sha256=record(successor/'run_manifest.json')['sha256'],
                  terminal_utc=platform['status']['finishTime'],platform_receipt=record(platform_path),
                  authorization=manifest['authorization'],historical_preemptible_label=platform['preemptible_label'],
                  prelaunch_promotion_receipt_found=False,retained_absence_scope='Q1 row27, ticket11 explicit no backfill instruction; original native launch records/manifests; no current formal32 qualification existed in these retained records. Not a global filesystem absence claim.',
                  historical_launch_order='BLOCKED_ON_DECISION; current draft cannot precede 2026-10-25 launch',
                  resource_interpretation='Original non-preemptible64 was explicitly authorized under D-H200-TEMP-MAINLINE. A new formal64 job must be wholly preemptible; admission of this historical reference to current acceptance requires YC to resolve the launch-order/reference-reuse question.'),
  recommended_next_action='Claude reviews the finished formal32 assessment and issues a timestamped receipt, then seeks/uses explicit native64 campaign authorization and runs a new wholly preemptible H200 reference. This preserves ticket19 as written. Existing-reference reuse is the zero-GPU alternative only through an explicit YC exception.',
  options=[dict(option='A',recommended=False,decision='YC explicitly waives launch order for the existing P2-64 reference, retaining its original temporary non-preemptible authorization and accepting reuse in current formal assessment.',
                gpu_hours=0,cost='Records/review only; no native rerun. Does not backdate or claim an old promotion receipt.',evidence=[str(TASK/'tickets/19-comparison-and-acceptance.md')+':149',str(platform_path),str(successor/'run_manifest.json')],result_if_approved='P2-64 may PASS if every other formal criterion passes; record explicit exception.'),
           dict(option='B',recommended=True,decision='Claude issues the qualified P2-32 receipt, then submits a new formal native P2-64 campaign through the authorized lead workflow.',
                estimated_gpu_hours=platform['container_gpu_hours'],estimate_basis='Original eight-node P2-64 clean campaign summed container durations (three full repeats, initialization, compile, warmup and teardown). Planning proxy, not a quote.',
                resource_class='64 wholly preemptible H200; 8 nodes x8 GPUs; step_main',timeout_seconds=manifest['resources']['timeout_seconds'],timeout_gpu_hour_ceiling=64*manifest['resources']['timeout_seconds']/3600,
                limitation='Queue time unknown; stronger all-pair/kernel/graph warm coverage, idle proofs and observers may increase cost. Retain existing basic validation; verify source/config identity and no new unmatched shapes.',
                evidence=[str(platform_path),str(successor/'run_manifest.json'),str(TASK/'requirements.md')+':156',str(TASK/'tickets/11-h200-regression-delivery.md')+':92'],result_if_completed='Meets prelaunch ordering and current resource class; rerun CPU normalize/warm gates against the new three native repeats. No new CPU simulator replay needed if inputs unchanged.')])
save('promotion.json',result)
print('numerical30',numerical,'formal predicate',result['predicate_status'],'existing64 order BLOCKED_ON_DECISION','new native64 proxy GPUh',platform['container_gpu_hours'])
