"""Check terminal runtime, native copies and the qualified Q2 domain, read-only."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
from assess_inputs import HERE, FORMAL, H3O, TASK, REPO, load, record, save, sha


def main(control=False):
    base=H3O/'two_fix_20261101' if control else FORMAL
    q2dir=H3O/'input_qualification_20261101/nixl_timing_q2'
    spec=importlib.util.spec_from_file_location('q2',q2dir/'analyze.py');q2=importlib.util.module_from_spec(spec);spec.loader.exec_module(q2)
    records_path=HERE/'scratch/q2_records.json'
    if records_path.exists(): records=load(records_path)
    else:
        q2.HERE=HERE/'scratch/q2_rebuilt';q2.HERE.mkdir(exist_ok=True);(q2.HERE/'scratch').mkdir(exist_ok=True)
        records,_,integrity=q2.measurement();records_path.write_text(json.dumps(records)+'\n');save('scratch/q2_integrity.json',integrity)
    adopted=load(q2dir/'proposed_input.json')
    native_inputs=TASK/'h200_mainline/diagnostic/inputs';manifest=load(native_inputs/'manifest.json')
    result={}
    for case in ['p2_32','p2_64']:
        cmd=load(base/f'{case}_command.json');assert cmd['status']=='FINISHED',(case,cmd['status'])
        root=Path(cmd['run_root']);receipt=load(root/'receipt.json');assert receipt['status']=='FINISHED'
        metrics=Path(receipt['metrics_dir']);config=load(metrics/'config.json');cl=config['cluster_config'];sc=cl['replica_scheduler_config']
        flags=dict(architecture=config['sys_arch'],parallel=config['enable_parallel_clusters'],thinking=config['enable_thinking_mode'],prefix=sc['enable_prefix_caching'],chunking=sc['enable_chunked_prefill'],graph=config['decode_cuda_graph_mode'],max_model_len=sc['max_model_len'],max_num_batched_tokens=sc['max_tokens_in_batch'],max_num_seqs=sc['batch_size_cap'],memory_pool_blocks=sc['num_blocks'],memory_utilization=sc['gpu_memory_utilization'],role_configs={role:cl[role+'_replica_config'] for role in ['prefill','decode']})
        assert flags['architecture']=='pd-disaggregation' and all(flags[k] is False for k in ['parallel','thinking','prefix','chunking'])
        assert flags['graph']=='full_and_piecewise' and flags['max_model_len']==7168 and flags['max_num_batched_tokens']==8192
        assert sc['num_blocks']==(109934 if case=='p2_32' else 112914)
        assert flags['max_num_seqs']==128 and flags['memory_utilization']==.85
        for role in ['prefill','decode']:
            rc=cl[role+'_replica_config']
            assert rc['device']=='h200' and rc['attn_tensor_parallel_size']==8 and rc['attn_dp']==(2 if case=='p2_32' else 4) and rc['moe_expert_parallel_size']==(16 if case=='p2_32' else 32) and rc['num_pipeline_stages']==1
            assert rc['speculative_decoding_config']['enabled'] is False and rc['speculative_decoding_config']['mtp_n_predict']==0
            assert rc['attention_backend']=='FLASHMLA_SPARSE_DSV4' and rc['kv_cache_dtype']==rc['indexer_kv_cache_dtype']=='fp8'
        assert receipt['a1']['enable_dummy_mode']=={'PREFILL':False,'DECODE':False}
        assert receipt['a1']['cc_backend_types']=={'PREFILL':'measured','DECODE':'measured'}
        assert receipt['a1']['no_step_cost_file_beside_inputs'] and receipt['a1']['step_cost_table_bound_by_predictor']=={'PREFILL':False,'DECODE':False}
        expected_count=23040 if case=='p2_32' else 39680
        assert receipt['a11']['forwards']['rows']==receipt['a11']['stage_end_events']==expected_count and receipt['a11']['stale_stage_end_events']==0
        assert sha(root/'forwards.jsonl.gz')==receipt['a11']['forwards']['sha256'] and sha(root/'collectives.json')==receipt['a11']['collectives']['sha256']
        completions=[]
        with (metrics/'metrics_ground_truth.jsonl').open() as f:
            for line in f:
                if '"request_completion"' in line: completions.append(json.loads(line))
        assert len(completions)==128 and len({x['request_id'] for x in completions})==128
        assert all(x['num_prefill_tokens']==6144 and x['num_decode_tokens']==1024 for x in completions)
        domain=q2.replay_windows(case+'_two_fix' if control else case+'_formal',root,adopted,records)
        domain.pop('per_transfer_rows');domain.pop('per_possible_post_rows')
        n='p2_32r' if case=='p2_32' else 'p2_64r';native=native_inputs/n
        copies=[]
        for entry in manifest[n]['files']:
            actual=record(native/entry['path']);assert actual['sha256']==entry['sha256'];copies.append(actual)
        r4=load(native/'run_check_status.groundtruth_clean.json');rr=next(iter(r4['runs'].values()));assert rr['status']=='PASS'
        engine_path=next((TASK/'h200_mainline'/('p2_32_reference' if case=='p2_32' else 'p2_64_reference')).glob('package_*/case/inputs/engine.json'))
        engine=load(engine_path)
        assert engine['max_model_len']==7168 and engine['max_num_batched_tokens']==8192
        assert engine['max_num_seqs']==128 and engine['gpu_memory_utilization']==.85
        assert engine['kv_cache_dtype']=='fp8' and engine['dtype']=='bfloat16'
        native_manifest=load(native/'run_manifest.json');assert sha(engine_path)==native_manifest['inputs_sha256']['inputs/engine.json']
        result[case]=dict(status='PASS',formal_command=record(base/f'{case}_command.json'),receipt=record(root/'receipt.json'),flags=flags,
                         nondummy_and_no_step_substitution='PASS',forward_count=expected_count,stage_end_count=expected_count,stale_stage_ends=0,
                         billed_collective_calls=receipt['a11']['collectives']['calls'],domain=domain,completion_count=128,exact_prompt_tokens=6144,exact_output_tokens=1024,
                         native_copies=copies,native_r4=record(native/'run_check_status.groundtruth_clean.json'),native_r4_status=rr['status'],native_idle_evidence=rr['checks']['requests_and_idle_lanes'],
                         native_effective_engine=record(engine_path),native_engine=engine,
                         memory_evidence=[record(H3O/'qualification_codex_20261031/qualification_report.md'),record(H3O/'qualification_codex_20261031/evidence/qualification_evidence.json')],
                         native_memory=dict(requested_bytes_per_rank=127603313869,available_kv_bytes_per_rank=[110181056717,110197833933] if case=='p2_32' else [113167401165,113184178381],transient_headroom_bytes=1957858304,workspace_bytes=1610612736 if case=='p2_32' else 3221225472,graph_memory_gib_rounded='2.35..2.38',independent_disjoint_phase_peaks='UNVERIFIED; not prerequisite to measured capacity isolation'),
                         kv_admission=receipt['kv_admission'],coverage_reason='Same qualified seven tables and identical complete forward/collective sequence relative to two-fix. Existing exact-point coverage and legal source zeros retained; no omitted workload admitted by fallback. See comparison for sequence identity.')
        print(case,'PASS','forward rows',expected_count,'completion count',len(completions),'domain',domain['matched_case'],'min gap',domain['min_subsequent_step_gap_ms'])
    save('scratch/control_runtime.json' if control else 'runtime.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control',action='store_true');main(p.parse_args().control)
