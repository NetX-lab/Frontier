"""Construct and admit ticket05's local, unmeasured sampling plan only."""
import hashlib
import json
import random

CONFIG_SHA = 'e535c5c9e6385e5c39d55daf346fc0b42dd93496bf5e540ea442168b2524bb63'
SOURCE_SHA = '4bcd69c5838bc9a52576e0fd518b47852d97181627127b4ea5fefa41903fdc46'
PLAN_SHA = '5ba53b2ae2c5598e719ba35876cfaf9688e4bc87c76c2b92ecf09d1822ad24d9'


def sample_cases(geometry, seed):
    r, b, w, k = (geometry[n] for n in ('r', 'B', 'W', 'K'))
    boundaries = sorted({r-1, r, r+1, 2*r-1, 2*r+1, b-1, b, b+1,
                         w-1, w, w+1, k*r-1, k*r, k*r+1})
    rng, cases = random.Random(seed), []
    def add(label, total, query, rows=1, valid=1, graph='eager', precision='bf16', partition='sample'):
        cases.append(dict(label=label, total=total, prefix=total-query, query=query,
            rows=rows, valid_rows=valid, graph=graph, precision=precision, partition=partition,
            layout='packed_content_LBNHC', table_pages=(total+b-1)//b,
            closed_regions=total//r, open_tokens=total%r, content_seed=rng.randrange(2**31),
            status='planned', measurement_status='unmeasured'))
    for n in boundaries:
        add(f'boundary_new_{n}', n, min(n, w))
        add(f'boundary_subsequent_{n}', n, 2)
    for q in (2, r-1, r+1):
        add(f'long_prefix_short_chunk_{q}', k*r+2*b+q, q)
    for valid in (r-1, r):
        add(f'local_graph_rows8_valid{valid}', k*r+1, 1, r, valid, 'isolated_replay')
    add('local_graph_rows9_edge', k*r+1, 1, r+1, r+1, partition='holdout')
    for precision in ('fp8_routed', 'bf16_excluded', 'fp32_norm_residual', 'fp32_router'):
        add(f'precision_{precision}', b+1, b+1, precision=precision)
    for n in (2*r+3, w+3, k*r+3):
        add(f'holdout_subsequent_{n}', n, 3, partition='holdout')
    return cases


def admit(manifest, task):
    """Reject invalid plans; this does not admit measured profiles or timings."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    def read(path):
        return json.loads((task/path).read_text())
    plan = task/'local_scope_sampling_01.json'
    require(hashlib.sha256(plan.read_bytes()).hexdigest() == PLAN_SHA, 'Changed preregistered sampling contract')
    require(manifest == read(plan.name), 'Changed owner, witness, coverage or measurement contract')
    for path, digest in manifest['construction_sources'].items():
        require(hashlib.sha256((task.parent.parent/path).read_bytes()).hexdigest() == digest, 'Changed construction source')
    require(hashlib.sha256((task/'reference_inputs/config.json').read_bytes()).hexdigest() == CONFIG_SHA, 'Metadata identity')
    require(hashlib.sha256((task/'native_source_export/observed/source_manifest.json').read_bytes()).hexdigest() == SOURCE_SHA, 'Source identity')
    lock, config = read('reference_lock.json'), read('reference_inputs/config.json')['text_config']
    env = read('native_g1_attention_state/observed_generation2/dsa_long/environment.json')
    identity = dict(image=lock['image_descriptor']['linux_amd64']['digest'],
        revision=lock['identities']['model_metadata']['revision'], config_sha256=CONFIG_SHA,
        source_manifest_sha256=SOURCE_SHA, hardware=env['device_name'], vllm=env['vllm'],
        torch=env['torch'], triton=env['triton'], native_python=env['python'],
        kernel_driver=env['kernel_module_label'], user_driver=env['mapped_libcuda'][0])
    require(manifest['identity'] == identity, 'Wrong native identity')
    require(read('g1_package_acceptance_01.json')['status'] == 'ACCEPTED_G1_RANK_LOCAL_ONLY', 'G1 prerequisite')
    dsa = read('native_g1_attention_state/observed_generation2/dsa_long/dsa_long.json')
    geometry = dict(r=config['sparse_config']['region_block_size'], B=dsa['spec']['block_size'],
                    W=config['sliding_window'], K=config['sparse_config']['topk'])
    require(manifest['geometry'] == geometry, 'Wrong region/page/window/selector units')
    require(manifest['cases'] == sample_cases(geometry, manifest['seed']), 'Missing features or changed sampling/holdouts')
    require(manifest['status'] == 'planned' and manifest['measurement_status'] == 'unmeasured', 'Unsupported acceptance')
    require(all(value is None for value in manifest['measurements'].values()), 'Fabricated timing/count')
    require(manifest['exclusive_target'] == dict(gpus=1, nic_activity='none', other_gpu_processes='none',
        observed=False, status='planned', concurrency='no overlap with ticket06'), 'Exclusive target not measured')
    expected = {
        'dsa_projection': ('UnquantizedLinearMethod', 'main_qkv_projection index_qkz_projection headwise_gate_projection'),
        'indexer_w': ('UnquantizedLinearMethod', 'indexer_w'),
        'dsa_prefill_prepare': ('fused_qknorm_rope_cache + indexer_norm_rope', 'main_norm_rope_raw_kv indexer_norm_rope_cast'),
        'dsa_decode_prepare': ('qkv_indexer_norm_rope/_joint_prepare', 'main_norm_rope_raw_kv indexer_norm_rope_cast summary_update'),
        'dsa_prefill_root': ('STEP5_DSA', 'summary_update page_staging summary_pack selector_score_select sparse_attention headwise_gate group_output_copy'),
        'dsa_decode_root': ('STEP5_DSA', 'summary_pack selector_score_select sparse_attention headwise_gate'),
        'swa_projection': ('UnquantizedLinearMethod', 'main_qkv_projection headwise_gate_projection'),
        'swa_prepare': ('fused_qknorm_rope_cache', 'main_norm_rope_raw_kv'),
        'swa_attention': ('STEP5_FLASH_ATTN/FA3', 'swa_attention'),
        'swa_gate': ('headwise_sigmoid_gate', 'headwise_gate'),
        'output_projection': ('UnquantizedLinearMethod', 'output_projection'),
        'dense_ffn': ('UnquantizedLinearMethod', 'gate_up_projection swiglu_activation down_projection'),
        'fp8_moe_root': ('Fp8MoeBackend.TRITON/TritonExperts', 'router routed_ffn shared_ffn activation combine'),
        'bf16_moe_root': ('UnquantizedMoeBackend.TRITON/TritonExperts', 'router routed_ffn shared_ffn activation combine'),
        'norm_residual': ('Step5RMSNorm/rms_norm', 'norm fp32_residual output_cast'),
    }
    require({s['id'] for s in manifest['scopes']} == set(expected), 'Missing or unsupported physical scope')
    claims, ids = set(), set()
    source = (task/'native_source_export/observed/vllm/models/step5/nvidia/model.py').read_text()
    require('weights, _ = owner.sparse_indexer_w(hidden_states)' in source, 'Missing separate indexer_w owner')
    for scope in manifest['scopes']:
        require(scope['id'] not in ids, 'Duplicate physical owner')
        ids.add(scope['id'])
        require(scope['status'] == 'planned' and scope['g1'] in manifest['evidence'], 'Unqualified scope')
        require(scope['backend'] == scope['required_backend'] == expected[scope['id']][0], 'Wrong backend')
        require(scope['includes'] == scope['required_includes'] == expected[scope['id']][1].split(), 'Wrong fused ownership')
        require(scope['topology'] == dict(tp=1, dp=1, ep=1, pp=1), 'Unmeasured native topology')
        require(scope['duration_charges'] == scope['kernel_count_charges'] == 1, 'Multiple physical charges')
        require(scope['source'] in manifest['evidence'] and scope['role'] == 'rank_local', 'Wrong scope identity')
        for branch in scope['branches']:
            for semantic in scope['includes']:
                key = (branch, semantic)
                require(key not in claims, 'Duplicate fused semantic charge')
                claims.add(key)
    sources = read('native_source_export/observed/source_manifest.json')['files']
    for row in sources:
        path = task/'native_source_export/observed'/row['relative_path']
        require(hashlib.sha256(path.read_bytes()).hexdigest() == row['source_sha256'], 'Changed installed source')
    for path, digest in manifest['evidence'].items():
        require(hashlib.sha256((task/path).read_bytes()).hexdigest() == digest, 'Changed evidence: '+path)
    require(all(c['status'] == 'unmeasured' and c['reason'] for c in manifest['coverage']), 'Unsupported coverage')
    require(manifest['sampling_origin'] == 'independent local shapes; no serving comparison inputs', 'Borrowed serving samples')
