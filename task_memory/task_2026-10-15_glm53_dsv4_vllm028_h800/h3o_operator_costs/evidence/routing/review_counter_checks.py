import bisect, collections, json, pathlib, sys
for base_arg in sys.argv[1:]:
    base = pathlib.Path(base_arg)
    name = base.parent.name if base.name == 'artifacts' else base.name
    for role in ('prefill', 'decode'):
        rows = forwards = disagree_rows = disagree_forwards = mode_rows = mode_forwards = 0
        examples = []
        for path in sorted(base.glob('nodes/*/serving/rank0.json')):
            text = path.read_text()
            data = json.loads(text)
            if data.get('role') != role:
                continue
            if not rows:
                print(name, role, 'inventory.model_runner =', data['inventory'].get('model_runner'))
            for index, row in enumerate(data['counters']['rows']):
                if row.get('kind') not in ('serving', 'dummy'):
                    continue
                de = row['dispatch_evidence']
                local, desired, uniforms, max_queries = de['dp_post_all_reduce']
                desc = de['dispatch']['descriptor']
                count = row.get('successful_forwards') or 0
                rows += 1
                forwards += count
                graph_branch = min(desired) != 0
                disagreement = graph_branch and len(set(uniforms)) > 1
                mode_difference = any(('NONE', 'PIECEWISE', 'FULL')[m] != desc['cg_mode'] for m in desired)
                if disagreement:
                    disagree_rows += 1
                    disagree_forwards += count
                if mode_difference:
                    mode_rows += 1
                    mode_forwards += count
                if disagreement and len(examples) < 2:
                    positions = [i for i, line in enumerate(text.splitlines(), 1) if '"dispatch_evidence":' in line]
                    line_ref = positions[index] if len(positions) == len(data['counters']['rows']) else None
                    examples.append({'file': str(path), 'row': index, 'line': line_ref,
                                     'kind': row['kind'], 'successful_forwards': count,
                                     'dp_post_all_reduce': de['dp_post_all_reduce'],
                                     'descriptor': desc,
                                     'synced_widths': de['dispatch']['num_tokens_across_dp']})
        print(name, role, 'rows', rows, 'forward_observations', forwards,
              'uniform_disagreement_graph_branch', (disagree_rows, disagree_forwards),
              'mode_difference', (mode_rows, mode_forwards))
        for example in examples:
            print(json.dumps(example, ensure_ascii=False))
