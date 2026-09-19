# SPDX-License-Identifier: AGPL-3.0-only
"""Freeze a source-pinned K3/K4/K5 candidate, without modifying a live kit.

MiaAI cooperative changes retain the original AGPL/MIT/ExLlamaV3 notices.
Preparing a kit is not evidence of GPU correctness, fit, or serving performance.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'dcp_overlap'))
from prepare import bounded_read, encoded, load_parent, safe_name, sha, MAX_FILE, MAX_TOTAL
from contracts import Policy, LAYOUT, COUNTERS
from readiness import transform as ready_transform


def transform(old, policy, binary, receipt, *, draft_binary=None, markov_add=False):
    if (receipt['status'] != 'native_experiment_built_cpu_only'
            or receipt['experiment'] != 401 or receipt['abi'] != 2
            or receipt['binary_sha256'] != sha(binary)):
        raise ValueError('Use the unchanged-arithmetic capacity36 build')
    payload = copy.copy(old)
    payload['serving/spark_combined_ready.py']=ready_transform(old['serving/spark_combined_ready.py'],policy)

    def edit(name, before, after):
        text = payload[name].decode()
        if text.count(before) != 1:
            raise ValueError('Changed source anchor in '+name+': '+before[:100])
        payload[name] = text.replace(before, after).encode()

    def assignment(name, key, *, literal=True):
        text = payload[name].decode()
        found = [n for n in ast.parse(text).body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == key for t in n.targets)]
        if len(found) != 1: raise ValueError('Changed assignment '+key)
        return ast.get_source_segment(text, found[0]), ast.literal_eval(found[0].value) if literal else None

    contract = 'serving/ds41/cooperative_contract.py'
    before, _ = assignment(contract, 'LAYOUT', literal=False)
    edit(contract, before, 'LAYOUT = '+repr(LAYOUT))
    edit(contract, '1 <= x_shape[0] <= 24', '1 <= x_shape[0] <= 36')
    edit('serving/ds41/cooperative_routes.py', 'ctr < 2547', 'ctr < 3819')
    edit('serving/ds41/cooperative_moe.py', '(48, 2547, 344)', '(48, 3819, 344)')
    for before,after in (('values.shape[0] <= 24','values.shape[0] <= 36'),
                         ('values.shape[0] * values.shape[1] > 24','values.shape[0] * values.shape[1] > 36'),
                         ('values.shape[1] <= 4','values.shape[1] <= 6'),
                         ('at most24 MXFP4 query rows','at most36 MXFP4 query rows')):
        edit('serving/ds41/dcp_indexer_graph.py',before,after)
    edit('serving/ds41/dcp_candidates_graph.py','1<=logits.shape[0]<=24','1<=logits.shape[0]<=36')
    edit('serving/spark_combined_miaai.py', 'cooperative_rows=list(range(1,25))',
         'cooperative_rows=list(range(1,37))')
    edit('serving/ds41/combined_dspark.py', "getattr(spec, 'num_speculative_tokens', None) != 3",
         "getattr(spec, 'num_speculative_tokens', None) != "+str(policy.draft_tokens))
    edit('serving/ds41/combined_dspark.py', 'three drafts and fixed verification',
         str(policy.draft_tokens)+' drafts and '+policy.verification+' verification')
    if policy.verification=='confidence':
        from confidence import cache_initializer_source
        cache_name = 'serving/ds41/vllm_v2_cache.py'
        payload[cache_name] = cache_initializer_source(old[cache_name])
        edit('serving/ds41/combined_dspark.py',
             "or getattr(spec, 'enable_adaptive_verification', False)",
             "or getattr(spec, 'enable_adaptive_verification', False) is not True")
        payload['serving/ds41/dspark_experiment/__init__.py']=b'# SPDX-License-Identifier: AGPL-3.0-only\n'
        payload['serving/ds41/dspark_experiment/confidence.py']=bounded_read(HERE/'confidence.py')
        edit('serving/ds41/combined_dspark.py',
             '    from .draft_exl3_serving import make_quantizer_patches\n    return [',
             '    from .draft_exl3_serving import make_quantizer_patches\n'
             '    from .dspark_experiment.confidence import make_patches as confidence_patches\n'
             '    return [\n        *confidence_patches(),')
    if policy.verification=='ema':
        payload['serving/ds41/dspark_experiment/__init__.py']=b'# SPDX-License-Identifier: AGPL-3.0-only\n'
        payload['serving/ds41/dspark_experiment/policy.py']=(
            '# SPDX-License-Identifier: AGPL-3.0-only\n'
            f'DRAFT_TOKENS = {policy.draft_tokens!r}\nVERIFICATION = {policy.verification!r}\n'
            f'PREFIX_LENGTHS = {policy.prefix_lengths!r}\n').encode()
        for name in ('adaptive.py','integration.py'):
            payload['serving/ds41/dspark_experiment/'+name]=bounded_read(HERE/name)
        edit('serving/ds41/combined_dspark.py',
             '    from .draft_exl3_serving import make_quantizer_patches\n    return [',
             '    from .draft_exl3_serving import make_quantizer_patches\n'
             '    from .dspark_experiment.integration import make_patches as prefix_patches\n'
             '    return [\n        *prefix_patches(),')
    if draft_binary is not None:
        payload['serving/ds41/dspark_experiment/__init__.py']=b'# SPDX-License-Identifier: AGPL-3.0-only\n'
        payload['serving/ds41/dspark_experiment/features.py']=(
            '# SPDX-License-Identifier: AGPL-3.0-only\n'
            f'KV_ONLY = True\nMARKOV_ADD = {markov_add!r}\nTOP3_SHA256 = {sha(draft_binary)!r}\n').encode()
        for name in ('kernel_integration.py','kv_projection.py','draft_top3.py','markov_sampling.py'):
            payload['serving/ds41/dspark_experiment/'+name]=bounded_read(HERE/name)
        payload['serving/dspark_draft_top3.so']=draft_binary
        edit('serving/ds41/combined_dspark.py','    from .draft_exl3_serving import make_quantizer_patches',
             '    from .dspark_experiment.kernel_integration import make_patches as draft_kernel_patches\n'
             '    from .draft_exl3_serving import make_quantizer_patches')
        edit('serving/ds41/combined_dspark.py','        *make_quantizer_patches(_draft_scope),',
             '        *draft_kernel_patches(),\n        *make_quantizer_patches(_draft_scope),')
        edit(contract,'    return eligible_shape(x_shape, ids_shape)',
             '    from .dspark_experiment.kernel_integration import selected_draft_shape\n'
             '    return eligible_shape(x_shape, ids_shape) or selected_draft_shape(x_shape, ids_shape)')
        edit('serving/ds41/cooperative_moe.py','    return call\n',
             '    from .dspark_experiment.kernel_integration import wrap_moe\n    return wrap_moe(call,root)\n')
        edit(contract,"mode='miaai_two_stage_cooperative'",
             "mode='ds41_draft_top3' if ids.shape[1]==3 else 'miaai_two_stage_cooperative'")
    if policy.verification in ('ema','confidence') or draft_binary is not None:
        backend='serving/spark_backend_attestation.py'
        before,pins=assignment(backend,'PRIVATE_SOURCES')
        for name,raw in payload.items():
            if name.startswith('serving/ds41/dspark_experiment/') or name=='serving/dspark_draft_top3.so':
                pins[name.removeprefix('serving/')]=sha(raw)
        edit(backend,before,'PRIVATE_SOURCES = '+repr(pins))
    edit('serving/ds41/combined_dspark.py', 'dspark_num_experts_per_tok=3, dspark_markov_rank=256)',
         'dspark_num_experts_per_tok=3, dspark_markov_rank=256, dspark_block_size=5)')
    before, _ = assignment('serving/ds41/combined_config.py', 'GRAPH_SIZES')
    edit('serving/ds41/combined_config.py', before, 'GRAPH_SIZES = '+repr(policy.graph_sizes()))
    edit('serving/ds41/combined_config.py', 'reviewed24-token bound',
         'reviewed'+str(max(policy.graph_sizes()))+'-token K'+str(policy.draft_tokens)+' bound')
    before, profile = assignment('tools/portable_node.py', 'PROFILE')
    spec = json.loads(profile['speculative-config'])
    if spec['num_speculative_tokens'] != 3 or spec['enable_adaptive_verification']:
        raise ValueError('Changed baseline speculation')
    spec['num_speculative_tokens'] = policy.draft_tokens
    spec['enable_adaptive_verification'] = policy.verification=='confidence'
    graph = json.loads(profile['compilation-config'])
    graph.update(cudagraph_capture_sizes=list(policy.graph_sizes()),
                 max_cudagraph_capture_size=max(policy.graph_sizes()))
    profile.update({'speculative-config': json.dumps(spec), 'compilation-config': json.dumps(graph)})
    edit('tools/portable_node.py', before, 'PROFILE = '+repr(profile))
    # The host preflight compares this exact YAML inventory to PROFILE.
    # Keep both explicit; do not bypass the comparison for experiments.
    payload['serving/conservative.yaml']=(''.join(k+': '+v+'\n' for k,v in profile.items())).encode()
    payload['serving/cooperative_moe.so'] = binary
    native = json.loads(payload['serving/cooperative-native.json'])
    native.update(binary_sha256=sha(binary), binary_bytes=len(binary),
                  experiment=401, rows_max=36, slots_max=216, counters=COUNTERS,
                  build_receipt_sha256=sha(encoded(receipt)), serving_qualified=False)
    payload['serving/cooperative-native.json'] = encoded(native)
    if len(binary) > 2819968:
        edit('serving/spark_backend_attestation.py', "'cooperative_moe.so': 2819968",
             "'cooperative_moe.so': "+str(len(binary)))

    # Propagate all changed source/binary hashes through existing fail-closed
    # loader, attestation and host checks. Never bypass a source pin.
    history = {name: {sha(raw)} for name, raw in old.items()
               if name.startswith(('serving/', 'tools/'))
               and name.endswith(('.py', '.json', '.so', '.yaml'))
               and not name.endswith('overlay-manifest.json')}
    # New files can themselves contain pins to changed parent files. Track
    # their initial and subsequent digests too, or PRIVATE_SOURCES retains a
    # pre-propagation hash despite the outer bundle manifest being correct.
    for name, raw in payload.items():
        if (name.startswith(('serving/', 'tools/'))
                and name.endswith(('.py', '.json', '.so', '.yaml'))
                and not name.endswith('overlay-manifest.json')):
            history.setdefault(name, set()).add(sha(raw))
    replacements = {}
    for _ in range(32):
        for name, values in history.items(): values.add(sha(payload[name]))
        updates = {digest: sha(payload[name]) for name, values in history.items()
                   for digest in values if digest != sha(payload[name])}
        replacements.update(updates)
        changed = False
        for name, raw in list(payload.items()):
            if (not name.startswith(('serving/', 'tools/')) or not name.endswith(('.py', '.json'))
                    or name.endswith('overlay-manifest.json')): continue
            for before, after in updates.items(): raw = raw.replace(before.encode(), after.encode())
            if raw != payload[name]: payload[name] = raw; changed = True
        if not changed: break
    else: raise ValueError('Source pins did not converge')

    def refresh(value):
        if isinstance(value, dict): return {k:refresh(v) for k,v in value.items()}
        if isinstance(value, list): return [refresh(v) for v in value]
        return replacements.get(value, value) if isinstance(value, str) else value
    requirements = refresh(json.loads(payload['runtime-requirements.json']))
    requirements['six_session_candidate']['cooperative_rows']=36
    requirements['dspark_candidate'] = dict(
        draft_tokens=policy.draft_tokens, verification=policy.verification,
        prefix_lengths=list(policy.prefix_lengths),
        graph_sizes=list(policy.graph_sizes()), status='unqualified',
        parent_qualification_applies_to_parent_only=True,
        serving_memory_limits_unchanged=True, new_persistent_moe_scratch_bytes=0,
        gpu_tests_run=False, serving_tests_run=False, publication_approved=False)
    requirements['dspark_candidate']['draft_kernels']=dict(top3=draft_binary is not None,
        kv_only=draft_binary is not None,markov_add=markov_add,full_markov_head=False)
    requirements['loaded_backend_verification']['sha256'] = sha(payload['serving/spark_backend_attestation.py'])
    payload['runtime-requirements.json'] = encoded(requirements)
    payload['serving/overlay-manifest.json'] = encoded({name.removeprefix('serving/'):sha(raw)
        for name,raw in payload.items() if name.startswith('serving/') and name != 'serving/overlay-manifest.json'})
    _, private_pins = assignment('serving/spark_backend_attestation.py', 'PRIVATE_SOURCES')
    for name, digest in private_pins.items():
        if sha(payload['serving/'+name]) != digest:
            raise ValueError('Candidate backend attestation pin did not converge: '+name)
    for name, raw in payload.items():
        if name.endswith('.py'): compile(raw, name, 'exec')
    return payload


def prepare(args):
    parent, out = args.parent.absolute(), args.output.absolute()
    if out.resolve() != out or out.exists() or not out.parent.is_dir() or out.is_relative_to(parent):
        raise ValueError('Use a fresh unredirected sibling output')
    previous, old = load_parent(parent, args.parent_sha256)
    receipt_raw = bounded_read(args.build.absolute()/'complete.json')
    receipt = json.loads(receipt_raw)
    prepared_raw = bounded_read(args.source.absolute()/'prepared.json')
    if sha(prepared_raw) != receipt['prepared_sha256']:
        raise ValueError('Build and corresponding source disagree')
    prepared = json.loads(prepared_raw)
    if prepared['files'] != receipt['source_files']:
        raise ValueError('Build source inventory changed')
    lengths=tuple(range(1,args.draft_tokens+1)) if args.verification!='fixed' else ()
    if getattr(args,'prefix_lengths',None) is not None:
        if args.verification!='ema':raise ValueError('Explicit prefix inventory is only supported for EMA')
        lengths=tuple(args.prefix_lengths)
    policy = Policy(args.draft_tokens,args.verification,lengths)
    draft_binary=None
    if args.draft_build:
        if not args.draft_source:raise ValueError('Corresponding draft source required')
        draft_receipt=json.loads(bounded_read(args.draft_build/'complete.json'))
        draft_source=bounded_read(args.draft_source/'prepared.json')
        draft_prepared=json.loads(draft_source)
        draft_binary=bounded_read(args.draft_build/'dspark_draft_top3.so')
        if (draft_receipt['abi']!=1 or draft_receipt['binary_sha256']!=sha(draft_binary)
                or draft_receipt['prepared_sha256']!=sha(draft_source)
                or draft_receipt['source_files']!=draft_prepared['files']):
            raise ValueError('Changed draft native build')
    elif args.draft_source or args.markov_add:raise ValueError('Combined draft kernels require source and build')
    payload = transform(old, policy, bounded_read(args.build.absolute()/'cooperative_moe.so'), receipt,
        draft_binary=draft_binary,markov_add=args.markov_add)
    if draft_binary is not None:
        for name,digest in draft_prepared['files'].items():
            raw=bounded_read(args.draft_source/safe_name(name))
            if sha(raw)!=digest:raise ValueError('Changed draft corresponding source: '+name)
            payload['experiments/dspark/top3/'+name]=raw
        payload['experiments/dspark/top3/prepared.json']=draft_source
        payload['experiments/dspark/top3/complete.json']=encoded(draft_receipt)
    for name, digest in prepared['files'].items():
        raw = bounded_read(args.source.absolute()/safe_name(name))
        if sha(raw) != digest: raise ValueError('Changed corresponding source: '+name)
        payload['experiments/dspark/native/'+name] = raw
    payload['experiments/dspark/native/prepared.json'] = prepared_raw
    payload['experiments/dspark/native/complete.json'] = receipt_raw
    for path in HERE.iterdir():
        if path.suffix in ('.py', '.md', '.cu'): payload['experiments/dspark/'+path.name] = bounded_read(path)
    if len(payload)>1000 or sum(map(len,payload.values()))>MAX_TOTAL or any(len(v)>MAX_FILE for v in payload.values()):
        raise ValueError('Candidate exceeds bounded kit limits')
    out.mkdir(mode=0o700)
    for name, raw in payload.items():
        path=out/safe_name(name); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f: f.write(raw)
    manifest = dict(format=previous['format'], standalone_runtime=False,
        clean_rebuild_qualified=False, publication_approved=False, serving_qualified=False,
        variant='experimental_dspark_k'+str(policy.draft_tokens)+'_'+policy.verification,
        parent_manifest_sha256=args.parent_sha256,
        files={name:dict(bytes=len(raw),sha256=sha(raw)) for name,raw in sorted(payload.items())})
    raw = encoded(manifest)
    with (out/'bundle-manifest.json').open('xb') as f:f.write(raw)
    load_parent(out,sha(raw))
    return dict(candidate=str(out), manifest_sha256=sha(raw), deployed=False)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('parent','source','build','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--parent-sha256',required=True)
    p.add_argument('--draft-tokens',type=int,choices=(3,4,5),required=True)
    p.add_argument('--verification',choices=('fixed','ema','confidence'),default='fixed')
    p.add_argument('--prefix-lengths',type=int,nargs='+',help='Explicit EMA graph inventory, e.g. 1 3 5')
    p.add_argument('--draft-build',type=Path)
    p.add_argument('--draft-source',type=Path)
    p.add_argument('--markov-add',action='store_true')
    print(json.dumps(prepare(p.parse_args()),indent=2))
