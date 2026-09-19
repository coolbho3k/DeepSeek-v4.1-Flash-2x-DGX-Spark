# SPDX-License-Identifier: AGPL-3.0-only
"""Keep readiness proof explicit for the selected draft/verification geometry."""
import hashlib

PARENT_SHA='605064f092620aa53ead5c6a86319aa92c02944492e2ba32967c1e7fde2d7220'


def transform(raw,policy):
    if hashlib.sha256(raw).hexdigest()!=PARENT_SHA:
        raise ValueError('Changed pinned graph readiness source')
    text=raw.decode();k=policy.draft_tokens
    def edit(before,after):
        nonlocal text
        if text.count(before)!=1:raise ValueError('Changed readiness anchor: '+before)
        text=text.replace(before,after)
    edit('graph_inventory(speculator.query_cudagraph_manager,3)',
         f'graph_inventory(speculator.query_cudagraph_manager,{k})')
    edit('graph_inventory(runner.cudagraph_manager, 4 if enabled else 1)',
         f'graph_inventory(runner.cudagraph_manager, {k+1} if enabled else 1)')
    edit("or 3 not in {r['tokens'] for r in result.get('draft_full_graphs',[])}",
         f"or {k} not in {{r['tokens'] for r in result.get('draft_full_graphs',[])}}")
    edit("or (4 if enabled else 1) not in {row['tokens'] for row in result.get('target_full_graphs', [])}",
         f"or ({k+1} if enabled else 1) not in {{row['tokens'] for row in result.get('target_full_graphs', [])}}")
    edit("rows.append(dict(tokens=key.num_tokens, requests=key.num_reqs))",
         "rows.append(dict(tokens=key.num_tokens, requests=key.num_reqs,\n"
         "            uniform_token_count=key.uniform_token_count, max_query_len=key.max_query_len))")
    old="""        for requests in range(1, PROFILE['max_num_seqs']+1):
            if (not any(r['tokens']==4*requests and r['requests']==requests for r in target_graphs)
                    or not any(r['tokens']==3*requests and r['requests']==requests for r in draft_graphs)):
                raise ValueError(f'Missing target/draft graphs for {requests} requests')"""
    draft_check=f"r['tokens']=={k}*requests and r['requests']==requests and r['uniform_token_count']=={k}"
    if policy.verification=='confidence':
        target_check=(f"r['tokens']>=(prefix+1)*requests and r['requests']>=requests "
                      f"and r['uniform_token_count'] is None and r['max_query_len']=={k+1}")
    else:
        target_check="r['tokens']==(prefix+1)*requests and r['requests']==requests and r['uniform_token_count']==prefix+1"
    new=f"""        for requests in range(1, PROFILE['max_num_seqs']+1):
            if not any({draft_check} for r in draft_graphs):
                raise ValueError(f'Missing full-K draft graph for {{requests}} requests')
            for prefix in {policy.prefix_lengths!r}:
                if not any({target_check} for r in target_graphs):
                    raise ValueError(f'Missing target graph for {{requests}} requests / {{prefix}} drafts')"""
    edit(old,new)
    edit("draft_layers=3 if enabled else 0, native_cache_groups_verified=True, gpu_utilization=INITIAL_UTILIZATION,",
         f"draft_layers=3 if enabled else 0, draft_tokens={k}, verification={policy.verification!r},\n"
         "        native_cache_groups_verified=True, gpu_utilization=INITIAL_UTILIZATION,")
    edit("if (result.get('shared_vocabulary_identity') is not True",
         f"if (result.get('draft_tokens')!={k} or result.get('verification')!={policy.verification!r}\n"
         "                or result.get('shared_vocabulary_identity') is not True")
    compile(text,'spark_combined_ready.py','exec')
    return text.encode()
