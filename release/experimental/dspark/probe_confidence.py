# SPDX-License-Identifier: AGPL-3.0-only
"""Maintenance-only native CPU/device query-mismatch and FP4 scoring test.

Uses the actual registered native metadata methods, with evenly budgeted CPU
lengths and differently distributed GPU lengths. This tests the flattened
indexer path, not the complete attention/cache writer or full-model sampler.
"""
import argparse
import gc
import json
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank',type=int,choices=(0,1),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier evidence')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU occupied; this probe never stops workloads')
    import torch
    torch.set_num_threads(2);torch.manual_seed(419190)
    runpy.run_path('/opt/ds41-serving/serve.py',run_name='confidence_metadata_probe')
    torch.cuda.set_per_process_memory_fraction(.02)
    torch.backends.cuda.matmul.allow_tf32=False
    from vllm.v1.attention.backends.mla import indexer
    from ds41.graph_validation import GraphOwner
    from ds41.dcp_indexer_graph import paged_logits
    from probe_attention import known
    cls=indexer.DeepseekV32IndexerMetadataBuilder
    if not indexer.DeepseekV32IndexerBackend.supports_device_cpu_query_lens_mismatch():
        raise ValueError('SM121 flattened confidence adapter missing')
    config=SimpleNamespace(speculative_config=SimpleNamespace(enable_adaptive_verification=True,
        num_speculative_tokens=5),num_speculative_tokens=5)
    if not indexer._use_flattening(config) or indexer._supports_varlen_paged_mqa_logits():
        raise ValueError('Expected native flattened rows, not SM100 varlen scorer')
    cases=[]
    report=dict(status='running',rank=args.rank,cases=cases,full_model_qualified=False,
        scope='native query flattening, per-token DCP/compression bounds, FP4 scoring and changed owned graph replay')
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    patterns=((3,),(1,5),(1,2,6),(1,2,3,6),(1,1,6,6,1),(1,2,3,4,5,3),(6,6,6,6,6,6))
    with torch.inference_mode():
        for lengths in patterns:
            batch=len(lengths);total=sum(lengths);cap=512;states=64;pages=32;columns=cap//states
            cpu_lengths=torch.tensor([total//batch+(i<total%batch) for i in range(batch)],dtype=torch.int32)
            for ratio in (1,2):
                for padding in (0,1):
                    if total+padding>36:continue
                    rows=total+padding
                    builder=SimpleNamespace(vllm_config=config,supports_varlen=False,
                        dcp_world_size=2,dcp_rank=args.rank,cp_kv_cache_interleave_size=1,
                        decode_seq_lens_buffer=torch.empty(36,device='cuda',dtype=torch.int32),
                        global_decode_seq_lens_buffer=torch.empty(36,device='cuda',dtype=torch.int32),
                        decode_lens_buffer=torch.empty(36,device='cuda',dtype=torch.int32),
                        expanded_block_table_buffer=torch.empty((36,columns),device='cuda',dtype=torch.int32),
                        arange_buffer=torch.arange(36,device='cuda',dtype=torch.int32))
                    q,qs,dense_q=known(torch,(rows,1,32))
                    keys,ks,dense_k=known(torch,(pages,states))
                    raw=torch.zeros((pages,4608),device='cuda',dtype=torch.uint8)
                    raw[:,:states*64].copy_(keys.view(torch.uint8).reshape(pages,-1))
                    raw[:,states*64:states*68].view(torch.int32).copy_(ks)
                    cache=raw[:,:states*68].view(pages,states,1,68)
                    table=(torch.arange(batch*columns,device='cuda').reshape(batch,columns)*7%pages).int()
                    weights=torch.rand(rows,1,32,device='cuda')*.01
                    device_lengths=torch.tensor(lengths,device='cuda',dtype=torch.int32)
                    context=torch.tensor([21+31*i for i in range(batch)],device='cuda',dtype=torch.int32)
                    def metadata():
                        starts=device_lengths.cumsum(0).int()-device_lengths
                        seq=context+device_lengths
                        global_flat=cls._prepare_global_decode_seq_lens(builder,seq,device_lengths,
                            cpu_lengths,starts,rows,False,int(cpu_lengths.max()))
                        flat,flat_table,_,flat_batch,requires_padding=cls._prepare_decode_tensors(builder,
                            seq,table,device_lengths,cpu_lengths,starts,batch,rows,False,6,int(cpu_lengths.max()))
                        if requires_padding or flat_batch!=rows:raise ValueError('Native flattening changed')
                        # Exactly the coordinated adapter's order: expand each
                        # causal bound, compress completed states, then localize.
                        if ratio>1:
                            flat//=ratio;global_flat//=ratio
                        flat=cls._dcp_localize_decode_seq_lens(builder,flat,batch,True)
                        return flat,global_flat,flat_table
                    def score(prepared):
                        flat,_,flat_table=prepared
                        return paged_logits((q,qs),cache,weights,flat[:,None],flat_table,None,max_model_len=cap)
                    def check(result,current):
                        logits,flat,global_flat,flat_table=result
                        expected=[];global_expected=[];tables=[]
                        contexts=context.cpu().tolist()
                        for request,count in enumerate(current):
                            for position in range(count):
                                bound=(contexts[request]+position+1)//ratio
                                global_expected.append(bound)
                                expected.append((bound+1-args.rank)//2)
                                tables.append(request)
                        expected += [0]*padding;global_expected += [0]*padding
                        torch.testing.assert_close(flat,torch.tensor(expected,device='cuda',dtype=torch.int32),rtol=0,atol=0)
                        torch.testing.assert_close(global_flat,torch.tensor(global_expected,device='cuda',dtype=torch.int32),rtol=0,atol=0)
                        for row,request in enumerate(tables):
                            torch.testing.assert_close(flat_table[row],table[request],rtol=0,atol=0)
                            restored=dense_k[table[request].long()].reshape(-1,128)
                            dense=(torch.matmul(dense_q[row,0],restored.T).relu()*weights[row,0,:,None]).sum(0)
                            reference=dense.masked_fill(torch.arange(cap,device='cuda')>=expected[row],-torch.inf)
                            torch.testing.assert_close(logits[row],reference,rtol=5e-5,atol=5e-5)
                        if padding and not torch.isneginf(logits[-1]).all():raise ValueError('Padding row became visible')
                    current=list(lengths)
                    prepared=metadata()
                    check((score(prepared),*prepared),current)
                    owner=GraphOwner(torch.device('cuda',0));graph=torch.cuda.CUDAGraph()
                    torch.cuda.synchronize()
                    # Native serving builds metadata outside the model graph.
                    # Its CPU scalar staging is not capture-safe and need not
                    # be: only the persistent outputs are read by replay.
                    with owner.execution(capture_only=True),torch.cuda.graph(graph):logits=score(prepared)
                    result=(logits,*prepared)
                    for _ in range(3):
                        current=current[1:]+current[:1]
                        device_lengths.copy_(torch.tensor(current,device='cuda',dtype=torch.int32))
                        context.add_(3);table.add_(3).remainder_(pages);weights.mul_(.97)
                        builder.decode_seq_lens_buffer.fill_(-100)
                        builder.global_decode_seq_lens_buffer.fill_(-100)
                        builder.expanded_block_table_buffer.fill_(-1)
                        refreshed=metadata()
                        if any(old.data_ptr()!=new.data_ptr() for old,new in zip(prepared,refreshed)):
                            raise ValueError('Native metadata output address changed across replay')
                        with owner.execution(capture_only=False):graph.replay()
                        check(result,current)
                    owner.wait_before_graph_destruction();graph.reset();owner.release_after_graph_destruction()
                    row=dict(requests=batch,rows=rows,compression=ratio,padding=padding,
                        cpu_lengths=cpu_lengths.tolist(),device_lengths=list(lengths),replays=3,
                        device_cpu_differ=list(lengths)!=cpu_lengths.tolist())
                    cases.append(row);save();print(json.dumps(row),flush=True)
                    del graph,owner,result,prepared,refreshed,logits,builder,q,qs,dense_q,keys,ks,dense_k,raw,cache,table,weights,device_lengths,context
                    gc.collect();torch.cuda.empty_cache()
    report.update(status='confidence_flattened_components_passed',peak_cuda_bytes=torch.cuda.max_memory_allocated())
    if report['peak_cuda_bytes']>=2*2**30:raise ValueError('Bounded metadata probe exceeded memory envelope')
    save()


if __name__=='__main__':main()
