# SPDX-License-Identifier: AGPL-3.0-only
"""Maintenance-only K4/K5 C6 indexer: dense reference and changing owned graphs.

Uses the DS41 graph/indexer work derived from the attributed MiaAI stack.
Do not run beside the resident server. No model weights are loaded.
"""
import argparse
import gc
import json
from pathlib import Path
import runpy
import subprocess


def known(torch,shape):
    lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],device='cuda')
    codes=torch.randint(0,16,(*shape,128),device='cuda',dtype=torch.int32)
    exponent=torch.randint(-4,2,(*shape,4),device='cuda',dtype=torch.int32)
    dense=lut[codes.long()]*torch.exp2(exponent.float()).repeat_interleave(32,dim=-1)
    packed=(codes[...,::2]|(codes[...,1::2]<<4)).to(torch.uint8).contiguous().view(torch.int8)
    scales=(exponent+127).to(torch.uint8).contiguous().view(torch.int32).squeeze(-1)
    return packed,scales,dense


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int,choices=(0,1),required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve earlier evidence')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU occupied; never stops workloads')
    import torch
    torch.set_num_threads(2)
    runpy.run_path('/opt/ds41-serving/serve.py',run_name='dspark_attention_probe')
    torch.cuda.set_per_process_memory_fraction(.02)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(419185)
    from ds41.dcp_indexer_graph import paged_logits
    from ds41.graph_validation import GraphOwner
    cases=[]
    with torch.inference_mode():
        for query in (5,6):
            for cap in (512,524288):
                batch=6;rows=batch*query;pages=32;states=64;columns=cap//states
                q,qs,dense_q=known(torch,(batch,query,32))
                keys,ks,dense_k=known(torch,(pages,states))
                raw=torch.zeros((pages,4608),device='cuda',dtype=torch.uint8)
                raw[:,:states*64].copy_(keys.view(torch.uint8).reshape(pages,-1))
                raw[:,states*64:states*68].view(torch.int32).copy_(ks)
                cache=raw[:,:states*68].view(pages,states,1,68)
                table=(torch.arange(batch*columns,device='cuda').reshape(batch,columns)*7%pages).int()
                flat_table=table.repeat_interleave(query,0)
                lengths=(torch.arange(rows,device='cuda').reshape(batch,query)*127%(cap+1)).int()
                lengths[0,:]=cap;lengths[1,0]=0
                weights=torch.rand(batch,query,32,device='cuda')*.01
                def grouped():return paged_logits((q,qs),cache,weights,lengths,table,None,max_model_len=cap)
                def flattened():return paged_logits((q.reshape(rows,1,32,64),qs.reshape(rows,1,32)),cache,
                    weights.reshape(rows,1,32),lengths.reshape(rows,1),flat_table,None,max_model_len=cap)
                def compare(actual):
                    expected=grouped()
                    torch.testing.assert_close(actual,expected,rtol=5e-5,atol=5e-5)
                    if cap==512:
                        for r in range(batch):
                            restored=dense_k[table[r].long()].reshape(-1,128)
                            dense=(torch.matmul(dense_q[r],restored.T).relu()*weights[r,:,:,None]).sum(1)
                            live=torch.arange(cap,device='cuda')[None,:]<lengths[r,:,None]
                            reference=dense.masked_fill(~live,-torch.inf)
                            torch.testing.assert_close(actual[r*query:(r+1)*query],reference,rtol=5e-5,atol=5e-5)
                compare(flattened())
                owner=GraphOwner(torch.device('cuda',0));graph=torch.cuda.CUDAGraph()
                torch.cuda.synchronize()
                with owner.execution(capture_only=True),torch.cuda.graph(graph):out=flattened()
                for step in range(3):
                    table.add_(3).remainder_(pages);flat_table.copy_(table.repeat_interleave(query,0))
                    lengths.copy_((lengths+17).remainder(cap+1));weights.mul_(.97)
                    with owner.execution(capture_only=False):graph.replay()
                    compare(out)
                owner.wait_before_graph_destruction();graph.reset();owner.release_after_graph_destruction()
                cases.append(dict(rows=rows,query=query,capacity=cap,dense_reference=cap==512,
                    grouped_flattened_match=True,changed_metadata_graph_replays=3))
                print(json.dumps(cases[-1]),flush=True)
                del graph,owner,out,q,qs,dense_q,keys,ks,dense_k,raw,cache,table,flat_table,lengths,weights
                gc.collect();torch.cuda.empty_cache()
    report=dict(status='dspark_k5_attention_component_passed',rank=a.rank,cases=cases,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),full_model_qualified=False)
    assert report['peak_cuda_bytes']<2*2**30
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
