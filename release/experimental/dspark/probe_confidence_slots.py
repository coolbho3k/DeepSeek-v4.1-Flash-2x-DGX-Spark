# SPDX-License-Identifier: AGPL-3.0-only
"""Maintenance-only real DCP slot-kernel regression for padded query starts.

No model load. Exercise all C1..C6 sizes, global DCP2, replicated SWA and
disabled rings, with poisoned tails and changed-input owned graph replay.
"""
import argparse
import inspect
import json
from pathlib import Path
import runpy
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank',type=int,choices=(0,1),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Preserve earlier evidence')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('GPU occupied; this probe never stops workloads')
    import torch
    torch.set_num_threads(2)
    runpy.run_path('/opt/ds41-serving/serve.py',run_name='confidence_slot_probe')
    torch.cuda.set_per_process_memory_fraction(.02)
    from ds41 import vllm_v2_cache
    from ds41.dspark_experiment.confidence import SLOT_VIEW
    from ds41.vllm_dcp import _compile
    from ds41.graph_validation import GraphOwner
    from vllm.v1.worker.gpu.block_table import BlockTables
    function=BlockTables.compute_slot_mappings
    # Reconstruct precisely the prior hook (only remove the view adaptation).
    # _compile dedents the nested compute() by four spaces.
    prior_view=tuple(part.replace('\n        ','\n    ').strip() for part in SLOT_VIEW)
    original=_compile(function,[(prior_view[1],prior_view[0])],
        inspect.getclosurevars(function).nonlocals)
    device=torch.device('cuda',0)
    tables=BlockTables([32,32,32],6,64,[16,16,16],device,[32,32,32],
        cp_size=2,cp_rank=args.rank,cp_interleave=1,slot_mapping_enabled=[True,True,False])
    tables._ds41_group_owners=(2,1,1)
    report=dict(status='running',rank=args.rank,cases=[],original_padded_rejections=0,
        full_model_qualified=False,scope='actual per-group slot mapping, not attention or sampling')
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    with torch.inference_mode():
        for requests in range(1,7):
            mapping=torch.arange(requests,device=device,dtype=torch.int64)
            starts=torch.empty(7,device=device,dtype=torch.int32)
            positions=torch.empty(36,device=device,dtype=torch.int64)
            expected=None
            def inputs(cycle):
                nonlocal expected
                lengths=[1+(i+cycle)%6 for i in range(requests)]
                ids=[(i+cycle)%6 for i in range(requests)]
                offsets=[0]
                pos=[]
                for i,length in enumerate(lengths):
                    offsets.append(offsets[-1]+length)
                    pos.extend(range(19+i*13+cycle*3,19+i*13+cycle*3+length))
                # Extra offsets are intentionally invalid: the slot kernel
                # must read ONLY the active prefix, without touching the tail.
                starts.fill_(-987654)
                starts[:requests+1].copy_(torch.tensor(offsets,device=device,dtype=torch.int32))
                mapping.copy_(torch.tensor(ids,device=device))
                positions.fill_(-987654)
                positions[:len(pos)].copy_(torch.tensor(pos,device=device))
                cpu_tables=[]
                for group,block in enumerate(tables.block_tables):
                    values=torch.arange(6*16,dtype=torch.int32).view(6,16)+100*group+cycle*5
                    cpu_tables.append(values)
                    block.gpu.copy_(values)
                expected=torch.full((3,64),-1,dtype=torch.int64)
                for request,idx in enumerate(ids):
                    for token in range(offsets[request],offsets[request+1]):
                        position=pos[token]
                        for group in (0,1):
                            if group==0 and position%2!=args.rank:continue
                            local=position//2 if group==0 else position
                            expected[group,token]=int(cpu_tables[group][idx,local//32])*32+local%32
                tables.slot_mappings.fill_(987654321)
                return lengths
            lengths=inputs(0)
            if requests<6:
                try:original(tables,mapping,starts,positions,36)
                except ValueError as error:
                    if str(error)!='Invalid V2 DCP slot mapping buffers':raise
                    report['original_padded_rejections']+=1
                else:raise ValueError('Original geometry failure was not reproduced')
            for invalid in (starts[:requests],starts.to(torch.int64),torch.empty(14,device=device,dtype=torch.int32)[::2]):
                try:function(tables,mapping,invalid,positions,36)
                except ValueError:pass
                else:raise ValueError('Invalid buffer admitted')
            function(tables,mapping,starts,positions,36)
            if not torch.equal(tables.slot_mappings.cpu(),expected):raise ValueError('Eager slot mismatch')
            owner=GraphOwner(device);graph=torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with owner.execution(capture_only=True),torch.cuda.graph(graph):
                result=function(tables,mapping,starts,positions,36)
            for cycle in (1,2,3):
                inputs(cycle)
                with owner.execution(capture_only=False):graph.replay()
                if not torch.equal(tables.slot_mappings.cpu(),expected):raise ValueError('Changed graph slot mismatch')
                if result.data_ptr()!=tables.slot_mappings.data_ptr():raise ValueError('Output storage copied')
            owner.wait_before_graph_destruction();graph.reset();owner.release_after_graph_destruction()
            report['cases'].append(dict(requests=requests,initial_lengths=lengths,changed_replays=3,
                padded_buffer_elements=7,replicated_swa_and_disabled_ring_checked=True))
            save()
    report.update(status='confidence_slot_components_passed',peak_cuda_bytes=torch.cuda.max_memory_allocated())
    if report['peak_cuda_bytes']>=2*2**30:raise ValueError('Slot probe exceeded bounded memory')
    save();print(json.dumps(report),flush=True)


if __name__=='__main__':main()
