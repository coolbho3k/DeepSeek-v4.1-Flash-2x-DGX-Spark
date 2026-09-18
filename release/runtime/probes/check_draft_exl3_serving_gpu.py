# SPDX-License-Identifier: AGPL-3.0-only
"""Both-rank packed loader, 128-expert/top3 numerics, native graph mutation proof."""
import argparse
import inspect
import json
from pathlib import Path
import runpy
from types import SimpleNamespace


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host-index',type=int,choices=(0,1),required=True)
    a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    runpy.run_path('/opt/ds41-serving/serve.py',run_name='draft_component_entry')
    import spark_combined_miaai as combined
    import spark_fused_moe as base
    from ds41 import combined_dspark as scoped
    from ds41.draft_exl3_serving import artifact,load_packed,inventory
    from ds41.exl3_moe import eager_moe,PackedExpert
    from vllm.models.deepseek_v4_1.quant_config import DeepseekV4FP8Config
    from check_combined_miaai_gpu import NativeGraph
    assert combined.register()['cooperative_moe']
    assert not torch.cuda.is_initialized()
    torch.cuda.set_per_process_memory_fraction(.06)
    method_type=inspect.getclosurevars(DeepseekV4FP8Config.get_quant_method).nonlocals['DraftEXL3MoEMethod']
    current=SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=2,
        enable_expert_parallel=False,enable_eplb=False))
    method_type.create_weights.__globals__['get_current_vllm_config']=lambda:current
    rank=1-a.host_index
    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.swiglu_limit=10.
            self.quant_method=method_type(SimpleNamespace(tp_rank=rank,tp_size=2),3)
        def _map_global_expert_id_to_local_expert_id(self,expert):return expert
    model=torch.nn.Module();model.model=torch.nn.Module();model.model.layers=torch.nn.ModuleList()
    token=scoped._draft_scope.set(SimpleNamespace())
    with torch.device('cuda'):
        try:
            for _ in range(3):
                layer=torch.nn.Module();layer.ffn=torch.nn.Module();layer.ffn.experts=torch.nn.Module()
                layer.ffn.experts.routed_experts=Experts()
                experts=layer.ffn.experts.routed_experts
                experts.quant_method.create_weights(experts,128,5120,1152,torch.bfloat16)
                layer.ffn.gate=torch.nn.Module()
                layer.ffn.gate.e_score_correction_bias=torch.nn.Parameter(torch.zeros(128,dtype=torch.float32),requires_grad=False)
                experts.e_score_correction_bias=layer.ffn.gate.e_score_correction_bias
                model.model.layers.append(layer)
        finally:scoped._draft_scope.reset(token)
    root,index=artifact()
    with torch.inference_mode():
        loaded=load_packed(model,root,index)
        assert len(loaded)==24
        for layer in model.model.layers:
            experts=layer.ffn.experts.routed_experts
            experts.quant_method.process_weights_after_loading(experts)
    from ds41.draft_exl3_serving import DESCRIPTOR_SHA
    model._ds41_draft_exl3=dict(descriptor_sha256=DESCRIPTOR_SHA)
    checked=inventory(model)
    total=checked['actual_parameter_bytes']
    assert checked['unchanged_router_bias_alias_bytes']==1536
    assert sum(p.numel()*p.element_size() for p in model.parameters())==total+1536
    assert total==2562494976,total
    from safetensors import safe_open
    from ds41.draft_exl3_contract import expected_keys
    # Independently construct eager experts from source tensors, not loader slices.
    refs={}
    for lid in range(3):
        refs[lid]={}
        with safe_open(root/f'draft-{lid+1:05d}-of-00003.safetensors',framework='pt',device='cpu') as shard:
            for eid in (0,63,127):
                tensors={key:shard.get_tensor(key).cuda() for key in expected_keys(lid,eid)}
                refs[lid][eid]=PackedExpert(tensors,f'mtp.{lid}.ffn.experts.{eid}',rank,2,limit=10.)
    torch.manual_seed(419163)
    dispatch=base._dispatcher;cases=[];replays=0
    def error(got,want):
        assert torch.isfinite(got).all() and torch.isfinite(want).all()
        nmse=((got.float()-want.float()).square().sum()/want.float().square().sum().clamp_min(1e-30)).item()
        assert nmse<=2e-5,nmse
        return nmse
    with torch.inference_mode():
        for lid,layer in enumerate(model.model.layers):
            bank=layer.ffn.experts.routed_experts._ds41_experts
            for rows in (1,2,3,4,6,8):
                for dtype in (torch.float16,torch.bfloat16):
                    x=torch.randn((rows,5120),device='cuda',dtype=dtype)*.2
                    ids=torch.tensor([0,63,127],device='cuda').expand(rows,3).contiguous()
                    weights=torch.rand((rows,3),device='cuda')*.5
                    expected=eager_moe(refs[lid],x,ids,weights)
                    actual=dispatch(bank,x,ids,weights)
                    cases.append(dict(layer=lid,rows=rows,dtype=str(dtype),nmse=error(actual,expected)))
                    if rows in (1,3,6):
                        graph=NativeGraph(lambda:dispatch(bank,x,ids,weights),tokens=rows)
                        for iteration in range(3):
                            ids.copy_(torch.tensor([127,0,63],device='cuda').expand_as(ids))
                            if iteration==1:ids[:,1]=-1
                            x.mul_(.91);weights.mul_(.9)
                            expected=eager_moe(refs[lid],x,ids,weights)
                            actual=graph.replay()
                            error(actual,expected);replays+=1
                        graph.close()
        # 40 target banks plus3 draft banks must fit without increasing scratch.
        retained=[]
        while len(dispatch.banks)<43:
            bank=dict(model.model.layers[0].ffn.experts.routed_experts._ds41_experts);retained.append(bank)
            dispatch(bank,x,ids,weights)
        assert len(dispatch.banks)==43
        assert dispatch.workspace.bytes==23470344
    result=dict(status='draft_exl3_serving_component_pass',host=a.host_index,tp_rank=rank,
        packed_parameter_bytes=total,cases=cases,native_graph_replays=replays,
        immutable_banks=len(dispatch.banks),workspace_bytes=dispatch.workspace.bytes,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),full_serving_qualified=False)
    (Path('/results')/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
