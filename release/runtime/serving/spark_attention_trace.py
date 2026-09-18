"""Private layer20 frontier extension; no numerical or scheduling fix.

Adds bounded observations to the already tested layer recorder. Original
callables, arguments, returned objects, cache writes and tensor precision are
preserved. Like the base trace, CPU snapshots perturb execution timing.
"""
import hashlib
from pathlib import Path

BASE_TRACE_SHA = '171e074857010bf0c5968c712ad8c4ab3443081924b6e484566fee98a99696c5'
SOURCES = {
    'models/deepseek_v4_1/attention.py':
        'ef13a8503b54172a63cca6932e2ee5a6d5d6ced81445949067f01b3f61ab6e5e',
    'models/deepseek_v4_1/nvidia/flashinfer_sparse.py':
        '19c8c2ebcffacd2e8fc370c005c7a2b619585597dfc1152e8010c7fecfd215c1',
}


def install_frontier(attn, recorder):
    """Install only after validating every needed layer20 observation point."""
    if (getattr(attn,'_ds41_attention_trace_installed',False) or attn.compress_ratio!=1
            or not attn.is_kv_source or not attn.is_index_source
            or attn.candidate_source_layer!=20 or attn.n_local_heads!=32 or attn.head_dim!=512
            or attn.indexer is None or attn.compressor is None
            or attn.topk_indices_buffer is None or attn.candidate_block_buffer is None):
        raise ValueError('Expected the unchanged ratio1 layer20 source, installed once')
    methods = ('_run_parallel_input_projections','forward_mqa')
    if any(not callable(getattr(attn,name,None)) for name in methods):
        raise ValueError('Missing native attention method')
    children = (
        ('main_q_projection',attn.wq_b),
        ('compressor_latent',attn.compressor),
        ('index_q_projection',attn.indexer.wq_b),
        ('index_k_projection',attn.indexer.wk),
    )
    if any(not callable(getattr(child,'register_forward_hook',None)) for _,child in children):
        raise ValueError('Missing native projection/compressor module')
    originals = {name:getattr(attn,name) for name in methods}

    def projects(*args,**kwargs):
        result = originals['_run_parallel_input_projections'](*args,**kwargs)
        recorder.record('layer.20.parallel_input_projections',result)
        return result

    def mqa(*args,**kwargs):
        values = dict(zip(('q','kv','positions','output'),args)) | kwargs
        if recorder.current is not None:
            if set(values)!={'q','kv','positions','output'}:
                raise ValueError('Unexpected native MQA call signature')
            recorder.record('layer.20.mqa.query',values['q'])
            recorder.record('layer.20.mqa.kv_input',values['kv'])
            recorder.record('layer.20.index.topk',attn.topk_indices_buffer)
            recorder.record('layer.20.index.candidate_blocks',attn.candidate_block_buffer)
        result = originals['forward_mqa'](*args,**kwargs)
        if recorder.current is not None:
            recorder.record('layer.20.mqa.output',values['output'])
        return result

    handles=[]
    for name,child in children:
        handles.append(child.register_forward_hook(
            lambda module,args,out,name=name: recorder.record('layer.20.'+name,out)))
    attn._run_parallel_input_projections = projects
    attn.forward_mqa = mqa
    attn._ds41_attention_trace_installed = True
    return dict(layer=20,methods=list(methods),module_outputs=[name for name,_ in children],
                cache_payloads_snapshotted=False,outputs_replaced=False,
                weights_modified=False,timing_perturbed=True), originals, handles


def attach(model,rank):
    import vllm
    import spark_layer_trace as base
    if hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()!=BASE_TRACE_SHA:
        raise ValueError('Previously qualified base trace changed')
    native=Path(vllm.__file__).resolve().parent
    for name,expected in SOURCES.items():
        if hashlib.sha256((native/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Unreviewed attention frontier source')
    if type(model).__name__!='DeepseekV41ForCausalLM':
        raise ValueError('Actual native V4.1 wrapper required')
    attn=model.language_model.model.layers[20].attn
    if type(attn).__name__!='DeepseekV4FlashInferSM120Attention':
        raise ValueError('Actual SM120 DCP attention required')
    base.attach(model,rank)
    recorder,_=model._ds41_private_layer_trace
    proof,originals,handles=install_frontier(attn,recorder)
    model._ds41_private_attention_trace=(originals,handles)
    base.exclusive(recorder.root/'attention-ready.json',dict(
        status='private_attention_frontier_attached_unarmed',rank=rank,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        base_trace_sha256=BASE_TRACE_SHA,native_source_sha256=SOURCES,
        original_base_trace_limits_preserved=True,publication_approved=False,**proof))
