"""Pinned vLLM CLI with the downward KV cap in parent and spawn interpreters."""
import os
from spark_context_free_metadata import register as register_context_free_metadata
register_context_free_metadata()

# Install before ANY native/vLLM import, in parent and spawn interpreters.
from spark_image_prefix import install as install_image_prefix
_image_prefix = install_image_prefix()

# Native on-demand initialization remains available; skip optional build children.
if os.environ.get('HUMMING_DISABLE_PARALLEL_BUILD', '1') != '1':
    raise ValueError('This candidate requires HUMMING_DISABLE_PARALLEL_BUILD=1')
os.environ['HUMMING_DISABLE_PARALLEL_BUILD'] = '1'

# Fixed private candidate: parent and spawn interpreters select the same path.
for _mode in ('DS41_ENABLE_FP4_MAIN_KV','DS41_ENABLE_FAST_INDEXER','DS41_ENABLE_FP4_INDEXER','DS41_ENABLE_PACKED_WOA','DS41_ENABLE_B12X_DENSE','DS41_ENABLE_INDEXER_K_PARITY','DS41_ENABLE_BACKEND_ATTESTATION','DS41_ENABLE_B12X_GRAPHS','DS41_B12X_NATIVE_SELECTION','DS41_B12X_FP32_REDUCER','DS41_ENABLE_FUSED_SPARSE_ATTENTION','DS41_ENABLE_FUSED_SPARSE_SLOTS','DS41_ENABLE_NATIVE_ENGRAM','DS41_ENABLE_GROUPED_PREFILL','DS41_ENABLE_DCP_COMMUNICATION','DS41_ENABLE_COMBINED_MIAAI','DS41_ENABLE_DSPARK'):
    if os.environ.get(_mode,'1') != '1':
        raise ValueError('This candidate requires '+_mode+'=1')
    os.environ[_mode] = '1'

# Fixed outer collective batch; the inner gather still processes32 tokens.
if os.environ.get('DS41_FP4_DCP_COLLECTIVE_CHUNK','512') != '512':
    raise ValueError('This candidate requires512-token outer DCP batches')
os.environ['DS41_FP4_DCP_COLLECTIVE_CHUNK']='512'

# Set before any B12X import in parent and spawned workers.
if os.environ.get('B12X_DENSE_SPLITK_TURBO','0') != '0':
    raise ValueError('This candidate requires FP32 B12X split-K reduction')
os.environ['B12X_DENSE_SPLITK_TURBO']='0'

if os.environ.get('DS41_ENABLE_SSD_VOCAB','1')!='1':
    raise ValueError('Input-vocabulary mode differs from the qualified runtime')
os.environ['DS41_ENABLE_SSD_VOCAB']='1'
if os.environ.get('VLLM_USE_BREAKABLE_CUDAGRAPH', '0') != '0':
    raise ValueError('Combined serving uses owned native graphs, not breakable graphs')
os.environ['VLLM_USE_BREAKABLE_CUDAGRAPH'] = '0'

if os.environ.get('DS41_ENABLE_COOPERATIVE_MOE', '1') != '1':
    raise ValueError('Combined draft candidate requires cooperative MoE')
os.environ['DS41_ENABLE_COOPERATIVE_MOE']='1'

from spark_combined_miaai import register as register_combined
register_combined()
from ds41.speculative_prefix_retention import register as register_speculative_prefix
register_speculative_prefix()
from spark_parent_heap import register as register_parent_heap
register_parent_heap()
from ds41.tokenizer_heap import register as register_tokenizer_heap
register_tokenizer_heap()

from spark_dcp_communication import register as register_dcp_communication
register_dcp_communication()

fp4_setting = os.environ.get('DS41_ENABLE_FP4_MAIN_KV', '0')
if fp4_setting not in ('0', '1'):
    raise ValueError('DS41_ENABLE_FP4_MAIN_KV must be exactly0 or1')
if fp4_setting == '1':
    from ds41.vllm_fp4_main import register as register_fp4_main
    register_fp4_main()

from spark_sparse_slots import register as register_sparse_slots
register_sparse_slots()

from spark_indexer_k_math import register as register_indexer_k_parity
register_indexer_k_parity()

from spark_b12x_linear import register as register_b12x
register_b12x()
from spark_b12x_decode_graph import register as register_b12x_graphs
register_b12x_graphs()
from spark_packed_wo_a import register as register_packed_wo_a
register_packed_wo_a()

from spark_cpu_profile import register as register_cpu_profile
register_cpu_profile()

from spark_native_engram import register as register_native_engram
register_native_engram()

from spark_grouped_prefill import register as register_grouped_prefill
register_grouped_prefill()

from spark_backend_attestation import register as register_backend_attestation
register_backend_attestation()

from ds41.display_kv import register as register_display_kv
register_display_kv()

from spark_kv_cap import register

# Deliberately outside the main guard: multiprocessing reexecutes this file
# as __mp_main__. Register before any engine can capture its planner binding.
register()

# Ensure the scheduler/API process also has the reviewed native hooks when
# plugins are disabled in bounded diagnostics. Repeat registration validates
# actual bindings; it does not silently repair an overwritten hook.
_image_prefix.register()

# Native AsyncLLM ignore_frontend missing-default compatibility.
import hashlib as _profile_hashlib
from pathlib import Path as _ProfilePath
import vllm.v1.engine.async_llm as _profile_async
if _profile_hashlib.sha256(_ProfilePath(_profile_async.__file__).read_bytes()).hexdigest() != '2395dcf299cc015e0e69476fe8d986ccc8b96a69a83b9fc51a9a8bc62d9a8b26':
    raise RuntimeError('Unreviewed native profiler API source')
if not hasattr(_profile_async.AsyncLLM, 'profiler'):
    _profile_async.AsyncLLM.profiler = None
elif _profile_async.AsyncLLM.profiler is not None:
    raise RuntimeError('Unexpected native frontend profiler class default')

if __name__ == '__main__':
    import sys
    from spark_mxfp4_config import configure_argv
    sys.argv = configure_argv(sys.argv)
    from vllm.entrypoints.cli.main import main
    main()
