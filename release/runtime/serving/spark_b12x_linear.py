"""Opt into measured native B12X MXFP8 linears with corrected FP32 reduction.

Only the non-BMM MXFP8 preference list changes. Native grouped wo_a selection
continues to choose Emulation; the separate packed wo_a hook handles it.
No weights, EXL3 experts, vision layers, or other quantization lists change.
"""
_installed = None


def register():
    global _installed
    from vllm.model_executor.kernels import linear
    from vllm.platforms import current_platform
    from spark_b12x_fp32_reduce import register as register_reduction

    if tuple(current_platform.get_device_capability() or ()) != (12, 1):
        raise ValueError('Private native B12X selection requires SM121')
    register_reduction()
    platform = current_platform._enum
    possible = linear._POSSIBLE_MXFP8_KERNELS[platform]
    backend = linear.B12xMxfp8LinearKernel
    if _installed is not None:
        if tuple(possible) != _installed:
            raise RuntimeError('Native B12X preference list changed')
        return
    if possible.count(backend) != 1:
        raise ValueError('Unexpected native MXFP8 kernel preference list')
    supported, reason = backend.is_supported()
    if not supported:
        raise ValueError('Native B12X is unavailable: '+str(reason))
    ordered = [backend, *(item for item in possible if item is not backend)]
    linear._POSSIBLE_MXFP8_KERNELS[platform] = ordered
    _installed = tuple(ordered)
