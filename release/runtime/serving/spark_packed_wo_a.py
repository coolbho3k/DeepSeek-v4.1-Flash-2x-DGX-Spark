"""Opt-in SM121 grouped MXFP8-weight/BF16-activation output projection.

Keep wo_a's original compact weights. Decode a tile to BF16 inside the
kernel and accumulate in FP32, rounding once at the output. Inverse RoPE
remains the native *unquantized* operation, as in the BF16 emulation path.
No serving entrypoint enables this experimental module yet.
"""
import functools

import torch


@functools.cache
def kernels():
    from vllm.triton_utils import triton, tl

    @triton.jit
    def gemv(X, W, S, P, M: tl.constexpr, G: tl.constexpr, N: tl.constexpr,
             K: tl.constexpr, XS0: tl.constexpr, XS1: tl.constexpr,
             XS2: tl.constexpr, SPLIT: tl.constexpr, BN: tl.constexpr,
             BK: tl.constexpr):
        ns = tl.program_id(0) * BN + tl.arange(0, BN)
        row = tl.program_id(1)
        group = tl.program_id(2) // SPLIT
        part = tl.program_id(2) % SPLIT
        ks = part * BK + tl.arange(0, BK)
        acc = tl.full((BN, BK), 0, tl.float32)
        for start in range(tl.cdiv(K, BK * SPLIT)):
            kk = ks + start * BK * SPLIT
            x = tl.load(X + row * XS0 + group * XS1 + kk * XS2,
                        mask=kk < K, other=0).to(tl.float32)
            w = tl.load(W + (group * N + ns[:, None]) * K + kk[None, :],
                        mask=(ns[:, None] < N) & (kk[None, :] < K), other=0.).to(tl.float32)
            scale = tl.load(S + (group * N + ns[:, None]) * (K // 32) + kk[None, :] // 32,
                            mask=(ns[:, None] < N) & (kk[None, :] < K), other=127).to(tl.float32)
            # Match native dequant_mxfp8_to_bf16 before multiplication.
            restored = (w * tl.exp2(scale - 127.)).to(tl.bfloat16).to(tl.float32)
            acc += restored * x[None, :]
        value = tl.sum(acc, axis=1)
        tl.store(P + ((part * M + row) * G + group) * N + ns, value, mask=ns < N)

    @triton.jit
    def gemm(X, W, S, P, M: tl.constexpr, G: tl.constexpr, N: tl.constexpr,
             K: tl.constexpr, XS0: tl.constexpr, XS1: tl.constexpr,
             XS2: tl.constexpr, SPLIT: tl.constexpr, BM: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr):
        ns = tl.program_id(0) * BN + tl.arange(0, BN)
        ms = tl.program_id(1) * BM + tl.arange(0, BM)
        group = tl.program_id(2) // SPLIT
        part = tl.program_id(2) % SPLIT
        ks = part * BK + tl.arange(0, BK)
        acc = tl.full((BM, BN), 0, tl.float32)
        for start in range(tl.cdiv(K, BK * SPLIT)):
            kk = ks + start * BK * SPLIT
            x = tl.load(X + ms[:, None] * XS0 + group * XS1 + kk[None, :] * XS2,
                        mask=(ms[:, None] < M) & (kk[None, :] < K), other=0)
            w = tl.load(W + (group * N + ns[None, :]) * K + kk[:, None],
                        mask=(ns[None, :] < N) & (kk[:, None] < K), other=0.).to(tl.float32)
            scale = tl.load(S + (group * N + ns[None, :]) * (K // 32) + kk[:, None] // 32,
                            mask=(ns[None, :] < N) & (kk[:, None] < K), other=127).to(tl.float32)
            restored = (w * tl.exp2(scale - 127.)).to(tl.bfloat16)
            acc = tl.dot(x, restored, acc)
        tl.store(P + ((part * M + ms[:, None]) * G + group) * N + ns[None, :],
                 acc, mask=(ms[:, None] < M) & (ns[None, :] < N))

    @triton.jit
    def finish(P, O, COUNT: tl.constexpr, SPLIT: tl.constexpr, B: tl.constexpr):
        offsets = tl.program_id(0) * B + tl.arange(0, B)
        parts = tl.arange(0, SPLIT)
        partials = tl.load(P + parts[:, None] * COUNT + offsets[None, :],
                           mask=offsets[None, :] < COUNT, other=0)
        tl.store(O + offsets, tl.sum(partials, axis=0), mask=offsets < COUNT)

    @triton.jit
    def restore(W, S, O, COUNT: tl.constexpr, B: tl.constexpr):
        offsets = tl.program_id(0) * B + tl.arange(0, B)
        values = tl.load(W + offsets, mask=offsets < COUNT, other=0.).to(tl.float32)
        codes = tl.load(S + offsets // 32, mask=offsets < COUNT, other=127).to(tl.float32)
        tl.store(O + offsets, values * tl.exp2(codes - 127.), mask=offsets < COUNT)

    return triton, gemv, gemm, finish, restore


def _reconstruct_weight(weight, scale):
    """One call-local 32 MiB BF16 buffer; never retained by a layer or module."""
    triton, _, _, _, restore = kernels()
    restored = torch.empty(weight.shape, dtype=torch.bfloat16, device=weight.device)
    restore[(triton.cdiv(weight.numel(), 1024),)](weight, scale, restored, weight.numel(),
                                               1024, num_warps=4)
    return restored


def grouped_projection(x, weight, scale, *, algorithm='auto'):
    """[tokens,4,4096] @ four [1024,4096] compact weight groups."""
    if (x.device.type != 'cuda' or weight.device != x.device or scale.device != x.device
            or x.dtype != torch.bfloat16 or weight.dtype != torch.float8_e4m3fn
            or scale.dtype != torch.uint8 or x.ndim != 3 or x.shape[1:] != (4, 4096)
            or weight.shape != (4096, 4096) or scale.shape != (4096, 128)
            or not weight.is_contiguous() or not scale.is_contiguous()
            or any(s <= 0 for s in x.stride()) or len(x) > 1056):
        raise ValueError('Packed wo_a requires the bounded TP2 DS41 BF16/MXFP8 layout')
    if torch.cuda.get_device_capability(x.device) != (12, 1):
        raise ValueError('Packed wo_a is qualified only on SM121')
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Packed wo_a graph capture is not qualified')
    if algorithm not in ('auto', 'gemv', 'gemm', 'reconstruct'):
        raise ValueError('Unknown packed wo_a algorithm')
    m, g, k = x.shape
    n = weight.shape[0] // g
    out = torch.empty((m, g, n), dtype=x.dtype, device=x.device)
    if not m:
        return out
    if algorithm == 'auto':
        algorithm = 'gemv' if m <= 6 else 'gemm' if m <= 16 else 'reconstruct'
    if algorithm == 'reconstruct':
        restored = _reconstruct_weight(weight, scale).view(g, n, k)
        torch.bmm(x.transpose(0, 1), restored.transpose(1, 2), out=out.transpose(0, 1))
        return out
    if algorithm == 'gemv' and m > 16:
        raise ValueError('GEMV is limited to small decode/speculative batches')
    split = 16 if algorithm == 'gemv' else 4 if m <= 16 else 1
    partials = (torch.empty((split, m, g, n), dtype=torch.float32, device=x.device)
                if split > 1 else out)
    triton, gemv, gemm, finish, _ = kernels()
    common = (x, weight, scale, partials, m, g, n, k, *x.stride(), split)
    if algorithm == 'gemv':
        gemv[(triton.cdiv(n, 16), m, g * split)](*common, 16, 256,
                                                 num_warps=4, enable_fp_fusion=False)
    else:
        gemm[(triton.cdiv(n, 64), triton.cdiv(m, 16), g * split)](
            *common, 16, 64, 64, num_warps=4)
    if split > 1:
        finish[(triton.cdiv(out.numel(), 256),)](partials, out, out.numel(), split, 256,
                                              num_warps=4)
    return out


def prepare_layer(layer):
    if (not getattr(layer, 'is_bmm', False) or getattr(layer, 'bmm_batch_size', None) != 4
            or not getattr(layer, 'prefix', '').endswith('.wo_a')
            or layer.weight.dtype != torch.float8_e4m3fn
            or layer.weight.shape != (4096, 4096)
            or layer.weight_scale.dtype != torch.uint8
            or layer.weight_scale.shape[0] != 4096 or layer.weight_scale.shape[1] < 128):
        raise ValueError('Unexpected packed wo_a loading layout')
    layer.weight = torch.nn.Parameter(layer.weight.detach().contiguous(), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(layer.weight_scale.detach()[:, :128].contiguous(),
                                           requires_grad=False)
    layer._ds41_packed_wo_a = True


def output_projection(attention, o, positions):
    from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import fused_inv_rope_fp8_quant

    if (attention.n_local_groups, attention.n_local_heads, attention.o_lora_rank,
            attention.nope_head_dim, attention.rope_head_dim) != (4, 32, 1024, 448, 64):
        raise ValueError('Unexpected DS41 grouped attention layout')
    transformed, scales = fused_inv_rope_fp8_quant(
        o, positions, attention.rotary_emb.cos_sin_cache,
        n_groups=4, heads_per_group=8, nope_dim=448, rope_dim=64,
        quant_group_size=attention._einsum_recipe[2],
        tma_aligned_scales=attention._tma_aligned_scales, quantize=False)
    z = grouped_projection(transformed, attention.wo_a.weight, attention.wo_a.weight_scale)
    return attention.wo_b(z.flatten(1))


_installed = None


def register():
    """Private opt-in; only change wo_a loading and its SM120 attention consumer."""
    global _installed
    from vllm.model_executor.kernels.linear.mxfp8.emulation import EmulationMxfp8LinearKernel
    from vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse import DeepseekV4FlashInferSM120Attention
    from vllm.platforms import current_platform

    if _installed is not None:
        if (EmulationMxfp8LinearKernel.process_weights_after_loading,
                DeepseekV4FlashInferSM120Attention._o_proj) != _installed:
            raise RuntimeError('Packed wo_a hooks were replaced')
        return
    if tuple(current_platform.get_device_capability() or ()) != (12, 1):
        raise ValueError('Packed wo_a requires SM121')
    original_load = EmulationMxfp8LinearKernel.process_weights_after_loading
    original_projection = DeepseekV4FlashInferSM120Attention._o_proj

    @functools.wraps(original_load)
    def load(kernel, layer):
        if getattr(layer, 'prefix', '').endswith('.wo_a') and getattr(layer, 'is_bmm', False):
            prepare_layer(layer)
        else:
            original_load(kernel, layer)

    @functools.wraps(original_projection)
    def project(attention, o, positions):
        if getattr(attention.wo_a, '_ds41_packed_wo_a', False):
            return output_projection(attention, o, positions)
        return original_projection(attention, o, positions)

    EmulationMxfp8LinearKernel.process_weights_after_loading = load
    DeepseekV4FlashInferSM120Attention._o_proj = project
    _installed = (load, project)
