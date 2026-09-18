// SPDX-License-Identifier: AGPL-3.0-only
// Standalone, narrowly bounded adapter. No DevCtx/cudaMalloc or global scratch:
// every work buffer comes from the caller's capped PyTorch allocator.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <vector>
#include <mutex>
namespace cg = cooperative_groups;
#include "quant/exl3_moe_kernel.cuh"

constexpr int DS41_HIDDEN = 5120;
constexpr int DS41_INTERMEDIATE = 1152;
constexpr int DS41_THREADS = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;
constexpr int DS41_LOCKS = MOE_SCHED_OFFSET + MOE_SCHED_INTS;

std::vector<int64_t> ds41_mul1_resources()
{
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    // Bounded single-device resource cache. Prewarm before graph capture;
    // repeated forwards never mutate function attributes inside a graph.
    static std::mutex resource_mutex;
    static int resource_device = -1;
    static std::vector<int64_t> resource_values;
    const std::lock_guard<std::mutex> lock(resource_mutex);
    if (!resource_values.empty()) {
        TORCH_CHECK(resource_device == device, "One visible device per combined worker");
        return resource_values;
    }
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream().stream(), &capture));
    TORCH_CHECK(capture == cudaStreamCaptureStatusNone, "Prewarm combined MoE resources before capture");
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    TORCH_CHECK(properties.major == 12 && properties.minor == 1, "DS41 MUL1 probe requires GB10");
    cudaFuncAttributes attributes;
    auto kernel = ds41_moe_mul1_kernel<3, 128, 2>;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    TORCH_CHECK(SMEM_MAX + attributes.sharedSizeBytes <= properties.sharedMemPerBlockOptin,
                "Fused kernel exceeds physical shared-memory capacity");
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX));
    int occupancy;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, DS41_THREADS, SMEM_MAX));
    TORCH_CHECK(occupancy >= 1 && properties.cooperativeLaunch, "Cooperative residency not supported");
    resource_device = device;
    resource_values = {properties.multiProcessorCount, occupancy, SMEM_MAX,
            static_cast<int64_t>(attributes.sharedSizeBytes), attributes.numRegs,
            static_cast<int64_t>(attributes.localSizeBytes), DS41_THREADS,
            DS41_LOCKS, MOE_SMS_PER_EXPERT};
    return resource_values;
}

void ds41_mul1_forward(
    const at::Tensor& x, const at::Tensor& out, const at::Tensor& counts,
    const at::Tensor& tokens, const at::Tensor& weights,
    const std::vector<at::Tensor>& ptrs, const std::vector<at::Tensor>& temps,
    const at::Tensor& locks)
{
    TORCH_CHECK(x.is_cuda(), "CUDA input required");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    const auto device = x.device();
    auto check = [&](const at::Tensor& t, at::ScalarType dtype) {
        TORCH_CHECK(t.device() == device && t.is_contiguous() && t.scalar_type() == dtype,
                    "Tensor device, layout or dtype differs from DS41 MUL1 contract");
    };
    check(x, at::kHalf); check(out, at::kFloat); check(counts, at::kLong);
    check(tokens, at::kLong); check(weights, at::kFloat); check(locks, at::kInt);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == DS41_HIDDEN && x.size(0) <= 2048,
                "Only bounded DS41 TP2 shapes are supported");
    TORCH_CHECK(out.sizes() == x.sizes(), "Output shape mismatch");
    TORCH_CHECK(counts.dim() == 1 && counts.size(0) >= 2 && counts.size(0) <= 385,
                "Counts must include one trailing sentinel");
    const int experts = counts.size(0) - 1;
    TORCH_CHECK(tokens.dim() == 1 && weights.dim() == 1 && tokens.numel() == weights.numel(),
                "Routing tables differ");
    TORCH_CHECK(tokens.numel() <= x.size(0) * 6, "Too many routed assignments");
    TORCH_CHECK(ptrs.size() == 9 && temps.size() == 4, "Nine pointer tables and four work buffers required");
    for (const auto& p : ptrs) {
        check(p, at::kLong);
        TORCH_CHECK(p.dim() == 1 && p.numel() == experts, "Expert pointer-table shape mismatch");
    }
    for (const auto& t : temps) check(t, at::kHalf);
    TORCH_CHECK(temps[0].dim() == 3 && temps[0].size(2) == DS41_HIDDEN,
                "Hidden scratch shape mismatch");
    const int concurrency = temps[0].size(0), rows = temps[0].size(1);
    TORCH_CHECK(concurrency >= 1 && concurrency <= 6 && rows >= 16 && rows <= 128 && rows % 16 == 0,
                "Unreviewed scratch concurrency or row capacity");
    TORCH_CHECK(temps[1].sizes() == temps[0].sizes() && temps[2].dim() == 3
                && temps[2].size(0) == concurrency && temps[2].size(1) == rows
                && temps[2].size(2) == DS41_INTERMEDIATE && temps[3].sizes() == temps[2].sizes(),
                "Intermediate scratch shape mismatch");
    TORCH_CHECK(locks.dim() == 1 && locks.numel() == DS41_LOCKS, "Lock workspace shape mismatch");
    if (x.size(0) == 0) return;
    TORCH_CHECK(tokens.numel() % x.size(0) == 0, "Nonrectangular routing table");
    const int topk = tokens.numel() / x.size(0);
    TORCH_CHECK(topk >= 1 && topk <= 6, "Unreviewed top-k");
    auto resources = ds41_mul1_resources();
    TORCH_CHECK(concurrency * MOE_SMS_PER_EXPERT <= resources[0], "Grid exceeds co-resident SM count");
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    // The owned Python dispatcher fences the caller-provided scratch.
    // Native CUDA graph capture preserves this cooperative kernel launch.

    void* px = x.data_ptr(); void* po = out.data_ptr();
    void* pc = counts.data_ptr(); void* pt = tokens.data_ptr(); void* pw = weights.data_ptr();
    void* work[4]; for (int i = 0; i < 4; ++i) work[i] = temps[i].data_ptr();
    void* tables[9]; for (int i = 0; i < 9; ++i) tables[i] = ptrs[i].data_ptr();
    void* pl = locks.data_ptr();
    const int hidden = DS41_HIDDEN, intermediate = DS41_INTERMEDIATE, activation = 0, bits = 3;
    const float limit = 10.0f;
    void* arguments[] = {&px, &work[0], &work[1], &work[2], &work[3], &po,
        &tables[0], &tables[1], &tables[2], &tables[3], &tables[4], &tables[5],
        &tables[6], &tables[7], &tables[8], &pc, &pt, &pw,
        const_cast<int*>(&hidden), const_cast<int*>(&intermediate), const_cast<int*>(&experts),
        const_cast<int*>(&topk), const_cast<int*>(&rows), const_cast<int*>(&concurrency),
        const_cast<float*>(&limit), const_cast<int*>(&activation), const_cast<int*>(&bits),
        const_cast<int*>(&bits), const_cast<int*>(&bits), &pl};
    // Cooperative launch enforces whole-grid residency for inter-CTA barriers.
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
        reinterpret_cast<void*>(ds41_moe_mul1_kernel<3, 128, 2>),
        dim3(MOE_SMS_PER_EXPERT, 1, concurrency), dim3(DS41_THREADS), arguments, SMEM_MAX, stream));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
