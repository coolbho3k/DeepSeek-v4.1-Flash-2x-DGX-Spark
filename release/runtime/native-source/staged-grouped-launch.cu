// SPDX-License-Identifier: AGPL-3.0-only
// Reuses the retained MIT EXL3 register decoder and DS41 precision epilogues.
// Two through four tokens. Independent CTAs; no cooperative grid or inter-CTA spinlock.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include <mutex>
#include <vector>
#include "util.h"
#include "util.cuh"
#include "staged_grouped_gemv.cuh"
#include "semantics.cuh"

namespace ds41_grouped_staged {
constexpr int H=5120,I=1152,TOP=6;
struct Tables {
    const int64_t* columns[9];
    template<typename T> __device__ const T* get(int c,int expert) const {
        return reinterpret_cast<const T*>(columns[c][expert]);
    }
};

__global__ void routes(const int64_t* counts,int64_t* meta,int experts,int assignments,int slots) {
    using Scan=cub::BlockScan<int64_t,512>;
    __shared__ typename Scan::TempStorage scan;
    const int expert=threadIdx.x;
    if(expert<slots)meta[expert]=-1;
    __syncthreads();
    // Same exclusive prefix of the compact expert histogram, in parallel.
    // The trailing missing-expert sentinel is excluded, as in the old scan.
    const int64_t count=expert<experts ? counts[expert] : 0;
    int64_t start;
    Scan(scan).ExclusiveSum(count,start);
    for(int64_t slot=start;slot<start+count && slot<assignments;++slot)
        if(slot>=0)meta[slot]=expert;
}

__global__ void gather(const half* x,half* g,half* u,const int64_t* meta,const int64_t* tokens,Tables t,int rows) {
    int slot=blockIdx.z,expert=meta[slot];
    int chunk=blockIdx.x*4+threadIdx.x/32;
    if(expert<0 || chunk>=H/128)return;
    const int off=chunk*128, dst=slot*rows*H+off;
    x += tokens[slot]*H;
    had_hf_r_128_inner<true,false>(x+off,g+dst,t.get<half>(1,expert)+off,0.088388347648f);
    had_hf_r_128_inner<true,false>(x+off,u+dst,t.get<half>(4,expert)+off,0.088388347648f);
}

template<int WK,int WNT,int PF,bool UP>
__global__ __launch_bounds__(WK*32) void gemv(
    half* g,half* u,half* ig,half* iu,const int64_t* meta,Tables t,int rows,int slots) {
    const int slot=UP ? blockIdx.z/2 : blockIdx.z;
    const int projection=UP ? blockIdx.z%2 : 2;
    const int expert=meta[slot];
    if(expert<0)return; // Uniform across the complete CTA.
    // The router emits contiguous equal-expert assignments. Partition each
    // run into groups of four, including duplicate routes within a token.
    // Every CTA agrees on the same leader and active row count.
    int begin=slot;
    while(begin>0 && meta[begin-1]==expert)--begin;
    if((slot-begin)%4!=0)return;
    int active_rows=1;
    while(active_rows<4 && slot+active_rows<slots
          && meta[slot+active_rows]==expert)++active_rows;
    const half* x=UP ? (projection==0 ? g : u)+slot*rows*H : ig+slot*rows*I;
    half* y=UP ? (projection==0 ? ig : iu)+slot*rows*I : g+slot*rows*H;
    // Literal column indices prevent a dynamically indexed local copy of
    // all nine pointer tables in every thread (80B stack in the first build).
    const uint16_t* w=UP ? (projection==0 ? t.get<uint16_t>(0,expert)
        : t.get<uint16_t>(3,expert)) : t.get<uint16_t>(6,expert);
    ds41_grouped_gemv_inner<WK,WNT,PF>(x,w,y,UP?H:I,UP?I:H,blockIdx.x,gridDim.x,active_rows);
}

__global__ void activate(half* ig,half* iu,const int64_t* meta,const float* weight,Tables t,int rows) {
    int slot=blockIdx.z,expert=meta[slot];
    int chunk=blockIdx.x*4+threadIdx.x/32;
    if(expert<0 || chunk>=I/128)return;
    const int off=chunk*128,dst=slot*rows*I+off;
    ds41_guad(ig+dst,iu+dst,t.get<half>(2,expert)+off,t.get<half>(5,expert)+off,
        t.get<half>(7,expert)+off,weight[slot]);
}

__global__ void finish(half* g,half* u,float* out,const int64_t* meta,const int64_t* tokens,Tables t,int rows) {
    int slot=blockIdx.z,expert=meta[slot];
    int chunk=blockIdx.x*4+threadIdx.x/32;
    if(expert<0 || chunk>=H/128)return;
    const int off=chunk*128,dst=slot*rows*H+off;
    ds41_down_out(g+dst,u+dst,t.get<half>(8,expert)+off,out+tokens[slot]*H+off);
}

template<int WK,int WNT,int PF,bool UP>
void resource(std::vector<int64_t>& result,int variant) {
    cudaFuncAttributes attr;
    auto kernel=gemv<WK,WNT,PF,UP>;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,kernel));
    constexpr int smem=WK*4*WNT*16*sizeof(float);
    int occupancy;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,kernel,WK*32,smem));
    TORCH_CHECK(occupancy>=1,"Staged GEMV has no legal CTA residency");
    result.insert(result.end(),{variant,UP?1:0,WK*32,smem,occupancy,attr.numRegs,
        static_cast<int64_t>(attr.localSizeBytes)});
}

template<bool UP> void resources(std::vector<int64_t>& r) {
    resource<4,2,4,UP>(r,0);resource<8,2,4,UP>(r,1);
    resource<8,4,2,UP>(r,2);resource<16,2,4,UP>(r,3);
}

template<int WK,int WNT,int PF,bool UP>
void launch(half* g,half* u,half* ig,half* iu,const int64_t* meta,Tables t,int rows,int slots,cudaStream_t stream) {
    gemv<WK,WNT,PF,UP><<<dim3((UP?I:H)/(WNT*16),1,UP?slots*2:slots),WK*32,
        WK*4*WNT*16*sizeof(float),stream>>>(g,u,ig,iu,meta,t,rows,slots);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<bool UP>
void select(int variant,half* g,half* u,half* ig,half* iu,const int64_t* meta,Tables t,int rows,int slots,cudaStream_t stream) {
    switch(variant) {
        case 0:launch<4,2,4,UP>(g,u,ig,iu,meta,t,rows,slots,stream);break;
        case 1:launch<8,2,4,UP>(g,u,ig,iu,meta,t,rows,slots,stream);break;
        case 2:launch<8,4,2,UP>(g,u,ig,iu,meta,t,rows,slots,stream);break;
        case 3:launch<16,2,4,UP>(g,u,ig,iu,meta,t,rows,slots,stream);break;
        default:TORCH_CHECK(false,"Unqualified staged GEMV variant");
    }
}
} // namespace ds41_grouped_staged

std::vector<int64_t> ds41_grouped_staged_resources() {
    static std::mutex mutex;
    static int selected_device=-1;
    static std::vector<int64_t> values;
    const std::lock_guard<std::mutex> lock(mutex);
    int device;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    if(!values.empty()) {
        TORCH_CHECK(device==selected_device,"One visible staged-decode device required");
        return values;
    }
    cudaStreamCaptureStatus capture;
    C10_CUDA_CHECK(cudaStreamIsCapturing(at::cuda::getCurrentCUDAStream().stream(),&capture));
    TORCH_CHECK(capture==cudaStreamCaptureStatusNone,"Prewarm staged resources before capture");
    cudaDeviceProp props;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props,device));
    TORCH_CHECK(props.major==12 && props.minor==1,"Staged DS41 requires SM121");
    ds41_grouped_staged::resources<true>(values);ds41_grouped_staged::resources<false>(values);
    selected_device=device;
    return values;
}

void ds41_grouped_staged_forward(const at::Tensor& x,const at::Tensor& out,
    const at::Tensor& counts,const at::Tensor& tokens,const at::Tensor& weights,
    const std::vector<at::Tensor>& ptrs,const std::vector<at::Tensor>& temps,
    const at::Tensor& meta,int64_t up_variant,int64_t down_variant) {
    using namespace ds41_grouped_staged;
    TORCH_CHECK(x.is_cuda(),"CUDA input required");
    const at::cuda::OptionalCUDAGuard guard(x.device());
    auto check=[&](const at::Tensor& a,at::ScalarType dtype) {
        TORCH_CHECK(a.device()==x.device() && a.is_contiguous() && a.scalar_type()==dtype,
            "Staged input device/dtype/layout mismatch");
    };
    check(x,at::kHalf);check(out,at::kFloat);check(counts,at::kLong);
    check(tokens,at::kLong);check(weights,at::kFloat);check(meta,at::kLong);
    TORCH_CHECK(x.dim()==2 && (x.size(0)>=2 && x.size(0)<=4) && x.size(1)==H && out.sizes()==x.sizes(),
        "Only two through four5120-wide tokens are supported");
    TORCH_CHECK(counts.dim()==1 && counts.numel()>=2 && counts.numel()<=385,"Invalid expert counts");
    TORCH_CHECK(tokens.dim()==1 && weights.dim()==1 && tokens.numel()==weights.numel()
        && tokens.numel()>=1 && tokens.numel()<=TOP*x.size(0),"Invalid small-batch routes");
    TORCH_CHECK(meta.dim()==1 && meta.numel()==TOP*x.size(0),"Six route slots per token required");
    TORCH_CHECK(ptrs.size()==9 && temps.size()==4,"Original nine tables/four workspaces required");
    Tables tables;
    const int experts=counts.numel()-1;
    for(int c=0;c<9;++c) {
        check(ptrs[c],at::kLong);
        TORCH_CHECK(ptrs[c].dim()==1 && ptrs[c].numel()==experts,"Pointer table shape mismatch");
        tables.columns[c]=ptrs[c].data_ptr<int64_t>();
    }
    for(auto& temp:temps)check(temp,at::kHalf);
    TORCH_CHECK(temps[0].dim()==3 && temps[0].size(0)==TOP && temps[0].size(2)==H,
        "Original six-group workspace required");
    const int rows=temps[0].size(1);
    TORCH_CHECK(rows>=16 && rows<=128 && rows%16==0 && temps[1].sizes()==temps[0].sizes()
        && temps[2].dim()==3 && temps[2].size(0)==TOP && temps[2].size(1)==rows
        && temps[2].size(2)==I && temps[3].sizes()==temps[2].sizes(),"Invalid scratch shape");
    TORCH_CHECK(up_variant>=0 && up_variant<=3 && down_variant>=0 && down_variant<=3,"Invalid variant");
    ds41_grouped_staged_resources();
    auto stream=at::cuda::getCurrentCUDAStream().stream();
    half* g=reinterpret_cast<half*>(temps[0].data_ptr());
    half* u=reinterpret_cast<half*>(temps[1].data_ptr());
    half* ig=reinterpret_cast<half*>(temps[2].data_ptr());
    half* iu=reinterpret_cast<half*>(temps[3].data_ptr());
    auto m=meta.data_ptr<int64_t>();
    routes<<<1,512,0,stream>>>(counts.data_ptr<int64_t>(),m,experts,tokens.numel(),TOP*x.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gather<<<dim3(10,1,TOP*x.size(0)),128,0,stream>>>(reinterpret_cast<const half*>(x.data_ptr()),g,u,m,tokens.data_ptr<int64_t>(),tables,1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    select<true>(up_variant,g,u,ig,iu,m,tables,1,TOP*x.size(0),stream);
    activate<<<dim3(3,1,TOP*x.size(0)),128,0,stream>>>(ig,iu,m,weights.data_ptr<float>(),tables,1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    select<false>(down_variant,g,u,ig,iu,m,tables,1,TOP*x.size(0),stream);
    finish<<<dim3(10,1,TOP*x.size(0)),128,0,stream>>>(g,u,out.data_ptr<float>(),m,tokens.data_ptr<int64_t>(),tables,1);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
