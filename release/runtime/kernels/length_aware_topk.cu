// SPDX-License-Identifier: AGPL-3.0-only
// DS41 exact length-aware selection. Uses NVIDIA CUB under its retained
// CUDA/CCCL distribution license; no CUB implementation is copied here.
#include <cuda_runtime.h>
#include <cub/block/block_radix_sort.cuh>
#include <cstdint>
#include <cmath>

template<int ITEMS>
__global__ void select_stage(const void* source, const int32_t* counts,
    uint64_t* destination, int32_t* output, int width, int stride, int dest_stride,
    int k, int level, bool first, bool final) {
    constexpr int THREADS=256, BLOCK=THREADS*ITEMS;
    using Sort=cub::BlockRadixSort<uint64_t,THREADS,ITEMS>;
    extern __shared__ __align__(16) unsigned char storage[];
    auto& temporary=*reinterpret_cast<typename Sort::TempStorage*>(storage);
    const int row=blockIdx.y, block=blockIdx.x;
    const int count=counts[row];
    if(count<=k) {
        if(final) for(int i=threadIdx.x;i<k;i+=THREADS)
            output[row*k+i]=i<count?i:-1;
        return;
    }
    int live=count;
    for(int l=0;l<level;l++)live=((live+BLOCK-1)/BLOCK)*k;
    if(block*BLOCK>=live)return;
    uint64_t values[ITEMS];
    #pragma unroll
    for(int i=0;i<ITEMS;i++) {
        const int col=block*BLOCK+threadIdx.x+i*THREADS;
        uint64_t key=0;
        if(col<width && col<live) {
            if(first) {
                float value=static_cast<const float*>(source)[size_t(row)*stride+col];
                if(value==0.f)value=0.f;
                uint32_t bits=__float_as_uint(value);
                uint32_t ordered=(bits&0x80000000u)?~bits:(bits^0x80000000u);
                if(isnan(value))ordered=0xffffffffu;
                key=(uint64_t(ordered)<<20)+uint64_t(1048575-col);
            } else key=static_cast<const uint64_t*>(source)[size_t(row)*stride+col];
        }
        values[i]=key;
    }
    Sort(temporary).SortDescending(values,0,52);
    #pragma unroll
    for(int i=0;i<ITEMS;i++) {
        const int lane=threadIdx.x*ITEMS+i;
        if(lane<k) {
            if(final)output[row*k+lane]=int32_t(1048575-(values[i]&1048575));
            else destination[size_t(row)*dest_stride+block*k+lane]=values[i];
        }
    }
}

static bool prepared=false;
extern "C" int ds41_topk_prepare(int* info) {
    if(!info)return int(cudaErrorInvalidValue);
    cudaDeviceProp p;
    int device;
    auto e=cudaGetDevice(&device);if(e!=cudaSuccess)return int(e);
    e=cudaGetDeviceProperties(&p,device);if(e!=cudaSuccess)return int(e);
    if(p.major!=12||p.minor!=1)return int(cudaErrorInvalidDevice);
    const int sizes[2]={sizeof(cub::BlockRadixSort<uint64_t,256,16>::TempStorage),
                        sizeof(cub::BlockRadixSort<uint64_t,256,32>::TempStorage)};
    void* funcs[2]={(void*)select_stage<16>,(void*)select_stage<32>};
    for(int n=0;n<2;n++) {
        if(sizes[n]>48*1024) {
            e=cudaFuncSetAttribute(funcs[n],cudaFuncAttributeMaxDynamicSharedMemorySize,sizes[n]);
            if(e!=cudaSuccess)return int(e);
        }
        cudaFuncAttributes attributes;
        e=cudaFuncGetAttributes(&attributes,funcs[n]);if(e!=cudaSuccess)return int(e);
        info[3*n]=sizes[n];info[3*n+1]=attributes.numRegs;info[3*n+2]=attributes.localSizeBytes;
    }
    prepared=true;return 0;
}
extern "C" int ds41_topk_stage(const void* source,const int32_t* counts,
    uint64_t* destination,int32_t* output,int rows,int width,int stride,int dest_stride,
    int k,int level,int first,int final,void* raw_stream) {
    if(!prepared||!source||!counts||!destination||!output||rows<1||rows>64||
       width<1||width>1048576||stride<width||level<0||level>12||
       (k!=512&&k!=1024&&k!=2048))return int(cudaErrorInvalidValue);
    const int block=k<=1024?4096:8192;
    const int blocks=(width+block-1)/block;
    if((final!=0)!=(blocks==1)||dest_stride<blocks*k)return int(cudaErrorInvalidValue);
    cudaStream_t stream=static_cast<cudaStream_t>(raw_stream);
    if(block==4096)select_stage<16><<<dim3(blocks,rows),256,
        sizeof(cub::BlockRadixSort<uint64_t,256,16>::TempStorage),stream>>>
        (source,counts,destination,output,width,stride,dest_stride,k,level,bool(first),bool(final));
    else select_stage<32><<<dim3(blocks,rows),256,
        sizeof(cub::BlockRadixSort<uint64_t,256,32>::TempStorage),stream>>>
        (source,counts,destination,output,width,stride,dest_stride,k,level,bool(first),bool(final));
    return int(cudaGetLastError());
}
