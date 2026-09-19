// SPDX-License-Identifier: AGPL-3.0-only
// DS41 draft specialization of the MiaAI/ExLlamaV3-derived staged FP32 path.
// Original copyright and MIT notices accompany the generated source bundle.
// This wrapper follows draft_grouped_kernels.cuh, generated from the pinned
// existing staged implementation without changing GEMM or epilogue math.
#include <cuda_runtime.h>
#include <cub/block/block_scan.cuh>
#include "util.h"
#include "util.cuh"
#include "staged_grouped_gemv.cuh"
#include "semantics.cuh"
#include "draft_grouped_kernels.cuh"

namespace draft = ds41_grouped_staged;
static bool prepared = false;
extern "C" int ds41_draft_top3_abi() { return 1; }

extern "C" int ds41_draft_top3_info(int* info) {
    if (!info) return int(cudaErrorInvalidValue);
    int device;
    auto err=cudaGetDevice(&device); if(err!=cudaSuccess)return int(err);
    cudaDeviceProp props;
    err=cudaGetDeviceProperties(&props,device);if(err!=cudaSuccess)return int(err);
    if(props.major!=12 || props.minor!=1)return int(cudaErrorInvalidDevice);
    void* kernels[2]={(void*)draft::gemv<8,4,2,true>,(void*)draft::gemv<8,4,2,false>};
    for(int i=0;i<2;++i) {
        cudaFuncAttributes attr;
        err=cudaFuncGetAttributes(&attr,kernels[i]);if(err!=cudaSuccess)return int(err);
        int blocks;
        err=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernels[i],256,8192);
        if(err!=cudaSuccess)return int(err);
        if(blocks<1)return int(cudaErrorLaunchOutOfResources);
        info[i*5]=256;info[i*5+1]=8192;info[i*5+2]=attr.numRegs;
        info[i*5+3]=int(attr.localSizeBytes);info[i*5+4]=blocks;
    }
    info[10]=30;info[11]=90;prepared=true;return 0;
}

// x/out/counts/tokens/weights, nine pointer tables, four shared temps, meta.
// Caller owns the serialized dispatcher and supplies zero-initialized out.
extern "C" int ds41_draft_top3_launch(void** t,int rows,int experts,void* stream_ptr) {
    if(!t || !prepared || rows<1 || rows>30 || experts<1 || experts>128)
        return int(cudaErrorInvalidValue);
    for(int i=0;i<19;++i)if(!t[i])return int(cudaErrorInvalidValue);
    draft::Tables tables;
    for(int i=0;i<9;++i)tables.columns[i]=(int64_t*)t[5+i];
    auto stream=(cudaStream_t)stream_ptr;
    half* g=(half*)t[14];half* u=(half*)t[15];half* ig=(half*)t[16];half* iu=(half*)t[17];
    auto meta=(int64_t*)t[18];auto tokens=(int64_t*)t[3];int slots=rows*3;
    draft::routes<<<1,512,0,stream>>>((int64_t*)t[2],meta,experts,slots,slots);
    auto err=cudaGetLastError();if(err!=cudaSuccess)return int(err);
    draft::gather<<<dim3(10,1,slots),128,0,stream>>>((half*)t[0],g,u,meta,tokens,tables,1);
    err=cudaGetLastError();if(err!=cudaSuccess)return int(err);
    draft::gemv<8,4,2,true><<<dim3(18,1,slots*2),256,8192,stream>>>(g,u,ig,iu,meta,tables,1,slots);
    err=cudaGetLastError();if(err!=cudaSuccess)return int(err);
    draft::activate<<<dim3(3,1,slots),128,0,stream>>>(ig,iu,meta,(float*)t[4],tables,1);
    err=cudaGetLastError();if(err!=cudaSuccess)return int(err);
    draft::gemv<8,4,2,false><<<dim3(80,1,slots),256,8192,stream>>>(g,u,ig,iu,meta,tables,1,slots);
    err=cudaGetLastError();if(err!=cudaSuccess)return int(err);
    draft::finish<<<dim3(10,1,slots),128,0,stream>>>(g,u,(float*)t[1],meta,tokens,tables,1);
    return int(cudaGetLastError());
}
