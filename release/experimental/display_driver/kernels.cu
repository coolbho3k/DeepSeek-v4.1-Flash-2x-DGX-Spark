// SPDX-License-Identifier: AGPL-3.0-only
// Bounded synthetic KV-like reads; this is NOT a serving-speed benchmark.
#include <stdint.h>
extern "C" __global__ void fill_pattern(uint4 *p,uint64_t n,uint32_t seed) {
    for(uint64_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        uint32_t j=(uint32_t)i;
        p[i]=make_uint4(j^seed,j*3U+seed,j*7U^seed,~j+seed);
    }
}
extern "C" __global__ void verify_pattern(const uint4 *p,uint64_t n,uint32_t seed,uint32_t *bad) {
    unsigned errors=0;
    for(uint64_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=gridDim.x*blockDim.x) {
        uint4 v=p[i];uint32_t j=(uint32_t)i;
        errors+=(v.x!=(j^seed))||(v.y!=(j*3U+seed))||(v.z!=(j*7U^seed))||(v.w!=(~j+seed));
    }
    if(errors)atomicAdd(bad,errors);
}
extern "C" __global__ void read_stream(const uint4 *p,uint64_t n,uint4 *out) {
    uint32_t t=blockIdx.x*blockDim.x+threadIdx.x;
    uint4 sum=make_uint4(0,0,0,0);
    for(uint64_t i=t;i<n;i+=gridDim.x*blockDim.x) {
        uint4 v=p[i];sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;
    }
    out[t]=sum;
}
extern "C" __global__ void gather_rows(const uint4 *p,uint64_t rows,uint32_t epoch,uint4 *out) {
    uint32_t t=blockIdx.x*blockDim.x+threadIdx.x;
    // 4096 independently selected 512-byte rows; one warp per row.
    if(t<4096*32) {
        uint64_t key=((uint64_t)(t/32)*8191U+epoch*131U)%rows;
        out[t]=p[key*32+t%32];
    }
}
