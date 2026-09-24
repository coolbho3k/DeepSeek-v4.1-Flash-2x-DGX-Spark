// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { CUresult r=(x);if(r!=CUDA_SUCCESS) { const char *s="unknown";cuGetErrorName(r,&s);printf("{\"gpu_probe_error\":\"%s\",\"code\":%d,\"name\":\"%s\"}\n",#x,(int)r,s);goto cleanup; } }while(0)
static CUresult launch(CUfunction f,void **args,CUstream stream) {
    return cuLaunchKernel(f,512,1,1,256,1,1,0,stream,args,NULL);
}
int display_gpu_probe(CUdeviceptr imported,size_t bytes) {
    CUmodule module=NULL;CUfunction fill,verify,read,gather;
    CUdeviceptr reference=0,output=0,bad=0;CUstream stream=NULL;
    CUevent start=NULL,stop=NULL;CUgraph graph=NULL;CUgraphExec executable=NULL;
    const size_t output_bytes=512*256*16;uint32_t *host=NULL;int ok=0;
    uint64_t n=bytes/16,rows=bytes/512;uint32_t seed=0x418375ab,bad_host=0,epoch=0;
    CHECK(cuModuleLoad(&module,"/kernels.cubin"));
    CHECK(cuModuleGetFunction(&fill,module,"fill_pattern"));
    CHECK(cuModuleGetFunction(&verify,module,"verify_pattern"));
    CHECK(cuModuleGetFunction(&read,module,"read_stream"));
    CHECK(cuModuleGetFunction(&gather,module,"gather_rows"));
    CHECK(cuMemAlloc(&reference,bytes));CHECK(cuMemAlloc(&output,output_bytes));CHECK(cuMemAlloc(&bad,4));
    CHECK(cuStreamCreate(&stream,CU_STREAM_NON_BLOCKING));
    CHECK(cuEventCreate(&start,CU_EVENT_DEFAULT));CHECK(cuEventCreate(&stop,CU_EVENT_DEFAULT));
    host=malloc(output_bytes);if(!host)goto cleanup;
    for(int which=0;which<4;which++) {
        CUdeviceptr input=(which&1)?reference:imported;const char *label=(which&1)?"cuda_device_control":"display_or_host_import";
        printf("{\"stage\":\"benchmark_pass\",\"pass\":%d,\"memory\":\"%s\"}\n",which,label);
        void *fill_args[]={&input,&n,&seed};void *verify_args[]={&input,&n,&seed,&bad};
        void *read_args[]={&input,&n,&output};void *gather_args[]={&input,&rows,&epoch,&output};
        CHECK(launch(fill,fill_args,stream));CHECK(cuStreamSynchronize(stream));
        CHECK(cuMemsetD32(bad,0,1));CHECK(launch(verify,verify_args,stream));CHECK(cuStreamSynchronize(stream));
        CHECK(cuMemcpyDtoH(&bad_host,bad,4));
        printf("{\"stage\":\"gpu_full_pattern_check\",\"memory\":\"%s\",\"bytes\":%zu,\"mismatched_vectors\":%u}\n",label,bytes,bad_host);
        if(bad_host)goto cleanup;
        for(int kind=0;kind<2;kind++) {
            CUfunction kernel=kind?gather:read;void **args=kind?gather_args:read_args;
            CHECK(launch(kernel,args,stream));CHECK(cuStreamSynchronize(stream));
            CHECK(cuEventRecord(start,stream));
            for(int repeat=0;repeat<20;repeat++)CHECK(launch(kernel,args,stream));
            CHECK(cuEventRecord(stop,stream));CHECK(cuEventSynchronize(stop));float ms=0;
            CHECK(cuEventElapsedTime(&ms,start,stop));
            size_t reads=kind?output_bytes:bytes;
            printf("{\"stage\":\"synthetic_read_bandwidth\",\"memory\":\"%s\",\"kind\":\"%s\",\"bytes_per_repeat\":%zu,\"repeats\":20,\"milliseconds\":%.6f,\"read_GBps\":%.6f}\n",label,kind?"gather_4096x512":"stream",reads,ms,(double)reads*20/(ms*1e6));
        }
        CHECK(cuStreamBeginCapture(stream,CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
        CHECK(launch(gather,gather_args,stream));CHECK(cuStreamEndCapture(stream,&graph));
        CHECK(cuGraphInstantiateWithFlags(&executable,graph,0));
        // Update actual backing memory outside the graph; verify replay reads
        // fresh values from the same imported pointer, not a stale copy.
        for(int replay=0;replay<6;replay++) {
            seed=0x418375abU+(uint32_t)replay*123457U;
            CHECK(launch(fill,fill_args,stream));CHECK(cuGraphLaunch(executable,stream));CHECK(cuStreamSynchronize(stream));
            CHECK(cuMemcpyDtoH(host,output,output_bytes));
            unsigned errors=0;
            for(uint32_t t=0;t<512*256;t++) {
                uint64_t key=((uint64_t)(t/32)*8191U+epoch*131U)%rows;
                uint32_t j=(uint32_t)(key*32+t%32),*v=host+4*t;
                errors+=(v[0]!=(j^seed))||(v[1]!=(j*3U+seed))||(v[2]!=(j*7U^seed))||(v[3]!=(~j+seed));
            }
            printf("{\"stage\":\"graph_gather_replay\",\"memory\":\"%s\",\"replay\":%d,\"mismatched_vectors\":%u}\n",label,replay,errors);
            if(errors)goto cleanup;
        }
        CHECK(cuGraphExecDestroy(executable));executable=NULL;CHECK(cuGraphDestroy(graph));graph=NULL;
    }
    ok=1;
cleanup:
    if(stream)cuStreamSynchronize(stream);
    if(executable)cuGraphExecDestroy(executable);
    if(graph)cuGraphDestroy(graph);
    if(start)cuEventDestroy(start);
    if(stop)cuEventDestroy(stop);
    if(stream)cuStreamDestroy(stream);
    if(bad)cuMemFree(bad);
    if(output)cuMemFree(output);
    if(reference)cuMemFree(reference);
    if(module)cuModuleUnload(module);
    free(host);
    return ok;
}
