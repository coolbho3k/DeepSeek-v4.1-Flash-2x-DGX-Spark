// SPDX-License-Identifier: AGPL-3.0-only
// Bounded owned DRM/CUDA registration diagnostic. No modesetting or driver changes.
#define _GNU_SOURCE
#include <cuda.h>
#include <drm/drm.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

static const size_t MIB=1024UL*1024;
int display_gpu_probe(CUdeviceptr,size_t);
static int result(const char *op,CUresult r) {
    const char *s="unknown"; cuGetErrorName(r,&s);
    printf("{\"operation\":\"%s\",\"rc\":%d,\"name\":\"%s\"}\n",op,r,s);
    return r==CUDA_SUCCESS;
}
static unsigned long long available(void) {
    FILE *f=fopen("/proc/meminfo","r");char line[256];unsigned long long n=0;
    if(!f)return 0;
    while(fgets(line,sizeof(line),f))if(sscanf(line,"MemAvailable: %llu",&n)==1)break;
    fclose(f);return n*1024;
}
static void snapshot(const char *stage) {
    size_t free=0,total=0;cuMemGetInfo(&free,&total);
    printf("{\"stage\":\"%s\",\"available\":%llu,\"cuda_free\":%zu,\"cuda_total\":%zu}\n",stage,available(),free,total);
}
static void mapping(void *p) {
    FILE *f=fopen("/proc/self/smaps","r");char line[512];unsigned long a,b;int selected=0;
    if(!f)return;
    while(fgets(line,sizeof(line),f)) {
        if(sscanf(line,"%lx-%lx",&a,&b)==2)selected=(uintptr_t)p>=a&&(uintptr_t)p<b;
        if(selected&&(!strncmp(line,"VmFlags:",8)||!strncmp(line,"Rss:",4)||!strncmp(line,"KernelPageSize:",15))) {
            line[strcspn(line,"\n")]=0;printf("{\"mapping\":\"%s\"}\n",line);
        }
    }
    fclose(f);
}
static int check_gpu(CUdeviceptr gpu,void *cpu,size_t bytes) {
    if(!result("device_fill",cuMemsetD8(gpu,0x6b,bytes))||!result("synchronize",cuCtxSynchronize()))return 0;
    size_t errors=0;
    for(size_t i=0;i<bytes;i+=4096)errors+=((volatile unsigned char*)cpu)[i]!=0x6b;
    errors+=((volatile unsigned char*)cpu)[bytes-1]!=0x6b;
    printf("{\"stage\":\"gpu_write_cpu_read\",\"errors\":%zu,\"bytes\":%zu}\n",errors,bytes);
    if(errors)return 0;
    return !getenv("DS41_PROBE_FULL_GPU") || display_gpu_probe(gpu,bytes);
}
int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);
    if(argc!=7) {fprintf(stderr,"usage: probe native|drm|anon MiB none|read|write|first|populate io|normal|portable fixed|direct register-MiB(0=all)\n");return 2;}
    const char *backend=argv[1],*touch=argv[3],*mode=argv[4],*layout=argv[5];
    char *end;unsigned long mib=strtoul(argv[2],&end,10);if(*end||!mib||mib>1792)return 2;
    unsigned long reg_mib=strtoul(argv[6],&end,10);if(*end||reg_mib>mib)return 2;
    if((strcmp(backend,"native")&&strcmp(backend,"drm")&&strcmp(backend,"anon"))||
       (strcmp(touch,"none")&&strcmp(touch,"read")&&strcmp(touch,"write")&&strcmp(touch,"first")&&strcmp(touch,"populate"))||
       (strcmp(mode,"io")&&strcmp(mode,"normal")&&strcmp(mode,"portable"))||
       (strcmp(layout,"fixed")&&strcmp(layout,"direct")))return 2;
    if(available()<8ULL*1024*1024*1024){fprintf(stderr,"Need idle host with 8 GiB available\n");return 3;}
    printf("{\"backend\":\"%s\",\"mib\":%lu,\"touch\":\"%s\",\"mode\":\"%s\",\"layout\":\"%s\",\"register_mib\":%lu}\n",backend,mib,touch,mode,layout,reg_mib);
    CUdevice dev;CUcontext ctx=NULL;int version=0,ok=0,fd=-1,registered=0;
    unsigned handle=0;size_t bytes=mib*MIB,reg_bytes=reg_mib?reg_mib*MIB:bytes;
    void *ptr=MAP_FAILED,*library=NULL,*pool=NULL;void (*destroy)(void*)=NULL;
    if(!result("cuInit",cuInit(0))||!result("cuDeviceGet",cuDeviceGet(&dev,0))||
       !result("primary_context",cuDevicePrimaryCtxRetain(&ctx,dev))||!result("set_context",cuCtxSetCurrent(ctx)))goto cleanup;
    cuDriverGetVersion(&version);Dl_info info={0};dladdr((void*)cuInit,&info);
    printf("{\"cuda_driver_api\":%d,\"libcuda\":\"%s\",\"page_size\":%ld}\n",version,info.dli_fname?info.dli_fname:"?",sysconf(_SC_PAGESIZE));
    snapshot("before_allocate");
    if(!strcmp(backend,"native")) {
        if(mib!=1792)return 2;
        library=dlopen("/original.so",RTLD_NOW|RTLD_LOCAL);if(!library){fprintf(stderr,"%s\n",dlerror());goto cleanup;}
        void *(*create)(size_t,size_t)=dlsym(library,"ds41_display_create");
        destroy=dlsym(library,"ds41_display_destroy");
        const char *(*error)(void)=dlsym(library,"ds41_display_error");
        uint64_t (*pointer)(void*)=dlsym(library,"ds41_display_pointer");
        if(!create||!destroy||!error||!pointer)goto cleanup;
        const char *ordinary_text=getenv("DS41_PROBE_ORDINARY_MIB");
        unsigned long ordinary=ordinary_text?strtoul(ordinary_text,&end,10):0;
        if(ordinary>1024||(ordinary_text&&*end))goto cleanup;
        printf("{\"ordinary_mib\":%lu}\n",ordinary);
        pool=create(ordinary*MIB,bytes);
        if(!pool){printf("{\"native_error\":\"%s\"}\n",error());goto cleanup;}
        snapshot("native_registered");ok=check_gpu(pointer(pool),(void*)(uintptr_t)pointer(pool),bytes+ordinary*MIB);goto cleanup;
    }
    if(!strcmp(backend,"anon"))ptr=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    else {
        fd=open("/dev/dri/card0",O_RDWR|O_CLOEXEC);if(fd<0){perror("open DRM");goto cleanup;}
        struct drm_mode_create_dumb c={.width=4096,.height=bytes/16384,.bpp=32};
        if(ioctl(fd,DRM_IOCTL_MODE_CREATE_DUMB,&c)){perror("create dumb");goto cleanup;}
        handle=c.handle;if(c.size!=bytes){fprintf(stderr,"Unexpected DRM size\n");goto cleanup;}
        snapshot("after_allocate");
        struct drm_mode_map_dumb m={.handle=handle};
        if(ioctl(fd,DRM_IOCTL_MODE_MAP_DUMB,&m)){perror("map dumb");goto cleanup;}
        if(!strcmp(layout,"fixed")) {
            ptr=mmap(NULL,bytes,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);if(ptr==MAP_FAILED)goto cleanup;
            if(mmap(ptr,bytes,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_FIXED,fd,m.offset)!=ptr){perror("map fixed");goto cleanup;}
        } else ptr=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_SHARED,fd,m.offset);
    }
    if(ptr==MAP_FAILED){perror("mmap");goto cleanup;}
    snapshot("after_mmap");mapping(ptr);
    if(!strcmp(touch,"populate")) {int rc=madvise(ptr,bytes,MADV_POPULATE_READ);printf("{\"populate_rc\":%d,\"errno\":%d}\n",rc,rc?errno:0);}
    else if(strcmp(touch,"none")) {
        size_t page=sysconf(_SC_PAGESIZE),limit=!strcmp(touch,"first")?1:bytes;unsigned char sum=0;
        for(size_t i=0;i<limit;i+=page) {
            volatile unsigned char *p=(volatile unsigned char*)ptr+i;
            if(!strcmp(touch,"write"))*p=0x35;else sum^=*p;
        }
        printf("{\"stage\":\"prefault\",\"checksum\":%u}\n",sum);
    }
    snapshot("before_register");mapping(ptr);
    unsigned flags=CU_MEMHOSTREGISTER_DEVICEMAP;
    if(strcmp(mode,"normal"))flags|=CU_MEMHOSTREGISTER_IOMEMORY;
    if(!strcmp(mode,"portable"))flags|=CU_MEMHOSTREGISTER_PORTABLE;
    if(!result("register",cuMemHostRegister(ptr,reg_bytes,flags)))goto cleanup;
    registered=1;CUdeviceptr gpu=0;
    if(!result("device_pointer",cuMemHostGetDevicePointer(&gpu,ptr,0)))goto cleanup;
    printf("{\"same_uva\":%s}\n",gpu==(CUdeviceptr)(uintptr_t)ptr?"true":"false");
    snapshot("after_register");ok=check_gpu(gpu,ptr,reg_bytes);
cleanup:
    if(registered)result("unregister",cuMemHostUnregister(ptr));
    if(ptr!=MAP_FAILED)munmap(ptr,bytes);
    if(handle){struct drm_gem_close c={.handle=handle};ioctl(fd,DRM_IOCTL_GEM_CLOSE,&c);}
    if(fd>=0)close(fd);
    if(pool&&destroy)destroy(pool);
    if(library)dlclose(library);
    if(ctx){snapshot("after_free");result("release_context",cuDevicePrimaryCtxRelease(dev));}
    printf("{\"status\":\"%s\"}\n",ok?"pass":"failed");return ok?0:1;
}
