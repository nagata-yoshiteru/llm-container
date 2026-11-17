// Three-rank GPU/NCCL regression probe; no model or framework imports.
// Compile against the patched NCCL and CUDA runtime with g++.
// Usage: probe <rank:0..2> <id-file>. Start rank 0, copy its 512-byte id-file
// to ranks 1/2 while rank 0 waits, then start them with the same NCCL settings.
// NCCL_NET_HOST_REGISTER=0/1 selects the original/fixed NET allocation path.
// Run under a timeout: peers cannot complete if another rank exits early.
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",#call,cudaGetErrorString(e));return 2;} }while(0)
#define NCCL(call) do { auto e=(call); if(e!=ncclSuccess){fprintf(stderr,"NCCL %s: %s\n",#call,ncclGetErrorString(e));return 3;} }while(0)
int main(int argc,char**argv){
  if(argc!=3)return 1;
  constexpr int groups=4;
  ncclUniqueId ids[groups];
  int rank=atoi(argv[1]);if(rank<0||rank>=3)return 5;
  if(rank==0){
    for(auto& id:ids)NCCL(ncclGetUniqueId(&id));
    FILE* f=fopen(argv[2],"wb");if(!f)return 4;
    bool ok=fwrite(ids,sizeof(ids),1,f)==1;fclose(f);if(!ok)return 4;
  }else{
    FILE* f=fopen(argv[2],"rb");if(!f)return 6;
    bool ok=fread(ids,sizeof(ids),1,f)==1;fclose(f);if(!ok)return 6;
  }
  CUDA(cudaSetDevice(0));
  cudaStream_t stream;CUDA(cudaStreamCreate(&stream));
  constexpr size_t max_count=262144;
  float *data;CUDA(cudaMalloc(&data,max_count*sizeof(float)));
  std::vector<float> host(max_count);
  ncclComm_t comms[groups];
  for(int group=0;group<groups;++group){
    if(group%2){ncclConfig_t config=NCCL_CONFIG_INITIALIZER;config.blocking=1;NCCL(ncclCommInitRankConfig(&comms[group],3,ids[group],rank,&config));}
    else NCCL(ncclCommInitRank(&comms[group],3,ids[group],rank));
    for(size_t count: {size_t(1),size_t(1024),max_count}){
      std::fill(host.begin(),host.end(),float(rank+1));
      CUDA(cudaMemcpyAsync(data,host.data(),count*sizeof(float),cudaMemcpyHostToDevice,stream));
      NCCL(ncclAllReduce(data,data,count,ncclFloat32,ncclSum,comms[group],stream));
      CUDA(cudaMemcpyAsync(host.data(),data,count*sizeof(float),cudaMemcpyDeviceToHost,stream));
      CUDA(cudaStreamSynchronize(stream));
      for(size_t i=0;i<count;++i)if(host[i]!=6.0f){fprintf(stderr,"WRONG rank=%d group=%d index=%zu got=%f\n",rank,group,i,host[i]);return 7;}
    }
    printf("PASS rank=%d group=%d sizes=1,1024,262144\n",rank,group);fflush(stdout);
  }
  for(int i=groups-1;i>=0;--i)NCCL(ncclCommDestroy(comms[i]));
  CUDA(cudaFree(data));CUDA(cudaStreamDestroy(stream));
  return 0;
}
