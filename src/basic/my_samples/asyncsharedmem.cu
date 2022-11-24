#include <cstdint>
#include "../tutor_book/error.cuh"
#include <cuda_pipeline.h>

template <typename T>
__global__ void pipeline_kernel_sync(T *global, uint64_t *clock, size_t copy_count) {
  extern __shared__ char s[];
  T *shared = reinterpret_cast<T *>(s);

  uint64_t clock_start = clock64();

  for (size_t i = 0; i < copy_count; ++i) {
    shared[blockDim.x * i + threadIdx.x] = global[blockDim.x * i + threadIdx.x];
  }

  uint64_t clock_end = clock64();

  atomicAdd(reinterpret_cast<unsigned long long *>(clock),
            clock_end - clock_start);
}



template <typename T>
__global__ void pipeline_kernel_async(T *global, uint64_t *clock, size_t copy_count) {
  extern __shared__ char s[];
  T *shared = reinterpret_cast<T *>(s);

  uint64_t clock_start = clock64();

  //pipeline pipe;
  for (size_t i = 0; i < copy_count; ++i) {
    __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x],
                            &global[blockDim.x * i + threadIdx.x], sizeof(T));
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  
  uint64_t clock_end = clock64();

  atomicAdd(reinterpret_cast<unsigned long long *>(clock),
            clock_end - clock_start);
}

int main(int argc, char const *argv[])
{
  char *d_vec;
  CHECK(cudaMalloc((void **)&d_vec, 32 * 1024 * 1024));

  // Warm Up
  for (size_t i = 0; i < 10; i++)
  {
      pipeline_kernel_sync<int><<<256, 256, 4096>>>((int *)d_vec, nullptr, 1024*1024/(sizeof(int)*256*256));
  }
  CHECK(cudaDeviceSynchronize());

  for (size_t Size = 1024*1024; Size < 32 * 1024 * 1024; Size += 1024*1024)
  {
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // test sync 4B case        
    uint64_t clk = 0; cudaEventRecord( start, 0 );
    for (size_t i = 0; i < 10; i++)
    {
        pipeline_kernel_sync<int><<<256, 256, Size/256>>>((int *)d_vec, &clk, Size/(sizeof(int)*256*256));
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    CHECK(cudaGetLastError());

    cudaEventElapsedTime( &time, start, stop );       
    printf("Size: %luB, elem: %luB, Sync: %fms, 10xCycle: %lu\n", Size, sizeof(int), time / 10.0f, clk);    

    // test async 4B case
    clk = 0; cudaEventRecord( start, 0 );    
    for (size_t i = 0; i < 10; i++)
    {
        pipeline_kernel_async<int><<<256, 256, Size/256>>>((int *)d_vec, &clk, Size/(sizeof(int)*256*256));
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    CHECK(cudaGetLastError());

    cudaEventElapsedTime( &time, start, stop );       
    printf("Size: %luB, elem: %luB, ASync: %fms, 10xCycle: %lu\n", Size, sizeof(int), time / 10.0f, clk);  

    // test sync 8B case
    clk = 0; cudaEventRecord( start, 0 );
    for (size_t i = 0; i < 10; i++)
    {
        pipeline_kernel_sync<int2><<<256, 256, Size/256>>>((int2 *)d_vec, &clk, Size/(sizeof(int2)*256*256));
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    CHECK(cudaGetLastError());

    cudaEventElapsedTime( &time, start, stop );       
    printf("Size: %luB, elem: %luB, Sync: %fms, 10xCycle: %lu\n", Size, sizeof(int2), time / 10.0f, clk);    

    // test async 8B case
    clk = 0; cudaEventRecord( start, 0 );    
    for (size_t i = 0; i < 10; i++)
    {
        pipeline_kernel_async<int2><<<256, 256, Size/256>>>((int2 *)d_vec, &clk, Size/(sizeof(int2)*256*256));
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    CHECK(cudaGetLastError());

    cudaEventElapsedTime( &time, start, stop );       
    printf("Size: %luB, elem: %luB, ASync: %fms, 10xCycle: %lu\n", Size, sizeof(int2), time / 10.0f, clk);  

    // test sync 16B case
    clk = 0; cudaEventRecord( start, 0 );
    for (size_t i = 0; i < 10; i++)
    {
        pipeline_kernel_sync<int4><<<256, 256, Size/256>>>((int4 *)d_vec, &clk, Size/(sizeof(int4)*256*256));
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    CHECK(cudaGetLastError());

    cudaEventElapsedTime( &time, start, stop );       
    printf("Size: %luB, elem: %luB, Sync: %fms, 10xCycle: %lu\n", Size, sizeof(int4), time / 10.0f, clk);    

    // test async 16B case
    clk = 0; cudaEventRecord( start, 0 );    
    for (size_t i = 0; i < 10; i++)
    {
        pipeline_kernel_async<int4><<<256, 256, Size/256>>>((int4 *)d_vec, &clk, Size/(sizeof(int4)*256*256));
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    CHECK(cudaGetLastError());

    cudaEventElapsedTime( &time, start, stop );       
    printf("Size: %luB, elem: %luB, ASync: %fms, 10xCycle: %lu\n", Size, sizeof(int4), time / 10.0f, clk); 
  }
  CHECK(cudaFree(d_vec));
  return 0;
}
