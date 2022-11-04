#include "reduce.cuh"
#include "error.cuh"

#include <cooperative_groups.h>
using namespace cooperative_groups;
namespace cg = cooperative_groups;

// 前置条件：*d_y 需要初始化为0
static __global__ void reduce_morework(const real *d_x, real *d_y, const int N) {
    extern __shared__ real s_y[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    int n = bid * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    real y = 0;
    for (; n < N; n += stride)  y += d_x[n];
    s_y[tid] = y;    __syncthreads();
    
    for (int i = blockDim.x >> 1; i>=32; i >>= 1) {
        if (tid < i)    s_y[tid] += s_y[tid + i];
        __syncthreads();
    }

    y = s_y[tid];
    thread_block_tile<32> g = tiled_partition<32>(this_thread_block());

    for (int i = g.size() >> 1; i; i >>= 1) {
        y += g.shfl_down(y, i);
    }    

    if(tid == 0) d_y[bid] = y;
}

extern __host__ real reduce7(const real *d_x, const int N) {
    const int GRID_SIZE = 10240;
    const int y_bytes = sizeof(real) * GRID_SIZE;

    real *d_y;
    CHECK(cudaMalloc((void**)&d_y, y_bytes));

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    reduce_morework<<<GRID_SIZE, BLOCK_SIZE, sizeof(real) * BLOCK_SIZE>>>(d_x, d_y, N);
    reduce_morework<<<1, 1024, sizeof(real) * 1024>>>(d_y, d_y, GRID_SIZE);
    cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
    float elapsedTime; cudaEventElapsedTime(&elapsedTime, start, stop);
    CHECK(cudaGetLastError());
    printf("Kernel with input size %d consumes %f ms.\n", N, elapsedTime);    

    real h_y[1] = {0.0};
    // 可以通过修改核函数并多次调用reduce_constmem来避免下面的CPU REDUCE
    CHECK(cudaMemcpy(h_y, d_y, sizeof(real), cudaMemcpyDeviceToHost));

    cudaFree(d_y);

    return h_y[0];     
}