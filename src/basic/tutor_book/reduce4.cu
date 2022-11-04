#include "reduce.cuh"
#include "error.cuh"

// 前置条件：*d_y 需要初始化为0
static __global__ void reduce_syncwarp(const real *d_x, real *d_y, const int N) {
    extern __shared__ real s_y[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;

    s_y[tid] = n < N ? d_x[n] : 0.0;    __syncthreads();
    for (int i = blockDim.x >> 1; i>=32; i >>= 1) {
        if (tid < i)    s_y[tid] += s_y[tid + i];
        __syncthreads();
    }

    for (int i = 16; i; i >>= 1) {
        if (tid < i)    s_y[tid] += s_y[tid + i];
        __syncwarp();
    }    

    if(tid == 0) atomicAdd(d_y, s_y[0]);
}

extern __host__ real reduce4(const real *d_x, const int N) {
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int y_bytes = sizeof(real) * 1;

    real *d_y;
    CHECK(cudaMalloc((void**)&d_y, y_bytes));
    CHECK(cudaMemset(d_y, 0, sizeof(real))); // d_y 必须初始化为0

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    reduce_syncwarp<<<GRID_SIZE, BLOCK_SIZE, sizeof(real) * BLOCK_SIZE>>>(d_x, d_y, N);
    cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
    float elapsedTime; cudaEventElapsedTime(&elapsedTime, start, stop);
    CHECK(cudaGetLastError());
    printf("Kernel with input size %d consumes %f ms.\n", N, elapsedTime);    

    real *h_y;
    CHECK(cudaMallocHost((void**)&h_y, y_bytes));
    // 可以通过修改核函数并多次调用reduce_constmem来避免下面的CPU REDUCE
    CHECK(cudaMemcpy(h_y, d_y, y_bytes, cudaMemcpyDeviceToHost));

    real ret = *h_y; 

    cudaFree(d_y);
    cudaFreeHost(h_y);

    return ret;      
}