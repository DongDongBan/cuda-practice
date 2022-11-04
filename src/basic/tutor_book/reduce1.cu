#include "reduce.cuh"
#include "error.cuh"

static __global__ void reduce_constmem(real *d_x, real *d_y, const int N) {
    const int tid = threadIdx.x;
    real * const p = d_x + blockIdx.x * blockDim.x;

    for (int i = blockDim.x >> 1; i; i >>= 1) {
        if (tid < i)    p[tid] += p[tid+i];
        __syncthreads();
    }

    if (tid == 0)   d_y[blockIdx.x] = p[0];
}

extern __host__ real reduce1(real *d_x, const int N) {
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int y_bytes = sizeof(real) * GRID_SIZE;

    real *d_y;
    CHECK(cudaMalloc((void**)&d_y, y_bytes));

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    reduce_constmem<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, d_y, N);
    cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
    float elapsedTime; cudaEventElapsedTime(&elapsedTime, start, stop);
    CHECK(cudaGetLastError());
    printf("Kernel with input size %d consumes %f ms.\n", N, elapsedTime);


    real *h_y;
    CHECK(cudaMallocHost((void**)&h_y, y_bytes));
    // 可以通过修改核函数并多次调用reduce_constmem来避免下面的CPU REDUCE
    CHECK(cudaMemcpy(h_y, d_y, y_bytes, cudaMemcpyDeviceToHost));

    real ret = 0; // 下面的循环也可以用CPU REDUCE代替
    for (int i = 0; i < GRID_SIZE; i++) ret += h_y[i];

    cudaFree(d_y);
    cudaFreeHost(h_y);

    return ret;    
} 