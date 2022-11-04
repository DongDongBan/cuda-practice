#include "reduce.cuh"
#include "error.cuh"

// 有一些DEBUG输出代码，不是很整洁就没用DEBUG宏判断开启，而是直接注释了
static __global__ void reduce_sharedmem(const real *d_x, real *d_y, const int N) {
    extern __shared__ real s_y[];
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = bid * blockDim.x + tid;

    s_y[tid] = n < N ? d_x[n] : 0.0;    __syncthreads();
    // if (bid == 0)   printf("%d: %lf", tid, s_y[tid]);
    for (int i = blockDim.x >> 1; i; i >>= 1) {
        if (tid < i)    s_y[tid] += s_y[tid + i];
        __syncthreads();
    }

    if(tid == 0)    d_y[bid] = s_y[0];
    // if(bid == gridDim.x - 1) printf("Last Block'%d: %lf\n", tid, s_y[tid]); 
}

extern __host__ real reduce2(const real *d_x, const int N) {
    const int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int y_bytes = sizeof(real) * GRID_SIZE;

    real *d_y;
    CHECK(cudaMalloc((void**)&d_y, y_bytes));

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    reduce_sharedmem<<<GRID_SIZE, BLOCK_SIZE, sizeof(real) * BLOCK_SIZE>>>(d_x, d_y, N);
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

    // int cnt = 0, minus = 0, plus = 0;
    // for (int i = 0; i < GRID_SIZE; i++) {
    //     if (h_y[i] <= 313.0 || h_y[i] >= 315.0) {
    //         if (cnt < 100)  printf("Block %d WA: %lf!", i, h_y[i]);
    //         else if (h_y[i] <= 313.0)   ++minus;
    //         else if (h_y[i] >= 315.0)   ++plus;
    //         if (true) {
    //             int beg = i * BLOCK_SIZE;
    //             real sx[BLOCK_SIZE]; CHECK(cudaMemcpy(sx, d_x+beg, sizeof(real)*BLOCK_SIZE, cudaMemcpyDeviceToHost));
    //             for (int j = 1; j < BLOCK_SIZE; ++j) {
    //                 if (sx[j] != 1.23) 
    //                     printf("d_x elem id %d: %lf", beg+j, sx[j]);
    //             }
    //         }
    //         ++cnt;
    //     }
        
    // }
    // printf("\n%d: -%d +%d\n", cnt, minus, plus);

    cudaFree(d_y);
    cudaFreeHost(h_y);

    return ret;      
}