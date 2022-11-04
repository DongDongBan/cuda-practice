#include "reduce.cuh"
#include <thrust/device_vector.h>
#include <cstdio>


extern __host__ real reduce8(real *d_x, const int N) {
    thrust::device_ptr<real> dev_ptr(d_x);

    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    real ans = thrust::reduce(dev_ptr, dev_ptr+N);
    cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
    float elapsedTime; cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Kernel with input size %d consumes %f ms.\n", N, elapsedTime);

    return ans;
}