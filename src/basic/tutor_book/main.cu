#include <iostream>
#include "reduce.cuh"
#include "error.cuh"
#include <chrono> // 简单性能测量用

__global__ void fill_with_value(real *p, const real val, const int N) {
    int stride = gridDim.x * blockDim.x;
    for (int n = threadIdx.x + blockIdx.x*blockDim.x; n < N; n += stride) 
        p[n] = val;
}

void test_kernelnum(int k, real *d_x, const int N) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    real ret;
    switch (k)
    {
        case 1: ret = reduce1(d_x, N); break;
        case 2: ret = reduce2(d_x, N); break;
        case 3: ret = reduce3(d_x, N); break;
        case 4: ret = reduce4(d_x, N); break;
        case 5: ret = reduce5(d_x, N); break;
        case 6: ret = reduce6(d_x, N); break;
        case 7: ret = reduce7(d_x, N); break;
        case 8: ret = reduce8(d_x, N); break;    
    }
    std::chrono::duration<double> diff = high_resolution_clock::now() - start;
    std::cout << "reduce" << k << " returns " << ret
              << " takes " << diff.count() << " s.\n";
    
    // Reset d_x device array
    fill_with_value<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_x, 1.23, N);
    CHECK(cudaGetLastError());
}

int main(int argc, char const *argv[])
{
    constexpr int N = 1e8;
    real *d_x;
    CHECK(cudaMalloc((void**)&d_x, sizeof(real) * N));
    fill_with_value<<<N/BLOCK_SIZE, BLOCK_SIZE>>>(d_x, 1.23, N);
    CHECK(cudaGetLastError());

    for (size_t i = 1; i <= 8; i++)
        test_kernelnum(i, d_x, N);

    cudaFree(d_x);    
    return 0;
}
