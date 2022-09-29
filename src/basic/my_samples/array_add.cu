#include <stdio.h>
#include "error.h"

__global__ void intAdd(const int *a, const int *b, int *c) {
  int tx = threadIdx.x;
  c[tx] = a[tx] + b[tx];
}

__host__ void gpuIntAdd(const int a, const int b, int *c, const unsigned int len) {
  int *dev_a, *dev_b, *dev_c; // device copies of a, b, c

  const unsigned int size = sizeof(int) * len; // bytes for and integer

  // allocate device copies of a, b, c
  HANDLE_ERROR(cudaMalloc((void**)&dev_a, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_b, size));
  HANDLE_ERROR(cudaMalloc((void**)&dev_c, size));

  // copy inputs to device
//   HANDLE_ERROR(cudaMemcpy(dev_a, &a, size, cudaMemcpyHostToDevice));
//   HANDLE_ERROR(cudaMemcpy(dev_b, &b, size, cudaMemcpyHostToDevice));

  // launch kernel intAdd()
  intAdd<<< 1, len >>>(dev_a, dev_b, dev_c);

  // copy device result back to host copy of c
  HANDLE_ERROR(cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost));

  // free device
  HANDLE_ERROR(cudaFree(dev_a));
  HANDLE_ERROR(cudaFree(dev_b));
  HANDLE_ERROR(cudaFree(dev_c));
}