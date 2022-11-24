#include <cstdio>

using std::printf;

__device__ void helloGPUDevice(void) {
  printf("[gpu]> Hello world! (device)\n");
}

__global__ void helloGPU(void) {
  printf("[gpu]> Hello world! (global)\n");
  helloGPUDevice();
}

__host__ void helloCPUFromHost(void) {
  printf("[cpu]> Hello world! (host)\n");
}

void helloCPU(void) {
  printf("[cpu]> Hello world! (normal)\n");
  helloCPUFromHost();
}

int main() {

  helloGPU<<< 1, 1 >>>();
  cudaDeviceSynchronize();

  helloCPU();

  return 0;
}
