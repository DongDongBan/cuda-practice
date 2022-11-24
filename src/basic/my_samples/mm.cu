#include "../tutor_book/error.cuh"
#define TILE_DIM 32

__global__ void simpleMultiply(float *a, float* b, float *c,
                               int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < TILE_DIM; i++) {
        sum += a[row*TILE_DIM+i] * b[i*N+col];
    }
    c[row*N+col] = sum;
}

__global__ void coalescedMultiply(float *a, float* b, float *c,
                                  int N)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    __syncwarp();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* b[i*N+col];
    }
    c[row*N+col] = sum;
}

__global__ void sharedABMultiply(float *a, float* b, float *c,
                                 int N)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM],
                     bTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N+col];
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
    }
    c[row*N+col] = sum;
}

int main(int argc, char const *argv[])
{
    constexpr int M = 4096, N = 4096;

    float *d_a, *d_b, *d_c;
    CHECK(cudaMalloc((void **)&d_a, M * TILE_DIM * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_b, N * TILE_DIM * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_c, M * N * sizeof(float)));

    dim3 grid_dim(M/32, N/32);
    dim3 block_dim(32, 32);

    // Warm Up
    for (size_t i = 0; i < 10; i++)
    {
        simpleMultiply<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
    }
    
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord( start, 0 );    
    for (size_t i = 0; i < 10; i++)
    {
        simpleMultiply<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    CHECK(cudaGetLastError());

    cudaEventElapsedTime( &time, start, stop );       
    printf("mm0 Used %fms\n", time / 10.0f);

    cudaEventRecord( start, 0 );    
    for (size_t i = 0; i < 10; i++)
    {
        coalescedMultiply<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    CHECK(cudaGetLastError());

    cudaEventElapsedTime( &time, start, stop );       
    printf("mm1 Used %fms\n", time / 10.0f);    

    cudaEventRecord( start, 0 );    
    for (size_t i = 0; i < 10; i++)
    {
        sharedABMultiply<<<grid_dim, block_dim>>>(d_a, d_b, d_c, N);
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    CHECK(cudaGetLastError());

    cudaEventElapsedTime( &time, start, stop );       
    printf("mm2 Used %fms\n", time / 10.0f);        

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    return 0;
}
