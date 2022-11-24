#include "../tutor_book/error.cuh"
#define TILE_DIM 32

__global__ void simpleMultiply(float *a, float *c, int M)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < TILE_DIM; i++) {
        sum += a[row*TILE_DIM+i] * a[col*TILE_DIM+i];
    }
    c[row*M+col] = sum;
}

__global__ void coalescedMultiply(float *a, float *c, int M)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM],
                     transposedTile[TILE_DIM][TILE_DIM];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    transposedTile[threadIdx.x][threadIdx.y] =
        a[(blockIdx.x*blockDim.x + threadIdx.y)*TILE_DIM +
        threadIdx.x];  // 这个转置的代码其实通用性很差，不过懒得改了
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* transposedTile[i][threadIdx.x];
    }
    c[row*M+col] = sum;
}

__global__ void nobankConflict(float *a, float *c, int M)
{
    __shared__ float aTile[TILE_DIM][TILE_DIM],
                     transposedTile[TILE_DIM][TILE_DIM+1];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
    transposedTile[threadIdx.x][threadIdx.y] =
        a[(blockIdx.x*blockDim.x + threadIdx.y)*TILE_DIM +
        threadIdx.x];  
    __syncthreads();
    for (int i = 0; i < TILE_DIM; i++) {
        sum += aTile[threadIdx.y][i]* transposedTile[i][threadIdx.x];
    }
    c[row*M+col] = sum;
}

int main(int argc, char const *argv[])
{
    constexpr int M = TILE_DIM;

    float *d_a, *d_c;
    CHECK(cudaMalloc((void **)&d_a, M * M * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_c, M * M * sizeof(float)));

    dim3 grid_dim(M/32, M/32);
    dim3 block_dim(32, 32);

    // Warm Up
    for (size_t i = 0; i < 10; i++)
    {
        simpleMultiply<<<grid_dim, block_dim>>>(d_a, d_c, M);
    }
    
    cudaEvent_t start, stop;
    float time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord( start, 0 );    
    for (size_t i = 0; i < 10; i++)
    {
        simpleMultiply<<<grid_dim, block_dim>>>(d_a, d_c, M);
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    CHECK(cudaGetLastError());

    cudaEventElapsedTime( &time, start, stop );       
    printf("mm0 Used %fms\n", time / 10.0f);

    cudaEventRecord( start, 0 );    
    for (size_t i = 0; i < 10; i++)
    {
        coalescedMultiply<<<grid_dim, block_dim>>>(d_a, d_c, M);
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    CHECK(cudaGetLastError());

    cudaEventElapsedTime( &time, start, stop );       
    printf("mm1 Used %fms\n", time / 10.0f);    

    cudaEventRecord( start, 0 );    
    for (size_t i = 0; i < 10; i++)
    {
        nobankConflict<<<grid_dim, block_dim>>>(d_a, d_c, M);
    }
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    CHECK(cudaGetLastError());

    cudaEventElapsedTime( &time, start, stop );       
    printf("mm2 Used %fms\n", time / 10.0f);        

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_c));

    return 0;
}