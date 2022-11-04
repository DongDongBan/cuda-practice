#ifdef USE_DOUBLE
    using real = double;
#else
    using real = float;
#endif

constexpr unsigned int FULL_MASK = 0xffffffff; // 线程束函数默认MASK
constexpr int BLOCK_SIZE = 256; // CUDA 编程指南中的推荐值

// 多版本REDUCE核函数例程的包装接口
// 注意reduce1,8的参数与后续的const参数不一致
extern __host__ real reduce1(real *d_x, const int N);
extern __host__ real reduce2(const real *d_x, const int N);
extern __host__ real reduce3(const real *d_x, const int N);
extern __host__ real reduce4(const real *d_x, const int N);
extern __host__ real reduce5(const real *d_x, const int N);
extern __host__ real reduce6(const real *d_x, const int N);
extern __host__ real reduce7(const real *d_x, const int N);
extern __host__ real reduce8(real *d_x, const int N);
