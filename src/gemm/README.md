矩阵乘法的重要性不言而喻，很多计算环节最终都可以用矩阵乘法来表达。
CUDA程序性能优化的第一手材料：[官方 CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
1. 学习了一份开源的GEMM性能优化流程[https://github.com/yzhaiustc/Optimizing-SGEMM-on-NVIDIA-Turing-GPUs]，同作者还有开源针对AVX512指令集的优化代码仓库
2. 从V100计算卡开始，NVIDIA引入了Tensor Core来加速矩阵乘法，只有CUBLAS和CUDNN提供了相关API。从短期来看，厂商和用户都获得了巨大性能收益，但是这打破了之前成熟的SIMT计算模型，长期影响有待商榷。