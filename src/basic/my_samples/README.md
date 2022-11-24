1. hello_world 第一个样例程序
2. l2cache 来自官方 Best Practice调优L2缓存参数（还有一些BUG待处理，未达到官方文档中的加速效果！）
3. mm 来自官方 Best Practice的矩阵乘法优化与带宽测量(与文档中的表现不一致，sharedABMultiply反而比coalescedMultiply性能要更低了……原因待探究！)
4. mmT-variant 来自官方 Best Practice的矩阵乘法优化与带宽测量
5. asyncsharedmem 来自官方 Best Practice比较同/异步加载到共享内存的性能（核函数实现还有一些BUG待处理，效果未知！）

**还有一些部署兼容性检测API的样例之后再补充！**
