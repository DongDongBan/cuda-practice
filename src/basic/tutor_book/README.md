这本书用来CUDA入门还是不错的，不足之处就是作者不是CS科班出身，缺乏对于GPU体系结构的深刻理解。
我只选录了对我有借鉴意义的源代码，想系统学习的请移步[原仓库](https://github.com/brucefan1983/CUDA-Programming)。

1. 只适用全局内存的reduce
2. 使用共享内存的reduce
3. 使用原子函数的reduce
4. 使用束内同步函数的reduce
5. 使用洗牌函数的reduce
6. 使用协作组的reduce
7. 增大上一步reduce中的线程利用率
8. 与Thrust中的reduce性能比较