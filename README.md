# Parallel-programming-Homework
南开大学2022春 并行程序设计作业。
Homework1是cache与流水线优化，接下来的实验均为对ANN的优化。
ANN.cpp中ANN为原始类，仅有一个隐藏层，SIMD实验在此基础上进行优化，实现了SSE，AVX，NOEN。
2022/5/2 对ANN类进行改进，ANN2.cpp为改进后的类，可指定任意多隐藏层;训练时可指定batch_size与epoch_num;使用独立的Layer，具见Layer.cpp。pthread实验在此基础上完成，也同时实现了单独的sse的SIMD优化与pthread+sse优化。
